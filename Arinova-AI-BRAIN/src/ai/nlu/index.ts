import db from "@/lib/client";
import { logger } from "@/ai/logger";
import { circuitBreaker } from "@/ai/circuit-breaker";
import { startSpan } from "@/ai/telemetry";
import { callOllama, LlmError } from "./ollama";
import {
  NLU_SYSTEM_PROMPT,
  buildUserPrompt,
  buildClarificationPrompt,
  buildClarificationQuestion,
  type NluContext,
} from "./prompts";
import {
  extractJsonObject,
  llmUpdateSchema,
  parsedUpdateSchema,
  type NluDeferReason,
  type NluResult,
  type ParsedUpdate,
} from "./schemas";

const PARSE_CONFIDENCE = 0.8;
const CLARIFY_CONFIDENCE = 0.5;

const PENALTY_TASK_MISSING = 0.2;
const PENALTY_STATUS_MISSING = 0.25;
const PENALTY_PROGRESS_MISSING = 0.1;
const PENALTY_NEXT_STEPS_MISSING = 0.05;
const PENALTY_BLOCKER_STATE_UNKNOWN = 0.05;
const PENALTY_BLOCKER_NO_DETAIL = 0.15;
const PENALTY_BLOCKER_LINK_UNVERIFIED = 0.05;
const PENALTY_UNKNOWN_TASK_REFERENCE = 0.3;
const PENALTY_CONFLICTING_UPDATE = 0.3;

function clamp01(value: number): number {
  return Math.min(1, Math.max(0, value));
}

function round2(value: number): number {
  return Math.round(value * 100) / 100;
}

function formatError(error: unknown): string {
  return error instanceof Error ? `${error.name}: ${error.message}` : String(error);
}

function shorten(text: string, maxLength = 400): string {
  const trimmed = text.trim();
  return trimmed.length > maxLength
    ? `${trimmed.slice(0, maxLength)}...`
    : trimmed;
}

function sanitizeMeta(value: unknown): unknown {
  if (value === null || value === undefined) return value;
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return value;
  }
  if (typeof value === "object") {
    if (Array.isArray(value)) return value.map(sanitizeMeta);
    const result: Record<string, unknown> = {};
    for (const [key, item] of Object.entries(value)) {
      result[key] = sanitizeMeta(item);
    }
    return result;
  }
  return String(value);
}

async function storeLowConfidenceUpdate(
  message: string,
  reason: NluDeferReason,
  projectId: string | null,
  meta: Record<string, unknown>
): Promise<string | null> {
  try {
    const record = await db.agentMemory.create({
      data: {
        projectId,
        type: "low_confidence_update",
        content: message,
        meta: sanitizeMeta({ reason, ...meta }) as object,
      },
    });
    logger.info(
      `[NLU][STEP3] Stored low-confidence record "${record.id}" (type=low_confidence_update) for human review.`
    );
    return record.id;
  } catch (error) {
    logger.error(
      `[NLU][STEP3] Failed to store low-confidence update in AgentMemory: ${formatError(error)}`
    );
    return null;
  }
}

function deferSafely(
  message: string,
  reason: NluDeferReason,
  projectId: string | null,
  meta: Record<string, unknown>,
  reasonPhrase: string
): Promise<NluResult> {
  return storeLowConfidenceUpdate(message, reason, projectId, meta).then(
    (recordId) => {
      logger.warn(
        `[NLU][STEP3] Deferred to human review (${reason}: ${reasonPhrase}). ` +
          `Raw message stored, task state NOT updated.`
      );
      return { status: "deferred", step: 3, reason, recordId };
    }
  );
}

async function verifyTaskReference(
  taskIdFromMessage: string | null,
  contextTaskId: string | null,
  basis: string[],
  penalty: number
): Promise<{ resolvedTaskId: string | null; penalty: number }> {
  if (taskIdFromMessage === null || taskIdFromMessage === contextTaskId) {
    return { resolvedTaskId: taskIdFromMessage ?? contextTaskId, penalty };
  }
  try {
    const task = await db.task.findUnique({
      where: { id: taskIdFromMessage },
      select: { id: true },
    });
    if (!task) {
      basis.push(
        `update references task "${taskIdFromMessage}", which does not exist`
      );
      return { resolvedTaskId: contextTaskId, penalty: penalty + PENALTY_UNKNOWN_TASK_REFERENCE };
    }
  } catch (error) {
    logger.warn(
      `[NLU] Task existence check failed (${formatError(error)}); skipping penalty.`
    );
  }
  return { resolvedTaskId: taskIdFromMessage, penalty };
}

export async function processEmployeeUpdate(
  message: string,
  taskId?: string | null,
  context?: NluContext
): Promise<NluResult> {
  const trimmed = (message ?? "").trim();
  const mergedContext: NluContext = {
    ...(context ?? {}),
    taskId: taskId ?? context?.taskId,
  };
  const contextTaskId = mergedContext.taskId ?? null;
  const span = startSpan("parse_employee_update", "nlu", {
    projectId: mergedContext.projectId ?? null,
    taskId: contextTaskId,
    attributes: { messageLength: message.length },
  });

  const circuitOk = await circuitBreaker.shouldAllow("read");
  if (!circuitOk) {
    const snapshot = await circuitBreaker.state;
    logger.warn(
      `[NLU] Skipped parsing because the circuit is ${snapshot.state.toUpperCase()}; update cannot be safely processed.`
    );
    span.end("CANCELLED", "circuit_open", { status: "deferred", reason: "circuit_breaker" });
    return {
      status: "deferred",
      step: 3,
      reason: "circuit_breaker",
      recordId: null,
    };
  }

  if (trimmed === "") {
    logger.warn(
      "[NLU][STEP3] Deferred to human review (empty_message: message was blank or whitespace-only). Task state NOT updated."
    );
    span.end("CANCELLED", "empty_message", { status: "deferred", reason: "empty_message" });
    return { status: "deferred", step: 3, reason: "empty_message", recordId: null };
  }

  const knownUserIds = new Set((mergedContext.knownUsers ?? []).map((user) => user.id));

  let llmText: string;
  try {
    const response = await callOllama({
      system: NLU_SYSTEM_PROMPT,
      prompt: buildUserPrompt(trimmed, mergedContext),
    });
    llmText = response.text;
  } catch (error) {
    logger.error(
      `[NLU] Ollama parsing failed (${formatError(error)}). Falling back to Step 3.`
    );
    if (error instanceof LlmError && (error.kind === "timeout" || error.kind === "network" || error.kind === "http")) {
      await circuitBreaker.recordFailure("llm_timeout", { reason: error.message });
    } else {
      await circuitBreaker.recordFailure("nlu_parse", { reason: formatError(error) });
    }
    return deferSafely(
      trimmed,
      "llm_error",
      mergedContext.projectId ?? null,
      { taskId: contextTaskId, error: formatError(error) },
      "the language model was unavailable or failed"
    );
  }

  const raw = extractJsonObject(llmText);
  if (raw === null) {
    logger.error(
      "[NLU] The language model returned no parseable JSON. Falling back to Step 3."
    );
    await circuitBreaker.recordFailure("nlu_parse", { reason: "no parseable JSON in model output" });
    return deferSafely(
      trimmed,
      "invalid_json",
      mergedContext.projectId ?? null,
      { taskId: contextTaskId, llmText: shorten(llmText) },
      "raw model output is not valid JSON"
    );
  }

  const llmParsed = llmUpdateSchema.safeParse(raw);
  if (!llmParsed.success) {
    logger.error(
      `[NLU] Model output failed Zod validation (${llmParsed.error.issues.length} issue(s)). Falling back to Step 3.`
    );
    await circuitBreaker.recordFailure("nlu_parse", { reason: "Zod schema validation failed" });
    return deferSafely(
      trimmed,
      "schema_validation",
      mergedContext.projectId ?? null,
      {
        taskId: contextTaskId,
        issues: llmParsed.error.issues.map(
          (issue) => `${issue.path.join(".")}: ${issue.message}`
        ),
      },
      "model output does not match the ParsedUpdate schema"
    );
  }

  const parsedRaw = llmParsed.data;
  const basis: string[] = [];
  let penalty = 0;
  let resolvedTaskId = parsedRaw.taskId ?? contextTaskId;

  const taskCheck = await verifyTaskReference(
    parsedRaw.taskId,
    contextTaskId,
    basis,
    0
  );
  resolvedTaskId = taskCheck.resolvedTaskId;
  penalty = taskCheck.penalty;

  if (resolvedTaskId === null) {
    basis.push("task could not be identified from the update or context");
    penalty += PENALTY_TASK_MISSING;
  }
  if (parsedRaw.status === null) {
    basis.push("task status could not be extracted");
    penalty += PENALTY_STATUS_MISSING;
  }
  if (parsedRaw.progressPercent === null) {
    basis.push("progress percentage could not be extracted");
    penalty += PENALTY_PROGRESS_MISSING;
  }
  if (parsedRaw.nextSteps === null) {
    basis.push("next steps could not be extracted");
    penalty += PENALTY_NEXT_STEPS_MISSING;
  }
  if (parsedRaw.blocker === null) {
    basis.push("blocker state could not be determined");
    penalty += PENALTY_BLOCKER_STATE_UNKNOWN;
  }
  if (parsedRaw.blocker === true && parsedRaw.blockerDescription === null) {
    basis.push("update is blocked but gives no description");
    penalty += PENALTY_BLOCKER_NO_DETAIL;
  }
  if (parsedRaw.blocker === true && parsedRaw.status === "completed") {
    basis.push("conflicting update: status is completed while work is blocked");
    penalty += PENALTY_CONFLICTING_UPDATE;
  }

  let blockedByUserId = parsedRaw.blockedByUserId;
  if (blockedByUserId !== null && !knownUserIds.has(blockedByUserId)) {
    basis.push(
      `blockedByUserId "${blockedByUserId}" is not a member of the known team`
    );
    blockedByUserId = null;
    penalty += PENALTY_BLOCKER_LINK_UNVERIFIED;
  }

  const llmConfidence = parsedRaw.confidence ?? 0;
  if (parsedRaw.confidence === null) {
    basis.push("the language model provided no confidence score");
  }

  const adjustedConfidence = round2(clamp01(llmConfidence - penalty));

  const parsed: ParsedUpdate = parsedUpdateSchema.parse({
    taskId: resolvedTaskId,
    progressPercent: parsedRaw.progressPercent,
    status: parsedRaw.status,
    blocker: parsedRaw.blocker,
    blockerDescription: parsedRaw.blockerDescription,
    blockedByUserId,
    nextSteps: parsedRaw.nextSteps,
    confidence: adjustedConfidence,
    basis,
  });

  if (adjustedConfidence >= PARSE_CONFIDENCE) {
    logger.info(
      `[NLU][STEP1] Parsed update with confidence ${adjustedConfidence} (>= ${PARSE_CONFIDENCE}). ` +
        `Returning structured update for the scheduler to apply.`
    );
    await circuitBreaker.recordSuccess();
    span.end("OK", null, { status: "parsed", confidence: adjustedConfidence, blocker: parsed.blocker });
    return { status: "parsed", step: 1, parsed };
  }

  if (adjustedConfidence >= CLARIFY_CONFIDENCE) {
    let question = buildClarificationQuestion(trimmed, parsed);
    try {
      const clarification = await callOllama({
        system: NLU_SYSTEM_PROMPT,
        prompt: buildClarificationPrompt(
          trimmed,
          basis.length > 0 ? basis : ["the overall confidence of the update"]
        ),
        formatJson: false,
      });
      const drafted = clarification.text.trim().replace(/^"|"$/g, "");
      if (drafted.length > 0 && drafted.length <= 160) {
        question = drafted;
      }
    } catch (error) {
      logger.warn(
        `[NLU][STEP2] Question phrasing via Ollama failed (${formatError(error)}); using a deterministic question.`
      );
    }
    logger.warn(
      `[NLU][STEP2] Clarify requested with confidence ${adjustedConfidence} (< ${PARSE_CONFIDENCE}). ` +
        `One follow-up question will be sent to the employee before the task state is updated.`
    );
    return { status: "clarify", step: 2, question, parsed };
  }

  logger.warn(
    `[NLU][STEP3] Confidence ${adjustedConfidence} < ${CLARIFY_CONFIDENCE}. ` +
      `Deferring update to human review (raw message stored, task state NOT updated).`
  );
  return deferSafely(
    trimmed,
    "low_confidence",
    mergedContext.projectId ?? null,
    { taskId: resolvedTaskId, confidence: adjustedConfidence, basis },
    `extracted confidence ${adjustedConfidence} is below the ${CLARIFY_CONFIDENCE} threshold`
  );
}