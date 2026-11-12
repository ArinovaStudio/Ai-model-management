import { z } from "zod";
import { logger } from "@/ai/logger";
import { callOllama, LlmError } from "@/ai/nlu/ollama";
import { circuitBreaker } from "@/ai/circuit-breaker";
import type { ProjectContext } from "@/ai/context/types";
import type { RiskAssessment, RiskNarrative } from "./types";

const riskNarrativeSchema = z.object({
  narrative: z.string().min(1),
  recommendations: z.array(z.string()),
});

export const RISK_ASSESSMENT_PROMPT = [
  "You are a project management advisor. A deterministic risk assessment was already computed from real data.",
  "Write a short human-readable narrative and give up to 3 concrete, specific recommendations.",
  "Only use the facts in the risk assessment. Do not invent dates, people, numbers, products, tools, or actions.",
  "Each recommendation must directly address one of the listed concerns and mention only what appears in the data.",
  "Respond with a single JSON object and nothing else, using exactly this shape:",
  '{"narrative": string, "recommendations": [string, string]}',
  "Keep the narrative under 120 words. Each recommendation must be one actionable sentence.",
].join("\n");

function buildRiskContext(
  context: ProjectContext,
  risk: RiskAssessment
): Record<string, unknown> {
  return {
    project: {
      name: context.project.name,
      progress: risk.progress,
      deadline: context.project.deadline,
      deadlineOverdue: context.project.deadlineOverdue,
    },
    risk: {
      level: risk.level,
      confidence: risk.confidence.score,
      concerns: risk.concerns.map((concern) => concern.label),
    },
    activeTaskCount: context.risks.activeTaskCount,
    blockedTaskCount: context.risks.blockedTaskCount,
    overdueTaskCount: context.risks.overdueTaskCount,
    topRiskBlockers: context.risks.topRiskBlockers.map((blocker) => ({
      title: blocker.title,
      severity: blocker.severity,
      status: blocker.status,
    })),
    teamCapacity: context.teamCapacity.map((member) => ({
      name: member.name,
      isLeader: member.isLeader,
      workloadPercent: member.workloadPercent,
    })),
  };
}

const GROUNDING_STOPWORDS = new Set([
  "with",
  "that",
  "this",
  "from",
  "they",
  "have",
  "them",
  "what",
  "will",
  "just",
  "make",
  "made",
  "more",
  "most",
  "then",
  "than",
  "into",
  "must",
  "your",
  "about",
  "being",
  "their",
  "there",
  "these",
  "those",
  "should",
  "would",
  "could",
  "also",
  "very",
  "task",
  "tasks",
  "task's",
  "team",
  "work",
  "time",
  "project",
  "currently",
  "already",
  "need",
  "needs",
  "action",
  "actions",
]);

function significantWords(value: string): Set<string> {
  const words = value.toLowerCase().match(/[a-z0-9]{4,}/g) ?? [];
  return new Set(words.filter((word) => !GROUNDING_STOPWORDS.has(word)));
}

const CONCERN_SYNONYMS: Record<string, string> = {
  deadline_overdue: "deadline overdue past due behind schedule",
  overdue_tasks: "overdue past due late behind schedule",
  blocked_tasks: "blocked blocking block dependencies dependency",
  unassigned_tasks: "unassigned assign assigned assignee unowned",
  team_capacity: "capacity workload overloaded overload bandwidth headcount",
  critical_blockers: "critical blocker blocking risk severity",
};

function synonymVocabulary(concerns: { id: string }[]): Set<string> {
  const vocabulary = new Set<string>();
  for (const concern of concerns) {
    const synonyms = CONCERN_SYNONYMS[concern.id];
    if (!synonyms) continue;
    for (const word of significantWords(synonyms)) vocabulary.add(word);
  }
  return vocabulary;
}

function isGrounded(recommendation: string, vocabulary: Set<string>): boolean {
  for (const word of significantWords(recommendation)) {
    if (vocabulary.has(word)) return true;
  }
  return false;
}

function groundRecommendations(
  recommendations: string[],
  groundingText: string,
  concerns: { id: string }[]
): string[] {
  const vocabulary = significantWords(groundingText);
  for (const word of synonymVocabulary(concerns)) vocabulary.add(word);
  const grounded: string[] = [];
  for (const recommendation of recommendations) {
    if (isGrounded(recommendation, vocabulary)) {
      grounded.push(recommendation);
    } else {
      logger.warn(
        `[REASONER][LLM] Dropped ungrounded recommendation (no overlap with the supplied data): "${recommendation}".`
      );
    }
  }
  return grounded;
}

function formatError(error: unknown): string {
  return error instanceof Error ? `${error.name}: ${error.message}` : String(error);
}

export async function assessRiskWithLlm(
  context: ProjectContext,
  risk: RiskAssessment
): Promise<RiskNarrative | null> {
    const riskContext = buildRiskContext(context, risk);
    const promptLines = [
      "Risk assessment (already computed deterministically):",
      JSON.stringify(riskContext),
      "Return the JSON object.",
    ];

    try {
      const response = await callOllama({
        system: RISK_ASSESSMENT_PROMPT,
        prompt: promptLines.join("\n"),
      });

    let parsed: unknown;
    try {
      parsed = JSON.parse(response.text);
    } catch {
      const start = response.text.indexOf("{");
      const end = response.text.lastIndexOf("}");
      if (start === -1 || end === -1 || end <= start) {
        logger.warn(
          "[REASONER][LLM] Risk narrative output was not valid JSON; falling back to deterministic explanation."
        );
        return null;
      }
      parsed = JSON.parse(response.text.slice(start, end + 1));
    }

    const validated = riskNarrativeSchema.safeParse(parsed);
    if (!validated.success) {
      logger.warn(
        `[REASONER][LLM] Risk narrative failed validation (issues: ${validated.error.issues.map((issue) => `${issue.path.join(".")}: ${issue.message}`).join("; ")}); falling back.`
      );
      return null;
    }

    return {
      narrative: validated.data.narrative,
      recommendations: groundRecommendations(
        validated.data.recommendations,
        JSON.stringify(riskContext),
        risk.concerns
      ),
      model: response.model,
    };
  } catch (error) {
    if (error instanceof LlmError && (error.kind === "timeout" || error.kind === "network" || error.kind === "http")) {
      await circuitBreaker.recordFailure("llm_timeout", { reason: error.message });
    } else {
      await circuitBreaker.recordFailure("nlu_parse", { reason: formatError(error) });
    }
    logger.warn(
      `[REASONER][LLM] Risk narrative generation failed (${formatError(error)}); falling back to deterministic explanation.`
    );
    return null;
  }
}