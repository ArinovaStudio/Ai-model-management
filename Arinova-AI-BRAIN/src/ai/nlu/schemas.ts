import { z } from "zod";

export const TASK_STATUS_VALUES = [
  "pending",
  "assigned",
  "in-progress",
  "completed",
] as const;

export const taskStatusSchema = z.enum(TASK_STATUS_VALUES);

export const llmUpdateSchema = z.object({
  taskId: z.string().trim().min(1).nullable(),
  progressPercent: z.coerce.number().min(0).max(100).nullable(),
  status: taskStatusSchema.nullable(),
  blocker: z.boolean().nullable(),
  blockerDescription: z.string().nullable(),
  blockedByUserId: z.string().trim().min(1).nullable(),
  nextSteps: z.string().nullable(),
  confidence: z.coerce.number().min(0).max(1).nullable(),
});

export type LlmUpdate = z.infer<typeof llmUpdateSchema>;

export const parsedUpdateSchema = z.object({
  taskId: z.string().nullable(),
  progressPercent: z.number().min(0).max(100).nullable(),
  status: taskStatusSchema.nullable(),
  blocker: z.boolean().nullable(),
  blockerDescription: z.string().nullable(),
  blockedByUserId: z.string().nullable(),
  nextSteps: z.string().nullable(),
  confidence: z.number().min(0).max(1),
  basis: z.array(z.string()),
});

export type ParsedUpdate = z.infer<typeof parsedUpdateSchema>;

export const clarificationSchema = z.object({
  question: z.string().min(1),
});

export type Clarification = z.infer<typeof clarificationSchema>;

export type NluDeferReason =
  | "empty_message"
  | "llm_error"
  | "invalid_json"
  | "schema_validation"
  | "low_confidence"
  | "circuit_breaker";

export type NluResult =
  | { status: "parsed"; step: 1; parsed: ParsedUpdate }
  | {
      status: "clarify";
      step: 2;
      question: string;
      parsed: Partial<ParsedUpdate>;
    }
  | {
      status: "deferred";
      step: 3;
      reason: NluDeferReason;
      recordId: string | null;
    };

function parseJsonSubstring(text: string): unknown | null {
  const start = text.indexOf("{");
  const end = text.lastIndexOf("}");
  if (start === -1 || end === -1 || end <= start) return null;
  try {
    return JSON.parse(text.slice(start, end + 1));
  } catch {
    return null;
  }
}

export function extractJsonObject(text: string): unknown | null {
  const trimmed = text.trim();
  if (trimmed === "") return null;
  try {
    return JSON.parse(trimmed);
  } catch {
    return parseJsonSubstring(trimmed);
  }
}