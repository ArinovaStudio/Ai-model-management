import { TASK_STATUS_VALUES } from "./schemas";
import type { ParsedUpdate } from "./schemas";

export interface NluContext {
  taskId?: string;
  taskTitle?: string;
  projectId?: string;
  knownUsers?: { id: string; name: string }[];
}

export const NLU_SYSTEM_PROMPT = [
  "You convert an employee's informal status update into a single JSON object.",
  "Extract only information that is clearly present or strongly implied by the update.",
  "If a field cannot be confidently extracted, set it to null. Never invent task IDs, user IDs, percentages, or statuses.",
  `Task status must be exactly one of: ${TASK_STATUS_VALUES.join(", ")}.`,
  'Set "blocker" to true only when the update states or clearly implies work is blocked; otherwise false or null if unclear.',
  'Set "blockedByUserId" to the exact user ID from the team list only when the update clearly names that person; otherwise set it to null.',
  'Set "taskId" to an exact task ID only when the update clearly references that task; otherwise set it to null.',
  'Rate "confidence" from 0 to 1 (0.9 for a clear update, 0.4 for a vague or conflicting one).',
  'Respond with a single JSON object and nothing else, using exactly this shape:',
  '{"taskId": string|null, "progressPercent": number|null, "status": "pending"|"assigned"|"in-progress"|"completed"|null, "blocker": boolean|null, "blockerDescription": string|null, "blockedByUserId": string|null, "nextSteps": string|null, "confidence": number}',
].join("\n");

export function buildUserPrompt(message: string, context: NluContext): string {
  const taskLine = context.taskId
    ? `Task reference: id="${context.taskId}"${
        context.taskTitle ? `, title="${context.taskTitle}"` : ""
      }.`
    : "No task reference was provided; use a task ID only if the update explicitly names one.";
  const teamLine =
    context.knownUsers && context.knownUsers.length > 0
      ? `Team (user IDs for "blockedByUserId"): ${context.knownUsers
          .map((user) => `${user.name} -> ${user.id}`)
          .join("; ")}.`
      : 'No team list provided; always set "blockedByUserId" to null.';
  const example = [
    'Example update: "checkout flow is stuck, waiting on Priya for the API keys, 60% done"',
    'Example output: {"taskId": null, "progressPercent": 60, "status": "in-progress", "blocker": true, "blockerDescription": "Waiting on API keys from Priya", "blockedByUserId": null, "nextSteps": null, "confidence": 0.8}',
    "",
  ].join("\n");
  return [
    taskLine,
    teamLine,
    example,
    `Employee status update: "${message}"`,
    "Return the JSON object.",
  ].join("\n");
}

export function buildClarificationPrompt(
  message: string,
  missingFields: string[]
): string {
  return [
    "Write ONE short, specific follow-up question for an employee whose status update was ambiguous.",
    `The employee wrote: "${message}"`,
    `The information that could not be extracted reliably: ${missingFields.join(
      ", "
    )}.`,
    "Focus on the single most important missing item. Keep the question under 20 words. Output only the question as plain text, with no quotes, prefixes, or labels.",
  ].join("\n");
}

export function buildClarificationQuestion(
  message: string,
  parsed: Partial<ParsedUpdate>
): string {
  const snippet = message.slice(0, 60);
  if (parsed.status === null || parsed.status === undefined) {
    return `What is the current status of "${snippet}"? Reply with one of: pending, assigned, in-progress, completed.`;
  }
  if (parsed.progressPercent === null || parsed.progressPercent === undefined) {
    return `What percent complete is "${snippet}"? Reply with a number from 0 to 100.`;
  }
  if (parsed.blocker) {
    return `What exactly is blocking "${snippet}", and who or what are you waiting for?`;
  }
  if (parsed.nextSteps === null || parsed.nextSteps === undefined) {
    return `What are your next steps for "${snippet}"?`;
  }
  return `Could you clarify what you meant by "${snippet}"?`;
}