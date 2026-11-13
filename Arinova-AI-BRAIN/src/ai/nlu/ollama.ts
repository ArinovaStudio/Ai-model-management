import { loadConfig } from "@/ai/config";
import { logger } from "@/ai/logger";

const MAX_ATTEMPTS = 2;
const RETRY_DELAY_MS = 1000;

export interface LlmRequest {
  system: string;
  prompt: string;
  formatJson?: boolean;
  maxTokens?: number;
}

export interface LlmResponse {
  text: string;
  model: string;
}

export type LlmErrorKind = "timeout" | "network" | "http" | "json";

export class LlmError extends Error {
  constructor(
    message: string,
    public readonly kind: LlmErrorKind
  ) {
    super(message);
    this.name = "LlmError";
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function classifyError(error: unknown, timeoutMs: number): LlmError {
  if (error instanceof LlmError) return error;
  if (error instanceof Error && error.name === "AbortError") {
    return new LlmError(
      `Ollama request timed out after ${timeoutMs}ms`,
      "timeout"
    );
  }
  return new LlmError(
    `Ollama request failed: ${error instanceof Error ? error.message : String(error)}`,
    "network"
  );
}

export async function callOllama(request: LlmRequest): Promise<LlmResponse> {
  const config = loadConfig();
  const url = `${config.ollamaBaseUrl}/api/chat`;
  const payload = {
    model: config.ollamaModel,
    messages: [
      { role: "system", content: request.system },
      { role: "user", content: request.prompt },
    ],
    stream: false,
    options: {
      temperature: 0,
      num_predict: request.maxTokens ?? config.nluMaxTokens,
    },
    ...(request.formatJson === false ? {} : { format: "json" as const }),
  };

  let lastError: LlmError | null = null;

  for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt += 1) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), config.llmTimeoutMs);
    try {
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });
      clearTimeout(timeout);

      if (!response.ok) {
        throw new LlmError(
          `Ollama responded with HTTP ${response.status}`,
          "http"
        );
      }

      let body: { message?: { content?: string }; model?: string };
      try {
        body = (await response.json()) as {
          message?: { content?: string };
          model?: string;
        };
      } catch {
        throw new LlmError("Ollama returned a non-JSON response", "json");
      }

      const text = body.message?.content ?? "";
      if (text.trim() === "") {
        throw new LlmError("Ollama returned an empty response", "json");
      }

      logger.info(
        `[NLU][OLLAMA] Model "${body.model ?? config.ollamaModel}" returned ${text.length} chars (attempt ${attempt}/${MAX_ATTEMPTS}).`
      );
      return { text, model: body.model ?? config.ollamaModel };
    } catch (error) {
      clearTimeout(timeout);
      lastError = classifyError(error, config.llmTimeoutMs);
      const willRetry =
        attempt < MAX_ATTEMPTS &&
        (lastError.kind === "timeout" ||
          lastError.kind === "network" ||
          lastError.kind === "http");
      if (willRetry) {
        logger.warn(
          `[NLU][OLLAMA] Attempt ${attempt}/${MAX_ATTEMPTS} failed (${lastError.kind}: ${lastError.message}). Retrying in ${RETRY_DELAY_MS}ms.`
        );
        await sleep(RETRY_DELAY_MS);
      } else {
        logger.warn(
          `[NLU][OLLAMA] Attempt ${attempt}/${MAX_ATTEMPTS} failed (${lastError.kind}: ${lastError.message}).`
        );
      }
    }
  }

  throw lastError ?? new LlmError("Ollama request failed", "network");
}