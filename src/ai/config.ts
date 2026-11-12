import "dotenv/config";

export interface AgentConfig {
  ollamaBaseUrl: string;
  ollamaModel: string;
  agentId: string;
  agentEmail: string;
  agentName: string;
  agentJwt: string | null;
  aiApiKey: string | null;
  promptStyle: string;
  nluMaxTokens: number;
  reasoningMaxTokens: number;
  llmTimeoutMs: number;
  maxDbRetries: number;
  dbRetryBaseDelayMs: number;
  shutdownTimeoutMs: number;
  heartbeatIntervalMs: number;
  logLevel: string;
}

const PROMPT_STYLES = ["conservative", "standard", "creative"] as const;

function parsePositiveInt(
  value: string | undefined,
  fallback: number,
  name: string
): number {
  if (value === undefined || value.trim() === "") return fallback;
  const parsed = Number.parseInt(value, 10);
  if (Number.isNaN(parsed) || parsed <= 0) {
    throw new Error(
      `[CONFIG] Invalid ${name}: expected a positive integer, got "${value}".`
    );
  }
  return parsed;
}

function deriveAgentEmail(env: NodeJS.ProcessEnv): string {
  if (env.AGENT_EMAIL && env.AGENT_EMAIL.trim() !== "") {
    return env.AGENT_EMAIL.trim();
  }
  if (env.APP_SUPPORT_EMAIL) {
    const domain = env.APP_SUPPORT_EMAIL.match(/@([^@\s>]+)/);
    if (domain) return `agent@${domain[1]}`;
  }
  return "agent@arinova.local";
}

function validateJwt(value: string): boolean {
  const jwtPattern = /^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$/;
  return jwtPattern.test(value);
}

export function loadConfig(
  env: NodeJS.ProcessEnv = process.env
): AgentConfig {
  const ollamaBaseUrl = (env.OLLAMA_BASE_URL?.trim() || "http://localhost:11434").replace(/\/+$/, "");
  const ollamaModel = env.OLLAMA_MODEL?.trim() || "llama3";
  const agentId = env.AGENT_ID?.trim() || "agent-service-001";
  const promptStyle = env.AGENT_PROMPT_STYLE?.trim() || "conservative";

  if (!env.DB_URI || env.DB_URI.trim() === "") {
    throw new Error(
      "[CONFIG] Required environment variable DB_URI is missing. The agent cannot start without a database connection."
    );
  }
  if (!/^https?:\/\//.test(ollamaBaseUrl)) {
    throw new Error(
      `[CONFIG] Invalid OLLAMA_BASE_URL "${env.OLLAMA_BASE_URL}": it must start with http:// or https://.`
    );
  }
  if (ollamaModel === "") {
    throw new Error("[CONFIG] OLLAMA_MODEL must not be empty.");
  }
  if (agentId === "") {
    throw new Error("[CONFIG] AGENT_ID must not be empty.");
  }
  if (!(PROMPT_STYLES as readonly string[]).includes(promptStyle)) {
    throw new Error(
      `[CONFIG] Invalid AGENT_PROMPT_STYLE "${promptStyle}". Allowed values: ${PROMPT_STYLES.join(", ")}.`
    );
  }

  const agentJwt = env.AGENT_JWT?.trim() || null;
  if (agentJwt !== null && !validateJwt(agentJwt)) {
    throw new Error(
      "[CONFIG] Invalid AGENT_JWT: expected a full JWT (header.payload.signature). Generate one or remove AGENT_JWT to have it created at startup."
    );
  }

  return {
    ollamaBaseUrl,
    ollamaModel,
    agentId,
    agentEmail: deriveAgentEmail(env),
    agentName: env.AGENT_NAME?.trim() || "Project Agent",
    agentJwt,
    aiApiKey: env.AI_API_KEY?.trim() || null,
    promptStyle,
    nluMaxTokens: parsePositiveInt(env.NLU_MAX_TOKENS, 500, "NLU_MAX_TOKENS"),
    reasoningMaxTokens: parsePositiveInt(env.REASONING_MAX_TOKENS, 1000, "REASONING_MAX_TOKENS"),
    llmTimeoutMs: parsePositiveInt(env.LLM_TIMEOUT_MS, 60000, "LLM_TIMEOUT_MS"),
    maxDbRetries: parsePositiveInt(env.AGENT_DB_MAX_RETRIES, 3, "AGENT_DB_MAX_RETRIES"),
    dbRetryBaseDelayMs: parsePositiveInt(env.AGENT_DB_RETRY_BASE_DELAY_MS, 1000, "AGENT_DB_RETRY_BASE_DELAY_MS"),
    shutdownTimeoutMs: parsePositiveInt(env.AGENT_SHUTDOWN_TIMEOUT_MS, 30000, "AGENT_SHUTDOWN_TIMEOUT_MS"),
    heartbeatIntervalMs: parsePositiveInt(env.AGENT_HEARTBEAT_INTERVAL_MS, 300000, "AGENT_HEARTBEAT_INTERVAL_MS"),
    logLevel: env.AGENT_LOG_LEVEL?.trim() || "info",
  };
}