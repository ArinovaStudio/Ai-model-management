import "dotenv/config";
import crypto from "crypto";
import bcrypt from "bcryptjs";
import db from "@/lib/client";
import { createToken } from "@/lib/jwt";
import { loadConfig, type AgentConfig } from "./config";
import { logger } from "./logger";
import { EventListener, registerDefaultEventHandlers } from "./events";
import { initStealthPoliciesFromDb } from "./policy";

const CIRCUIT_BREAKER_SINGLETON_ID = "agent-circuit-breaker";
const SIGNALS: NodeJS.Signals[] = ["SIGTERM", "SIGINT"];

interface OllamaTag {
  name: string;
}

interface OllamaTagsResponse {
  models?: OllamaTag[];
}

let config: AgentConfig;
let shuttingDown = false;
let heartbeatTimer: NodeJS.Timeout | null = null;
let forceExitTimer: NodeJS.Timeout | null = null;
let listener: EventListener | null = null;

function formatError(error: unknown): string {
  return error instanceof Error ? `${error.name}: ${error.message}` : String(error);
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function connectDatabase(): Promise<void> {
  let attempt = 0;
  while (true) {
    attempt += 1;
    try {
      await db.$connect();
      await db.$queryRaw`SELECT 1`;
      logger.info(`Database connection established (attempt ${attempt}/${config.maxDbRetries}).`);
      return;
    } catch (error) {
      const isLastAttempt = attempt >= config.maxDbRetries;
      if (isLastAttempt) {
        logger.error(
          `[FATAL] Could not connect to the database after ${config.maxDbRetries} attempts. ` +
            `Error: ${formatError(error)}. Verify that DB_URI is set and the Neon host is reachable, then restart the agent.`
        );
        throw error;
      }
      const delay = config.dbRetryBaseDelayMs * 2 ** (attempt - 1);
      logger.warn(
        `Database connection attempt ${attempt}/${config.maxDbRetries} failed: ${formatError(error)}. ` +
          `Retrying in ${delay}ms (exponential backoff).`
      );
      await sleep(delay);
    }
  }
}

async function ensureServiceAccount(): Promise<void> {
  const existing = await db.user.findUnique({
    where: { id: config.agentId },
  });

  if (existing) {
    logger.info(
      `Service account found: "${existing.email}" (id: ${existing.id}).`
    );
    return;
  }

  const emailTaken = await db.user.findUnique({
    where: { email: config.agentEmail },
  });
  if (emailTaken) {
    throw new Error(
      `[FATAL] Cannot create service account: email "${config.agentEmail}" is already assigned to user ` +
        `"${emailTaken.id}". Set AGENT_EMAIL (and/or AGENT_ID) to unique values in the environment.`
    );
  }

  const randomPassword = crypto.randomBytes(24).toString("hex");
  const hashedPassword = await bcrypt.hash(randomPassword, 10);

  const created = await db.user.create({
    data: {
      id: config.agentId,
      email: config.agentEmail,
      name: config.agentName,
      password: hashedPassword,
      workingAs: "AI Agent",
      department: "AI",
      isDev: true,
      isLogin: false,
      employeeId: `agent-${config.agentId}`,
    },
  });

  logger.info(
    `Created service account: "${created.email}" (id: ${created.id}). ` +
      `A random password was set so it cannot be used for interactive login.`
  );
}

async function initCircuitBreaker(): Promise<void> {
  const existing = await db.circuitBreakerState.findUnique({
    where: { id: CIRCUIT_BREAKER_SINGLETON_ID },
  });

  if (existing) {
    logger.info(
      `Circuit breaker initialized: state="${existing.state}", failureCount=${existing.failureCount}.`
    );
    return;
  }

  await db.circuitBreakerState.create({
    data: {
      id: CIRCUIT_BREAKER_SINGLETON_ID,
      state: "closed",
      failureCount: 0,
    },
  });

  logger.info(
    `Circuit breaker initialized: state="closed", failureCount=0 (singleton row ${CIRCUIT_BREAKER_SINGLETON_ID}).`
  );
}

function modelIsAvailable(modelName: string, tags: OllamaTag[]): boolean {
  return tags.some(
    (tag) =>
      tag.name === modelName ||
      tag.name === `${modelName}:latest` ||
      tag.name.startsWith(`${modelName}:`)
  );
}

async function checkOllama(): Promise<boolean> {
  const tagsUrl = `${config.ollamaBaseUrl}/api/tags`;

  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), config.llmTimeoutMs);
    let response: Response;
    try {
      response = await fetch(tagsUrl, { signal: controller.signal });
    } finally {
      clearTimeout(timeout);
    }

    if (!response.ok) {
      logger.error(
        `[OLLAMA] Server at ${tagsUrl} responded with HTTP ${response.status}. ` +
          `Check that OLLAMA_BASE_URL is correct.`
      );
      return false;
    }

    const body = (await response.json()) as OllamaTagsResponse;
    const tags = body.models ?? [];

    if (tags.length === 0) {
      logger.error(
        `[OLLAMA] Server is running but no models are installed. Pull the model with: ollama pull ${config.ollamaModel}`
      );
      return false;
    }

    if (!modelIsAvailable(config.ollamaModel, tags)) {
      logger.error(
        `[OLLAMA] Model "${config.ollamaModel}" is not installed. Installed models: ${tags
          .map((tag) => tag.name)
          .join(", ")}. Pull it with: ollama pull ${config.ollamaModel}`
      );
      return false;
    }

    logger.info(
      `Ollama reachable at ${config.ollamaBaseUrl}; model "${config.ollamaModel}" is available.`
    );
    return true;
  } catch (error) {
    if (error instanceof Error && error.name === "AbortError") {
      logger.error(
        `[OLLAMA] Request to ${tagsUrl} timed out after ${config.llmTimeoutMs}ms. ` +
          `Ensure the server is running (ollama serve) and reachable.`
      );
    } else {
      logger.error(
        `[OLLAMA] Server is not reachable at ${config.ollamaBaseUrl}. ` +
          `Start it with "ollama serve" and verify OLLAMA_BASE_URL. Error: ${formatError(error)}`
      );
    }
    return false;
  }
}

async function resolveAgentJwt(): Promise<string> {
  if (config.agentJwt) {
    logger.info("Using AGENT_JWT provided in the environment.");
    return config.agentJwt;
  }
  if (!process.env.JWT_SECRET || process.env.JWT_SECRET.trim() === "") {
    logger.warn(
      "No AGENT_JWT and no JWT_SECRET: the agent cannot authenticate to internal APIs until one is configured."
    );
    return "";
  }
  const token = createToken({
    userId: config.agentId,
    email: config.agentEmail,
    role: "AGENT",
  });
  logger.info("Generated agent JWT from JWT_SECRET (role=AGENT).");
  return token;
}

function startHeartbeat(): void {
  const startedAt = Date.now();
  heartbeatTimer = setInterval(() => {
    const uptimeSeconds = Math.floor((Date.now() - startedAt) / 1000);
    const label = `Agent alive (uptime ${uptimeSeconds}s)`;
    void db.circuitBreakerState
      .findUnique({ where: { id: CIRCUIT_BREAKER_SINGLETON_ID } })
      .then((breaker) => {
        if (breaker) {
          logger.info(`${label}; circuit breaker state="${breaker.state}".`);
        } else {
          logger.info(label);
        }
      })
      .catch((error: unknown) => {
        logger.warn(`${label}; circuit breaker read failed: ${formatError(error)}`);
      });
  }, config.heartbeatIntervalMs);
}

function stopHeartbeat(): void {
  if (heartbeatTimer) {
    clearInterval(heartbeatTimer);
    heartbeatTimer = null;
  }
}

async function shutdown(signal: string, exitCode: number): Promise<void> {
  if (shuttingDown) return;
  shuttingDown = true;
  logger.info(`Received ${signal}. Starting graceful shutdown...`);

  forceExitTimer = setTimeout(() => {
    logger.error(
      `Graceful shutdown did not complete within ${config.shutdownTimeoutMs}ms. Forcing exit.`
    );
    process.exit(1);
  }, config.shutdownTimeoutMs);

  try {
    stopHeartbeat();
    if (listener) {
      await listener.stop();
    }
    logger.info("Stopped accepting scheduled work; draining in-flight operations...");
    await db.$disconnect();
    logger.info("Database connection closed cleanly. Agent stopped.");
    clearTimeout(forceExitTimer);
    process.exit(exitCode);
  } catch (error) {
    logger.error(`Error during shutdown: ${formatError(error)}`);
    clearTimeout(forceExitTimer);
    process.exit(1);
  }
}

async function main(): Promise<void> {
  config = loadConfig();
  logger.info(
    `AI agent bootstrap starting (id=${config.agentId}, model=${config.ollamaModel}, promptStyle=${config.promptStyle}).`
  );

  if (!config.aiApiKey) {
    logger.warn(
      "AI_API_KEY is not set: reasoning through the AI SDK will be limited; the agent will rely on Ollama."
    );
  }

  for (const signal of SIGNALS) {
    process.on(signal, () => {
      void shutdown(signal, 0);
    });
  }

  process.on("unhandledRejection", (reason: unknown) => {
    logger.error(`Unhandled promise rejection: ${formatError(reason)}`);
  });
  process.on("uncaughtException", (error: Error) => {
    logger.error(`Uncaught exception: ${formatError(error)}`);
    void shutdown("uncaughtException", 1);
  });

  await connectDatabase();
  await ensureServiceAccount();
  await initCircuitBreaker();
  await initStealthPoliciesFromDb();
  await resolveAgentJwt();

  const ollamaOk = await checkOllama();
  if (!ollamaOk) {
    logger.warn(
      "Agent is starting in DEGRADED mode: deterministic rules will be used and no autonomous LLM reasoning until Ollama is available."
    );
  }

  startHeartbeat();

  listener = new EventListener(config.agentId);
  registerDefaultEventHandlers(listener);
  await listener.start();

  logger.info(
    `🤖 AI agent is ready (id=${config.agentId}). Listening for scheduler and event triggers.`
  );
}

main().catch((error: unknown) => {
  logger.error(`Agent failed to start: ${formatError(error)}`);
  process.exit(1);
});