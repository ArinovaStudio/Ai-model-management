/** @jest-environment node */

// The LLM is mocked so tests never touch Ollama. `callOllama` is the single
// entrypoint used by both NLU and the reasoner's risk narrative.
jest.mock("@/ai/nlu/ollama", () => {
  const actual = jest.requireActual("@/ai/nlu/ollama");
  return { ...actual, callOllama: jest.fn() };
});

import db from "@/lib/client";
import { processEmployeeUpdate } from "@/ai/nlu";
import { callOllama, LlmError } from "@/ai/nlu/ollama";
import { circuitBreaker } from "@/ai/circuit-breaker";
import { makePrefix, cleanupTestData, createUser, resetBreaker } from "./setup";
import { seedProject, seedTask } from "./fixtures";

const mockCallOllama = jest.mocked(callOllama);

function llmReturn(update: unknown): { text: string; model: string } {
  return { text: JSON.stringify(update), model: "mock-llm" };
}

const FULL_UPDATE = {
  taskId: null,
  progressPercent: 80,
  status: "in-progress",
  blocker: false,
  blockerDescription: null,
  blockedByUserId: null,
  nextSteps: "write integration tests",
  confidence: 0.9,
};

describe("ai/nlu processEmployeeUpdate", () => {
  const prefix = makePrefix("nlu");
  let projectId: string;
  let taskId: string;
  let memberId: string;

  beforeAll(async () => {
    const project = await seedProject(prefix, "NLU Project");
    projectId = project.projectId;
    const member = await createUser(prefix, "dev");
    memberId = member.userId;
    const task = await seedTask(prefix, projectId, "api", {
      title: "Build the payment API",
      status: "in-progress",
      userId: memberId,
      assignee: "Dev One",
    });
    taskId = task.taskId;
  });

  beforeEach(async () => {
    mockCallOllama.mockReset();
    await resetBreaker();
  });

  afterAll(async () => {
    await cleanupTestData(prefix);
  });

  it("returns a parsed update at high confidence", async () => {
    mockCallOllama.mockResolvedValueOnce(llmReturn(FULL_UPDATE));

    const result = await processEmployeeUpdate(
      "Payment API is 80% done, in progress. Next: write integration tests.",
      taskId,
      { taskId, taskTitle: "Build the payment API", projectId, knownUsers: [{ id: memberId, name: "Dev One" }] }
    );

    expect(result.status).toBe("parsed");
    if (result.status !== "parsed") return;
    expect(result.step).toBe(1);
    expect(result.parsed.taskId).toBe(taskId);
    expect(result.parsed.progressPercent).toBe(80);
    expect(result.parsed.status).toBe("in-progress");
    expect(result.parsed.confidence).toBe(0.9);
  });

  it("requests clarification for a usable-but-ambiguous update", async () => {
    mockCallOllama
      .mockResolvedValueOnce(
        llmReturn({
          taskId: null,
          progressPercent: 50,
          status: "in-progress",
          blocker: false,
          blockerDescription: null,
          blockedByUserId: memberId,
          nextSteps: null,
          confidence: 0.75,
        })
      )
      .mockResolvedValueOnce({ text: "What are your next steps?", model: "mock-llm" });

    const result = await processEmployeeUpdate(
      "Halfway there, blocked by waiting on Priya's keys.",
      taskId,
      { taskId, taskTitle: "Build the payment API", projectId, knownUsers: [{ id: memberId, name: "Dev One" }] }
    );

    expect(result.status).toBe("clarify");
    if (result.status !== "clarify") return;
    expect(result.step).toBe(2);
    expect(result.question.length).toBeGreaterThan(0);
    expect(result.question.length).toBeLessThanOrEqual(160);
    expect(result.parsed.taskId).toBe(taskId);
  });

  it("defers a low-confidence update to human review and stores it", async () => {
    mockCallOllama.mockResolvedValueOnce(
      llmReturn({ ...FULL_UPDATE, progressPercent: 50, confidence: 0.4 })
    );

    const result = await processEmployeeUpdate(
      "Punched along a bit.",
      taskId,
      { taskId, taskTitle: "Build the payment API", projectId, knownUsers: [{ id: memberId, name: "Dev One" }] }
    );

    expect(result.status).toBe("deferred");
    if (result.status !== "deferred") return;
    expect(result.reason).toBe("low_confidence");
    expect(result.recordId).not.toBeNull();

    const memory = await db.agentMemory.findFirst({
      where: { projectId, type: "low_confidence_update" },
    });
    expect(memory).not.toBeNull();
    expect(memory!.content).toContain("Punched along a bit.");
  });

  it("defers an empty message", async () => {
    const result = await processEmployeeUpdate("   ", taskId);
    expect(result.status).toBe("deferred");
    if (result.status === "deferred") expect(result.reason).toBe("empty_message");
  });

  it("defers when the model returns no parseable JSON", async () => {
    mockCallOllama.mockResolvedValueOnce({ text: "sure thing, lets wrap it up", model: "mock-llm" });

    const result = await processEmployeeUpdate("Basically done.", taskId);
    expect(result.status).toBe("deferred");
    if (result.status === "deferred") expect(result.reason).toBe("invalid_json");
  });

  it("defers when the model output fails schema validation", async () => {
    mockCallOllama.mockResolvedValueOnce(llmReturn({ nonsense: true }));

    const result = await processEmployeeUpdate("Done, give me a new one.", taskId);
    expect(result.status).toBe("deferred");
    if (result.status === "deferred") expect(result.reason).toBe("schema_validation");
  });

  it("defers when the LLM call fails and records a timeout failure", async () => {
    mockCallOllama.mockRejectedValueOnce(new LlmError("Ollama unreachable", "network"));

    const result = await processEmployeeUpdate("35% done with the login flow.", taskId);
    expect(result.status).toBe("deferred");
    if (result.status === "deferred") expect(result.reason).toBe("llm_error");

    const snapshot = await circuitBreaker.state;
    expect(snapshot.failureCounts.llm_timeout ?? 0).toBeGreaterThanOrEqual(1);
  });

  it("falls back to the context task when the update references an unknown task", async () => {
    mockCallOllama.mockResolvedValueOnce(llmReturn({ ...FULL_UPDATE, taskId: "nonexistent-task-xyz" }));

    const result = await processEmployeeUpdate(
      "The payment API work is 80% done.",
      taskId,
      { taskId, taskTitle: "Build the payment API", projectId, knownUsers: [{ id: memberId, name: "Dev One" }] }
    );

    expect(result.status).toBe("clarify");
    if (result.status === "clarify") {
      expect(result.parsed.taskId).toBe(taskId);
      expect(result.parsed.basis?.some((b) => b.includes("does not exist"))).toBe(true);
    }
  });
});