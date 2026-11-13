/** @jest-environment node */
import { circuitBreaker, TRIP_THRESHOLDS } from "@/ai/circuit-breaker";
import { makePrefix, cleanupTestData, resetBreaker } from "./setup";
import { seedProject } from "./fixtures";

const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

describe("ai/circuit-breaker (live DB)", () => {
  const prefix = makePrefix("breaker");

  beforeAll(async () => {
    await resetBreaker();
    await seedProject(prefix, "Breaker Probe Project");
  });

  beforeEach(async () => {
    await resetBreaker();
  });

  afterAll(async () => {
    await cleanupTestData(prefix);
  });

  it("starts closed and allows writes", async () => {
    expect(await circuitBreaker.isClosed()).toBe(true);
    expect(await circuitBreaker.shouldAllow("execute")).toBe(true);
  });

  it("trips open when a failure type crosses its threshold", async () => {
    await circuitBreaker.recordFailure("llm_timeout");
    await circuitBreaker.recordFailure("llm_timeout");
    expect(await circuitBreaker.isOpen()).toBe(false);

    await circuitBreaker.recordFailure("llm_timeout");
    expect(await circuitBreaker.isOpen()).toBe(true);
  });

  it("blocks writes but lets reads pass while OPEN", async () => {
    await circuitBreaker.recordFailure("fatal", { reason: "force open" });
    expect(await circuitBreaker.isOpen()).toBe(true);
    expect(await circuitBreaker.shouldAllow("execute")).toBe(false);
    expect(await circuitBreaker.shouldAllow("write")).toBe(false);
    expect(await circuitBreaker.shouldAllow("read")).toBe(true);
  });

  it("tracks per-failure-type counts", async () => {
    await circuitBreaker.recordFailure("execution");
    await circuitBreaker.recordFailure("execution");
    await circuitBreaker.recordFailure("verification_mismatch");

    const snapshot = await circuitBreaker.state;
    expect(snapshot.failureCounts.execution).toBe(2);
    expect(snapshot.failureCounts.verification_mismatch).toBe(1);
    expect(snapshot.state).toBe("closed");
  });

  it("unknown failure types fall back to fatal", async () => {
    await circuitBreaker.recordFailure("bogus-type", { reason: "unknown" });
    expect(await circuitBreaker.isOpen()).toBe(true);
  });

  it("recovers to half_open after the dwell window, then closes on a successful probe", async () => {
    await circuitBreaker.recordFailure("fatal", { reason: "force open" });
    expect(await circuitBreaker.isOpen()).toBe(true);

    await sleep(1300);

    const probed = await circuitBreaker.probe();
    expect(probed).toBe(true);
    expect(await circuitBreaker.isClosed()).toBe(true);
  });

  it("resets failure counters on a success", async () => {
    await circuitBreaker.recordFailure("execution");
    await circuitBreaker.recordFailure("execution");
    await circuitBreaker.recordSuccess();

    const snapshot = await circuitBreaker.state;
    expect(snapshot.failureCount).toBe(0);
    expect(snapshot.failureCounts).toEqual({});
  });

  it("exposes default failure thresholds", () => {
    expect(TRIP_THRESHOLDS.fatal).toBe(1);
    expect(TRIP_THRESHOLDS.llm_timeout).toBe(3);
    expect(TRIP_THRESHOLDS.execution).toBeGreaterThan(1);
  });
});