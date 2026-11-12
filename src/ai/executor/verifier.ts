import type { VerificationOutcome, VerificationResult } from "./types";

export function compareField(
  actual: unknown,
  expected: unknown,
  fieldName: string
): VerificationResult {
  const normalized = (value: unknown): unknown => {
    if (typeof value === "number" && Number.isFinite(value)) {
      return String(value);
    }
    if (value instanceof Date) return value.toISOString();
    return value;
  };

  const actualNorm = normalized(actual);
  const expectedNorm = normalized(expected);
  const passed = actualNorm === expectedNorm;

  return {
    passed,
    outcome: passed ? "pass" : "mismatch",
    detail: passed
      ? `field "${fieldName}" matches (${JSON.stringify(expectedNorm)})`
      : `field "${fieldName}" mismatch: expected ${JSON.stringify(expectedNorm)}, actual ${JSON.stringify(actualNorm)}`,
    actual: actualNorm,
    expected: expectedNorm,
  };
}

export function compareFields(
  actual: Record<string, unknown>,
  expected: Record<string, unknown>
): VerificationResult[] {
  return Object.entries(expected).map(([field, value]) =>
    compareField(actual[field], value, field)
  );
}

export function mergeVerification(results: VerificationResult[]): VerificationResult {
  const failed = results.find(
    (result) => result.outcome === "mismatch" || result.outcome === "field_missing"
  );
  if (failed) {
    return {
      passed: false,
      outcome: failed.outcome,
      detail: failed.detail,
      actual: failed.actual,
      expected: failed.expected,
    };
  }
  return {
    passed: true,
    outcome: "pass",
    detail: `all ${results.length} field(s) matched`,
    actual: null,
    expected: null,
  };
}

export function detectConflict(
  actual: unknown,
  expected: unknown,
  prior: unknown,
  fieldName: string
): VerificationResult {
  const equalsExpected = compareField(actual, expected, fieldName).passed;
  if (equalsExpected) {
    return {
      passed: true,
      outcome: "pass",
      detail: `field "${fieldName}" matches the expected value ${JSON.stringify(expected)}`,
      actual,
      expected,
    };
  }
  const equalsPrior = compareField(actual, prior, fieldName).passed;
  if (equalsPrior) {
    return {
      passed: false,
      outcome: "mismatch",
      detail: `field "${fieldName}" still equals the prior value ${JSON.stringify(prior)}; the mutation did not take effect`,
      actual,
      expected,
    };
  }
  return {
    passed: false,
    outcome: "conflict",
    detail: `field "${fieldName}" was concurrently changed by another actor: expected ${JSON.stringify(expected)}, but found ${JSON.stringify(actual)} (prior value was ${JSON.stringify(prior)})`,
    actual,
    expected,
  };
}

export type { VerificationOutcome };