import { NextResponse } from "next/server";
import db from "@/lib/client";
import { getUserId } from "@/lib/auth";
import type { PolicyDecision } from "@/ai/policy/types";

export class ApiError extends Error {
  status: number;
  code: string;
  details?: unknown;

  constructor(status: number, code: string, message: string, details?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.details = details;
  }
}

export function ok(data: unknown, status = 200): NextResponse {
  return NextResponse.json({ success: true, ...(data as Record<string, unknown>) }, { status });
}

export function errorResponse(error: unknown): NextResponse {
  if (error instanceof ApiError) {
    const body: Record<string, unknown> = {
      success: false,
      error: error.code,
      message: error.message,
    };
    if (error.details !== undefined) body.details = error.details;
    return NextResponse.json(body, { status: error.status });
  }
  const message = error instanceof Error ? error.message : String(error);
  return NextResponse.json(
    { success: false, error: "INTERNAL", message: "Internal Server Error", details: message },
    { status: 500 }
  );
}

export async function authedUser(
  req: Request
): Promise<{ id: string; role: string; name: string }> {
  let userId: string;
  try {
    userId = await getUserId(req);
  } catch {
    throw new ApiError(401, "UNAUTHORIZED", "Authentication required: provide a valid token.");
  }
  const user = await db.user.findUnique({
    where: { id: userId },
    select: {
      id: true,
      name: true,
      subRole: { select: { role: { select: { name: true } } } },
    },
  });
  if (!user) {
    throw new ApiError(401, "UNAUTHORIZED", "The authenticated user no longer exists.");
  }
  return { id: user.id, name: user.name, role: user.subRole?.role?.name ?? "CLIENT" };
}

export async function assertProjectExists(projectId: string): Promise<void> {
  const project = await db.project.findUnique({ where: { id: projectId }, select: { id: true } });
  if (!project) {
    throw new ApiError(400, "PROJECT_NOT_FOUND", `Project "${projectId}" does not exist.`);
  }
}

export async function requireProjectRead(
  userId: string,
  role: string,
  projectId: string
): Promise<void> {
  await assertProjectExists(projectId);
  if (role === "ADMIN") return;
  const member = await db.projectMember.findFirst({
    where: { projectId, userId },
    select: { id: true },
  });
  if (!member) {
    throw new ApiError(403, "FORBIDDEN", "You do not have read access to this project.");
  }
}

export async function requireProjectManagerOrAdmin(
  userId: string,
  role: string,
  projectId: string
): Promise<void> {
  await assertProjectExists(projectId);
  if (role === "ADMIN" || role === "MANAGER") return;
  const leader = await db.projectMember.findFirst({
    where: { projectId, userId, isLeader: true },
    select: { id: true },
  });
  if (!leader) {
    throw new ApiError(
      403,
      "FORBIDDEN",
      "PROJECT_MANAGER or ADMIN role is required for this project."
    );
  }
}

export async function requireProjectMemberOrHigher(
  userId: string,
  role: string,
  projectId: string
): Promise<void> {
  await assertProjectExists(projectId);
  if (role === "ADMIN" || role === "MANAGER") return;
  const member = await db.projectMember.findFirst({
    where: { projectId, userId },
    select: { id: true },
  });
  if (!member) {
    throw new ApiError(403, "FORBIDDEN", "Project membership is required for this action.");
  }
}

export function requireAdmin(role: string): void {
  if (role !== "ADMIN") {
    throw new ApiError(403, "FORBIDDEN", "ADMIN role is required.");
  }
}

export const LOW_RISK_POLICY: PolicyDecision = {
  allowed: true,
  requiresApproval: false,
  status: "allowed",
  riskLevel: "LOW",
  action: "send_notification",
  projectId: null,
  reason: "Endpoint-level authorization granted by the API layer.",
  checks: [
    {
      name: "api_endpoint_authorization",
      passed: true,
      detail: "Endpoint performed its own role/membership authorization.",
    },
  ],
};

export function formatError(error: unknown): string {
  return error instanceof Error ? `${error.name}: ${error.message}` : String(error);
}

export const CHECK_IN_HORIZON_MS = 24 * 60 * 60 * 1000;
