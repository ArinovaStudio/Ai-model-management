import { NextResponse } from "next/server";
import { getStealthPolicy, setStealthPolicy, initStealthPoliciesFromDb } from "@/ai/policy";
import type { StealthPolicy } from "@/ai/policy/types";
import { triggerActiveAllocations } from "@/ai/events";

let hydrated = false;

export async function GET(request: Request) {
  try {
    if (!hydrated) {
      await initStealthPoliciesFromDb();
      hydrated = true;
    }
    const { searchParams } = new URL(request.url);
    const projectId = searchParams.get("projectId");
    const policy = getStealthPolicy(projectId);
    
    return NextResponse.json({
      success: true,
      policy: policy ?? {
        projectId: projectId ?? "*",
        isActive: false,
        minConfidenceScore: 0.75,
        startDate: null,
        endDate: null,
        modeType: "AUTO",
        selectedTaskIds: [],
        isException: false,
      },
    });
  } catch (error) {
    return NextResponse.json(
      { success: false, error: error instanceof Error ? error.message : "Failed to fetch stealth policy" },
      { status: 500 }
    );
  }
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { projectId, isActive, minConfidenceScore, startDate, endDate, modeType, selectedTaskIds, isException } = body;

    if (!projectId || typeof projectId !== "string") {
      return NextResponse.json(
        { success: false, error: "projectId is required" },
        { status: 400 }
      );
    }

    const policy: StealthPolicy = {
      projectId,
      isActive: Boolean(isActive),
      minConfidenceScore: typeof minConfidenceScore === "number" ? Math.max(0, Math.min(1, minConfidenceScore)) : 0.75,
      startDate: startDate || null,
      endDate: endDate || null,
      modeType: modeType === "SELECT" ? "SELECT" : "AUTO",
      selectedTaskIds: Array.isArray(selectedTaskIds) ? selectedTaskIds : [],
      isException: Boolean(isException),
    };

    setStealthPolicy(policy);

    if (policy.isActive) {
      void triggerActiveAllocations(projectId).catch((err) => {
        console.error("[STEALTH_POLICY] Failed to trigger allocations on policy save:", err);
      });
    }

    return NextResponse.json({
      success: true,
      message: "Stealth policy updated successfully",
      policy,
    });
  } catch (error) {
    return NextResponse.json(
      { success: false, error: error instanceof Error ? error.message : "Failed to update stealth policy" },
      { status: 500 }
    );
  }
}
