import { NextResponse } from "next/server";
import { getUniversalConfig, setUniversalConfig, initStealthPoliciesFromDb } from "@/ai/policy";

let hydrated = false;

export async function GET() {
  try {
    if (!hydrated) {
      await initStealthPoliciesFromDb();
      hydrated = true;
    }
    const config = getUniversalConfig();
    return NextResponse.json({
      success: true,
      config,
    });
  } catch (error) {
    return NextResponse.json(
      { success: false, error: error instanceof Error ? error.message : "Failed to fetch universal config" },
      { status: 500 }
    );
  }
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { isUniversalActive, globalMinConfidence, exceptionProjectIds } = body;

    const config = {
      isUniversalActive: Boolean(isUniversalActive),
      globalMinConfidence: typeof globalMinConfidence === "number" ? Math.max(0, Math.min(1, globalMinConfidence)) : 0.75,
      exceptionProjectIds: Array.isArray(exceptionProjectIds) ? exceptionProjectIds : [],
    };

    setUniversalConfig(config);

    return NextResponse.json({
      success: true,
      message: "Universal stealth configuration updated successfully",
      config,
    });
  } catch (error) {
    return NextResponse.json(
      { success: false, error: error instanceof Error ? error.message : "Failed to update universal config" },
      { status: 500 }
    );
  }
}
