import { NextResponse } from "next/server";
import db from "@/lib/client";

export async function GET() {
  try {
    const decisions = await db.agentDecision.findMany({
      orderBy: { createdAt: "desc" }
    });
    
    const decisionsWithTaskInfo = await Promise.all(decisions.map(async (dec) => {
      const basisObj = (typeof dec.basis === "string" ? JSON.parse(dec.basis) : dec.basis) as any;
      let taskTitle = "";
      let projectName = "";
      
      const targetTaskId = basisObj?.taskId ?? basisObj?.evidence?.[0]?.id;
      if (targetTaskId) {
        const task = await db.task.findUnique({
          where: { id: targetTaskId },
          select: { title: true, Project: { select: { name: true } } }
        });
        if (task) {
          taskTitle = task.title;
          projectName = task.Project?.name || "";
        }
      }

      if (!projectName && dec.projectId) {
        const proj = await db.project.findUnique({
          where: { id: dec.projectId },
          select: { name: true }
        });
        if (proj) projectName = proj.name;
      }
      
      return {
        ...dec,
        taskTitle: taskTitle || "Task Assignment Recommendation",
        projectName
      };
    }));

    return NextResponse.json({ success: true, decisions: decisionsWithTaskInfo });
  } catch (error) {
    console.error("Failed to fetch agent decisions:", error);
    return NextResponse.json({ success: false, error: "Failed to fetch decisions" }, { status: 500 });
  }
}
