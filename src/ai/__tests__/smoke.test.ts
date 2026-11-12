/** @jest-environment node */
import db from "@/lib/client";
import { makePrefix, cleanupTestData } from "./setup";
import { seedProject, seedTask, daysFromNow } from "./fixtures";

describe("smoke: db connectivity + fixtures", () => {
  const prefix = makePrefix("smoke");

  afterAll(async () => {
    await cleanupTestData(prefix);
  });

  it("can seed and read a project with a task", async () => {
    const { projectId } = await seedProject(prefix, "Smoke Project");
    const { taskId } = await seedTask(prefix, projectId, "a", {
      status: "pending",
      dueDate: daysFromNow(3),
    });

    const task = await db.task.findUnique({ where: { id: taskId } });
    expect(task).not.toBeNull();
    expect(task!.projectId).toBe(projectId);

    const project = await db.project.findUnique({ where: { id: projectId } });
    expect(project!.name).toBe("Smoke Project");
  });
});