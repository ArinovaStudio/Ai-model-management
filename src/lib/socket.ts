/**
 * src/lib/socket.ts
 *
 * Standalone socket helper for Arinova-AI-BRAIN.
 * Pushes events to the central socket server via HTTP /emit.
 */

const SOCKET_SERVER_URL = (() => {
  const envUrl = process.env.SOCKET_WEBHOOK_URL?.replace("/api/webhook/ai-event", "")
    || process.env.SOCKET_URL
    || "http://arinova_socket_server:3001";
  return envUrl.replace(/\/+$/, "");
})();

export async function emitEvent(
  userId: string,
  event: string,
  data: any
): Promise<void> {
  try {
    await fetch(`${SOCKET_SERVER_URL}/emit`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ clientId: userId, event, data }),
    });
  } catch (err: any) {
    // Fire-and-forget: socket broadcast failures must never crash the AI engine
  }
}

export async function emitToUser(
  userId: string,
  notification: any
): Promise<void> {
  try {
    await fetch(`${SOCKET_SERVER_URL}/emit`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ clientId: userId, notification }),
    });
  } catch (err: any) {}
}
