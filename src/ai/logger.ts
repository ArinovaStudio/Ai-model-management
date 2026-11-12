import "dotenv/config";
import winston from "winston";

export const logger = winston.createLogger({
  level: process.env.AGENT_LOG_LEVEL || "info",
  format: winston.format.combine(
    winston.format.timestamp({ format: "YYYY-MM-DD HH:mm:ss.SSS" }),
    winston.format.errors({ stack: true }),
    winston.format.printf((info) => {
      const { timestamp, level, message, ...meta } = info;
      const metaString =
        Object.keys(meta).length > 0 ? ` ${JSON.stringify(meta)}` : "";
      return `${timestamp} [${level}] ${message}${metaString}`;
    })
  ),
  transports: [new winston.transports.Console()],
});

export type AgentLogger = typeof logger;