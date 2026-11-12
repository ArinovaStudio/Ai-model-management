import { PrismaClient } from "@prisma/client";
import { Pool } from "pg";
import { PrismaPg } from "@prisma/adapter-pg";

const pool = new Pool({
  connectionString: process.env.DB_URI,
});

const adapter = new PrismaPg(pool);

const globalForPrisma = globalThis as any;

const db = globalForPrisma.prisma as PrismaClient || new PrismaClient({ adapter });

if (process.env.NODE_ENV !== "production") globalForPrisma.prisma = db;

export default db;
