-- PostgreSQL initialization script for RealEstateAI
-- Creates pgvector extension and sets initial configurations

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Verify vector extension is loaded
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_extension WHERE extname = 'vector'
    ) THEN
        RAISE NOTICE 'Vector extension successfully initialized.';
    ELSE
        RAISE EXCEPTION 'Failed to initialize vector extension.';
    END IF;
END $$;
