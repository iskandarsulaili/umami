-- Enable PostgreSQL extensions for Umami
-- This runs on every container startup

-- 1. TimescaleDB - time-series hypertables
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- 2. Apache AGE - graph database (multi-model)
CREATE EXTENSION IF NOT EXISTS age CASCADE;

-- 3. pgvector - vector similarity search (for ML embeddings)
CREATE EXTENSION IF NOT EXISTS vector CASCADE;

-- 4. pgcrypto - cryptographic functions (existing umami requirement)
CREATE EXTENSION IF NOT EXISTS pgcrypto CASCADE;

-- 5. pg_stat_statements - query performance monitoring
CREATE EXTENSION IF NOT EXISTS pg_stat_statements CASCADE;

-- Load AGE into search path
ALTER DATABASE umami SET search_path TO '$user', public, ag_catalog;
