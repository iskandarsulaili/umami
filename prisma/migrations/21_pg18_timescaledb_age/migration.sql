-- Upgrades Umami to PostgreSQL 18 with TimescaleDB 2.28 and Apache AGE
-- This migration enables PostgreSQL extensions and configures hypertables.
-- TimescaleDB and AGE must be installed at the system level.

-- ============================================================
-- Step 1: Enable PostgreSQL 18 extensions
-- ============================================================

-- TimescaleDB (time-series hypertables)
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Apache AGE (graph database for visitor journey analysis)
CREATE EXTENSION IF NOT EXISTS age CASCADE;

-- pgvector (vector similarity search for ML embeddings)
CREATE EXTENSION IF NOT EXISTS vector CASCADE;

-- Query performance monitoring
CREATE EXTENSION IF NOT EXISTS pg_stat_statements CASCADE;

-- ============================================================
-- Step 2: Create Apache AGE graph for analytics
-- ============================================================

SELECT ag_catalog.create_graph('umami_analytics');

-- ============================================================
-- Step 3: Convert tables to TimescaleDB hypertables
-- These tables are time-series heavy and benefit from
-- automatic partitioning, compression, and continuous aggregates.
-- ============================================================

-- Helper: convert to hypertable if not already
DO $$
DECLARE
  hypertable_count INTEGER;
BEGIN
  -- website_event
  SELECT COUNT(*) INTO hypertable_count
  FROM timescaledb_information.hypertables WHERE hypertable_name = 'website_event';
  IF hypertable_count = 0 THEN
    PERFORM create_hypertable('website_event', 'created_at',
      chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE, migrate_data => TRUE);
    PERFORM add_dimension('website_event', 'website_id', number_partitions => 4, if_not_exists => TRUE);
  END IF;

  -- event_data
  SELECT COUNT(*) INTO hypertable_count
  FROM timescaledb_information.hypertables WHERE hypertable_name = 'event_data';
  IF hypertable_count = 0 THEN
    PERFORM create_hypertable('event_data', 'created_at',
      chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE, migrate_data => TRUE);
    PERFORM add_dimension('event_data', 'website_id', number_partitions => 4, if_not_exists => TRUE);
  END IF;

  -- session_data
  SELECT COUNT(*) INTO hypertable_count
  FROM timescaledb_information.hypertables WHERE hypertable_name = 'session_data';
  IF hypertable_count = 0 THEN
    PERFORM create_hypertable('session_data', 'created_at',
      chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE, migrate_data => TRUE);
    PERFORM add_dimension('session_data', 'website_id', number_partitions => 4, if_not_exists => TRUE);
  END IF;

  -- session
  SELECT COUNT(*) INTO hypertable_count
  FROM timescaledb_information.hypertables WHERE hypertable_name = 'session';
  IF hypertable_count = 0 THEN
    PERFORM create_hypertable('session', 'created_at',
      chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE, migrate_data => TRUE);
    PERFORM add_dimension('session', 'website_id', number_partitions => 4, if_not_exists => TRUE);
  END IF;

  -- session_replay
  SELECT COUNT(*) INTO hypertable_count
  FROM timescaledb_information.hypertables WHERE hypertable_name = 'session_replay';
  IF hypertable_count = 0 THEN
    PERFORM create_hypertable('session_replay', 'created_at',
      chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE, migrate_data => TRUE);
    PERFORM add_dimension('session_replay', 'website_id', number_partitions => 2, if_not_exists => TRUE);
  END IF;

  -- heatmap_event
  SELECT COUNT(*) INTO hypertable_count
  FROM timescaledb_information.hypertables WHERE hypertable_name = 'heatmap_event';
  IF hypertable_count = 0 THEN
    PERFORM create_hypertable('heatmap_event', 'created_at',
      chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE, migrate_data => TRUE);
    PERFORM add_dimension('heatmap_event', 'website_id', number_partitions => 4, if_not_exists => TRUE);
  END IF;

  -- revenue
  SELECT COUNT(*) INTO hypertable_count
  FROM timescaledb_information.hypertables WHERE hypertable_name = 'revenue';
  IF hypertable_count = 0 THEN
    PERFORM create_hypertable('revenue', 'created_at',
      chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE, migrate_data => TRUE);
  END IF;
END $$;

-- ============================================================
-- Step 4: Create continuous aggregates for common queries
-- ============================================================

-- Hourly page views (auto-refreshed materialized view)
CREATE MATERIALIZED VIEW IF NOT EXISTS website_event_hourly
WITH (timescaledb.continuous) AS
SELECT
  time_bucket('1 hour', created_at) AS bucket,
  website_id,
  url_path,
  COUNT(*) AS pageviews,
  COUNT(DISTINCT session_id) AS unique_sessions,
  COUNT(DISTINCT visit_id) AS unique_visits
FROM website_event
GROUP BY bucket, website_id, url_path
WITH NO DATA;

SELECT add_continuous_aggregate_policy('website_event_hourly',
  start_offset => INTERVAL '3 days',
  end_offset => INTERVAL '1 hour',
  schedule_interval => INTERVAL '10 minutes',
  if_not_exists => TRUE);

-- Daily page views aggregate
CREATE MATERIALIZED VIEW IF NOT EXISTS website_event_daily
WITH (timescaledb.continuous) AS
SELECT
  time_bucket('1 day', created_at) AS bucket,
  website_id,
  url_path,
  COUNT(*) AS pageviews,
  COUNT(DISTINCT session_id) AS unique_sessions,
  COUNT(DISTINCT visit_id) AS unique_visits
FROM website_event
GROUP BY bucket, website_id, url_path
WITH NO DATA;

SELECT add_continuous_aggregate_policy('website_event_daily',
  start_offset => INTERVAL '30 days',
  end_offset => INTERVAL '1 day',
  schedule_interval => INTERVAL '1 hour',
  if_not_exists => TRUE);

-- ============================================================
-- Step 5: Configure data retention policies
-- ============================================================

SELECT add_retention_policy('website_event', INTERVAL '2 years', if_not_exists => TRUE);
SELECT add_retention_policy('event_data', INTERVAL '2 years', if_not_exists => TRUE);
SELECT add_retention_policy('session_data', INTERVAL '2 years', if_not_exists => TRUE);
SELECT add_retention_policy('session', INTERVAL '2 years', if_not_exists => TRUE);
SELECT add_retention_policy('session_replay', INTERVAL '1 year', if_not_exists => TRUE);
SELECT add_retention_policy('heatmap_event', INTERVAL '1 year', if_not_exists => TRUE);
