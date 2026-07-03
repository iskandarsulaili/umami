-- Convert umami time-series tables to TimescaleDB hypertables
-- This is idempotent - safe to run on every startup

DO $$
BEGIN
  -- 1. website_event - core analytics table
  IF NOT EXISTS (
    SELECT 1 FROM timescaledb_information.hypertables WHERE hypertable_name = 'website_event'
  ) THEN
    PERFORM create_hypertable('website_event', 'created_at', chunk_time_interval => INTERVAL '1 day',
      if_not_exists => TRUE, migrate_data => TRUE);
    -- Additional space partitioning by website_id for parallel I/O
    PERFORM add_dimension('website_event', 'website_id', number_partitions => 4, if_not_exists => TRUE);
  END IF;

  -- 2. event_data - custom event properties (time-series)
  IF NOT EXISTS (
    SELECT 1 FROM timescaledb_information.hypertables WHERE hypertable_name = 'event_data'
  ) THEN
    PERFORM create_hypertable('event_data', 'created_at', chunk_time_interval => INTERVAL '1 day',
      if_not_exists => TRUE, migrate_data => TRUE);
    PERFORM add_dimension('event_data', 'website_id', number_partitions => 4, if_not_exists => TRUE);
  END IF;

  -- 3. session_data - session-level properties
  IF NOT EXISTS (
    SELECT 1 FROM timescaledb_information.hypertables WHERE hypertable_name = 'session_data'
  ) THEN
    PERFORM create_hypertable('session_data', 'created_at', chunk_time_interval => INTERVAL '1 day',
      if_not_exists => TRUE, migrate_data => TRUE);
    PERFORM add_dimension('session_data', 'website_id', number_partitions => 4, if_not_exists => TRUE);
  END IF;

  -- 4. session (visits/sessions)
  IF NOT EXISTS (
    SELECT 1 FROM timescaledb_information.hypertables WHERE hypertable_name = 'session'
  ) THEN
    PERFORM create_hypertable('session', 'created_at', chunk_time_interval => INTERVAL '1 day',
      if_not_exists => TRUE, migrate_data => TRUE);
    PERFORM add_dimension('session', 'website_id', number_partitions => 4, if_not_exists => TRUE);
  END IF;

  -- 5. session_replay - user interaction recordings
  IF NOT EXISTS (
    SELECT 1 FROM timescaledb_information.hypertables WHERE hypertable_name = 'session_replay'
  ) THEN
    PERFORM create_hypertable('session_replay', 'created_at', chunk_time_interval => INTERVAL '7 days',
      if_not_exists => TRUE, migrate_data => TRUE);
    PERFORM add_dimension('session_replay', 'website_id', number_partitions => 2, if_not_exists => TRUE);
  END IF;

  -- 6. heatmap_event - click/scroll coordinates
  IF NOT EXISTS (
    SELECT 1 FROM timescaledb_information.hypertables WHERE hypertable_name = 'heatmap_event'
  ) THEN
    PERFORM create_hypertable('heatmap_event', 'created_at', chunk_time_interval => INTERVAL '1 day',
      if_not_exists => TRUE, migrate_data => TRUE);
    PERFORM add_dimension('heatmap_event', 'website_id', number_partitions => 4, if_not_exists => TRUE);
  END IF;

  -- 7. revenue - financial transactions
  IF NOT EXISTS (
    SELECT 1 FROM timescaledb_information.hypertables WHERE hypertable_name = 'revenue'
  ) THEN
    PERFORM create_hypertable('revenue', 'created_at', chunk_time_interval => INTERVAL '7 days',
      if_not_exists => TRUE, migrate_data => TRUE);
  END IF;
END $$;

-- Create continuous aggregates for common queries
DO $$
BEGIN
  -- Hourly page views aggregate
  IF NOT EXISTS (
    SELECT 1 FROM timescaledb_information.continuous_aggregates WHERE view_name = 'website_event_hourly'
  ) THEN
    CREATE MATERIALIZED VIEW website_event_hourly
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

    -- Refresh policy - every 10 minutes, look back 2 hours
    PERFORM add_continuous_aggregate_policy('website_event_hourly',
      start_offset => INTERVAL '3 days',
      end_offset => INTERVAL '1 hour',
      schedule_interval => INTERVAL '10 minutes');
  END IF;

  -- Daily page views aggregate
  IF NOT EXISTS (
    SELECT 1 FROM timescaledb_information.continuous_aggregates WHERE view_name = 'website_event_daily'
  ) THEN
    CREATE MATERIALIZED VIEW website_event_daily
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

    PERFORM add_continuous_aggregate_policy('website_event_daily',
      start_offset => INTERVAL '30 days',
      end_offset => INTERVAL '1 day',
      schedule_interval => INTERVAL '1 hour');
  END IF;
END $$;

-- Data retention policy: auto-drop chunks older than 2 years
SELECT add_retention_policy('website_event', INTERVAL '2 years', if_not_exists => TRUE);
SELECT add_retention_policy('event_data', INTERVAL '2 years', if_not_exists => TRUE);
SELECT add_retention_policy('session_data', INTERVAL '2 years', if_not_exists => TRUE);
SELECT add_retention_policy('session', INTERVAL '2 years', if_not_exists => TRUE);
SELECT add_retention_policy('session_replay', INTERVAL '1 year', if_not_exists => TRUE);
SELECT add_retention_policy('heatmap_event', INTERVAL '1 year', if_not_exists => TRUE);
