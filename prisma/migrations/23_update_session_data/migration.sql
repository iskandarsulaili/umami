-- Fork adaptation for TimescaleDB: session_data is partitioned on created_at,
-- so every unique index MUST include created_at (TimescaleDB TS103 rule).
-- Upstream's original created UNIQUE INDEX on (session_id, data_key) alone,
-- which TimescaleDB rejects.

-- Deduplicate session_data rows (keep the newest per session_id+data_key)
WITH ranked_session_data AS (
  SELECT
    "session_data_id",
    ROW_NUMBER() OVER (
      PARTITION BY "session_id", "data_key"
      ORDER BY "created_at" DESC NULLS LAST, "session_data_id" DESC
    ) AS row_num
  FROM "session_data"
)
DELETE FROM "session_data"
USING ranked_session_data
WHERE "session_data"."session_data_id" = ranked_session_data."session_data_id"
  AND ranked_session_data.row_num > 1;

-- Unique index including created_at (TimescaleDB-compatible composite)
CREATE UNIQUE INDEX "session_data_session_id_data_key_key"
ON "session_data"("session_id", "data_key", "created_at");
