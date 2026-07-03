-- Fix primary keys to include created_at for TimescaleDB compatibility.
-- TimescaleDB requires the partitioning column (created_at) in all unique
-- indexes and primary keys.

-- website_event: PK (event_id, created_at)
ALTER TABLE website_event DROP CONSTRAINT IF EXISTS website_event_pkey CASCADE;
ALTER TABLE website_event ADD PRIMARY KEY (event_id, created_at);

-- event_data: PK (event_data_id, created_at)
ALTER TABLE event_data DROP CONSTRAINT IF EXISTS event_data_pkey CASCADE;
ALTER TABLE event_data ADD PRIMARY KEY (event_data_id, created_at);

-- session: PK (session_id, created_at)
ALTER TABLE session DROP CONSTRAINT IF EXISTS session_pkey CASCADE;
ALTER TABLE session ADD PRIMARY KEY (session_id, created_at);

-- session_data: PK (session_data_id, created_at)
ALTER TABLE session_data DROP CONSTRAINT IF EXISTS session_data_pkey CASCADE;
ALTER TABLE session_data ADD PRIMARY KEY (session_data_id, created_at);

-- session_replay: PK (replay_id, created_at) 
ALTER TABLE session_replay DROP CONSTRAINT IF EXISTS session_replay_pkey CASCADE;
ALTER TABLE session_replay ADD PRIMARY KEY (replay_id, created_at);

-- heatmap_event: PK (heatmap_event_id, created_at)
ALTER TABLE heatmap_event DROP CONSTRAINT IF EXISTS heatmap_event_pkey CASCADE;
ALTER TABLE heatmap_event ADD PRIMARY KEY (heatmap_event_id, created_at);

-- revenue: PK (revenue_id, created_at)
ALTER TABLE revenue DROP CONSTRAINT IF EXISTS revenue_pkey CASCADE;
ALTER TABLE revenue ADD PRIMARY KEY (revenue_id, created_at);
