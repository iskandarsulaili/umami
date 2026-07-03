-- Apache AGE graph setup for Umami analytics
-- Creates graph structures for visitor journey analysis
-- AGE must be loaded via shared_preload_libraries before this runs

DO $$
BEGIN
  -- Create the main analytics graph if it doesn't exist
  PERFORM * FROM ag_catalog.create_graph('umami_analytics');
EXCEPTION WHEN OTHERS THEN
  -- Graph may already exist
  NULL;
END $$;

-- Set up AGE search path (commented; applied at DB level in init-extensions.sql)
-- ALTER DATABASE umami SET search_path TO '$user', public, ag_catalog;
