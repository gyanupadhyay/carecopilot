-- The least-privilege role used by Text-to-SQL.
--
-- This role is deliberately NOT the owner of any table. PostgreSQL exempts
-- a table's owner from that table's row-level security policies, so owning
-- the schema would silently void the entire authorization boundary that
-- migration 0001 installs.
--
-- Three independent restrictions apply, in order of how hard they are to
-- get wrong:
--
--   1. no write grants at all — DML fails on privileges alone;
--   2. default_transaction_read_only — writes fail even if a grant leaks;
--   3. row-level security — reads outside app.patient_id return no rows.
--
-- The SQL validator in the application is a fourth layer and a usability
-- feature (it produces a clear error), not the security boundary.

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'carecopilot_ro') THEN
        CREATE ROLE carecopilot_ro LOGIN PASSWORD 'carecopilot_ro';
    END IF;
END
$$;

-- Password is a local-development default. Override it in any deployed
-- environment; ANALYTICS_DATABASE_URL is the only place it belongs.
ALTER ROLE carecopilot_ro SET default_transaction_read_only = on;
ALTER ROLE carecopilot_ro SET statement_timeout = '3s';
ALTER ROLE carecopilot_ro SET idle_in_transaction_session_timeout = '5s';

-- Nothing may be created by this role, and it inherits no future grants.
REVOKE CREATE ON SCHEMA public FROM carecopilot_ro;
REVOKE ALL ON ALL TABLES IN SCHEMA public FROM carecopilot_ro;

GRANT USAGE ON SCHEMA public TO carecopilot_ro;

-- SELECT grants on the analytics allowlist are issued by migration 0001,
-- alongside the RLS policies they are paired with, so that adding a table
-- to the allowlist is a single reviewable change in one file.
