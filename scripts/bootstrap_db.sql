-- Bootstrap for a natively installed PostgreSQL (no Docker).
--
-- Run once, as a superuser:
--
--     psql -U postgres -f scripts/bootstrap_db.sql
--
-- It creates the two roles and the database, then installs the extensions
-- inside it. Everything after this point is the application's own Alembic
-- migration, which runs as the unprivileged carecopilot role.
--
-- Docker users do not need this file: docker/initdb/*.sql does the same
-- work automatically on first container start.

\set ON_ERROR_STOP on

-- Application role: owns the schema, read/write.
SELECT 'CREATE ROLE carecopilot LOGIN PASSWORD ''carecopilot'''
WHERE NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'carecopilot')\gexec

-- Analytics role: Text-to-SQL only. Owns nothing, so RLS applies to it.
SELECT 'CREATE ROLE carecopilot_ro LOGIN PASSWORD ''carecopilot_ro'''
WHERE NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'carecopilot_ro')\gexec

SELECT 'CREATE DATABASE carecopilot OWNER carecopilot'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'carecopilot')\gexec

ALTER ROLE carecopilot_ro SET default_transaction_read_only = on;
ALTER ROLE carecopilot_ro SET statement_timeout = '3s';
ALTER ROLE carecopilot_ro SET idle_in_transaction_session_timeout = '5s';

\connect carecopilot

-- pgvector is a separate install and is frequently absent from a stock
-- Windows PostgreSQL build. Its absence is a known, supported configuration
-- (VECTOR_BACKEND=array), so it must not abort the rest of the bootstrap --
-- the roles and grants below matter regardless of which backend is used.
--
-- To install it on Windows you build it, with VS Build Tools and the
-- PostgreSQL server headers already on the machine:
--
--     git clone --branch v0.8.6 https://github.com/pgvector/pgvector.git
--     call "...\VC\Auxiliary\Build\vcvars64.bat"
--     set "PGROOT=C:\Program Files\PostgreSQL\18"
--     nmake /F Makefile.win
--     nmake /F Makefile.win install      <- needs an elevated prompt
--
-- On Linux/macOS it is a package (apt install postgresql-18-pgvector,
-- brew install pgvector) and the Docker image already carries it.
DO $$
BEGIN
    EXECUTE 'CREATE EXTENSION IF NOT EXISTS vector';
    RAISE NOTICE 'pgvector installed: VECTOR_BACKEND=pgvector is available.';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE
        'pgvector is NOT available on this server. Set VECTOR_BACKEND=array '
        'in .env before running migrations; embeddings will be stored as '
        'double precision[] and ranked by cc_cosine_distance.';
END
$$;

CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- The app role owns the public schema and therefore bypasses the RLS
-- policies on its own tables — which is the intended asymmetry.
GRANT ALL ON SCHEMA public TO carecopilot;
GRANT CONNECT ON DATABASE carecopilot TO carecopilot_ro;
GRANT USAGE ON SCHEMA public TO carecopilot_ro;
REVOKE CREATE ON SCHEMA public FROM carecopilot_ro;

\echo 'Bootstrap complete. Next: cd backend && alembic upgrade head'
