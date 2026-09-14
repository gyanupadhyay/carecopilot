-- Extensions must be created by a superuser, which the application role is
-- not. Docker runs everything in /docker-entrypoint-initdb.d as the
-- bootstrap superuser, so this is the only place they can be installed
-- without handing the app elevated rights.
--
-- Migration 0001 also issues CREATE EXTENSION IF NOT EXISTS vector; that is
-- a no-op here and a clear, early failure on a server that lacks pgvector.

CREATE EXTENSION IF NOT EXISTS vector;

-- Trigram matching backs fuzzy lookups on medication and test names. Not
-- required by the MVP retrieval path, but free to install now and awkward
-- to add later on a running database.
CREATE EXTENSION IF NOT EXISTS pg_trgm;
