#!/bin/bash
# Runs once, when the Postgres volume is first created: one database per app, so the runtime's own
# tables and each app's migration chain stay separate. On a host, `make db APP=x` does the same later.
set -e
for db in shop finance; do
  psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -tc "select 1 from pg_database where datname = '$db'" | grep -q 1 \
    || psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -c "create database \"$db\""
done
