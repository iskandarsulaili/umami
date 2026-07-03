FROM postgres:18-bookworm

LABEL description="PostgreSQL 18 with TimescaleDB 2.28, Apache AGE, and pgvector"
LABEL maintainer="umami-software"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install build dependencies
RUN set -eux \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        gnupg \
        lsb-release \
    && rm -rf /var/lib/apt/lists/*

# Add TimescaleDB GPG key and repo
RUN set -eux \
    && curl -fsSL https://packagecloud.io/timescale/timescaledb/gpgkey | gpg --dearmor -o /usr/share/keyrings/timescaledb-archive-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/timescaledb-archive-keyring.gpg] https://packagecloud.io/timescale/timescaledb/ubuntu/ $(lsb_release -cs) main" > /etc/apt/sources.list.d/timescaledb.list

# Install PostgreSQL extensions
RUN set -eux \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        postgresql-18-timescaledb \
        postgresql-18-age \
        postgresql-18-pgvector \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy custom PostgreSQL configuration
COPY db/postgresql/postgresql.conf /etc/postgresql/18/main/postgresql.conf

# Copy initialization scripts
COPY db/postgresql/init-extensions.sql /docker-entrypoint-initdb.d/01-extensions.sql
COPY db/postgresql/init-hypertables.sql /docker-entrypoint-initdb.d/02-hypertables.sql
