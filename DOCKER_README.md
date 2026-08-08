# vector-db-benchmark

A Rust-based benchmarking tool for vector databases. Measures upload throughput, search QPS, latency percentiles, and recall for various vector search engines.

## Supported Engines

| Engine | Client Library | Protocol |
|--------|---------------|----------|
| **Redis** (RediSearch) | `redis` 0.27 | Redis protocol |
| **VectorSets** | `redis` 0.27 | Redis protocol |
| **Elasticsearch** | `elasticsearch` 8.15 | HTTP/REST |
| **OpenSearch** | `opensearch` 2.3 | HTTP/REST |
| **Qdrant** | `qdrant-client` 1.17 | gRPC |
| **PgVector** | `postgres` 0.19 + `pgvector` 0.4 | PostgreSQL |
| **Weaviate** | `tonic` 0.12 / `prost` 0.13 (gRPC) + `reqwest` (REST) | gRPC (search) + HTTP/REST (schema) |
| **Milvus** | `reqwest` (REST API v2) | HTTP/REST |
| **MongoDB** (Atlas Search) | `mongodb` 3 (sync) | MongoDB protocol |
| **Valkey** (Valkey Search) | `redis` 0.27 | RESP protocol |
| **Turbopuffer** | `turbopuffer-client` 0.0.4 | HTTP/REST (cloud) |

## Quick Start

```bash
# Show help
docker run --rm redis/vector-db-benchmark --help

# List available datasets
docker run --rm redis/vector-db-benchmark --describe datasets

# List available engine configurations
docker run --rm redis/vector-db-benchmark --describe engines
```

## Running a Benchmark

```bash
# Start Redis
docker run -d --name redis -p 6379:6379 redis:8.8.0

# Run benchmark against localhost Redis
docker run --rm --network host \
  -v $(pwd)/datasets:/code/datasets \
  -v $(pwd)/results:/code/results \
  redis/vector-db-benchmark \
  --host localhost --engines 'redis*' --datasets 'glove-25-angular'
```

## CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `--engines <PATTERN>` | Engine patterns to run (supports wildcards) | `*` |
| `--datasets <PATTERN>` | Dataset patterns to run (supports wildcards) | `*` |
| `--host <HOST>` | Redis / engine host | `localhost` |
| `--skip-upload` | Reuse the corpus already on the server (no configure, no upload); the runner verifies it server-side first | — |
| `--allow-partial-corpus` | Downgrade the `--skip-upload` reuse check from a hard error to a warning | — |
| `--skip-search` | Skip the search phase | — |
| `--skip-if-exists` | Skip if results already exist | — |
| `--parallels <N,N,...>` | Filter by parallel thread counts | — |
| `--ef-runtime <N,N,...>` | Filter by ef runtime values | — |
| `--update-search-ratio <U:S>` | Mixed benchmark: interleave U updates per S searches | — |
| `--describe <TYPE>` | Describe `datasets` or `engines` | — |
| `-v, --verbose` | Verbose output for `--describe` | — |
| `--exit-on-error` | Stop on first error | — |
| `--timeout <SECS>` | Timeout in seconds | `86400` |

## Mixed Benchmarks (Update + Search)

Interleave vector updates with searches to measure write impact on search performance:

```bash
# 1 update per 10 searches against Redis
docker run --rm --network host \
  -v $(pwd)/datasets:/code/datasets \
  -v $(pwd)/results:/code/results \
  redis/vector-db-benchmark \
  --host localhost --engines redis-docker-test --datasets random-100 \
  --update-search-ratio 1:10
```

The ratio format is `U:S` (e.g., `1:10` = 1 update per 10 searches per cycle). Supported engines: Redis, VectorSets, Valkey. Results include separate search and update metrics (RPS, latencies). Omitting the flag runs standard search-only benchmarks.

## Volume Mounts

| Container Path | Purpose |
|----------------|---------|
| `/code/datasets` | Dataset files (HDF5, NPY, JSONL) |
| `/code/results` | Benchmark result JSON files |

The image includes the `random-100` dataset (228KB) for quick smoke tests. For larger datasets, mount your local `datasets/` directory.

## Docker Compose Example

```yaml
services:
  redis:
    image: redis:8.8.0
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 1s
      timeout: 3s
      retries: 10

  benchmark:
    image: redis/vector-db-benchmark:latest
    depends_on:
      redis:
        condition: service_healthy
    volumes:
      - ./datasets:/code/datasets
      - ./results:/code/results
    command:
      - "--engines"
      - "redis*"
      - "--datasets"
      - "glove-25-angular"
      - "--host"
      - "redis"
```

## Source Code

[github.com/RedisLabs/vector-db-benchmark](https://github.com/RedisLabs/vector-db-benchmark)
