# Agent Instructions for vector-db-benchmark

## Project Overview

Rust implementation of a vector database benchmarking tool. Supports Redis (RediSearch) and VectorSets engines.

## Build & Test Commands

All commands go through the Makefile:

```bash
make check              # Run formatting (rustfmt) + linting (clippy) — MUST pass before any PR
make vector-db-benchmark # Build the main CLI binary (release mode)
make build              # Build all binaries (release mode)
make test               # Run unit tests (no Docker needed)
make integration-test   # Run integration tests (starts redis:8.8.0 Docker on port 6399)
make v0-check           # Compare Rust vs Python v0 (precision, QPS, latency) — starts Docker
make fmt                # Auto-format code
make clean              # Clean build artifacts
```

## Workflow Rules

1. **Always run `make check` after code changes** — formatting and clippy must pass
2. **Always run `make vector-db-benchmark`** to verify the binary builds
3. **Run `make test` after any change to readers, config, or dataset code**
4. **Run `make integration-test` after changes to engine code** (redis, vectorsets)
5. **Never bypass `make` targets** — use them for all build/test/check operations

## Project Structure

```
src/
  lib.rs                          # Library: readers, config, redis_client
  readers/
    mod.rs                        # Reader exports + unit tests
    hdf5_reader.rs                # HDF5 format (.hdf5, .h5)
    jsonl_reader.rs               # JSONL format (vectors.jsonl, queries.jsonl, neighbours.jsonl)
    npy_reader.rs                 # NPY format (vectors.npy)
    compound_reader.rs            # Compound/TAR format (vectors.npy + payloads.jsonl + tests.jsonl)
    metadata.rs                   # Metadata types (MetadataItem, MetadataValue)
  config.rs                       # RedisConfig from environment
  redis_client.rs                 # Redis connection management
  bin/
    vector_db_benchmark/
      main.rs                     # CLI entry point
      cli.rs                      # Clap argument parsing
      config.rs                   # Dataset/engine config loading from JSON
      dataset.rs                  # Dataset wrapper (path resolution, reading, auto-download)
      download.rs                 # HTTP download + tgz extraction
      experiment.rs               # Experiment runner (upload + search loop)
      engine/
        mod.rs                    # Engine trait + SearchResults
        redis.rs                  # Redis/RediSearch engine (FT.CREATE, HSET, FT.SEARCH)
        valkey.rs                 # Valkey Search engine (RESP, uses redis crate)
        vectorsets.rs             # VectorSets engine (VADD, VSIM)
        redis_utils.rs            # Shared utils: commandstats validation
tests/
  integration_redis.rs            # Integration tests (requires redis:8.8.0 on port 6399)
datasets/
  datasets.json                   # Dataset registry (names, paths, download links)
experiments/
  configurations/*.json           # Engine configurations (HNSW params, search params)
```

## Dataset Formats

- **HDF5** (`.hdf5`): Single file with `train`, `test`, `neighbors` datasets
- **JSONL** (`type: "jsonl"`): Directory with `vectors.jsonl`, `queries.jsonl`, `neighbours.jsonl`
- **Compound/TAR** (`type: "tar"`): Directory with `vectors.npy`, `payloads.jsonl`, `tests.jsonl`

Datasets are auto-downloaded from the `link` URL in `datasets.json` when not found locally.

## Key Patterns

- **Parallel upload/search**: `thread::scope` + `AtomicUsize` work-stealing across batches
- **Vector encoding**: f32 little-endian bytes for both Redis and VectorSets
- **Score conversion**: VectorSets uses `1.0 - score` (1=identical, 0=opposite)
- **Config resolution**: `./datasets/` → site-packages → `v0/datasets/`

## Environment

- `HDF5_DIR`: Path to HDF5 library (default: `/usr/lib/x86_64-linux-gnu/hdf5/serial`)
- `REDIS_HOST`, `REDIS_PORT`, `REDIS_PASSWORD`: Redis connection (default: `localhost:6379`)
- Docker: `tests/docker-compose.test.yml` runs redis:8.8.0 on port 6399 for integration tests

## Migration from Python (v0/)

The `v0/` directory contains the original Python implementation. The Rust version is a partial migration — only Redis (RediSearch) and Redis (VectorSets) engines are ported.

### Engine coverage

| Engine | Python (v0/) | Rust | Client Library |
|--------|:---:|:---:|----------------|
| Redis / RediSearch | `redis` | `redis` | `redis` 0.27 |
| VectorSets | `vectorsets` | `vectorsets` | `redis` 0.27 |
| Elasticsearch | `elasticsearch` | `elasticsearch` | `elasticsearch` 8.15 |
| Milvus | `milvus` | `milvus` | `reqwest` (REST API v2) |
| OpenSearch | `opensearch` | `opensearch` | `opensearch` 2.3 |
| pgvector | `pgvector` | `pgvector` | `postgres` 0.19 + `pgvector` 0.4 |
| Qdrant | `qdrant` | `qdrant` | `qdrant-client` 1.17 (gRPC) |
| Weaviate | `weaviate` | `weaviate` | `tonic`/`prost` (gRPC search) + `reqwest` (REST schema) |
| MongoDB | — | `mongodb` | `mongodb` 3 (sync) |
| Valkey | — | `valkey` | `redis` 0.27 \* |
| Turbopuffer | — | `turbopuffer` | `turbopuffer-client` 0.0.4 |

\* Valkey GLIDE has no Rust crate ([valkey-io/valkey-glide#828](https://github.com/valkey-io/valkey-glide/issues/828), closed NOT_PLANNED). GLIDE maintainers recommend `redis-rs` for Rust.

Python engine configs use engine names like `redis`, `vectorsets`. Rust engine configs use the same.

### Precision validation

When migrating or modifying search logic, **always compare precision output** between the Rust and Python versions to verify correctness. Both versions save results as JSON files in `results/`.

**Python v0 search result JSON fields:**
```json
{
  "total_time": 1.23,
  "mean_time": 0.0012,
  "mean_precisions": 0.95,
  "std_time": 0.0003,
  "min_time": 0.0008,
  "max_time": 0.0021,
  "rps": 820.5,
  "p50_time": 0.0011,
  "p95_time": 0.0018,
  "p99_time": 0.0020,
  "precisions": [0.9, 1.0, ...],
  "latencies": [0.001, 0.0012, ...]
}
```

**Rust search result JSON** uses the same field names as Python v0 for timing/throughput (`rps`, `p50_time`, …). Both versions write to `results/` with filename format: `{engine}-{dataset}-search-{id}-{pid}-{timestamp}.json`.

#### Quality metric keys: `mean_precisions` is NOT ours (#217)

Python v0 — and upstream `qdrant/vector-db-benchmark`, `engine/base_client/search.py` — computes `len(ids & expected[:top]) / top`, i.e. **recall@top**, and publishes it under the key `mean_precisions`. Our Rust build emits, since **schema version 2**:

| key | formula | notes |
|---|---|---|
| `mean_precision_at_returned` | `hits / |results returned|` | was `mean_precisions` before schema v2 — the rename is what closed #217 |
| `mean_recall` | `hits / |valid ground-truth ids in expected[:top]|` | equals Python/upstream `mean_precisions` **only** when every ground-truth row has >= `top` valid ids |
| `precisions_at_returned` | per-query array of the above precision | only under `--dump-raw-latencies`; was `precisions` |
| `precision_at_returned_dist` | digest of that array | was `precision_dist` |

Every result file carries a top-level `metrics_schema` block with these formulas plus a `ground_truth` width profile and a `comparable_to_upstream_mean_precisions` field naming the key (if any) that can be overlaid on upstream numbers. **We never emit a key named `mean_precisions`** — the same name for two formulas is the state that must not ship.

Calibration (`calibration_precision`) targets `mean_precision_at_returned`; when the dataset's ground truth is narrower than `top` the target can be unreachable by construction, which the run now warns about and records in `params.calibration.reached_target`.

### How to compare precision

1. Run the same dataset + engine config on both versions:
   ```bash
   # Python v0
   cd v0 && poetry run vector-db-benchmark --engines "redis-m-16-ef-128" --datasets "h-and-m-2048-angular-filters"
   # Rust
   ./target/release/vector-db-benchmark --engines "redis-m-16-ef-128" --datasets "h-and-m-2048-angular-filters"
   ```
2. Compare `mean_precisions` (Python) vs **`mean_recall`** (Rust) — the matching pair; values should agree within floating-point tolerance on full-width ground truth. `scripts/v0_check.sh` does this mapping for you. Comparing Python `mean_precisions` against Rust `mean_precision_at_returned` compares recall with precision.
3. If it differs, check: score conversion, neighbor ordering, distance metric, top-k cutoff logic — and whether the dataset's ground-truth rows are shorter than `top` (`metrics_schema.ground_truth.queries_with_fewer_than_top_neighbours` in the Rust output), in which case the two denominators legitimately disagree.
