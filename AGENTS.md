# Agent Instructions for vector-db-benchmark

## Project Overview

Rust implementation of a vector database benchmarking tool. **15 engines**:
Redis (RediSearch), VectorSets, Valkey, Dragonfly, KiviDB, Elasticsearch,
OpenSearch, Qdrant, pgvector, Weaviate, Milvus, MongoDB, Chroma, Turbopuffer,
Vertex AI.

The thing this repo optimises for is **not being wrong**. Its dominant historical
bug class is a run that completes, reports a plausible number, and measured
something other than what its config name claims. Most of the unusual-looking
rigour below exists because of a specific incident.

## Verify your change with ONE command

```bash
make agent-check
```

That is the entire no-Docker CI gate — fmt, clippy `-D warnings`, and the unit /
binary / invariant suites **in release** — in CI's order and profile. If it
passes, the non-Docker CI jobs will pass.

### Read this before trusting a green local run

- **`make check` runs NO TESTS.** It is `fmt-check` + `lint` only. It is not a
  sufficient gate; `make agent-check` is.
- **CI runs the test suites with `--release`.** A `debug_assert!` therefore does
  not exist in CI, or in the binaries this repo ships. If you write a guard, make
  it a returned `Err` or a real `assert!`, and run the suite under both profiles.
- **Match CI's Rust toolchain.** CI uses default `stable`. A newer stable adds
  clippy lints your local version does not have, so `make check` can be clean
  locally and fail CI on `-D warnings`. `rustup update stable` before concluding
  a lint failure is spurious.
- **Rebuild `--release` before copying a binary anywhere.** `cargo test` builds
  debug; `scp target/release/<bin>` after it ships the *previous* build.

## Other Make targets

```bash
make vector-db-benchmark # Build the main CLI binary (release)
make build               # Build all binaries (release)
make test                # Unit tests only, debug profile (no Docker)
make check               # fmt + clippy ONLY — see the warning above
make fmt                 # Auto-format
make prepare-msmarco     # Build an MS MARCO corpus (see below)
make v0-check            # Compare Rust vs Python v0 (precision, QPS, latency)
make clean
```

Per-engine integration tests, each starts and stops its own containers:

```bash
make integration-test                 # Redis
make integration-test-{valkey,dragonfly,kividb,chroma}
make integration-test-{pgvector,qdrant,elasticsearch,opensearch}
make integration-test-{weaviate,milvus,mongodb}
make integration-test-vertex          # cloud-only, needs real GCP credentials
make integration-test-no-docker
```

## Workflow Rules

1. **Run `make agent-check` after any code change.** Not `make check` alone.
2. **Run the matching `make integration-test-<engine>`** after touching an
   engine. Changing shared code under `engine/` (e.g. `redis_utils.rs`,
   `filter_guard.rs`, `index_naming.rs`) affects several engines — run each.
3. **Put logic in `src/` (the library), not in `src/bin/`.** The library is
   unit-tested; the binaries are meant to be I/O only. The one regression in the
   MS MARCO work was offset-selection logic that ended up in a binary and so had
   no test to mutate.
4. **Mutate your guard, not just your code.** Break the thing the guard is named
   after and confirm the suite goes red. A guard that passes either way is
   decoration — this is the repo's most common review finding.
5. **Never bypass `make` targets** for build/test/check.

## Project Structure

```
src/
  lib.rs              # Library root — put LOGIC here, it is what gets unit-tested
  readers/            # hdf5, jsonl, npy, compound(tar), sparse, multivector, metadata
  msmarco.rs          # MS MARCO corpus prep: sampling, oracle, NPY writers (unit-tested)
  synthetic.rs        # Generators for the synthetic-* fixtures
  query_filter.rs     # The ONE door from a dataset's `conditions` JSON to a filter (#219)
  metrics.rs          # recall / precision / MRR / NDCG — see the metric-naming note below
  start_gate.rs       # Worker barrier for parallel search
  config.rs, redis_client.rs, parsers.rs
  bin/
    vector_db_benchmark/
      main.rs cli.rs config.rs dataset.rs download.rs experiment.rs ground_truth.rs
      engine/         # 15 engines + shared: mod.rs redis_utils.rs filter_guard.rs
                      #   index_naming.rs geo.rs
    generate_dataset.rs   # -> `generate-dataset`, writes the synthetic-* fixtures
    prepare_msmarco.rs    # -> `prepare-msmarco`, builds/verifies MS MARCO corpora
tests/                # integration_<engine>.rs, plus harness_invariants +
                      #   overhead_invariants (both run in CI, no Docker)
datasets/datasets.json        # Dataset registry
experiments/configurations/   # Engine configs (HNSW params, search params)
```

## Dataset Formats

- **HDF5** (`.hdf5`): Single file with `train`, `test`, `neighbors` datasets
- **JSONL** (`type: "jsonl"`): Directory with `vectors.jsonl`, `queries.jsonl`, `neighbours.jsonl`
- **Compound/TAR** (`type: "tar"`): Directory with `vectors.npy`, `payloads.jsonl`, `tests.jsonl`

Datasets are auto-downloaded from the `link` URL in `datasets.json` when not found locally.

Entries with **no `link`** do not exist until a tool writes them:

- `synthetic-*` — `cargo run --release --bin generate-dataset`
(The `msmarco-cohere-1024-*` entries used to be in this list. **All six now have
`link`s** and auto-download like any other dataset — see below.)

## MS MARCO datasets

`msmarco-cohere-1024-{100K,1M,10M}-cosine` and their `-crc32-` twins: MS MARCO
v2.1 passages + Cohere embed-v3 vectors **and** real document metadata. All six
auto-download from S3; `prepare-msmarco` rebuilds them from Hugging Face.

- Logic lives in `src/msmarco.rs` (unit-tested); the binary is I/O only. That
  convention is load-bearing — the one regression in this area was
  offset-selection logic that ended up in the binary and so had no test.
- Size is fixed by the dataset name, never a flag. The `-crc32-` variants declare
  their REALIZED count (99,964 / 1,000,044 / 9,999,959), since a hash threshold
  cannot land on a round number.
- Ground truth is brute-forced per corpus and cross-checked against the upstream
  global top-1k before anything is written; `--verify` re-runs that check against
  an already-prepared dataset.
- `--discover-crc32` re-derives the sampling thresholds (26.9 GB metadata scan).

## Traps that have actually cost time here

Each of these produced a real incident, and none of them look wrong while you are
doing them.

- **A config key that is parsed and then never applied.** The run completes and
  the result JSON names a tuning knob it did not use. Recall-based tests cannot
  catch this — the numbers stay plausible. After adding a knob, assert the SERVER
  reports it (`FT.INFO`, `indexes/describe`, …), not that the code read it.
- **A guard written one dimension narrower than the bug it names**, so it passes
  either way and its name convinces everyone the case is covered. Always mutate
  the guard.
- **Ground truth that is subtly for a different corpus.** Row counts match, ids
  are in range, recall comes out "reasonable" and is meaningless. Ground truth
  must be derived from, or cross-checked against, the exact bytes uploaded.
- **`top` derived from the ground-truth width.** A search config without `top`
  takes it from the dataset's neighbour count. On a 1000-wide dataset that runs
  every point at k=1000, and HNSW raises effective breadth to at least k — so an
  `ef` sweep publishes one measurement under several config names. Set `top`.
- **Two agents on one branch or worktree** silently clobber each other's commits.
  One nearly reverted an already-merged PR. Verify the SHA you push is the SHA
  you gated, and check the tree contents, not just a green build.
- **`--skip-upload` against a partially-loaded server** scores recall over a
  fraction of the corpus without erroring. The completeness gate exists for this.

## Datasets: what is generated vs downloaded

Most entries in `datasets/datasets.json` auto-download from their `link`. Two
families do not ship that way by default:

- `synthetic-*` — `cargo run --release --bin generate-dataset`. Small,
  fixed-seed, exist to exercise sparse / hybrid / multivector / filter code paths.
- `msmarco-cohere-1024-*` — see the MS MARCO section below.

A dataset whose `path` names its own size (`…/1M`) must declare a matching
`vector_count`; `config.rs` enforces this, because a mismatch once made a sweep
score recall over 0.01% of a corpus. A deliberate subset gets a path that does
not claim a size it does not have.

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

The `v0/` directory contains the original Python implementation. The Rust port
now covers more engines than `v0/` did; the table below is the mapping.

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
| Dragonfly | — | `dragonfly` | `redis` 0.27 (RESP) |
| KiviDB | — | `kividb` | `redis` 0.27 (RESP) |
| Chroma | — | `chroma` | `reqwest` (REST v2) |
| Vertex AI | — | `vertex` | `tonic`/`prost` (gRPC) + `reqwest` — cloud-only |

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
