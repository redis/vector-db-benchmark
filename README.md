# vector-db-benchmark

A benchmarking tool for vector databases, written in Rust. Measures upload throughput, search QPS, latency percentiles (p50/p95/p99), and recall for vector search engines.

## Supported Engines

| Engine | Client Library | Protocol | Distance Metrics | Metadata Filters |
|--------|---------------|----------|-----------------|-----------------|
| **Redis** (RediSearch) | `redis` 1.3 | Redis protocol | L2, Cosine, IP | Yes |
| **VectorSets** | `redis` 1.3 | Redis protocol | L2, Cosine | Yes |
| **Elasticsearch** | `elasticsearch` 8.15 | HTTP/REST | L2, Cosine | Yes |
| **OpenSearch** | `opensearch` 2.4 | HTTP/REST | L2, Cosine | Yes [\*\*\*\*\*\*\*](#opensearch-note) |
| **Qdrant** | `qdrant-client` 1.18 | gRPC | L2, Cosine, Dot | Yes |
| **PgVector** | `postgres` 0.19 + `pgvector` 0.4 | PostgreSQL | L2, Cosine | Yes |
| **Weaviate** | `tonic` 0.12 / `prost` 0.13 (gRPC) + `reqwest` (REST) | gRPC (search) + HTTP/REST (schema) [\*\*](#weaviate-protocol-note) | L2, Cosine, Dot | Yes |
| **Milvus** | `reqwest` (REST API v2) | HTTP/REST | L2, Cosine, IP | Yes |
| **MongoDB** (Atlas Search) | `mongodb` 3 (sync) | MongoDB protocol | Euclidean, Cosine, Dot | Yes |
| **Valkey** (Valkey Search) | `redis` 1.3 [\*](#valkey-client-note) | RESP protocol | L2, Cosine, IP | Yes |
| **Turbopuffer** | `turbopuffer-client` 0.0.4 | HTTP/REST (cloud) | Cosine, Euclidean | Yes |
| **Dragonfly** (Dragonfly Search) | `redis` 1.3 | RESP protocol | L2, Cosine, IP | Yes [\*\*\*](#dragonfly-note) |
| **Vertex AI** (Vector Search) | `reqwest` (Vertex AI REST v1) | HTTP/REST (cloud) | L2, Cosine, Dot | Yes [\*\*\*\*](#vertex-note) |
| **Chroma** | `reqwest` (Chroma v2 REST API) | HTTP/REST | L2, Cosine, IP | Yes [\*\*\*\*\*](#chroma-note) |
| **KiviDB** | `redis` 1.3 | RESP protocol | L2, Cosine, IP | Yes [\*\*\*\*\*\*](#kividb-note) |

<a id="valkey-client-note"></a>
\* **Valkey client note:** Valkey GLIDE has no published Rust crate ([valkey-io/valkey-glide#828](https://github.com/valkey-io/valkey-glide/issues/828), closed NOT_PLANNED). The GLIDE maintainers recommend using `redis-rs` for Rust and upstream their improvements to it. The `redis` crate works with Valkey since it speaks the same RESP protocol.

<a id="weaviate-protocol-note"></a>
\*\* **Weaviate protocol note:** Vector search runs over Weaviate's **gRPC** API (port 50051) by default — the high-throughput query path used by the official clients (packed binary vectors). Schema management, upload, and search-time `ef` tuning use the REST API (v1). Filtered searches also run over gRPC — metadata filter conditions are translated to the gRPC `Filters` message. The tool falls back to the slower GraphQL-over-HTTP search path only when a filter condition can't be expressed in gRPC or when `WEAVIATE_USE_GRAPHQL` is set. Override the gRPC port with `WEAVIATE_GRPC_PORT`.

<a id="dragonfly-note"></a>
\*\*\* **Dragonfly note:** Uses **Dragonfly Search** (Beta), the RediSearch-compatible `FT.*` subset Dragonfly ships (`FT.CREATE`/`FT.SEARCH`/`FT.INFO`/`FT.DROPINDEX`, `VECTOR` FLAT/HNSW, `*=>[KNN k @field $blob AS score]`). This engine supports **vector KNN + metadata filtering** — Dragonfly Search implements RediSearch TAG/NUMERIC/TEXT hybrid filtered KNN (`(prefilter)=>[KNN...]`, verified live against `df-v1.38.1`), so the engine indexes the dataset's metadata schema and applies filter `conditions` exactly like Redis/Valkey: keyword/int/float/bool/datetime/uuid datatypes, `match`/`match_any`/`range`, and AND/OR/nested boolean. **GEO** is the one unsupported filter type (Dragonfly's geo-query parser rejects the `$param` placeholders the shared RediSearch builder emits, like Chroma/Milvus). No mixed (search+update) workload, no quantization. Dragonfly Search supports only the **float32** vector type, so vectors are always encoded as FLOAT32. Runs over the RESP protocol via Docker; set the host port with `DRAGONFLY_PORT` (default `6385`). Upload concurrency/batch come from the engine config but can be overridden at runtime with `DRAGONFLY_UPLOAD_PARALLEL` / `DRAGONFLY_UPLOAD_BATCH_SIZE` (env takes precedence) — managed **Dragonfly Cloud** resets connections under the default 100-thread upload burst on larger-dimensional datasets, so cloud runs set `DRAGONFLY_UPLOAD_PARALLEL=16`; search throughput is unaffected by upload concurrency.

<a id="vertex-note"></a>
\*\*\*\* **Vertex AI note:** Uses **Vertex AI Vector Search** (Google Cloud) — a STREAM_UPDATE tree-AH index streamed with `upsertDatapoints`, queried with `findNeighbors`. Cloud-only (no local server), like Turbopuffer. **Metadata filters** are supported: on upload, string/`labels` fields become categorical `restricts` and int/float fields become `numericRestricts` on each datapoint; on query, `match`/`range` conditions translate to Vertex query restrictions over REST **and** both gRPC transports. Vertex restrictions AND across fields and OR within a field's `allowList`, so an `and` of per-field conditions maps directly — but a filter Vertex cannot express (cross-field `or`, nested boolean, a numeric `match_any` IN-list, or geo) is a **hard error** rather than a silently partial filter. **Mixed workload** (`--update-search-ratio`) is supported: each worker interleaves S `findNeighbors` searches with U single-datapoint `upsertDatapoints` updates, reporting search recall/latency alongside update RPS/latency. Required: `VERTEX_PROJECT`; auth is `VERTEX_ACCESS_TOKEN` if set, else `gcloud auth print-access-token`. Optional: `VERTEX_REGION` (default `us-central1`), `VERTEX_MACHINE_TYPE` (default `e2-standard-16`), `VERTEX_DEPLOY_TIMEOUT_SECS` (default `3600`), and index-tuning knobs `VERTEX_APPROX_NEIGHBORS` / `VERTEX_LEAF_EMBEDDING_COUNT` / `VERTEX_LEAF_SEARCH_PERCENT`. **Deploying an index takes tens of minutes**; to skip the create+deploy step, point at an already-deployed index with `VERTEX_INDEX`, `VERTEX_INDEX_ENDPOINT`, and `VERTEX_DEPLOYED_INDEX_ID` (in that case the tool leaves those resources in place on cleanup). Query-time recall/latency is tuned per search config via `search_params.fraction_leaf_nodes_to_search_override` (0..1) and `num_candidates` (→ `approximateNeighborCount`). A config's `num_candidates` is honored (clamped to `top`, which Vertex requires as the floor); when it's **unset** the query runs at the index's own configured `approximateNeighborsCount` sent **explicitly** — never Vertex's silent `0` "use index default" sentinel — and the effective knobs are **logged per config** (`Vertex effective search knobs: approximateNeighborCount=… (config|index-default), …`) so a sweep point is honestly labeled rather than silently measured at the default (fairness gate, #200). Upload streams `upsertDatapoints` concurrently (`upload_params.parallel`); each `upsertDatapoints` request is bounded by payload size, so **very wide datasets may need a smaller `batch_size`** than the default 1000 to stay under the request limit. **Batch ingest (experimental, #187):** setting `VERTEX_GCS_STAGING_BUCKET` switches the index to `BATCH_UPDATE` and, on upload, stages every datapoint to a JSONL object under `gs://<bucket>/vdbb-batch/<display-name>/` and triggers a single index rebuild via `contentsDeltaUri` — avoiding the per-project streaming write quota on large corpora (`VERTEX_BATCH_THRESHOLD`, default `100000`, is the recommended cross-over size). This path was **live-validated** (1000×8d ingested to `vectorsCount=1000` in ~1 min on a fresh `BATCH_UPDATE` index); it ships behind the opt-in env var, so leave the bucket unset for the default streaming ingest.

<a id="chroma-note"></a>
\*\*\*\*\* **Chroma note:** Uses **Chroma** (OSS) via its **v2 REST API** — a collection of records (`ids` + `embeddings` + scalar `metadatas`) queried with `query` + a `where` document. **Metadata filters** map directly onto the canonical model: `match` → `$eq`, `match_any` → `$in`, `range` → `$gte`/`$gt`/`$lte`/`$lt`, and **AND / OR / nested boolean** to Chroma's native `$and` / `$or` (which nest arbitrarily). Supported datatypes: **keyword, int, float, bool, uuid, datetime** (stored as epoch-seconds int, like Milvus, so numeric range operators apply). **Full-text** (`{match:{text}}`) is supported via `where_document` `$contains` — the `text`-typed field's value is uploaded as each record's Chroma `document`. **NOT supported** by Chroma's metadata engine (those conditions are dropped, like Dragonfly's documented limits): **geo-radius** and multi-valued **`labels` arrays** (Chroma metadata values are scalar only). Runs over HTTP/REST via Docker; set the host port with `CHROMA_PORT` (default `8000`, test compose maps `8003`), and optionally `CHROMA_COLLECTION` / `CHROMA_TENANT` / `CHROMA_DATABASE`. Distance space (`l2`/`cosine`/`ip`) is set per collection from the dataset metric.

<a id="kividb-note"></a>
\*\*\*\*\*\* **KiviDB note:** [KiviDB](https://kividb.io) is a Redis-wire-compatible (RESP2) data store that implements a RediSearch-compatible `FT.*` subset (`FT.CREATE`/`FT.SEARCH`/`FT.INFO`/`FT.DROPINDEX`, `VECTOR` HNSW/FLAT, `*=>[KNN k @field $blob AS score]`). Supports **vector KNN + metadata filtering**: TAG/NUMERIC/TEXT datatypes, `match`/`match_any`/`range` (inclusive **and** exclusive bounds), and AND/OR/nested boolean, all verified live against filtered ground truth. KiviDB genuinely **pre-filters** (measured recall ~1.0 down to ~2% filter selectivity), so its filtered numbers are meaningful. KiviDB's `FT.CREATE` supports only the **float32** vector type.

The filter path is **not** the shared RediSearch builder used by Redis/Valkey/Dragonfly — KiviDB's `FT.SEARCH` diverges from RediSearch in ways that would otherwise produce silently wrong recall (issue #205), so the engine has its own emitter (`kividb_filter` in `kividb.rs`, where every divergence is documented with the measurement behind it). The headline one: **KiviDB does not substitute `$param` placeholders in a filter expression, in any query form** — a plain `FT.SEARCH` fails identically to a hybrid one, so this is not a prefilter-specific quirk — and all filter values are therefore inlined as literals. It also does not accept RediSearch backslash escaping, RediSearch's `(` exclusive-range marker, or `-` negation, and it reads a spaced intra-brace TAG-OR (`@f:{a | b}`) as match-all.

**Not supported on KiviDB**, and therefore a hard **error** rather than a silently dropped condition (following `vertex.rs`; a dropped clause would run kNN over the whole corpus and publish a recall for a filter that was never applied):
- **GEO** — KiviDB's index schema has no GEO field type at all, so the field is never indexed (unlike Dragonfly, which has a GEO type but rejects this tool's `$param` geo bounds at the query-parser level).
- **Multi-valued `labels` arrays** — a KiviDB TAG value is *atomic*, never split on a separator (`FT.CREATE` rejects the `SEPARATOR` modifier, and `@f:{b}` does not match a stored `a;b;c`), so `match_any` over a labels field could only ever match the whole joined string.
- **Multi-term or non-alphanumeric full-text terms** — KiviDB's TEXT matcher handles a single alphanumeric term per `@field:(...)` clause and returns zero documents otherwise.
- **TAG filter values containing `|`, `@`, `(` or `)`** — inexpressible with no escaping mechanism. `|` is always the TAG-OR separator (under-match), `@` degrades the whole query to match-all (over-match), and `(`/`)` are parsed as grouping as soon as the clause sits inside a group, which every non-trivial filter does (`@t:{zz(zz} @u:{blue}` returns 0; `(@t:{zz)zz})` returns the whole corpus). This is a **query-side** limit only: a *stored* value containing one of these is inert — measured, a stored `a|b` is not split, and a corpus seeded with such values returns identical counts to a clean one for every clause shape the builder emits — so uploads are never rejected for it. That matters in practice: h-and-m stores a paren in `prod_name` on 3,900 documents and benchmarks correctly, because none of the 586 distinct filter values in its shipped queries contains one.

The first two are decidable from the dataset schema, so KiviDB rejects them in the **configure** phase — before the index is created and before the corpus is read — leaving no partial upload, no orphan result file and no populated keyspace behind.

> **Which shipped datasets this excludes.** **3 of the 53** datasets in `datasets/datasets.json` cannot run on KiviDB at all and now abort instead of producing a result file: `random-geo-radius-100-angular-filters` and `random-geo-radius-2048-angular-filters` (geo), and `arxiv-titles-384-angular-filters` (multi-valued `labels` — the only shipped dataset that declares a `labels` field). Before this was fixed these three *did* run and published a recall near 0.0, which was visibly wrong; they now publish nothing at all, and no result file records the rejection, so `--plot` simply shows a gap for KiviDB on those datasets. If you are sweeping KiviDB across everything, either exclude the three from `--datasets` (`arxiv-titles-384-angular-no-filters` is the pure-KNN twin of the arxiv one) or pass `--exit-on-error false` so the rest of the sweep continues. Every other dataset — including all filtered ones — is unaffected.

One real protocol difference worth knowing: KiviDB's `FT.INFO` does not expose RediSearch's `num_docs`/`percent_indexed` — it reports HNSW graph state directly instead (`hnsw_live_count`, `hnsw_compaction_in_progress`), because it builds each vector's HNSW entry synchronously inside the `HSET` that stores it (no async backfill phase exists to report progress on); `wait_for_indexing` polls those fields instead and returns immediately as a result. RESP2 only (no RESP3 opt-in). Set the host port with `KIVIDB_PORT` (default `6380` — KiviDB's own default listen port, **not** Redis's 6379).

<a id="opensearch-note"></a>
\*\*\*\*\*\*\* **OpenSearch note:** Connection is set with `OPENSEARCH_PORT` (default `9200`), `OPENSEARCH_INDEX` (default `bench`), `OPENSEARCH_TIMEOUT` seconds (default `300`), and `OPENSEARCH_USER` / `OPENSEARCH_PASSWORD`.

**Shard count** is read from `collection_params.number_of_shards`. Leave it unset to inherit the cluster default — but note that default is **1 on open-source OpenSearch and historically 5 on Amazon OpenSearch Service**, and shard count materially changes vector indexing speed and precision, so a run that leaves it unset is not comparable across engine versions or deployments. `elasticsearch.rs` pins `number_of_shards: 1`, so pin it here too for an apples-to-apples ES-vs-OS comparison. The effective value is printed per config (`OpenSearch: HNSW { … }, number_of_shards: 5|cluster-default`), and a present-but-non-integer value is a hard error rather than a silent fallback to the default.

**Retries against a managed domain.** Amazon OpenSearch Service returns states an open-source single-node cluster never produces: HTTP 429 on bulk (its `knn.algo_param.index_thread_qty` defaults to 1, so HNSW graph building is single-threaded and cannot drain as fast as a parallel uploader pushes), 400 `snapshot_in_progress_exception` on delete (automated snapshots cannot be disabled), 503 `process_cluster_event_timeout_exception` on create, and 502/504 from the front door on force-merge. All four paths retry with jittered exponential backoff, tunable with:

| Variable | Default | Applies to |
|---|---|---|
| `OPENSEARCH_BULK_MAX_RETRIES` | `8` | bulk upload |
| `OPENSEARCH_BULK_RETRY_BASE_MS` | `500` | bulk upload |
| `OPENSEARCH_INDEX_OP_MAX_RETRIES` | `10` | create / delete / refresh / force-merge |
| `OPENSEARCH_INDEX_OP_RETRY_BASE_MS` | `2000` | create / delete / refresh / force-merge |
| `OPENSEARCH_SEARCH_MAX_RETRIES` | `5` | search |
| `OPENSEARCH_SEARCH_RETRY_BASE_MS` | `50` | search |
| `OPENSEARCH_SEARCH_RETRY_BUDGET_MS` | `2000` | search (total wall-clock ceiling per query) |

Search retries are deliberately short and additionally bounded by a wall-clock budget, because a single transport attempt can block for `OPENSEARCH_TIMEOUT`. **Backoff is excluded from the reported latency**: a retried query is billed only for the attempt that produced its result, so the percentiles stay comparable with engines that do not retry. Queries that needed a retry are counted and reported separately (`⚠ N of M search queries succeeded only after a retry`), since clean latency figures would otherwise hide that the server was shedding load.

Dropped queries are handled uniformly across all engines, not here: `failed_queries` is recorded in the results JSON and warned about, and `--fail-on-dropped-queries` makes it fatal. See that flag's help for why it is off by default.

<details>
<summary><b>Runbook: benchmarking against Vertex AI</b></summary>

```bash
# 1. Auth + enable the API (once per project).
gcloud config set project <your-project>
gcloud services enable aiplatform.googleapis.com
export VERTEX_PROJECT=<your-project>
export VERTEX_REGION=us-central1
# VERTEX_MACHINE_TYPE defaults to e2-standard-16 (the smallest type the default
# shard size accepts — e2-standard-2 is rejected at deploy).

# 2. Full run — creates + DEPLOYS a fresh index (slow, ~30-40 min), uploads,
#    searches, then tears the resources back down.
vector-db-benchmark --engines vertex-default --datasets random-100 --skip-if-exists false

# 3. Fast iteration — reuse an already-deployed index and skip the deploy.
#    (grab the ids the first run printed; cleanup then LEAVES them in place)
export VERTEX_INDEX=projects/P/locations/us-central1/indexes/ID
export VERTEX_INDEX_ENDPOINT=projects/P/locations/us-central1/indexEndpoints/EID
export VERTEX_DEPLOYED_INDEX_ID=vdb_benchmark_deployed
vector-db-benchmark --engines vertex-default --datasets random-100 --skip-if-exists false
```

The gated `integration_vertex` test drives this same flow and asserts a recall floor; it self-skips unless `VERTEX_PROJECT` is set:

```bash
VERTEX_PROJECT=<your-project> \
  cargo test --test integration_vertex --release -- --nocapture
```
</details>

```
docker run --rm --network=host redis/vector-db-benchmark:latest \
  --host localhost --engines 'redis-single*' --datasets glove-25-angular
(...)
============================================================
Running experiment: redis-single-node - glove-25-angular
============================================================
Experiment stage: Configure
Using algorithm hnsw with config {'M': 16, 'EF_CONSTRUCTION': 128}
Experiment stage: Upload
Reading dataset from datasets/glove-25-angular/...
Read 1183514 vectors (25d) in 0.82s
Upload time: 12.3s (96,220 records/sec)
Experiment stage: Search
  Running search 0: ef=128, parallel=4
  → QPS: 3214.5, Precision: 0.9785
  Running search 1: ef=128, parallel=8
  → QPS: 5891.2, Precision: 0.9785
Experiment stage: Done
```

> [View published results](https://redis.io/blog/benchmarking-results-for-vector-databases/)

## Quick Start

### Docker (recommended)

```bash
# Show help
docker run --rm redis/vector-db-benchmark:latest --help

# List available datasets and engines
docker run --rm redis/vector-db-benchmark:latest --describe datasets
docker run --rm redis/vector-db-benchmark:latest --describe engines

# Run a benchmark against a local Redis instance
docker run --rm --network=host \
  -v $(pwd)/datasets:/code/datasets \
  -v $(pwd)/results:/code/results \
  redis/vector-db-benchmark:latest \
  --host localhost --engines 'redis-single*' --datasets glove-25-angular
```

### Using with Redis

```bash
# Start Redis
docker run -d --name redis -p 6379:6379 redis:8.8.0

# Run benchmark
docker run --rm --network=host \
  -v $(pwd)/datasets:/code/datasets \
  -v $(pwd)/results:/code/results \
  redis/vector-db-benchmark:latest \
  --host localhost --engines redis-docker-test --datasets random-100

# Clean up
docker stop redis && docker rm redis
```

### Using Docker Compose

```bash
# Full integration test (downloads h-and-m dataset ~200MB)
make docker-integration

# Fast smoke test (uses random-100 dataset baked into image, 228KB)
make docker-integration-fast
```

### Build from source

```bash
# Prerequisites: Rust toolchain, libhdf5-dev, pkg-config
cargo build --release --bin vector-db-benchmark

# Run
./target/release/vector-db-benchmark --help
./target/release/vector-db-benchmark --describe datasets
./target/release/vector-db-benchmark \
  --host localhost --engines 'redis-single*' --datasets glove-25-angular
```

## CLI Options

```
Usage: vector-db-benchmark [OPTIONS]

Options:
    --engines <PATTERN>        Engine config patterns (wildcards supported) [default: *]
    --engines-file <PATH>      Path to JSON file with custom engine configs
    --datasets <PATTERN>       Dataset patterns (wildcards supported) [default: *]
    --host <HOST>              Redis/engine host [default: localhost]
    --parallels <N,N,...>      Filter by parallel thread counts
    --ef-runtime <N,N,...>     Filter by ef runtime values
    --skip-upload              Skip the upload phase
    --skip-search              Skip the search phase
    --skip-if-exists           Skip if results already exist
    --exit-on-error            Stop on first error
    --timeout <SECS>           Timeout in seconds [default: 86400]
    --update-search-ratio <U:S> Mixed benchmark: interleave U updates per S searches
    --describe <TYPE>          Describe available 'datasets' or 'engines'
    -v, --verbose              Verbose output for --describe
    --plot <OUTPUT.svg>        Render a QPS-vs-precision trade-off chart from results/
    -h, --help                 Print help
```

### Per-config index isolation (Redis / Valkey / Dragonfly / KiviDB)

Each engine config gets its **own** RediSearch index and keyspace, derived from the
config `name`: index `"<base>:<config>"` (base `idx`) with docs keyed
`"<config>:<id>"`. This lets an M×EF_CONSTRUCTION **sweep** run all its configs
against one server and coexist, so you can upload every config once and then
search each in a later `--skip-upload` pass — each config reads its own graph, and
memory is reported per-index via `FT.INFO` (issue #151-4).

- `REDIS_INDEX_NAME` / `VALKEY_INDEX_NAME` / `DRAGONFLY_INDEX_NAME` / `KIVIDB_INDEX_NAME`
  now set the **base namespace**, not the whole index name; the config name is always appended.
  Set `<VAR>_EXACT=1` to use the base verbatim (single-config "point at an
  out-of-band index" case — combining it with >1 config for that engine is a
  startup error).
- Indexes/keys written by any **pre-#151-4** binary are incompatible — re-upload.
  `--skip-upload` against a missing/mismatched index now **hard-errors** instead of
  silently writing a `recall 0.0` file.
- Two-phase coexistence sweep: `… --keep-data` (upload + search all configs,
  keep the data), then `… --skip-upload --keep-data --skip-if-exists false`
  (search each against its own index). Per-config prefixing stores N copies of
  otherwise-identical sweep docs, so keyspace bytes scale ×N — the intended trade
  for isolation.
- **Shared-corpus (upload-once / build-many) mode — Redis, opt-in (#188):** for a
  sweep over **one** dataset where only the index params (M / EF_construction)
  differ, the corpus is identical across configs, so re-uploading it per config is
  wasted work (N× ingest — dominant at 10M+). Set `REDIS_KEY_PREFIX=<shared>:` to
  make **all** configs share ONE corpus keyspace: the first config uploads it, and
  every later config **skips the re-upload** (detected by a corpus key-count check)
  and just builds its own per-config index over the shared docs (the index name
  stays per-config). In this mode the index is dropped **without `DD`** so the
  shared corpus survives across configs; flush the DB (or the shared prefix) when
  the sweep finishes. Unset (the default) keeps full per-config isolation — no
  behavior change.

### Charts

Render a QPS-vs-precision trade-off plot (SVG, no dependencies) from existing `*-summary.json` results — one colored series per engine, filtered by `--engines`/`--datasets`:

```bash
# Compare all engines on one dataset
vector-db-benchmark --plot tradeoff.svg --engines '*' --datasets glove-100-angular
```

## Client resources & concurrency

**Concurrency model.** A search config's `parallel: N` (and the `--parallels` filter) runs **N OS threads that share a single in-memory copy** of the dataset and query set — not N processes. Raising `parallel` from 1 to 100 adds ~N worker threads and N engine connections (on the order of tens to low-hundreds of MB), **not** N× the dataset. Each worker accumulates only its own small latency/quality samples; nothing per-query is retained beyond scalar metrics.

**Peak client memory ≈ raw dataset size, during upload.** The client loads the whole dataset into RAM for the upload phase — roughly `vector_count × dim × 4 bytes` (e.g. `cohere-768-1M` ≈ 1M × 768 × 4 ≈ 3 GB, plus reader/allocator overhead) — then **frees it before the search phase**. Search holds only the (far smaller) query set plus the per-thread sample buffers, so search-phase memory is largely independent of `parallel`. A client with RAM ≥ ~2× the raw dataset size runs comfortably; increasing search concurrency does not materially change client memory.

> If you saw the client machine hang/OOM specifically when the client count rose (e.g. to 100) — as reported for the original **Python** `vector-db-benchmark`, which used a process-per-client model that copied the dataset into every worker process — this Rust implementation does not reproduce that: workers are threads sharing one copy, and the uploaded vectors are released before searching. Size the client for the upload peak above, not for the client count.

## Mixed Benchmarks (Update + Search)

The `--update-search-ratio` flag enables mixed workload benchmarks that interleave vector updates with searches. This measures how search performance is affected by concurrent write operations.

```bash
# 1 update per 10 searches
vector-db-benchmark --engines redis-docker-test --datasets random-100 \
  --update-search-ratio 1:10

# 1 update per 5 searches (heavier write load)
vector-db-benchmark --engines vectorsets-docker-test --datasets h-and-m-2048-angular \
  --update-search-ratio 1:5
```

The ratio format is `U:S` where U = number of updates and S = number of searches per cycle. Each worker thread performs S searches followed by U updates in a loop.

**Supported engines**: Redis, VectorSets, Valkey

Results JSON includes separate metrics for both operation types:

```json
{
  "results": {
    "rps": 5891.2,
    "precision": 0.9785,
    "p50_time": 0.00032,
    "p95_time": 0.00089,
    "p99_time": 0.00142,
    "update_rps": 589.1,
    "update_mean_time": 0.00045,
    "update_p50_time": 0.00041,
    "update_p95_time": 0.00098,
    "update_p99_time": 0.00156,
    "update_search_ratio": "1:10"
  }
}
```

Omitting the flag preserves the standard search-only benchmark behavior.

## Multi-tenancy

Multi-tenancy benchmarks model many tenants sharing **one** index: every search is scoped to a single tenant via an exact keyword-equality filter on a `tenant` field (`schema: { "tenant": "keyword" }`), and recall is measured against the nearest neighbours **within that tenant only**. This reuses the standard keyword-TAG filter path (no engine-specific code) and mirrors upstream qdrant/vector-db-benchmark's `random-768-*-tenants` scenario (registered here as `random-768-25-tenants`). The per-query filter looks like `{"and":[{"tenant":{"match":{"value":"tenant_7"}}}]}`. Because ground truth is tenant-local, recall is a strong isolation signal — a leaked cross-tenant document displaces a correct neighbour and lowers recall — and the tests assert **exact** per-query recall (`== 1.0` against an exact search, one query per tenant), so any single cross-tenant leak fails the check. Redis and Valkey are covered end-to-end (over both RESP2 and RESP3) by the `test_binary_{redis,valkey}_tenancy` integration tests. (Recall is necessary but not by itself sufficient to *prove* zero leakage; strict per-id membership checking is a possible future hardening.)

## Datasets

Most datasets are automatically downloaded on first use. The image includes `random-100` (228KB) for quick smoke tests. (Exception: `random-768-25-tenants` is a locally-generated placeholder with no public download link yet — see the Multi-tenancy section.)

| Dataset                                                                                                     | Dimensions |  Train size | Test size | Neighbors | Distance  |
| ----------------------------------------------------------------------------------------------------------- | ---------: |  ---------: | --------: | --------: | --------- |
| **LAION Image Embeddings (512D)**                                                                          |            |             |           |           |           |
| [LAION-1M: subset of LAION 400M English (image embedings)](https://laion.ai/blog/laion-400-open-dataset/)   |        512 |   1,000,000 |    10,000 |       100 | Cosine    |
| [LAION-10M: subset of LAION 400M English (image embedings)](https://laion.ai/blog/laion-400-open-dataset/)  |        512 |  10,000,000 |    10,000 |       100 | Cosine    |
| [LAION-20M: subset of LAION 400M English (image embedings)](https://laion.ai/blog/laion-400-open-dataset/)  |        512 |  20,000,000 |    10,000 |       100 | Cosine    |
| [LAION-40M: subset of LAION 400M English (image embedings)](https://laion.ai/blog/laion-400-open-dataset/)  |        512 |  40,000,000 |    10,000 |       100 | Cosine    |
| [LAION-100M: subset of LAION 400M English (image embedings)](https://laion.ai/blog/laion-400-open-dataset/) |        512 | 100,000,000 |    10,000 |       100 | Cosine    |
| [LAION-200M: subset of LAION 400M English (image embedings)](https://laion.ai/blog/laion-400-open-dataset/) |        512 | 200,000,000 |    10,000 |       100 | Cosine    |
| [LAION-400M: from LAION 400M English (image embedings)](https://laion.ai/blog/laion-400-open-dataset/)      |        512 | 400,000,000 |    10,000 |       100 | Cosine    |
| **LAION Image Embeddings (768D)**                                                                          |            |             |           |           |           |
| [LAION-1M: 768D image embeddings](https://laion.ai/blog/laion-400-open-dataset/)                           |        768 |   1,000,000 |    10,000 |       100 | Cosine    |
| [LAION-1B: 768D image embeddings](https://laion.ai/blog/laion-400-open-dataset/)                           |        768 | 1,000,000,000|   10,000 |       100 | Cosine    |
| **Standard Benchmarks**                                                                                    |            |             |           |           |           |
| [GloVe-25: Word vectors](http://ann-benchmarks.com)                                                        |         25 |   1,183,514 |    10,000 |       100 | Cosine    |
| [GloVe-100: Word vectors](http://ann-benchmarks.com)                                                       |        100 |   1,183,514 |    10,000 |       100 | Cosine    |
| [Deep Image-96: CNN image features](http://ann-benchmarks.com)                                             |         96 |   9,990,000 |    10,000 |       100 | Cosine    |
| [GIST-960: Image descriptors](http://ann-benchmarks.com)                                                   |        960 |   1,000,000 |     1,000 |       100 | L2        |
| **Text and Knowledge Embeddings**                                                                          |            |             |           |           |           |
| [DBpedia OpenAI-1M: Knowledge embeddings](https://www.dbpedia.org/)                                       |      1,536 |   1,000,000 |    10,000 |       100 | Cosine    |
| [LAION Small CLIP: Small CLIP embeddings](https://laion.ai/blog/laion-400-open-dataset/)                   |        512 |     100,000 |     1,000 |       100 | Cosine    |
| **Yandex Datasets**                                                                                        |            |             |           |           |           |
| [Yandex T2I: Text-to-image embeddings](https://research.yandex.com/)                                      |        200 |   1,000,000 |   100,000 |       100 | Dot       |
| **Random and Synthetic**                                                                                   |            |             |           |           |           |
| Random-100: Small synthetic dataset                                                                        |        100 |         100 |         9 |         9 | Cosine    |
| Random-100-Euclidean: Small synthetic dataset                                                              |        100 |         100 |         9 |         9 | L2        |
| **Filtered Search Datasets**                                                                               |            |             |           |           |           |
| H&M-2048: Fashion product embeddings (with filters)                                                        |      2,048 |     105,542 |     2,000 |       100 | Cosine    |
| H&M-2048: Fashion product embeddings (no filters)                                                          |      2,048 |     105,542 |     2,000 |       100 | Cosine    |
| ArXiv-384: Academic paper embeddings (with filters)                                                        |        384 |   2,205,995 |    10,000 |       100 | Cosine    |
| ArXiv-384: Academic paper embeddings (no filters)                                                          |        384 |   2,205,995 |    10,000 |       100 | Cosine    |
| Random Match Keyword-100: Synthetic keyword matching (with filters)                                        |        100 |   1,000,000 |    10,000 |       100 | Cosine    |
| Random Match Keyword-100: Synthetic keyword matching (no filters)                                          |        100 |   1,000,000 |    10,000 |       100 | Cosine    |
| Random Match Int-100: Synthetic integer matching (with filters)                                            |        100 |   1,000,000 |    10,000 |       100 | Cosine    |
| Random Match Int-100: Synthetic integer matching (no filters)                                              |        100 |   1,000,000 |    10,000 |       100 | Cosine    |
| Random Range-100: Synthetic range queries (with filters)                                                   |        100 |   1,000,000 |    10,000 |       100 | Cosine    |
| Random Range-100: Synthetic range queries (no filters)                                                     |        100 |   1,000,000 |    10,000 |       100 | Cosine    |
| Random Geo Radius-100: Synthetic geo queries (with filters)                                                |        100 |   1,000,000 |    10,000 |       100 | Cosine    |
| Random Geo Radius-100: Synthetic geo queries (no filters)                                                  |        100 |   1,000,000 |    10,000 |       100 | Cosine    |
| Random Match Keyword-2048: Large synthetic keyword matching (with filters)                                 |      2,048 |     100,000 |     1,000 |       100 | Cosine    |
| Random Match Keyword-2048: Large synthetic keyword matching (no filters)                                   |      2,048 |     100,000 |     1,000 |       100 | Cosine    |
| Random Match Int-2048: Large synthetic integer matching (with filters)                                     |      2,048 |     100,000 |     1,000 |       100 | Cosine    |
| Random Match Int-2048: Large synthetic integer matching (no filters)                                       |      2,048 |     100,000 |     1,000 |       100 | Cosine    |
| Random Range-2048: Large synthetic range queries (with filters)                                            |      2,048 |     100,000 |     1,000 |       100 | Cosine    |
| Random Range-2048: Large synthetic range queries (no filters)                                              |      2,048 |     100,000 |     1,000 |       100 | Cosine    |
| Random Geo Radius-2048: Large synthetic geo queries (with filters)                                         |      2,048 |     100,000 |     1,000 |       100 | Cosine    |
| Random Geo Radius-2048: Large synthetic geo queries (no filters)                                           |      2,048 |     100,000 |     1,000 |       100 | Cosine    |
| Random Match Keyword Small Vocab-256: Small vocabulary keyword matching (with filters)                     |        256 |   1,000,000 |    10,000 |       100 | Cosine    |
| Random Match Keyword Small Vocab-256: Small vocabulary keyword matching (no filters)                       |        256 |   1,000,000 |    10,000 |       100 | Cosine    |

### Generating local datasets

The sparse-vector, hybrid (dense+sparse fusion), and multi-datatype filter code
paths ship with **locally-generated** synthetic datasets — small, deterministic
(fixed-seed) fixtures with **no public download link**. Generate them once with:

```bash
cargo run --release --bin generate-dataset          # writes into ./datasets
# or a subset / custom location:
cargo run --release --bin generate-dataset -- --only sparse --out-dir /tmp/ds
```

This writes four datasets under `datasets/` (each in the exact on-disk layout
its reader expects), registered in [`datasets/datasets.json`](./datasets/datasets.json):

| Dataset                     | Type     | Dims | Distance | Layout                                                                                  |
| --------------------------- | -------- | ---: | -------- | --------------------------------------------------------------------------------------- |
| `synthetic-sparse-300`      | `sparse` |  300 | dot      | `data.csr` + `queries.csr` + `neighbours.jsonl` (dot/MIPS ground truth)                 |
| `synthetic-hybrid-16`       | `hybrid` |   16 | l2       | `vectors.npy` + `queries.npy` + `data.csr` + `queries.csr` + shared `neighbours.jsonl`  |
| `synthetic-filter-32`       | `tar`    |   32 | l2       | `vectors.npy` + `payloads.jsonl` + `tests.jsonl` (per-query `conditions` + filtered GT) |
| `synthetic-selectivity-32`  | `tar`    |   32 | l2       | `vectors.npy` + `payloads.jsonl` + `tests.jsonl` (one `rank < K` range query per selectivity rung) |

`synthetic-filter-32`'s per-query `conditions` rotate through **keyword**, **int**,
**bool** and **datetime** filters (schema `color:keyword, size:int, flag:bool, ts:datetime`),
each with ground truth brute-forced over only the matching documents, so a high
recall proves the engine actually applied the filter. `synthetic-selectivity-32`
(2000 docs) instead holds one `rank < K` range query per rung of a **selectivity
ladder** (1% / 2% / 5% / 10% / 25% / 50% / 90%), each row annotated with its
`selectivity` / `n_matching`, so recall/latency can be reported as a function of
filter selectivity. The generated files are git-ignored — regenerate them on any
checkout with the command above.

### Filter features

Across the filtering engines, metadata `conditions` support the datatypes
**keyword**, **int**, **float**, **bool**, **datetime** (ISO-8601 range),
**uuid**, **geo-radius**, and **full-text**; the compositions **`match`** (exact),
**`match_any`** (IN-set), **`range`**, and boolean **AND**, **OR**, and
**nested/grouped** trees (e.g. `(A∧B)∨(C∧D)`); plus **multi-tenancy** (per-tenant
scoped filters). Not every engine supports every feature natively — see each
engine's note above for its exceptions (e.g. Dragonfly is KNN-only; Chroma has no
geo/full-text/array metadata; Vertex errors on cross-field `or`/nested/geo). Each
`(engine × feature)` combination is covered by an end-to-end `tests/integration_*`
recall test that scores against filtered brute-force ground truth, so an engine
that silently drops or mis-applies a filter fails its test.

Example runs against a generated dataset (start the engine first, e.g.
`docker compose -f tests/docker-compose.test.yml up -d qdrant redis`):

```bash
# Sparse (Qdrant):
cargo run --release --bin vector-db-benchmark -- \
  --engines qdrant-default --datasets synthetic-sparse-300
# Hybrid dense+sparse fusion (Qdrant):
cargo run --release --bin vector-db-benchmark -- \
  --engines qdrant-hybrid --datasets synthetic-hybrid-16
# Filter datatypes (Redis):
cargo run --release --bin vector-db-benchmark -- \
  --engines redis-docker-test --datasets synthetic-filter-32
```

## Engine Configurations

Engine configurations live in [`experiments/configurations/`](./experiments/configurations/). Each JSON file defines one or more experiment configurations specifying the engine, index parameters, search parameters, and upload parallelism.

Example (`redis-docker-test.json`):

```json
[
  {
    "name": "redis-docker-test",
    "engine": "redis",
    "connection_params": {},
    "collection_params": {
      "hnsw_config": { "M": 16, "EF_CONSTRUCTION": 128 }
    },
    "search_params": [
      { "parallel": 1, "search_params": { "ef": 128 } }
    ],
    "upload_params": { "parallel": 8 }
  }
]
```

Use `--engines` with wildcard patterns to select configurations:

```bash
vector-db-benchmark --engines 'redis-single*' --datasets 'glove*'
vector-db-benchmark --engines 'vectorsets*' --datasets 'h-and-m*'
vector-db-benchmark --engines 'elasticsearch*' --datasets 'glove*'
vector-db-benchmark --engines 'qdrant*' --datasets 'deep-image*'
```

Or provide a custom file with `--engines-file`:

```bash
vector-db-benchmark --engines-file my_engines.json --datasets glove-25-angular
```

### Qdrant hybrid (dense + sparse) search

`experiments/configurations/qdrant-hybrid.json` runs Qdrant's server-side
reciprocal-rank fusion (RRF) of a dense-vector prefetch and a sparse-vector
prefetch. It **requires a `type: "hybrid"` dataset** — running it against an
ordinary dense dataset silently degrades to a plain dense search (there is no
sparse vector to fuse). A hybrid dataset directory must contain all of:

```
vectors.npy      # dense document vectors  (npy, row i == point id i)
queries.npy      # dense query vectors      (npy)
data.csr         # sparse document vectors  (binary CSR, row-aligned with vectors.npy)
queries.csr      # sparse query vectors     (binary CSR, row-aligned with queries.npy)
neighbours.jsonl # fused ground truth: one JSON array of ids per query line
```

Register it in `datasets/datasets.json` with `"type": "hybrid"` and the dense
`vector_size`/`distance`. The end-to-end hybrid path (collection with a named
`dense` + named `sparse` vector, dual-vector upsert, and RRF fusion) is covered
by `tests/integration_qdrant.rs::test_binary_qdrant_hybrid`, which also
generates a tiny hybrid fixture you can consult for the exact layout.

## How to register a dataset?

Datasets are configured in [`datasets/datasets.json`](./datasets/datasets.json). The tool automatically downloads datasets on first use if a download link is provided.

## Development

### Prerequisites

The quickest way to install all dependencies (Linux/macOS):

```bash
make setup    # installs libhdf5, pkg-config, and Rust toolchain
```

Or install manually:

- **Rust toolchain** (install via [rustup](https://rustup.rs/))
- **libhdf5-dev** and **pkg-config**

```bash
# Ubuntu/Debian
sudo apt-get install libhdf5-dev pkg-config

# macOS
brew install hdf5 pkg-config
```

### Build and test

```bash
make build              # Build release binary
make test               # Run unit tests
make check              # Clippy + rustfmt
make docker-build       # Build Docker image
```

### Integration tests

Each engine has a dedicated integration test that runs against a Docker container:

```bash
make integration-test                  # Redis 8.8.0 (default)
make integration-test-elasticsearch    # Elasticsearch 9.4.3
make integration-test-opensearch       # OpenSearch 3.7.0
make integration-test-pgvector         # PgVector (PostgreSQL 18)
make integration-test-qdrant           # Qdrant v1.18.2
make integration-test-weaviate         # Weaviate 1.38.2
make integration-test-milvus           # Milvus v2.6.19
make integration-test-mongodb          # MongoDB Atlas Local 8.0.17
make integration-test-valkey           # Valkey Bundle (latest)
```

Each target starts the engine via `docker compose -f tests/docker-compose.test.yml`, runs the tests, then stops the container.

### Fuzzing

The untrusted dataset parsers (sparse CSR, NPY, JSONL, and metadata/JSON readers) are fuzzed with [`cargo-fuzz`](https://github.com/rust-fuzz/cargo-fuzz) / libFuzzer to ensure malformed input returns `Err` instead of panicking/overflowing/OOMing. Run locally with a nightly toolchain:

```bash
cargo +nightly fuzz run sparse_reader -- -max_total_time=60 -rss_limit_mb=2048
```

A nightly GitHub Actions workflow (`.github/workflows/fuzz.yml`) fuzzes each parser at higher effort. See [`fuzz/README.md`](fuzz/README.md) for details.

**Turbopuffer** is cloud-only and requires an API key:
```bash
TURBOPUFFER_API_KEY=your-key ./target/release/vector-db-benchmark \
  --engines 'turbopuffer*' --datasets random-100
```

### Project structure

```
src/
  lib.rs                              # Library: readers, data formats
  readers/                            # HDF5, NPY, JSONL, compound readers
  bin/
    vector_db_benchmark/
      main.rs                         # CLI entry point
      config.rs                       # Configuration loading
      dataset.rs                      # Dataset resolution and reading
      experiment.rs                   # Experiment runner with calibration
      engine/
        mod.rs                        # Engine trait and factory
        redis.rs                      # Redis (RediSearch) engine
        vectorsets.rs                 # VectorSets engine
        elasticsearch.rs              # Elasticsearch engine
        opensearch.rs                 # OpenSearch engine
        qdrant.rs                     # Qdrant engine (gRPC)
        pgvector.rs                   # PgVector engine (PostgreSQL)
        weaviate_grpc.rs              # Weaviate gRPC client (generated from vendored v1 protos)
        weaviate.rs                   # Weaviate engine (gRPC search + REST schema)
        milvus.rs                     # Milvus engine (REST)
        mongodb_engine.rs             # MongoDB Atlas Search engine
        valkey.rs                     # Valkey engine (RESP protocol)
        kividb.rs                     # KiviDB engine (RESP protocol)
        turbopuffer.rs                # Turbopuffer engine (cloud API)
        redis_utils.rs                # Shared utils for Redis-protocol engines
experiments/configurations/           # Engine configuration JSON files
datasets/datasets.json                # Dataset definitions
tests/
  docker-compose.test.yml             # Docker services for integration tests
  integration_redis.rs                # Redis integration tests
  integration_elasticsearch.rs        # Elasticsearch integration tests
  integration_opensearch.rs           # OpenSearch integration tests
  integration_pgvector.rs             # PgVector integration tests
  integration_qdrant.rs               # Qdrant integration tests
  integration_weaviate.rs             # Weaviate integration tests
  integration_milvus.rs               # Milvus integration tests
  integration_mongodb.rs              # MongoDB integration tests
  integration_valkey.rs               # Valkey integration tests
```
