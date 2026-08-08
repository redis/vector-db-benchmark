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
| **MongoDB** (Atlas Search) | `mongodb` 3 (sync) | MongoDB protocol | Euclidean, Cosine, Dot | Yes [\*\*\*\*\*\*\*\*](#mongodb-note) |
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
\*\*\*\*\* **Chroma note:** Uses **Chroma** (OSS) via its **v2 REST API** — a collection of records (`ids` + `embeddings` + scalar `metadatas`) queried with `query` + a `where` document. **Metadata filters** map directly onto the canonical model: `match` → `$eq`, `match_any` → `$in`, `range` → `$gte`/`$gt`/`$lte`/`$lt`, and **AND / OR / nested boolean** to Chroma's native `$and` / `$or` (which nest arbitrarily). Supported datatypes: **keyword, int, float, bool, uuid, datetime** (stored as epoch-seconds int, like Milvus, so numeric range operators apply). **Full-text** (`{match:{text}}`) is supported via `where_document` `$contains` — the `text`-typed field's value is uploaded as each record's Chroma `document`. **NOT supported** by Chroma's metadata engine: **geo-radius** and multi-valued **`labels` arrays** (Chroma metadata values are scalar only). Geo is a *permanent* gap, not a to-do: Chroma's `where` is a closed enum of `field OP literal` comparisons with no geospatial operator, no attribute-vs-attribute comparison and no arithmetic, and a spherical cap is not an axis-aligned box in any query-independent coordinate system — so there is no exact encoding, and an inexact bounding box would under-constrain the query while looking filtered (#223). Since #219 these are a hard **error**, not a silent drop — and a condition tree containing a geo leaf is refused *whole*, so a `geo AND keyword` query cannot quietly run as keyword-only. See the note below. Runs over HTTP/REST via Docker; set the host port with `CHROMA_PORT` (default `8000`, test compose maps `8003`), and optionally `CHROMA_COLLECTION` / `CHROMA_TENANT` / `CHROMA_DATABASE`. Distance space (`l2`/`cosine`/`ip`) is set per collection from the dataset metric.

<a id="unexpressible-filter-note"></a>
### Filters an engine cannot express are a hard error (#219)

A query's `conditions` are resolved through one guarded choke point
(`src/query_filter.rs`). If a dataset declares a filter and the engine's builder
produces nothing for it, the run **stops** instead of issuing an unfiltered
search whose recall would then be scored against filtered ground truth. "No
filter declared" — absent, `null`, or `{}` — stays a normal unfiltered run, so
every `*_no_filters` dataset is unaffected.

Which shipped dataset/engine pairs this stops today:

| dataset | engines that now abort | why |
|---|---|---|
| `random-geo-radius-100-angular-filters`, `random-geo-radius-2048-angular-filters` | turbopuffer, chroma | neither filter DSL has a geo operator, an attribute-vs-attribute comparison, or any arithmetic, so an exact radius has no representation at all (#223). A bounding box is a *widening*, not a substitute. |
| the same two | dragonfly, valkey | neither `configure()` declares a GEO field, so the shared RediSearch geo clause targeted a field that does not exist. Previously it was emitted anyway (#223) — Dragonfly's is a builder limitation (it rejects `$param` placeholders and integer literals, not geo itself), Valkey's is an engine one (no GEO field type). |
| the same two | kividb, vertex | already aborted before #219 |
| `arxiv-titles-384-angular-filters` | kividb | multi-valued `labels` (see the KiviDB note) |

Two operational consequences worth knowing:

* **The check runs in `search()`, after `configure()` and `upload()`.** On the
  geo datasets, chroma/turbopuffer/dragonfly/valkey will ingest the whole corpus
  and *then*
  abort. KiviDB is the exception — it rejects in
  `configure()`, before any ingest. Moving the check earlier for those four needs
  the dataset's conditions at configure time and is tracked separately; until
  then, budget the ingest time or exclude those datasets up front.
* **With the default `--exit-on-error true`, one refusal ends the sweep.** A
  `--engines '*' --datasets 'random-geo*'` run aborts at the first refusal.
  Pass `--exit-on-error false` to let the rest of the matrix finish; each
  failure then prints `Experiment failed: [config=… dataset=…] …` so the
  refusal can be traced back to its sweep point.

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

<a id="mongodb-note"></a>
\*\*\*\*\*\*\*\* **MongoDB note — which knobs MongoDB actually honours:** MongoDB Vector Search exposes **build-time** HNSW tuning and **no query-time `ef`**.

`collection_params.hnsw_config` is forwarded into the `vectorSearch` index as the `hnswOptions` sub-document — MongoDB's own spelling, since it rejects the HNSW-generic names outright (`unrecognized fields ["m", "efConstruction"]`):

| benchmark key | MongoDB key | server-enforced bounds |
|---|---|---|
| `M` | `hnswOptions.maxEdges` | `[16..64]` |
| `EF_CONSTRUCTION` | `hnswOptions.numEdgeCandidates` | `[100..3200]` |

Values are forwarded **verbatim, never clamped**: an out-of-range value fails index creation loudly rather than silently benchmarking a different index. Note the server **elides default-valued options** when reporting a definition back, so `{maxEdges:16, numEdgeCandidates:100}` (the defaults) is indistinguishable from an untuned index.

At query time, **`numCandidates` is the only recall/latency dial** — there is no `ef`/`efSearch`, and `$vectorSearch` *silently ignores* unknown stage fields, so a stray `search_params.ef` is a no-op rather than an error. The benchmark sends `numCandidates = top × search_params.num_candidates`, i.e. **`num_candidates` is a multiplier here**, unlike `elasticsearch.rs` and `vertex.rs` where the same key is an absolute count. `experiments/configurations/mongodb-single-node.json` therefore sweeps `M` × `EF_CONSTRUCTION` at build time and `num_candidates` at query time; it previously swept a `search_params.ef` that MongoDB never honoured (issue #216).

<a id="opensearch-note"></a>
\*\*\*\*\*\*\* **OpenSearch note:** Connection is set with `OPENSEARCH_PORT` (default `9200`), `OPENSEARCH_INDEX` (default `bench`), `OPENSEARCH_TIMEOUT` seconds (default `300`), and `OPENSEARCH_USER` / `OPENSEARCH_PASSWORD`.

**Shard count** is read from `collection_params.number_of_shards`. Leave it unset to inherit the cluster default — but note that default is **1 on open-source OpenSearch and historically 5 on Amazon OpenSearch Service**, and shard count materially changes vector indexing speed and precision, so a run that leaves it unset is not comparable across engine versions or deployments. The effective value is printed per config (`OpenSearch: HNSW { … }, number_of_shards: 5|cluster-default`), and a present-but-non-integer value is a hard error rather than a silent fallback to the default.

Every OpenSearch config shipped **in this (Rust) tree** pins it explicitly, so no published run inherits a default (#211). The resolved value is also written into every result file as `params.number_of_shards` (`"cluster-default"` if a custom config left it unpinned), so a run can be audited without reverse-engineering its config name. The legacy `v0/` Python tree is *not* pinned and cannot be: `v0/engine/clients/opensearch/configure.py` hardcodes its index settings and ignores `number_of_shards` entirely, while `v0/engine/clients/elasticsearch/configure.py` pins 1 — so v0's own ES-vs-OS pairing still has the #211 asymmetry.

| Config file | `number_of_shards` | Use for |
|---|---|---|
| `experiments/configurations/opensearch-single-node.json` | `1` | Open-source / single-node OpenSearch. Matches `elasticsearch.rs`, which pins `ES_NUMBER_OF_SHARDS = 1`, so this is the ES-vs-OS head-to-head pairing (see the caveat below). |
| `experiments/configurations/opensearch-5-shard.json` | `5` | A 5-shard index — the per-index default Amazon OpenSearch Service inherited from Elasticsearch and still applies on legacy/ES-derived domains. Same HNSW sweep, config names prefixed `opensearch-5-shard-`. |

The 5-shard file is named for what it *does*, not for a deployment: a modern Amazon OpenSearch Service domain defaults to **1** shard per index, not 5, so "managed" would have been a claim about someone else's default that is no longer reliably true. Run it on a managed domain if you want to reproduce a legacy default, and set `number_of_shards` to whatever your own domain actually reports if you want to model it.

**Do not compare `opensearch-5-shard-*` numbers against Elasticsearch.** `elasticsearch.rs` always builds a 1-shard index, so a 5-shard OpenSearch result against an ES result differs in shard count as well as engine and is not a head-to-head. Pair ES only with `opensearch-single-node.json`. Even that pairing is not yet fully matched: ES still bulk-loads with `refresh_interval: "10s"` while OpenSearch uses `-1`, so ES pays refresh work during ingest that OpenSearch does not — tracked in #240.

Pick one file per run — they are exact and survive future renames:

```
--engines-file experiments/configurations/opensearch-single-node.json
--engines-file experiments/configurations/opensearch-5-shard.json
```

Avoid selecting by glob here: `--engines 'opensearch-m-*'` matches only 6 of the 7 sweep points — it silently drops the `opensearch-default` (m=16, ef_construction=100) baseline, whose name lacks the `opensearch-m-` prefix — and a bare `--engines 'opensearch-*'` matches both files, sweeping two different shard counts into one report.

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
| `OPENSEARCH_FORCE_MERGE_TIMEOUT` | `max(OPENSEARCH_TIMEOUT, 3600)` | force-merge (seconds, per attempt; `0` = no client-side deadline) |
| `OPENSEARCH_FORCE_MERGE_BUDGET` | `2x OPENSEARCH_FORCE_MERGE_TIMEOUT` | force-merge (seconds, total wall-clock ceiling across all attempts; `0` = unlimited) |

Force-merge needs its own two bounds because it is not sized like the other requests. It merges the index down to a single segment, which rewrites the whole corpus: measured at 1,077–1,312 s for 1.18M vectors, well past the 300 s client-wide `OPENSEARCH_TIMEOUT`. Sharing that bound would abort every attempt client-side while the merge kept running on the server, and spend the whole `OPENSEARCH_INDEX_OP_MAX_RETRIES` budget failing a merge that was always going to succeed — discarding a completed multi-hour ingest at the last step. The wall-clock budget then bounds the retrying itself, since a per-attempt bound of an hour and eleven attempts is otherwise an ~11 h worst case. Both reject a malformed value loudly rather than silently falling back to the default.

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
    --allow-partial-configs    Run even if some experiments/configurations/*.json
                               failed to load (off by default; see "Engine
                               Configurations" below)
    --datasets <PATTERN>       Dataset patterns (wildcards supported) [default: *]
    --host <HOST>              Redis/engine host [default: localhost]
    --parallels <N,N,...>      Filter by parallel thread counts
    --ef-runtime <N,N,...>     Filter by ef runtime values
    --skip-upload              Reuse the corpus already on the server (no configure,
                               no upload); verified against the live server first
    --allow-partial-corpus     Downgrade that verification from a hard error to a warning
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

### `--skip-upload`: reuse means reuse (issue #238)

`--skip-upload` asserts *"the server already holds the corpus I want"*. Three
rules follow, and all three are enforced:

1. **The configure phase never runs.** `configure()` is destructive on **14 of the
   15 engines** — `FT.DROPINDEX … DD` (Redis), `FT.DROPINDEX` + `SCAN`/`UNLINK`
   (Valkey / Dragonfly / KiviDB), `collection.drop()` (MongoDB), `DROP TABLE`
   (pgvector), `DELETE /collections/<n>` (Qdrant / Chroma / Weaviate / Milvus /
   Turbopuffer), `indices.delete` (Elasticsearch / OpenSearch), `DEL <key>`
   (VectorSets). Only Vertex is non-destructive. Until #238 the
   `--skip-upload --skip-vector-index` combination still called it: the flags that
   mean *"do not upload, do not build an index, just use what is there"* deleted
   the corpus and then benchmarked the empty index — printing a QPS number and
   exiting 0.
2. **The cleanup phase never runs either.** `--keep-data` defaults to **false**,
   so `--skip-upload` on its own used to reuse a corpus, measure it, and then
   `engine.delete()` it. A run that did not create the corpus does not delete it,
   whatever `--keep-data` says.
3. **The corpus is verified before anything is measured.** The runner reads the
   row count back off the live server (`FT.INFO num_docs` / `hnsw_live_count`,
   `GET /collections/<n>` → `points_count`, `_count`, `countDocuments`,
   `reltuples`, `VCARD`) and compares it with the corpus **measured on disk**:
   - **fewer rows than the dataset holds — including zero, i.e. a missing index —
     is a hard error.** Recall/precision are scored against ground truth for the
     FULL corpus, so a short corpus publishes a wrong number under a config name
     that claims otherwise.
   - more rows → warning (often deliberate: a shared prefix, a superset upload).
   - an **estimate** that comes up short → warning only. pgvector's count is a
     planner figure (see below); aborting a run on a number that is allowed to be
     wrong just trades one silent failure for a noisy one.
   - a **probe failure** (unreachable server, `NOPERM`, missing `SELECT`, a closed
     ES index, a shard that did not answer) → a hard error that says *probe
     failure*. It is never reported as "the corpus is empty": that names the wrong
     problem and invites a re-upload over a corpus that was fine.
   - the **dataset's own expected row count cannot be determined** → a hard error
     as well (issue #290), because it is the same outcome as a short corpus with
     *less* information. Until #290 it printed a note and ran anyway, and over an
     empty corpus that publishes `mean_recall: 0.0` with `failed_queries: 0` and
     exit 0. Where the count comes from depends on the layout, and so does the
     remedy:
     - measurable layouts (`tar`, `h5`, `jsonl`, `hybrid`, and `sparse` — 56 of
       the 57 shipped datasets) read it from the corpus file's header or line
       count. **The check resolves the dataset first**, using the same
       `get_path()` the search phase is about to call, so a dataset that is
       simply not downloaded yet is fetched rather than rejected. Two things it
       does *not* do, both worth knowing: it resolves only **after** the
       server-side probe succeeds, so an unreachable server still aborts without
       paying for a download; and it does **not** re-fetch a dataset whose
       directory already exists. That second one is the realistic failure — a
       corpus file deleted to reclaim disk while `tests.jsonl` (queries + ground
       truth) stayed behind, so the run would still search and still publish. A
       valid `link` will not repair it; delete the directory or restore the file.
       The error says which of the two you have.
     - `h5-multi` is the one layout with no cheap row count — its total is the
       sum of 100 part headers — so its number comes from `vector_count` in
       `datasets.json`, and, unlike the shared-corpus upload gate, **without**
       requiring every part to be present locally, since `--skip-upload` never
       uploads. Missing `vector_count` is the error, and the message says so.
       `sparse` uses the same fallback only when `data.csr` is absent; when it is
       present the header is read and the declaration is checked, not trusted.

     `--allow-partial-corpus` waives it, and the waiver is recorded.
   - no count read back for the engine → a printed note that the reuse went
     unverified, **whatever the dataset side says**: with neither side available
     there is nothing to compare in either direction, so this never escalates to
     an abort. A probe is implemented for Redis, Valkey, Dragonfly, KiviDB,
     VectorSets, Qdrant, Elasticsearch, OpenSearch, pgvector and MongoDB. Chroma,
     Milvus and Weaviate all expose a count we have simply not wired up yet;
     Turbopuffer and Vertex are the only genuinely uncountable ones. (Qdrant's
     probe is wired up but can reply without a `points_count`, which lands here
     too — hence "no count read back" rather than "not implemented".)

   **On a sweep, the right remedy for a rejected config is `--exit-on-error false`**
   (`--exit-on-error` defaults to true, so one stale config otherwise kills the
   whole run and the *correct* measurements from the other configs are lost too).
   `--allow-partial-corpus` downgrades the rejection to a warning and measures the
   partial corpus — it publishes exactly the number the check exists to suppress,
   so reach for it last, and only deliberately.

The verdict is recorded in every result file it applies to, under
`params.corpus_reuse` (`status` — one of `verified`, `surplus`, `short`,
`corpus_size_unknown`, `unverified`, or `probe_failed`, which is written by the
probe-failure branch and carries `detail` instead of the count fields — plus
`expected_rows`, `actual_rows`, `actual_is_estimate`,
`waived_by_allow_partial_corpus`), and a rejected
experiment is listed in the summary under `rejected_experiments` — otherwise a
sweep that lost most of its configs produces a summary and a chart
indistinguishable from a complete one.

**What the check cannot do: cardinality is not identity.** Only the five
Redis-wire engines address a per-config object: Redis / Valkey / Dragonfly /
KiviDB each own an index plus a keyspace (#151-4), and VectorSets owns the single
key `idx:<config>` (#236 — a vector set *is* one key). MongoDB, pgvector,
Elasticsearch/OpenSearch, Qdrant and Milvus each have ONE corpus object per
server, shared by every config *and every dataset* — so a full-size corpus
uploaded by a sibling config, or by a different dataset, passes. The count proves
the corpus is the right SIZE, never that it is the right corpus.

On the five per-config engines a sibling's corpus does **not** pass, because the
count is read from this config's own object. What still passes there is the same
config re-run against a **different dataset** of equal size — the key is derived
from the config name alone, not from the dataset.

**pgvector's count is an estimate on purpose.** The engine never `VACUUM`s and
forces `STORAGE PLAIN`, so `SELECT count(*) FROM items` plans a Seq Scan over the
whole heap *immediately before the search phase*, silently turning a cold-cache
run warm and changing the published number. Measured on a cold 1M-row / 768-dim
table built like this engine's (3906 MB, 500000 relpages):

| | heap blocks touched | cache primed | cold wall clock |
|---|---|---|---|
| `SELECT count(*)` | 500,000 | 3906 MB | 1.58 s |
| `ANALYZE items` | 30,001 | 234 MB | 0.84 s |

The point is not the 16.7x ratio but that ANALYZE's sample is capped at
`300 * default_statistics_target` = 30,000 rows **regardless of table size**, so
its footprint is bounded while `count(*)` grows linearly — at 1536 dims (~7.8 GB)
`count(*)` primes all of it and ANALYZE still touches ~30,000 blocks. Bounded is
not free (~6% of this heap), so the perturbation is capped, not eliminated. The
check uses `ANALYZE` + `reltuples` and can therefore only warn, never abort.

Two-phase workflows are unchanged, and now provably non-destructive: upload with
`--keep-data`, then re-run with `--skip-upload --skip-if-exists false`. A
filter-only two-phase run must pass `--skip-vector-index` in **both** phases —
the flag renames the config to `<engine>-no-vector`, so phase 1 writes exactly the
schema-only index phase 2 reuses.

### Per-config index isolation (Redis / Valkey / Dragonfly / KiviDB / VectorSets)

Each engine config gets its **own** RediSearch index and keyspace, derived from the
config `name`: index `"<base>:<config>"` (base `idx`) with docs keyed
`"<config>:<id>"`. This lets an M×EF_CONSTRUCTION **sweep** run all its configs
against one server and coexist, so you can upload every config once and then
search each in a later `--skip-upload` pass — each config reads its own graph, and
memory is reported per-config as `index_memory_bytes` (issue #151-4).

**VectorSets** joined this in #236 with the same derivation and the same knobs, but
a different shape: it has no RediSearch index and no doc keyspace — its entire
corpus is the **single key** `"<base>:<config>"` that `VADD`/`VSIM`/`VINFO`/`VCARD`/`DEL`
address, so that key *is* the namespace and there is no doc-key prefix. Before #236
every VectorSets config used the literal key `idx` and `configure()` opened with
`DEL idx`, so starting a second config **deleted the first's entire corpus**.

**Reading the memory numbers.** Every upload result file carries two figures, and
on a coexisting sweep they mean different things:

| field | source | scope |
|---|---|---|
| `index_memory_bytes` | `FT.INFO` (Redis / Valkey / Dragonfly / KiviDB), `MEMORY USAGE <key>` (VectorSets, #236) | **this config only** — the number to quote |
| `used_memory` | `INFO memory` | **server-wide**: the SUM over every config resident at that moment |

Before per-config isolation the two were interchangeable for VectorSets, because
exactly one corpus could be resident. They are not interchangeable now, so a
`used_memory` figure from a pre-#236 run and one from a current sweep are not
comparable quantities. Note also that nothing deletes a legacy `idx` left by an old
binary — it stays resident and inflates `used_memory` in every later result file
until you `DEL` it.

- `REDIS_INDEX_NAME` / `VALKEY_INDEX_NAME` / `DRAGONFLY_INDEX_NAME` / `KIVIDB_INDEX_NAME` /
  `VECTORSETS_INDEX_NAME` now set the **base namespace**, not the whole index name;
  the config name is always appended.
  Set `<VAR>_EXACT=1` to use the base verbatim (single-config "point at an
  out-of-band index" case — combining it with >1 config for that engine is a
  startup error).
- Indexes/keys written by any **pre-#151-4** binary are incompatible — re-upload.
  For VectorSets the equivalent cut-off is **pre-#236**: a corpus left under the bare
  `idx` key is neither found nor deleted by a current binary (`DEL idx` to reclaim it).
  `--skip-upload` against a missing/mismatched index **hard-errors** instead of
  silently writing a `recall 0.0` file — including for VectorSets, where `VSIM`
  against a missing key returns an **empty array with no error**, so nothing
  downstream could otherwise tell "no data" from "no matches". The `VCARD`-based
  corpus-reuse check (#238/#271) is what makes that promise reach the fifth member
  of this family.

  **The guarantee is conditional on the tool knowing how many rows to expect** —
  and since #290 it stops rather than guesses. When the expected count cannot be
  determined (an unmeasurable layout with no `vector_count`, or a corpus that is
  neither on this machine nor fetchable), the verdict is `corpus_size_unknown`
  and the run is a **hard error**, waivable with `--allow-partial-corpus`. Before
  #290 that path printed `Reuse check — SKIPPED` and let the run proceed, so a
  missing corpus was still measured and still published as `recall 0.0` at exit 0.
  One softer case remains and still prints `SKIPPED`: when no server-side count is
  read back at all (`status = "unverified"`, e.g. Chroma/Milvus/Weaviate/
  Turbopuffer/Vertex), there is nothing to compare in either direction, so the run
  continues — treat that `SKIPPED` line as "this run's corpus was never verified"
  and check it yourself. (Inherited from #238/#271, not specific to VectorSets.)
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

### VectorSets: datetime corpora written before #230 must be re-uploaded

VSIM `FILTER` has no date type and its comparison operators are numeric, so since
PR #230 a datetime metadata value is stored as an **epoch-seconds number** on
`VADD … SETATTR`, and range/equality filters compare against epoch numbers. A
corpus uploaded by any **pre-#230** binary holds ISO-8601 **strings** instead, and
VSIM coerces a non-numeric attribute to `0` in a numeric comparison — so a
`--skip-upload` search against such a corpus silently returns the wrong documents
(a two-sided range matches nothing → recall 0; a one-sided `lt`/`lte` matches
everything). **Re-upload before searching.**

`--skip-upload` checks that the corpus is **present and complete** on VectorSets
(`VCARD idx:<config>`, see above), but it cannot check that the corpus is
*current*: a pre-#230 corpus is the right size and the wrong encoding, so it is
found, the run succeeds, and only the recall number is wrong.

**In practice #236 makes that case unreachable by default**, because #230 shipped
before #236: any corpus old enough to hold ISO-8601 strings is also old enough to
sit under the bare key `idx`, where a current binary does not look for it. Such a
corpus is **not** found — `VCARD idx:<config>` answers 0, the reuse check
classifies it `Short`, and the run is rejected. That is deliberate: `VSIM` against
a missing key returns an **empty array with rc=0**, so the run would otherwise have
published `recall 0.0` at an *inflated* QPS (an empty search returns instantly —
measured on a 400-vector corpus, `ef=400 parallel=1`: **637.6 QPS at recall 1.000
with the corpus present, 2679.3 QPS at recall 0.000 with the key deleted** — 4.2×
*faster* for measuring nothing), with `failed_queries: 0` in both runs and every guard keying off
it silent. Re-upload, or `DEL idx` and re-upload.

Exactly two things get you back to the wrong-encoding case, and neither is a
migration path this README recommends:

- `VECTORSETS_INDEX_NAME_EXACT=1` with the base left at `idx`, which pins a single
  config onto the legacy key; or
- renaming the legacy key by hand (`RENAME idx idx:<config>`).

Both hand the old corpus to a current binary under a name it will accept. If you do
either, you own the encoding check — the size check will pass.

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
    "mean_precision_at_returned": 0.9785,
    "mean_recall": 0.9785,
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

Multi-tenancy benchmarks model many tenants sharing **one** index: every search is scoped to a single tenant via an exact keyword-equality filter on a tenant field (`schema: { "<field>": "keyword" }` — the field is named `a` in `random-768-100-tenants` and `tenant` in `random-768-25-tenants`), and recall is measured against the nearest neighbours **within that tenant only**. This reuses the standard keyword-TAG filter path (no engine-specific code) and mirrors upstream qdrant/vector-db-benchmark's `random-768-*-tenants` scenario. Two are registered: **`random-768-100-tenants`** (1M points over 100 tenants) downloads, and is the reproducible one to benchmark with — note its tenant field is named `a` (`schema: { "a": "keyword" }`), matching the published tarball; `random-768-25-tenants` is a locally-generated placeholder using a `tenant` field. The per-query filter looks like `{"and":[{"tenant":{"match":{"value":"tenant_7"}}}]}` for the 25-tenant fixture, and `{"and":[{"a":{"match":{"value":"WLRCI"}}}]}` for the downloadable 100-tenant set (whose tenant ids are random 5-character strings). Because ground truth is tenant-local, recall is a strong isolation signal — a leaked cross-tenant document displaces a correct neighbour and lowers recall — and the tests assert **exact** per-query recall (`== 1.0` against an exact search, one query per tenant), so any single cross-tenant leak fails the check. Redis and Valkey are covered end-to-end (over both RESP2 and RESP3) by the `test_binary_{redis,valkey}_tenancy` integration tests. (Recall is necessary but not by itself sufficient to *prove* zero leakage; strict per-id membership checking is a possible future hardening.)

## Datasets

Most datasets are automatically downloaded on first use. The image includes `random-100` (228KB) for quick smoke tests. (Exception: `random-768-25-tenants` is a locally-generated placeholder with no public download link — use the downloadable `random-768-100-tenants` instead; see the Multi-tenancy section.)

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
| [DBpedia OpenAI-100K: Knowledge embeddings](https://www.dbpedia.org/)                                     |      1,536 |     100,000 |     5,000 |        10 | Cosine    |
| [LAION Small CLIP: Small CLIP embeddings](https://laion.ai/blog/laion-400-open-dataset/)                   |        512 |     100,000 |     1,000 |       100 | Cosine    |
| **Sparse Vectors** (learned/lexical sparse embeddings — Qdrant is the only engine with a sparse path)        |            |             |           |           |           |
| [MS MARCO Sparse-100K: SPLADE-style sparse embeddings](https://microsoft.github.io/msmarco/)                |   *sparse* |     100,000 |     6,980 |        10 | Dot       |
| [MS MARCO Sparse-1M: SPLADE-style sparse embeddings](https://microsoft.github.io/msmarco/)                  |   *sparse* |   1,000,000 |     6,980 |        10 | Dot       |
| **Yandex Datasets**                                                                                        |            |             |           |           |           |
| [Yandex T2I: Text-to-image embeddings](https://research.yandex.com/)                                      |        200 |   1,000,000 |   100,000 |       100 | Dot       |
| **Random and Synthetic**                                                                                   |            |             |           |           |           |
| Random-100: Small synthetic dataset                                                                        |        100 |         100 |         9 |         9 | Cosine    |
| Random-100-Euclidean: Small synthetic dataset                                                              |        100 |         100 |         9 |         9 | L2        |
| **Filtered Search Datasets**                                                                               |            |             |           |           |           |
| H&M-2048: Fashion product embeddings (with filters)                                                        |      2,048 |     105,542 |    10,000 |    ≤ 25 † | Cosine    |
| H&M-2048: Fashion product embeddings (no filters)                                                          |      2,048 |     105,542 |    10,000 |        10 | Cosine    |
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
| **Multi-Tenancy** (many tenants share one index; every query scoped to one tenant)                          |            |             |           |           |           |
| Random-768-100-tenants: 100 tenants, per-tenant scoped queries (tenant field `a`)                          |        768 |   1,000,000 |       200 |        25 | Cosine    |

† **The "Neighbors" column is the ground-truth width, and it bounds what recall can mean.** H&M-2048 (with filters) is not uniform: its 10,000 queries carry between **1 and 25** true neighbours (mean 23.4; 931 queries have fewer than 25), because a filtered query only has as many true neighbours as the filter admits. Our `mean_recall` divides by the neighbours that actually exist, so a 1-neighbour query can still score 1.0; upstream `qdrant/vector-db-benchmark` always divides by `top`, so the same query caps at `1/top`. A run's `metrics_schema.ground_truth` block reports the measured profile and the resulting ceiling — at `top: 100` H&M's ceiling is 0.233, so a `calibration_precision` above that is unreachable by construction. Configs that do not set `top` derive it from the ground-truth row width (25 here, 10 for the no-filters variant), which is why this matters mostly when `top` is set explicitly.

### Generating local datasets

The sparse-vector, hybrid (dense+sparse fusion), and multi-datatype filter code
paths ship with **locally-generated** synthetic datasets — small, deterministic
(fixed-seed) fixtures with **no public download link**. (For sparse vectors at
realistic scale, prefer the downloadable `msmarco-sparse-100K` / `-1M` in the
table above; the synthetic fixture exists to exercise the code path fast. Both
work: a sparse dataset's ground truth is read from either `neighbours.jsonl` —
one JSON array of ids per query line, what the generator writes — or the binary
`results.gt` those downloads ship.)
Generate them once with:

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
geo or array metadata; Turbopuffer has no geo; Vertex errors on cross-field
`or`/nested/geo). **geo-radius** is expressed by
Qdrant/Redis/Elasticsearch/OpenSearch/Weaviate/pgvector natively, by Milvus via a
`Geometry` column and `ST_DWITHIN`, by MongoDB via `geoWithin` inside a `$search`
pre-filter, and by VectorSets via an exact unit-vector dot product in its
`FILTER` expression (#223). **Valkey, Dragonfly, Chroma, Turbopuffer, KiviDB and
Vertex do not express it** and refuse a geo dataset rather than running it
under-filtered: Valkey Search has no GEO field type at all; Dragonfly Search
*does* support geo but rejects the `$param` placeholders and integer literals
this repo's shared RediSearch builder emits, so the field is not declared; the
rest per their notes above. Each
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

Two rules are enforced at load time, and breaking either aborts the **run** rather than
quietly changing what gets measured. `--describe engines` is deliberately exempt from the
second one: its job is to diagnose the configuration directory, so it lists the unloadable
files first and then shows the configurations that did load. (A duplicate name aborts
`--describe` too — there is nothing useful to show when one name means two things.)

- **`name` must be unique across every file in the directory**, not just within one
  file. The name is what `--engines` selects and what the result JSON is keyed by, so
  two definitions sharing a name would mean the reported number came from whichever
  one happened to load last. A duplicate is an error naming both definitions (#239).
  The same rule applies to dataset names in `datasets/datasets.json`.
- **Every `*.json` in the directory must parse.** `serde` rejects a whole file on one
  bad entry, so a single typo removes every configuration that file defines — and
  under a wildcard `--engines` (the default is `*`) the sweep would just get smaller
  and still exit 0. Pass `--allow-partial-configs` to run anyway; the run then records
  the offending files under `skipped_config_files` in every summary JSON it writes.

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
