# MS MARCO text/scalar scale-up (#389, #390)

These profiles compare the existing Cohere-embedded MS MARCO v2.1 **passages**.
They are not the Document v2 corpus, and result agreement is not relevance truth.
Deep1B and filtered KNN are separate phases. A completed 100M cloud comparison
remains pending until its actual raw evidence is linked here.

## Profile contract

`experiments/configurations/msmarco-compound.json` contains separate
`{redis,opensearch}-msmarco-compound-{parity,english}-{no-vector,full-index}` configs.
Choose exactly one engine config per invocation. Each has 11 families at 1 and
100 clients: segment text, title AND segment, tag AND text, text/tag/range,
text exclusion, tag equality, numeric point, numeric range, repeated-value
range, tag AND range, and range OR range.

The first five families select terms and bounds deterministically from up to
32 real passages in the first 4096 payload rows. For each selected segment term,
exclusions use the most frequent co-occurring title term absent from the seed's
title, with a lexical tie-break. The scanned window must contain both a retained
seed and an excluded matching row; seeds without an exclusion witness are
skipped. The remaining six reuse all
31 prepared scalar queries and their corpus-wide exact counts. This is a
reproducible diagnostic suite, not a representative sample of user traffic.
Each result embeds the complete selected query AST, stable names/families, and
a SHA-256 fingerprint; per-query dumps retain the family and exact match count.
Dumps also carry the query fingerprint, analysis mode, and SHA-256 of
`PREPARED.json`; comparison rejects mismatched identities and pairs families by
stable search ID. Compare only the same dataset/checksum, analysis, and query
fingerprint. Deliberate full-index/no-vector comparisons must retain and report
both layout labels; count/rank agreement does not imply equal index costs.

Scalar counts must equal preparation truth. Generated parity text queries require
at least one match. English exclusions allow zero because stemming may turn an
excluded term into a token present in the seed title; explicit authored count
assertions remain unchanged. Tiny live fixtures check exact text
membership, Boolean precedence, escaping, case, missing fields, empty matches,
and inclusive/exclusive bounds. Corpus text counts may differ between analyzers;
overlap and ranking agreement do not establish relevance/recall. Pure scalar
hits can have tied scores, so arbitrary top-k membership is not a correctness
failure when exact counts agree.

`parity` pins both text fields, disables stemming, and normalizes whitespace at
upload. Parity still retains documented differences in Unicode punctuation and
infix tokenization (RediSearch separators versus OpenSearch's standard tokenizer);
it does not guarantee identical whole-corpus text counts. Inspect the emitted
`known_differences` and exact totals. `english` retains the existing pinned English pipelines and their
known tokenizer/stemming differences. They need separate indexes/corpora;
never share an English corpus with a parity index. All compound queries use
unit field boosts, Redis BM25STD / OpenSearch BM25 scoring, score-descending
order with unspecified ties, top 10, exact totals, and IDs/scores only.
OpenSearch tag/numeric leaves run in non-scoring `bool.filter` context, including
inside mixed OR branches; on OpenSearch only positive text contributes to BM25
scores. Redis retains native BM25STD, including native TAG/nontext iterator
contributions that may be nonzero. This difference is emitted in
`known_differences`; no identical score or ranking contract is claimed. The
`native-engine-scoring-v1` contract is recorded and hashed into query identity,
so earlier scoring-context diagnostic dumps cannot pair silently with these profiles.
Unsupported predicates/types are refused, never silently discarded. Existing
ranked BM25 profiles keep their original OR semantics.

`no-vector` means **no vector index**, not removal of stored vectors: the upload
retains the complete payload and vector bytes. `full-index` also builds HNSW
(M=16, construction EF=64). Measure and label their footprints separately.
The shipped OpenSearch baseline has one primary shard; any scale override must
be committed in the run config, paired deliberately with the Redis deployment,
and recorded with the actual replica/shard counts. Multi-shard text scores use
per-shard statistics and are not identical to single-shard BM25.

## Preparation and integrity

Run compute on an AWS host with a 2 TiB working disk and the orchestration
repository's required FinOps ownership tags on every supported resource, including
disks. Cloud provisioning and private archive
upload belong to cloud-benchmarks-private, not this driver.

The published 100M family contains 100,000,027 rows, about 410 GB of float32
vectors plus 184 GB payloads. These are source-file sizes, **not measured index
sizes**. Existing verified private S3 artifacts can be restored with that
repository's `scripts/restore-msmarco-dataset.py`; no recomputation is needed.
The registry paths are:

| Variant | Path under `datasets/` | Rows |
|---|---|---:|
| 100K | msmarco-cohere-1024/crc32-t886 | 99,964 |
| 1M | msmarco-cohere-1024/crc32-t8805 | 1,000,044 |
| 10M | msmarco-cohere-1024/crc32-t88075 | 9,999,959 |
| 100M | msmarco-cohere-1024/crc32-t880865 | 100,000,027 |

If preparing anew, from the checkout root:

```bash
make build
./target/release/prepare-msmarco --dataset msmarco-cohere-1024-100K-cosine --workloads text-scalar
./target/release/prepare-msmarco --dataset msmarco-cohere-1024-100K-cosine --verify
```

Use `--dataset msmarco-cohere-1024-100M-cosine` for the large corpus. The sealed
preparation manifest binds source revision, files, hashes, counts and capabilities.
Keep `COMPLETE`, `PREPARED.json`, and all manifest-listed files together. Verify
restored bytes before upload. Never fabricate KNN truth for a scalar run.

## Ingest, correctness, and reuse

Set connection secrets in the process environment (never in a committed command
or result). Redis uses `REDIS_PORT` / `REDIS_AUTH`; OpenSearch uses
`OPENSEARCH_PORT` / `OPENSEARCH_USER` / `OPENSEARCH_PASSWORD`. Hosts below are
placeholders for the actual private endpoints. Run from the checkout root.
Set a dedicated OpenSearch index before ingestion and keep it unchanged for
reuse; choose a different name for English or full-index runs:

```bash
export OPENSEARCH_INDEX=msmarco-parity-no-vector
```

```bash
/usr/bin/time -v -o ingest-redis.time ./target/release/vector-db-benchmark \
  --engines-file experiments/configurations/msmarco-compound.json \
  --engines redis-msmarco-compound-parity-no-vector \
  --datasets msmarco-cohere-1024-100K-cosine --host "$REDIS_HOST" \
  --upload-reader stream --skip-search --keep-data

/usr/bin/time -v -o ingest-opensearch.time ./target/release/vector-db-benchmark \
  --engines-file experiments/configurations/msmarco-compound.json \
  --engines opensearch-msmarco-compound-parity-no-vector \
  --datasets msmarco-cohere-1024-100K-cosine --host "$OPENSEARCH_HOST" \
  --upload-reader stream --skip-search --keep-data
```

Ingestion waits for index readiness; retain the separate upload/index-wait/total
phase fields and force-merge diagnostics. OpenSearch force merge belongs to the
indexing phase, never timed query throughput. Capture `FT.INFO` / `INFO MEMORY`
and OpenSearch `_stats` / `_settings` / `_mapping` / node statistics **before**
teardown. Report primary logical data/index sizes separately from replicas and
physical RAM/flash/EBS capacity; unavailable managed-service values stay unknown.

```bash
./target/release/vector-db-benchmark \
  --engines-file experiments/configurations/msmarco-compound.json \
  --engines redis-msmarco-compound-parity-no-vector \
  --datasets msmarco-cohere-1024-100K-cosine --host "$REDIS_HOST" \
  --skip-upload --skip-if-exists false --queries=-1 --repetitions 1 \
  --dump-results dumps/redis-smoke --fail-on-dropped-queries
```

Repeat with the OpenSearch engine/host and a distinct dump directory. Reuse
checks server corpus count and the actual text/scalar/vector schema; it does not
prove the contents of an arbitrary foreign same-size corpus. Bind endpoint,
config hash, dataset manifest hash and driver SHA in the orchestration record.
Never use `--allow-partial-corpus` for comparison data. Use dedicated indexes;
`--skip-upload` preserves them, so orchestration must explicitly clean them up.

```bash
./target/release/vector-db-benchmark compare-results dumps/redis-smoke dumps/opensearch-smoke \
  --reference opensearch --json comparison.json --markdown comparison.md
```

Before accepting a pair, verify identical query fingerprints and dataset hashes
in the result metadata. Inspect counts/errors per query, not just aggregate QPS.

## Pilots, 100M, and repeated measurements

1. Pass deterministic live fixtures and the real 100K pair, including reuse.
2. Repeat ingestion on 1M, then 10M. Record wall time, whole-process peak RSS,
   client CPU/network/disk saturation, errors/retries, and both engine footprints.
3. Estimate 100M capacity from both pilots, recording nonlinearity and explicit
   headroom. A 500 GB Flex plan is not evidence of fit. Confirm successful Redis
   Cloud provisioning and record actual versions, shards, proxies, RAM/flash and
   limits before committing the large run.
4. Run 100M separately by replacing the dataset in the exact commands above.
   Use a **32 GiB client memory limit** enforced by orchestration/cgroups and
   `--upload-reader stream`; record observed RSS. Allocation unit tests alone
   do not establish that the full 100M execution fits.
5. Run one untimed warmup invocation, then at least three separate measured
   invocations, each with `--repetitions 1`, fresh output/dump directories, and
   `--skip-if-exists false`. Keep engine order and cache policy explicit. Start
   with one complete query cycle; calibrate the count on the pilot so **every family at
   each concurrency runs at least 60 seconds**, and reject shorter headline
   cells. Do not treat a one-pass smoke result as performance evidence. Preserve
   every repetition; the driver's default best-of-three is not this protocol.

Use `--search-ids 0` (or another zero-based ID from the committed config) to
calibrate and measure one family/concurrency without renumbering its dump ID.
This selector supports dedicated fulltext configs only and cannot combine with `--parallels` or `--ef-runtime`. Keep the
complete engine config unchanged so index identity and schema remain fixed.
A separate invocation per ID allows a different calibrated request count for
each point. Preserve the identical query definitions and uniform cyclic mix on
both engines, but calibrate request counts independently so a large speed gap
does not force an hours-long slow-arm run. Run `--queries=1000` as initial
warmup/calibration (reduce it for a slow pilot), then choose
`ceil(max(observed_engine_qps * 75, 10 * parallel) / family_query_count) * family_query_count`
requests per engine. Complete cycles retain identical query weights; the minimum
request count avoids a short batch that never sustains the configured concurrency.
Very slow families can therefore take longer than 75 seconds. Keep three
independent runs and enforce the 60-second floor on each actual result; increase
that arm's count and repeat it if it finishes early. Record both requested and
completed counts. These are count-based closed-loop runs; the fulltext
harness rejects `--target-qps` and `--search-duration` rather than ignoring them.

Use the same query definitions, cyclic order and relative weights on both
engines. These closed-loop measurements compare the declared concurrency and
achieved throughput; they do not establish latency at matched offered load. Do not claim a cold-cache run without
observable cache control. Capture retries, throttling and client saturation;
count-only requests with tiny result pages can saturate a client first.

Archive: exact driver SHA/build info, config and query fingerprints, preparation
manifest/checksums, row count, ingestion/readiness/force-merge phases, raw search
JSON and per-query dumps, per-run timestamps/RSS, engine/module versions, actual
shard/replica/index settings, deployment limits, index statistics and monitoring.
Keep raw evidence private, exclude credentials, verify archive read-back, then
verify cloud-resource cleanup.

Filtered KNN is a later, separately named phase. Use the preparation tool's
`--add-knn-truth` and, if needed, `--add-hybrid-queries`, then `--verify`; it must
preserve corpus identity and supply the required truth capabilities. Missing
truth must fail before timed KNN. Compare hybrid runs at comparable measured
recall; do not relabel text/scalar agreement as vector recall.
