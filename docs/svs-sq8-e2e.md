# SVS and HNSW SQ8 end-to-end comparison

Use `experiments/configurations/redis-svs-sq8-comparison.json` on a dedicated
Redis instance with HNSW SQ8 support. These are client-to-Redis benchmarks.
The [SQ8 execution guide](hnsw-sq8-e2e.md) covers binary provenance, schema
readback, complete backend drain, cleanup, and repeated independent builds.

## Paired workload

Each input type (FLOAT32 and FLOAT16) has four cases:

| Case suffix | Algorithm | Compression | HNSW training threshold |
| --- | --- | --- | --- |
| plain | HNSW | none | not applicable |
| sq8-zero | HNSW | SQ8 | 0 |
| sq8-trained | HNSW | SQ8 | 10240 |
| svs-lvq8 | SVS-VAMANA | requested LVQ8 | not applicable |

All cases use the same complete corpus, upload parallelism 8, batch size 64,
k=10 and k=100, and 1/8/100 search clients. Each search setting sweeps
100/200/400/800/1600 using HNSW `ef` or SVS `SEARCH_WINDOW_SIZE`, followed by
calibration targets 0.95 and 0.99. This is 42 search points per index, 8 indexes.
The existing k=10-only SQ8 file remains useful for datasets with narrow ground truth.

HNSW uses M=32 and EF_CONSTRUCTION=200; SVS uses GRAPH_MAX_DEGREE=32 and
CONSTRUCTION_WINDOW_SIZE=200. These are starting construction settings:
equal numeric knobs do not imply equal graph structure or equal recall.
Report each algorithm's measured speed/quality tradeoff.

On the pinned dorer-arm development build, requested SVS LVQ8 reports
`GlobalSQ8` in FT.INFO and scalar quantization in VECSIM_INFO. Label results
with that effective compression, CPU and build. They do not measure Intel LVQ
performance. Read back the SVS graph degree, construction window, compression
and actual training threshold; the HNSW threshold field does not configure SVS.

## Dataset and workers

Start with registered dataset `dbpedia-openai-975K-1536-angular-100neighbors`.
This corrects the previous `dbpedia-openai-1M-1536-angular-100neighbors` entry:
the archive contains 975,000 vectors, not 1,000,000. Its source URL is unchanged.
Update saved commands to the corrected dataset name; existing downloads can be
moved to `datasets/dbpedia-openai-975K-1536-angular-100neighbors/dbpedia_openai_975K`.

Before measuring, verify the downloaded corpus count and that every measured
query has at least 100 valid ground-truth neighbors. The registry name alone is
not verification. The existing DBpedia 100K download has only 10 neighbors and
must not be used with this combined k=10/k=100 matrix.

For other SVS datasets, apply the same complete matrix to each corpus separately
after checking its ground truth. Do not compare different embedding models as
if only corpus size or dimension changed, shrink the corpus without recomputing
ground truth, or truncate vectors.

Run independent full builds with `search-workers=0` and `search-workers=4`,
reading back the setting before each build. Worker count is a server setting;
`upload_params.parallel` and search `parallel` are client counts. Repeat each
build at least three times and alternate algorithm order. Keep only one case's
index and keys resident when collecting memory measurements.

## Run one case

Build the harness with `make vector-db-benchmark` on the chosen build host.
The harness writes to the checkout's `results/` directory. Archive it after each
case, worker setting and repetition, then start the next case with an empty
`results/` directory. Pin the harness commit, Redis/Search hashes, dataset
hashes and effective settings.
The example assumes the dedicated server already has the intended worker
setting and `search-on-timeout=FAIL`.

```sh
export REDIS_PORT=14960
export REDIS_QUERY_TIMEOUT=90000
CASE=redis-compare-float32-sq8-trained
DATASET=dbpedia-openai-975K-1536-angular-100neighbors
CONFIG=experiments/configurations/redis-svs-sq8-comparison.json

./target/release/vector-db-benchmark \
  --engines-file "$CONFIG" --engines "$CASE" --datasets "$DATASET" \
  --skip-search --keep-data --skip-if-exists false
```

Before searching, verify FT.INFO has the requested schema, complete document
count and no indexing errors. Inspect `_FT.DEBUG VECSIM_INFO idx:$CASE vector`:
frontend size must be 0, backend size must equal the corpus count and background
indexing must be 0. Record upload plus additional backend-drain time separately
from dataset download. The harness upload timer alone does not prove completion.

```sh
./target/release/vector-db-benchmark \
  --engines-file "$CONFIG" --engines "$CASE" --datasets "$DATASET" \
  --skip-upload --keep-data --skip-if-exists false \
  --fail-on-dropped-queries --repetitions 1 --search-duration 30
```

Copy results and telemetry, then drop only this case's index and owned documents
with `FT.DROPINDEX idx:$CASE DD` before the next case. Repeat for
`redis-compare-float32-svs-lvq8`, both HNSW controls, and their FLOAT16 counterparts.

## Compare results

Report full build time through backend drain, vector-index memory, total Redis
memory, QPS, p50/p95/p99 latency and `mean_recall` for each k and client count.
FLOAT16 uses the original dataset ground truth, so its quality includes input
conversion. Compare compression within each input type.

Calibration uses the existing harness's `mean_precision_at_returned`, not recall,
and searches breadth in [top, 1000]. Its `params.calibration.reached_target`
flag is therefore not proof of recall at the requested target. Check the final
measurement's `mean_recall`, full ground-truth width and absence of failed queries.
Precision and recall agree only when all queries return k distinct results
against k valid reference neighbors.

For recall targets 0.95 and 0.99, select the best measured QPS that actually meets
the target within the same dataset, dtype, k, worker count and client count.
Include both fixed-sweep and calibrated points. The fixed sweep extends to 1600,
beyond calibration's ceiling. Report a target as unachieved if no measured point
qualifies; never infer a crossing or use precision as a substitute for recall.
Retain all repetitions and use an identical-binary control before attributing
small speed differences.

This matrix measures complete ingestion and steady-state search. Threshold
crossing and responsiveness during training are covered separately by the
[Redis transition runner](https://github.com/dor-forer/vector-db-benchmark/blob/dor-forer-MOD-14960-sq8-transition/docs/redis-transition-benchmark.md),
which currently lives on a separate follow-up branch.

The `redis-svs-sq8-comparison.json` configuration is generated by
`experiments/configurations/create-hnsw-sq8.py`.
