# MS MARCO preparation without nearest-neighbor truth (#388)

`--workloads text-scalar` skips **both** unfiltered and filtered KNN oracles.
It keeps the existing hash-selected corpus, vectors, document IDs, query text,
query embeddings, and scalar predicates with exact expected counts. It does not
eliminate embedding downloads or the approximately 410 GB `vectors.npy` at 100M.

Build and prepare on a provisioned preparation host, from the driver checkout:

```bash
make build
make prepare-msmarco MSMARCO_DATASET=msmarco-cohere-1024-100K-cosine \
  MSMARCO_ARGS='--workloads text-scalar'
./target/release/prepare-msmarco \
  --dataset msmarco-cohere-1024-100K-cosine --verify
```

The default remains `--workloads full`. For text/scalar, `tests.jsonl` carries
query inputs but **omits** `closest_ids`/`closest_scores`; there is no empty or
fabricated recall. `hybrid_queries.jsonl` is absent. `PREPARED.json.preparation`
version 1 declares `vectors`, `text`, `scalar`, `vector_truth`, `filtered_truth`
and a byte count and SHA-256 for each file. `COMPLETE` is published last.
`--verify` checks every file digest and regenerates scalar fields/counts, without
running a KNN oracle. The loader checks capabilities, identity, sizes, query
schema and the NPY shape before engine creation. Hash verification is an explicit
streaming pass; run it after transporting or restoring a dataset.

Existing full artifacts remain supported, including the published v2 archives
whose manifests use the old `-crc32-` names. A ranking-free artifact without its manifest is refused. New manifests with
incomplete capabilities or a missing completion marker are also refused.

## Redis/OpenSearch smoke and reuse

The example `docs/msmarco-text-scalar.json` config uses separate parity BM25
and scalar indexes. Parity normalizes uploaded text, so sharing it with scalar
workloads is intentionally refused. The text profile retains the existing full
schema including its vector index; the scalar profile uses a no-vector index.
Omitting KNN truth does not mean omitting the text profile's vector index. A
dedicated text-only layout is a separate change (#363).

With Redis on port 6379 and OpenSearch on port 9200:

```bash
for profile in text scalar; do
  ./target/release/vector-db-benchmark --engines "redis-msmarco-no-truth-$profile" \
    --engines-file docs/msmarco-text-scalar.json \
    --datasets msmarco-cohere-1024-100K-cosine --host localhost \
    --upload-reader stream --queries 64 --repetitions 1 --keep-data \
    --fail-on-dropped-queries --dump-results "/tmp/msmarco-dumps/redis/$profile"
  OPENSEARCH_INDEX="msmarco-no-truth-$profile" \
  ./target/release/vector-db-benchmark --engines "opensearch-msmarco-no-truth-$profile" \
    --engines-file docs/msmarco-text-scalar.json \
    --datasets msmarco-cohere-1024-100K-cosine --host http://localhost \
    --upload-reader stream --queries 64 --repetitions 1 --keep-data \
    --fail-on-dropped-queries --dump-results "/tmp/msmarco-dumps/opensearch/$profile"
  ./target/release/vector-db-benchmark compare-results \
    "/tmp/msmarco-dumps/redis/$profile" "/tmp/msmarco-dumps/opensearch/$profile" \
    --reference redis --json "/tmp/msmarco-comparison-$profile.json"
done
```

Keep each OpenSearch profile on its own `OPENSEARCH_INDEX`: the driver otherwise
uses one index named `bench`, and the next profile replaces it. Repeat either
benchmark command with `--skip-upload --skip-if-exists false` and a different dump
folder to exercise the existing completeness/index-compatibility checks. The
scalar path verifies exact expected counts per query. Text comparison reports
ranking overlap; it is not judged relevance or vector recall. These short runs
are correctness checks, not headline performance measurements.

## Add KNN truth later without rebuilding the corpus

```bash
./target/release/prepare-msmarco \
  --dataset msmarco-cohere-1024-100K-cosine --add-knn-truth
./target/release/prepare-msmarco \
  --dataset msmarco-cohere-1024-100K-cosine --verify
```

This validates the recorded hashes and pinned query identity (allowing up to
2e-6 coordinate drift from float32 normalization across builds), preserves and
scores the stored query vectors, and computes truth from
the existing vectors, cross-checks against upstream hits remapped by docid,
then adds filtered truth. It rewrites only queries and manifest, never vectors
or payloads. An interrupted publication has no `COMPLETE` marker and is refused
by benchmark loaders. Rerun the preparation/upgrade command to finish a sealed
publication: it verifies every candidate file hash, corpus identity, NPY shape
and query schema before renaming any remaining `.part` files, then restores
`COMPLETE`. Missing or corrupt candidates remain refused; unchanged corpus
files are not rebuilt.
A vector/hybrid workload on a text/scalar artifact fails before creating an
engine and prints the upgrade command. `--add-derived-fields` still invalidates
filtered truth and updates capabilities; rerun `--add-knn-truth` afterward.

## Avoid repeated upstream scans

Build the largest required variant once, then stream nested samples from it:

```bash
./target/release/prepare-msmarco \
  --dataset msmarco-cohere-1024-100M-cosine --workloads text-scalar --shard-workers 48
./target/release/prepare-msmarco \
  --dataset msmarco-cohere-1024-10M-cosine --workloads text-scalar \
  --from-prepared datasets/msmarco-cohere-1024/crc32-t880865
```

The default is 16 concurrent upstream shards; `--shard-workers` accepts 1–64.
On the 96-vCPU AWS preparation host, use 48 to exploit the network/disk capacity.
Retries honor upstream `Retry-After`; shard order, selection and IDs do not
depend on concurrency. Use the same source for 1M and 100K. Source and destination must differ. The
source must be an equal/larger registered corpus with the matching hash rule
and realized count; query inputs must match the pinned upstream file. New
sources are hash-verified before sampling. Legacy source archives have no file
hashes; retain their download provenance and verify them separately. Sampling
streams vectors and payloads in source order and applies the same docid hash
threshold, preserving the IDs that a fresh preparation of that size assigns.
The nested artifact records its source dataset and manifest SHA-256, and reports
zero upstream shards read for that preparation.

Allow roughly 600 GB for the 100M final corpus, plus temporary shard files and
smaller variants. A 2 TiB working disk gives room for preparation and S3
verification. Measure `/usr/bin/time -v` peak RSS, actual disk usage and transfer
times on the host; batch buffers alone are not a process memory measurement.
Cloud host provisioning, private S3 publication/read-back, and cleanup remain
in the orchestration repository.

Scalar validation parses batches in parallel, using the existing field validator
and preserving corpus/lookup-key order. Each batch is capped at 8,192 rows or
16 MiB (plus the final row). The compact derived-field table still uses 16 bytes
per passage, approximately 1.6 GB at 100M, in addition to process overhead.

## Recover an interrupted upstream scan

Keep the same attempt's `parts.tmp` directory and run:

```bash
./target/release/prepare-msmarco \
  --dataset msmarco-cohere-1024-100M-cosine --workloads text-scalar \
  --shard-workers 48 --resume-parts
```

Recovery rereads the complete pinned metadata for each cached shard and compares
all selected payloads in order, then checks the exact raw vector byte count and
all floats for finite values. A truncated, mismatched or reordered shard is
redownloaded. Only use parts produced by this preparation attempt: they have no
independent vector checksum until the final manifest is sealed. Completed files
are still SHA-256 verified before publication. HTTP retries honor `Retry-After`;
interrupted body streams retry the affected shard twice. A one-hour HTTP timeout
allows large streams to finish while sharing bandwidth with the other shards.
