# Cross-cutting nitpick taxonomy — vector-db-benchmark, real precedent only

Grounded in actual GitHub history on `redis/vector-db-benchmark` (197 merged PRs, 100 issues sampled, as of
September 2026) plus this repo's own `AGENTS.md`, `README.md`, and `Makefile`. This is a multi-engine benchmark
harness (15 vector-DB backends in Rust), and the real finding here is that its recurring bugs are
**overwhelmingly cross-cutting invariants that get violated per-engine**, not fifteen independent per-engine
bug categories — the same three or four shapes of mistake keep recurring as each new engine or feature gets
added. Items 1–6 below are the cross-cutting invariants, evidenced across many engines each. The per-engine
table at the end covers real, documented protocol/DSL limits that are *not* bugs — don't flag them as such.

## 1. Silent wrong-answer beats a hard error — the dominant real bug class here

By far the most-repeated real bug shape in this repo's history: a code path that **cannot fully honor what a
config asked for keeps running anyway and reports a number**, instead of failing the run. Real, evidenced
instances, across nearly every engine:

- The founding case: `query_filter.rs`'s hard-error choke point (issue/PR #219) — before this, an engine whose
  filter builder couldn't express a condition would silently drop it and search unfiltered, then that recall
  got scored against *filtered* ground truth. Now: "no filter declared" (absent/`null`/`{}`) stays a normal
  unfiltered run; "filter declared but unbuildable" is a hard error. This is the single most-cited invariant in
  the repo's issue history.
- VectorSets datetime range filters ran completely unfiltered — recall silently dropped from 1.000 to 0.520
  (PR #230, closes #220).
- Qdrant: an empty/partially-dropped filter clause was a behavioral no-op that widened the query invisibly
  (PR #233, closes #222/#219) — the PR body itself is worth reading as the template for how to isolate and
  quantify this class of bug (see `voice-profiles.md`).
- KiviDB ignored `$params` in a hybrid prefilter entirely (PR #237). Milvus ran filtered searches with no
  scalar index on the filtered column, silently understating the engine (PR #242). MongoDB's HNSW build knobs
  were parsed and never forwarded, with a fake `ef` sweep (PR #241). OpenSearch/Elasticsearch had un-pinned
  shard counts and mismatched refresh semantics between the two engines being compared (PR #235, #246, #248).
  Redis/Valkey silently **delete** non-string members of a `match_any` list rather than erroring (issue #234,
  open at mining time — worth checking whether a PR under review actually closes this).

**When reviewing:** any new or modified filter/query builder, config-knob plumbing, or index-creation path
should be checked for exactly this — does an input the code can't fully honor produce an error, or does it
produce a number?

## 2. Per-config isolation — the "#151-4" invariant

Issue #151 item 4 is the founding, extremely well-documented case: `configured_index_name()` returned one
fixed name and `create_index()` ran `FLUSHALL` on every upload, so a 12-config Redis M×EF_CONSTRUCTION sweep
silently measured **only the last config's index** for all 12 result rows (*"observed a flat 0.917 for
`redis-m-16-ef-64` through `redis-m-64-ef-512`"*). The fix — derive index/collection/key names per-config, with
an explicit collision guard at startup — became a named, cross-referenced invariant (`#151-4`) that the repo's
own PR bodies cite by number when a new engine or feature adopts it (PR #308's MongoDB fix explicitly follows
"the Redis-family convention exactly... the same precedent #151-4 set"). Real recurrences: VectorSets used a
hardcoded key and clobbered concurrent runs (issue #236); MongoDB never adopted per-config naming at all
(issue #306/#307, PR #308); the startup collision guard itself has real gaps (`--skip-vector-index` collapsing
every config to one name, issue #283; a second bare `StartGate` slipping past a per-file invariant check, issue
#276).

**When reviewing:** any change that adds a benchmark sweep dimension, a new engine, or touches an
index/collection/key-naming function should be checked against this — does every distinct config in a sweep
get a distinct, derived name, and is there a guard that fails loudly on collision rather than silently
overwriting?

## 3. Corpus-size / "is it actually there" trust bugs

A recurring, real pattern: trusting a **count** as a proxy for **identity or readiness**, when the two can
diverge. Real instances: `--skip-upload`'s reuse guard checks corpus *size*, not corpus *identity*, so a
same-count-but-different-content corpus passes unnoticed (issue #279); the same guard is inert when a row count
can't be determined at all, publishing recall 0.0 at inflated QPS instead of failing (issue #290, PR #295);
MongoDB's index-catch-up probe stopped waiting at 10,000 docs regardless of true corpus size, letting 50 of 57
datasets publish recall against a ~1%-indexed index (issue #305, PR #309); ES/OS's `_count` is refresh-scoped
and ignores `_shards.failed` (issue #284). The mixed-workload update counter had the same shape: it counted
client-side update *attempts*, not what the server actually accepted (PR #298, closes an open issue on this).

**When reviewing:** anywhere a PR uses a count, a size, or "the request didn't error" as evidence that data is
present/correct/ready, ask whether it's checking the thing that actually serves queries, or a proxy for it that
can legitimately diverge.

## 4. MongoDB Atlas Search timing/race bugs — a real, recent, still-active cluster

MongoDB Atlas Search is a separately-provisioned, eventually-consistent index over the write store, and this
mismatch has produced a real, tight cluster of timing bugs in August 2026: a fixed-duration `sleep` before
declaring an index ready is not enough at scale (issue #313, PR #315 — "120s is not enough"); a catch-up probe
whose exhaustive `$vectorSearch` query itself exceeds a wire limit above ~888k documents (issue #313, PR #314 —
partition the probe by `_id` range); two of three timed harnesses had no start gate and could silently lose
workers (issue #307, PR #310); Atlas rejects the `:` separator in a generated search-index name (PR #311). If a
PR touches MongoDB's index-readiness polling, `_id`-range partitioning, or search-index naming, this is real,
recent, evidenced precedent for scrutinizing it more than a comparable change to a simpler engine.

## 5. Harness measurement-fidelity and concurrency bugs

The harness's own timed, multi-worker paths have a real recurring bug shape: synchronization that can silently
degrade instead of failing loudly. Evidenced: eight timed fan-out harnesses had no synchronized start at all
(issue #266); `StartGate::wait_ready` has no deadline, so a worker hung inside setup hangs the whole run (issue
#267); a thread-spawn failure or worker panic could deadlock the search-start barrier rather than failing the
run (PR #263); `WorkerPool::start` counted sample-buffer collection inside the measured `total_time`, biasing
latency (issue #270); `tests/overhead_invariants.rs` — the suite meant to guard exactly this class — had never
actually run in CI (issue #265, closed by wiring it into `ci.yml`). This is a Rust concurrency codebase; a PR
touching `thread::scope`, `AtomicUsize` work-stealing, or any timed loop should be checked for whether setup
work, sample collection, or barrier waits leak into the measured window, and whether a worker-side panic/error
degrades to a wrong-but-quiet result or fails the run.

## 6. Destructive test-suite safety

Real, evidenced: `integration_redis` unconditionally `FLUSHALL`ed whatever was listening on the default test
port (6399), and an old port-override env var silently no-op'd instead of erroring, so a misconfigured test run
could wipe a server it didn't own (issue #292). The fix (PR #300) made destructive integration suites refuse to
`FLUSHALL` a server they don't provably own. Any new integration test that connects to a real engine and
performs a destructive setup step (flush, drop-index, drop-collection) should be checked for the same
ownership/scope guard.

## 7. New-engine review checklist (from the one real evidenced case, PR #203)

When a PR adds a new engine (this repo has added several in 2026: Chroma, KiviDB, Turbopuffer, Vertex AI), the
real, evidenced review pass (`voice-profiles.md`'s PR #203 writeup) checked, specifically: RESP/wire-protocol
parsing, vector encoding, filter translation, concurrency/connection lifecycle, per-config namespacing (item 2
above), search-`ef`/tuning-knob gating, distance-metric mapping, error handling, config schema, trait
completeness, test coverage, and a secrets/injection sweep. Two real "before merge" defect classes were found
that are worth checking on any new engine, not just KiviDB: a **socket/read timeout shorter than the engine's
own graceful polling timeout** (so a slow server hard-aborts instead of hitting the intended graceful path),
and a **regression-guard test that reimplements the production parsing logic inline** instead of calling the
real function, so it can't actually catch a production regression in what it claims to guard.

## Per-engine known real limits — not bugs, don't flag as such

These are documented, live-verified protocol/DSL limits (mostly in `README.md`'s per-engine notes), not gaps to
"fix" in an unrelated PR:

| Engine | Real, documented limit |
|---|---|
| Chroma, Turbopuffer | No geo operator in either's filter DSL at all — a `geo` leaf is a **permanent**, hard-error gap (#223), not a to-do. |
| Dragonfly, Valkey | No `GEO` field type / rejects `$param` placeholders in the shared geo clause — hard error since #219, not silent degradation. |
| KiviDB | Multi-valued `labels` arrays unsupported (`arxiv-titles-384-angular-filters` hard-errors as designed). |
| Vertex AI | Cross-field `or`, nested boolean, numeric `match_any` IN-list, and geo are hard errors — Vertex's `restricts`/`numericRestricts` model can't express them, by design, not by oversight. |
| Weaviate | Falls back from gRPC to slower GraphQL only when a filter can't be expressed over gRPC, or when `WEAVIATE_USE_GRAPHQL` is explicitly set — not a bug if seen in a trace. |
| MongoDB | 255-byte namespace ceiling on `<db>.<collection>` is a real, live-verified MongoDB limit (not this codebase's bug) — over-budget names are hashed (FNV-1a-64), not truncated, specifically so two configs with a shared long prefix can't collide. |

## What this taxonomy is honestly thin or silent on

- **Multi-person review disagreement.** Outside of the CI bots, this repo's real review record is either one
  person (`fcostaoliveira`) reviewing his own work rigorously, or the same person doing a genuine deep review
  of one outside contribution (PR #203). There is no real evidence of two humans disagreeing with each other in
  this repo's history — don't write dialogue between named contributors that never happened.
- **Community bug triage.** 98 of the 100 most-recently-sampled issues are self-filed by `fcostaoliveira` as an
  engineering audit trail. There is exactly one real external bug report in the sample (#52) and it predates
  the Rust rewrite. An issue-triage automation built on the assumption of a steady stream of non-maintainer
  reports is solving for a pattern that doesn't really exist here yet — say so if asked, rather than assuming
  otherwise.
- **Python (`v0/`) code review standards.** The Python implementation is now explicitly legacy (`AGENTS.md`:
  "partial migration" reference implementation for precision comparison only) — this taxonomy is about the
  actively-developed Rust surface. A PR touching only `v0/` is out of scope for most of the above; say so.
- **Style/formatting nits.** `cargo fmt --check` and `cargo clippy --all-targets -- -D warnings` are a real,
  blocking CI gate (`ci.yml`) — no real reviewer comment in this repo's sample re-litigates formatting or lint
  findings CI already enforces.
