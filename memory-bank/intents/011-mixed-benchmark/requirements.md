---
intent: 011-mixed-benchmark
phase: inception
status: complete
created: 2026-03-05T10:00:00Z
updated: 2026-03-05T10:00:00Z
---

# Requirements: Mixed Benchmark (Concurrent Search + Update)

## Intent Overview

Add a mixed workload benchmark mode that interleaves vector updates with searches at a configurable ratio. This simulates real-world scenarios where a vector database serves queries while simultaneously receiving data updates. The `--update-search-ratio` CLI flag controls the interleaving (e.g., `1:10` = 1 update per 10 searches). Initial engine support: Redis (RediSearch), VectorSets, Valkey.

## Business Goals

| Goal | Success Metric | Priority |
|------|----------------|----------|
| Measure search performance under write pressure | Report search QPS/latency degradation vs search-only baseline | Must |
| Measure update throughput during mixed workload | Report update RPS and latency percentiles separately | Must |
| Reproducible mixed benchmarks | Same ratio + seed = same results across runs | Must |
| Support ratio tuning | Users can sweep ratios (1:1, 1:10, 1:100) to find degradation curves | Should |

---

## Functional Requirements

### FR-1: CLI Flag `--update-search-ratio`
- **Description**: Accept `--update-search-ratio <U>:<S>` on the CLI. For every S searches, perform U updates. Example: `1:10` means 1 update per 10 searches.
- **Acceptance Criteria**: Flag parses correctly; invalid ratios (0:0, negative, non-numeric) produce clear errors; omitting the flag runs search-only (existing behavior).
- **Priority**: Must

### FR-2: Interleaved Execution
- **Description**: Each worker thread interleaves updates and searches based on the ratio. With ratio `1:10`, a worker performs 10 searches then 1 update, repeating until all queries are exhausted.
- **Acceptance Criteria**: The interleaving pattern is deterministic given the same inputs. Workers share a global query counter (existing pattern) plus a global update counter.
- **Priority**: Must

### FR-3: Update Operation (Vector + Metadata)
- **Description**: An update replaces both the vector embedding and metadata for an existing ID. Uses the same ingestion vectors (re-insert same ID with same data). The update pool is all ingested vectors — each update picks the next vector from a seeded deterministic sequence.
- **Acceptance Criteria**: Updates use the engine's native upsert path (HSET for Redis/Valkey, VADD with existing ID for VectorSets). The update sequence is reproducible given the same seed.
- **Priority**: Must

### FR-4: Deterministic Update Sequence
- **Description**: A seeded PRNG (e.g., `rand::SeedableRng` with fixed seed) generates the sequence of vector IDs to update. All vectors are eligible. The sequence wraps around if more updates than vectors.
- **Acceptance Criteria**: Two runs with the same seed and ratio produce the same update sequence.
- **Priority**: Must

### FR-5: Separate Metrics Reporting
- **Description**: Report update and search metrics independently in the results JSON: `search_rps`, `search_mean_time`, `search_p50`, `search_p95`, `search_p99`, `update_rps`, `update_mean_time`, `update_p50`, `update_p95`, `update_p99`.
- **Acceptance Criteria**: Results JSON contains both metric sets. Summary output displays both. Precision is still computed from search results only.
- **Priority**: Must

### FR-6: Engine Support — Redis (RediSearch)
- **Description**: Implement mixed benchmark for the Redis engine. Update = `HSET` with new vector blob + metadata fields for the given ID.
- **Acceptance Criteria**: Mixed benchmark runs on Redis with correct interleaving and metrics.
- **Priority**: Must

### FR-7: Engine Support — VectorSets
- **Description**: Implement mixed benchmark for VectorSets. Update = `VADD` with existing element name (upsert semantics) + `SETATTR` for metadata.
- **Acceptance Criteria**: Mixed benchmark runs on VectorSets with correct interleaving and metrics.
- **Priority**: Must

### FR-8: Engine Support — Valkey
- **Description**: Implement mixed benchmark for Valkey. Update = `HSET` (same as Redis, RESP compatible).
- **Acceptance Criteria**: Mixed benchmark runs on Valkey with correct interleaving and metrics.
- **Priority**: Must

### FR-9: Results JSON Schema Extension
- **Description**: Extend the search results JSON to include `update_search_ratio`, `update_count`, and the update latency metrics alongside existing search metrics. When ratio is not specified, these fields are omitted (backward compatible).
- **Acceptance Criteria**: Existing result parsers/scripts don't break. New fields only appear when mixed mode is active.
- **Priority**: Must
- **AMENDED BY #293**: this document is the original intent and is left as
  written, but `update_count`'s meaning has since changed and this file no longer
  describes it. It was implemented as a count of client-side loop iterations; it
  now counts writes the **server** accepted, with failures excluded into
  `update_failures`, the attribution tier published as `update_attribution`, and
  writes the server could not attribute to an existing corpus row counted in
  `update_unattributed`. The README's "Mixed Benchmarks" section is the current
  schema reference; the same caveat applies to FR-5 above and to
  `units/001-mixed-benchmark-core/stories/{002,004}-*.md` and
  `bolts/006-mixed-benchmark-core/implementation-plan.md`.

### FR-10: Engine Trait Extension
- **Description**: Add an `update` method to the Engine trait (or a `MixedBenchmarkSupport` sub-trait) that takes a vector ID, vector data, and optional metadata, and performs the upsert. Engines that don't support mixed mode return an error.
- **Acceptance Criteria**: The trait method has a default implementation returning `Err("not supported")`. Redis, VectorSets, and Valkey override it.
- **Priority**: Must

### FR-11: Precision Invariance Under Mixed Workload
- **Description**: Since updates re-insert the same vectors with the same data (same ID, same embedding, same metadata), the search precision must remain identical to a search-only benchmark. The mixed workload must not degrade recall.
- **Acceptance Criteria**: Running the same dataset with `--update-search-ratio 1:10` produces the same mean_precision (within floating-point tolerance) as running without the flag. This serves as a correctness validation that updates are not corrupting the index.
- **Priority**: Must

---

## Non-Functional Requirements

### Performance
| Requirement | Metric | Target |
|-------------|--------|--------|
| Mixed mode overhead | Framework overhead vs search-only | < 5% (excluding actual update cost) |
| Update latency reporting | Timer accuracy | Microsecond precision (same as search) |

### Reliability
| Requirement | Metric | Target |
|-------------|--------|--------|
| Determinism | Same seed + ratio = same sequence | 100% reproducible |
| Error handling | Failed updates | Report count, don't abort entire benchmark |

---

## Constraints

### Technical Constraints

**Intent-specific constraints**:
- Must reuse existing parallel search infrastructure (thread::scope, AtomicUsize counters)
- Update vectors come from the same dataset used for ingestion (no separate update dataset)
- The interleaving happens within each worker thread, not globally — actual ratio may vary slightly under parallelism but the pattern per-thread is deterministic

### Business Constraints
- Initial scope limited to Redis, VectorSets, Valkey (engines already using `redis` crate)
- Other engines (Qdrant, ES, etc.) can be added later via the trait method

---

## Assumptions

| Assumption | Risk if Invalid | Mitigation |
|------------|-----------------|------------|
| VADD with existing element name does upsert | Updates silently fail or error | Test with redis-cli before implementation |
| HSET on existing hash key does upsert | Unexpected behavior | Already confirmed by RediSearch semantics |
| Interleaved updates don't cause index corruption | Data integrity issues | Validate precision post-mixed-run |

---

## Open Questions

| Question | Owner | Due Date | Resolution |
|----------|-------|----------|------------|
| Should we support `--update-seed` flag for custom seeds? | TBD | TBD | Default to fixed seed (42), add flag later if needed |
| Should summary output show a comparison table (mixed vs search-only)? | TBD | TBD | Could — nice to have for degradation analysis |
