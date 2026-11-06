//! Hybrid search, BM25 full-text + vector KNN fused by Reciprocal Rank Fusion
//! (workload `hybrid-rrf`). Not to be confused with [`crate::hybrid`], the
//! #341 prefilter + KNN workload.
//!
//! Everything the Redis and OpenSearch arms must agree on lives here, so the
//! two arms in the binary are I/O only: the semantics, the config block, the
//! query set, the request each engine is sent, the reply checks, the
//! client-side reference and the per-query score.
//!
//! # Semantics (`rrf-v1`), identical on both engines
//!
//! For one query, with page size `top`, candidate window `W` and rank
//! constant `k`:
//!
//! 1. **BM25 candidates**: the top `W` documents of the ranked full-text
//!    query of [`crate::fulltext`] — the same query terms, field boosts,
//!    analyzer (`english` or `parity`), stopwords and scorer as the
//!    `fulltext` workload, from the search config's own `fulltext` block.
//! 2. **KNN candidates**: the top `W` documents of the KNN query on the
//!    vector field, HNSW breadth `ef` (required, `ef >= W`).
//! 3. **Fusion**: each document scores `sum over the lists it is in of
//!    1 / (k + rank)`, rank 1-based within its list; a list it is not in
//!    contributes nothing. Return the `top` highest.
//!
//! Server-side only: Redis 8.4+ `FT.HYBRID … COMBINE RRF 4 CONSTANT k WINDOW W`
//! (the SEARCH subquery is capped at `WINDOW`, and `KNN K W` is capped at it
//! too), OpenSearch 2.19+ `hybrid` query with `pagination_depth: W` and a
//! search pipeline whose `score-ranker-processor` combines with `rrf` at
//! `rank_constant: k`. Both engines rank 1-based and add `1/(k + rank)`
//! (RediSearch `HybridRRFScore`; neural-search `RRFNormalizationTechnique`).
//! A server that cannot run it is refused before the timed window; there is
//! no client-side fusion fallback. `k` is an integer in
//! [`MIN_RANK_CONSTANT`]..=[`MAX_RANK_CONSTANT`] because OpenSearch accepts
//! nothing else (Redis takes any positive double).
//!
//! # Reference ("ground truth")
//!
//! Computed client-side and deterministically, before the timed window, by
//! [`reference`]: RRF at the same `k` and `W` over
//!
//! - the **exact BM25** top `W`: the engine's own ranked full-text query
//!   (same request as step 1, an inverted-index scan, which is exact). BM25
//!   scores are engine-defined (see `fulltext::KNOWN_DIFFERENCES`), so this
//!   arm of the reference is per engine; and
//! - the **exact KNN** top `W`: the dataset's brute-forced neighbours
//!   (`tests.jsonl`), the same truth the vector workload is scored against —
//!   engine independent.
//!
//! The run therefore measures what the server-side fusion loses against
//! fusing exact candidate lists: the approximate KNN arm (HNSW at `ef`),
//! anything in the fusion itself, and equal-BM25-score documents inside the
//! window that the server's BM25 arm and the reference order differently
//! (tie-aware recall absorbs only ties in the FUSED score). Cross-engine agreement is a separate
//! question, answered by `compare-results` on the dumps.
//!
//! # Score
//!
//! RRF produces ties by construction (a document ranked `r` by BM25 only and
//! one ranked `r` by KNN only score the same), and the engines break them
//! differently. [`score_query`] is therefore tie-aware: a returned document is
//! a hit when its reference score is at least the reference score at position
//! `top` (within [`TIE_TOLERANCE`], which absorbs OpenSearch's float32
//! scores). `mean_recall` is that tie-aware recall@top; the strict recall
//! against the id-tie-broken reference list is published next to it.
//!
//! Every returned score must also be an RRF score at this `k` and `W` (in
//! range and on the `1/(k+a) [+ 1/(k+b)]` lattice, [`check_rrf_score`]): a
//! server that ignored the fusion (raw BM25, linear or min-max scores) is
//! refused, never scored.

use crate::fulltext::{self, Analysis, FieldBoost, FulltextSpec};
use crate::workload::Workload;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};

/// Key of the fusion block inside a `hybrid-rrf` search config.
pub const SPEC_KEY: &str = "rrf";

/// Name of the semantics, recorded in every result.
pub const SEMANTICS: &str = "rrf-v1";

/// OpenSearch's `rank_constant` bounds (`RRFNormalizationTechnique`).
pub const MIN_RANK_CONSTANT: u32 = 1;
pub const MAX_RANK_CONSTANT: u32 = 10_000;

/// Largest window: OpenSearch's default `index.max_result_window`, which
/// bounds `pagination_depth`.
pub const MAX_WINDOW: usize = 10_000;

/// Relative tolerance for comparing RRF scores. OpenSearch computes each
/// `1/(k+rank)` to 10 decimals and sums in float32 (relative error ~1e-7 at
/// these magnitudes); Redis prints doubles. Distinct single-list RRF values
/// differ by about `1/(k+W)` relatively, far above this.
pub const TIE_TOLERANCE: f64 = 1e-5;

/// The `rrf` block of a `hybrid-rrf` search config. Both keys are required:
/// no engine default decides the semantics.
///
/// ```json
/// {"workload": "hybrid-rrf", "top": 10, "search_params": {"ef": 200},
///  "fulltext": {"fields": {"title": 1.0, "segment": 1.0}},
///  "rrf": {"window": 100, "rank_constant": 60}}
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RrfSpec {
    /// Candidates taken from EACH of the BM25 and KNN lists (`W`).
    pub window: usize,
    /// RRF rank constant (`k`).
    pub rank_constant: u32,
}

/// Parse and validate the `rrf` block. Absent is an error.
pub fn parse_spec(block: Option<&Value>) -> Result<RrfSpec, String> {
    let block = block.ok_or_else(|| {
        format!(
            "hybrid-rrf needs an explicit `{SPEC_KEY}` block: {{\"window\": W, \"rank_constant\": k}}"
        )
    })?;
    let spec: RrfSpec = serde_json::from_value(block.clone())
        .map_err(|e| format!("invalid `{SPEC_KEY}` block: {e}"))?;
    if spec.window == 0 || spec.window > MAX_WINDOW {
        return Err(format!(
            "`{SPEC_KEY}.window` must be in 1..={MAX_WINDOW} (OpenSearch's default \
             index.max_result_window bounds pagination_depth), got {}",
            spec.window
        ));
    }
    if !(MIN_RANK_CONSTANT..=MAX_RANK_CONSTANT).contains(&spec.rank_constant) {
        return Err(format!(
            "`{SPEC_KEY}.rank_constant` must be an integer in \
             {MIN_RANK_CONSTANT}..={MAX_RANK_CONSTANT} (OpenSearch's bounds), got {}",
            spec.rank_constant
        ));
    }
    Ok(spec)
}

/// The `fulltext` block of a `hybrid-rrf` config: the BM25 arm. Same parser
/// as the fulltext workload; the compound suite and the text/scalar layouts
/// are refused (they have no vector field, or no ranked BM25 query).
pub fn parse_text_spec(block: Option<&Value>) -> Result<FulltextSpec, String> {
    let spec = fulltext::parse_spec(block)?;
    check_text_spec(&spec)?;
    Ok(spec)
}

/// The BM25-arm restrictions of [`parse_text_spec`] on an already-parsed spec.
pub fn check_text_spec(spec: &FulltextSpec) -> Result<(), String> {
    if spec.compound.is_some() {
        return Err(
            "hybrid-rrf fuses the ranked BM25 query; `fulltext.compound` (a Boolean match \
             suite) cannot be its text arm"
                .into(),
        );
    }
    if spec.allow_multi_shard {
        return Err(
            "hybrid-rrf refuses `fulltext.allow_multi_shard`: each shard keeps its own top W \
             per subquery while the reference is the global top W, and multi-shard RRF ranking \
             has not been verified live — run it on one shard"
                .into(),
        );
    }
    if spec.no_vector_index || spec.scalar_indexing.is_reduced() {
        return Err(
            "hybrid-rrf needs the vector index; `fulltext.no_vector_index` / text-tag-only \
             layouts cannot serve it"
                .into(),
        );
    }
    Ok(())
}

/// Startup check, before anything is downloaded or connected: an `rrf` block
/// only on a `hybrid-rrf` config, and a `hybrid-rrf` config with a valid one.
/// (Its `fulltext` block is checked by `fulltext::validate_declared`.)
pub fn validate_declared(workload: Workload, block: Option<&Value>) -> Result<(), String> {
    match (workload, block) {
        (Workload::HybridRrf, b) => parse_spec(b).map(|_| ()),
        (w, Some(_)) => Err(format!(
            "a `{SPEC_KEY}` block is only valid on a hybrid-rrf search config, not workload \"{w}\""
        )),
        (_, None) => Ok(()),
    }
}

/// Redis deployment check, at engine construction: hybrid-rrf is refused
/// through the OSS Cluster API. The reference is the GLOBAL top W of each
/// list. How the RediSearch coordinator merges FT.HYBRID's subquery lists
/// across shards before fusing is not verified live. A shard-local fusion
/// still yields scores on the RRF lattice, so the probe could not tell. This
/// is the Redis counterpart of the OpenSearch `allow_multi_shard` refusal.
pub fn check_redis_deployment(cluster_api: bool, workloads: &[Workload]) -> Result<(), String> {
    if cluster_api && workloads.contains(&Workload::HybridRrf) {
        return Err(
            "hybrid-rrf refuses the Redis Cluster API (cluster_api / REDIS_CLUSTER): the \
             reference is the global top W per list and FT.HYBRID's cross-shard merge is not \
             verified live — run it against a single-shard endpoint"
                .into(),
        );
    }
    Ok(())
}

/// Startup check of a `hybrid-rrf` config's typed keys. Returns `top`.
///
/// - `top` is required (never derived from the ground-truth width) and at
///   most `W`;
/// - `search_params.ef` is required and at least `W`: it is the KNN arm's
///   HNSW breadth on both engines, and an engine default (Redis 10, Lucene
///   `k`) must not decide it;
/// - query-count scheduling only, like the fulltext workload.
pub fn check_params(
    top: Option<i64>,
    ef: Option<i64>,
    spec: &RrfSpec,
    target_qps: Option<f64>,
    duration_seconds: Option<f64>,
) -> Result<usize, String> {
    fulltext::check_execution_params(target_qps, duration_seconds)
        .map_err(|e| e.replace("fulltext", "hybrid-rrf"))?;
    let top = match top {
        Some(t) if t >= 1 => t as usize,
        Some(t) => return Err(format!("hybrid-rrf: `top` must be >= 1, got {t}")),
        None => {
            return Err(
                "hybrid-rrf needs an explicit `top` (the fused page size); it is never taken \
                 from the ground-truth width"
                    .into(),
            )
        }
    };
    if top > spec.window {
        return Err(format!(
            "hybrid-rrf: `top` ({top}) is larger than `{SPEC_KEY}.window` ({}): the fused page \
             cannot be filled from W candidates per list deterministically",
            spec.window
        ));
    }
    match ef {
        None => Err(format!(
            "hybrid-rrf needs `search_params.ef` (the KNN arm's HNSW breadth, >= window {}); \
             an engine default must not decide it",
            spec.window
        )),
        Some(ef) if ef < spec.window as i64 || ef > i32::MAX as i64 => Err(format!(
            "hybrid-rrf: `search_params.ef` ({ef}) must be >= `{SPEC_KEY}.window` ({}) — the \
             KNN arm returns W candidates — and fit in 32 bits",
            spec.window
        )),
        Some(_) => Ok(top),
    }
}

/// Fingerprint at the dispatch point: a `hybrid-rrf` result carries its
/// [`result_metadata`] block, no other workload's does.
pub fn verify_fingerprint(workload: Workload, has_metadata: bool) -> Result<(), String> {
    match (workload, has_metadata) {
        (Workload::HybridRrf, false) => Err(
            "workload hybrid-rrf returned a result without hybrid_rrf metadata — the engine did \
             not run its hybrid-rrf path"
                .into(),
        ),
        (w, true) if w != Workload::HybridRrf => Err(format!(
            "workload \"{w}\" returned hybrid_rrf metadata — the engine ran its hybrid-rrf path"
        )),
        _ => Ok(()),
    }
}

/// Builder for `QueryConditions::try_resolve_all`: hybrid-rrf has no filter
/// clause, so any DECLARED condition is an error (its KNN truth would be the
/// filtered one). Absent conditions never reach this function.
pub fn refuse_conditions(conditions: &Value) -> Result<Option<()>, String> {
    Err(format!(
        "hybrid-rrf has no filter clause, but the query declares conditions {conditions}; its \
         KNN ground truth is filtered, so running it unfiltered would score against the wrong \
         truth — use a dataset without per-query conditions"
    ))
}

/// One runnable query.
#[derive(Debug, Clone, PartialEq)]
pub struct HybridRrfQuery {
    /// Line of `tests.jsonl` (also the index of the query vector).
    pub query_id: usize,
    /// The BM25 arm's terms (`fulltext::query_terms`).
    pub terms: Vec<String>,
    /// The exact KNN top `W` (dataset neighbours, row ids).
    pub knn_truth: Vec<i64>,
}

/// The query set: every `tests.jsonl` line with at least one searchable term.
#[derive(Debug, Clone, PartialEq)]
pub struct QuerySet {
    pub queries: Vec<HybridRrfQuery>,
    /// Query ids with no term left after `fulltext::query_terms`; skipped on
    /// both engines (a hybrid query without a BM25 arm is not this workload).
    pub skipped: Vec<usize>,
    pub total: usize,
}

impl QuerySet {
    /// The fulltext view of the BM25 arm, for `fulltext::result_metadata_for`.
    pub fn text_view(&self) -> fulltext::QuerySet {
        fulltext::QuerySet {
            queries: self
                .queries
                .iter()
                .map(|q| fulltext::TextQuery {
                    compound: None,
                    expression: None,
                    query_id: q.query_id,
                    terms: q.terms.clone(),
                })
                .collect(),
            preparation_sha256: None,
            skipped: self.skipped.clone(),
            total: self.total,
        }
    }
}

/// Build the query set from `tests.jsonl`'s `query_text` (line order) and the
/// dataset's neighbours (same order). Errors: the two disagree on the query
/// count (not the same `tests.jsonl`), or a query's ground truth holds fewer
/// than `W` valid ids (the KNN arm of the reference would be truncated).
pub fn build_query_set(
    texts: &[String],
    neighbours: &[Vec<i64>],
    spec: &RrfSpec,
) -> Result<QuerySet, String> {
    if texts.len() != neighbours.len() {
        return Err(format!(
            "hybrid-rrf: tests.jsonl has {} query texts but the dataset returned {} ground-truth \
             rows — they must be the same queries",
            texts.len(),
            neighbours.len()
        ));
    }
    let text = fulltext::build_query_set(texts)?;
    let mut queries = Vec::with_capacity(text.queries.len());
    for q in text.queries {
        let mut seen = HashSet::new();
        let truth: Vec<i64> = neighbours[q.query_id]
            .iter()
            .copied()
            .filter(|&id| id >= 0 && seen.insert(id))
            .take(spec.window)
            .collect();
        if truth.len() < spec.window {
            return Err(format!(
                "hybrid-rrf: query {} has {} valid ground-truth neighbours, but `{SPEC_KEY}.window` \
                 is {}; the exact KNN arm of the reference needs W of them (lower the window or \
                 use a dataset with deeper ground truth)",
                q.query_id,
                truth.len(),
                spec.window
            ));
        }
        queries.push(HybridRrfQuery {
            query_id: q.query_id,
            terms: q.terms,
            knn_truth: truth,
        });
    }
    Ok(QuerySet {
        queries,
        skipped: text.skipped,
        total: text.total,
    })
}

/// Reciprocal Rank Fusion of ranked id lists: each id scores the sum over
/// the lists it appears in of `1 / (k + rank)`, rank 1-based (a repeated id
/// counts at its first rank). Sorted by score descending, ties by id
/// ascending — deterministic.
pub fn rrf_fuse(lists: &[&[i64]], rank_constant: u32) -> Vec<(i64, f64)> {
    let k = f64::from(rank_constant);
    let mut scores: HashMap<i64, f64> = HashMap::new();
    for list in lists {
        let mut seen = HashSet::new();
        let mut rank = 0usize;
        for &id in *list {
            if !seen.insert(id) {
                continue;
            }
            rank += 1;
            *scores.entry(id).or_insert(0.0) += 1.0 / (k + rank as f64);
        }
    }
    let mut fused: Vec<(i64, f64)> = scores.into_iter().collect();
    fused.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    fused
}

/// The client-side reference of one query: RRF over the exact candidates.
#[derive(Debug, Clone, PartialEq)]
pub struct Reference {
    /// Every candidate of either list, fused and ranked (see [`rrf_fuse`]).
    pub fused: Vec<(i64, f64)>,
    /// Exact BM25 candidates the engine returned (`<= W`).
    pub bm25_candidates: usize,
    /// Ids in both candidate lists.
    pub in_both: usize,
}

/// Build the reference from the engine's exact BM25 top `W` (row ids, rank
/// order) and the dataset's exact KNN top `W`. More than `W` BM25 ids, or a
/// duplicate, means the request was not what this module built: an error.
pub fn reference(bm25: &[i64], knn: &[i64], spec: &RrfSpec) -> Result<Reference, String> {
    if bm25.len() > spec.window {
        return Err(format!(
            "hybrid-rrf reference: {} BM25 candidates for a window of {}",
            bm25.len(),
            spec.window
        ));
    }
    let bm25_set: HashSet<i64> = bm25.iter().copied().collect();
    if bm25_set.len() != bm25.len() {
        return Err("hybrid-rrf reference: duplicate id in the BM25 candidates".into());
    }
    let knn = &knn[..knn.len().min(spec.window)];
    Ok(Reference {
        fused: rrf_fuse(&[bm25, knn], spec.rank_constant),
        bm25_candidates: bm25.len(),
        in_both: knn.iter().filter(|id| bm25_set.contains(id)).count(),
    })
}

/// One query's quality against its [`Reference`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QueryScore {
    /// Tie-aware recall@top (the published `recall`).
    pub recall: f64,
    /// Recall@top against the id-tie-broken reference list.
    pub strict_recall: f64,
    /// Tie-aware hits / ids returned.
    pub precision: f64,
}

/// Score one fused page (row ids, rank order) against its reference. See the
/// module docs: the denominator is `min(top, |reference|)`; a hit is a
/// returned id whose reference score is at least the reference score at that
/// position (within [`TIE_TOLERANCE`]).
pub fn score_query(returned: &[i64], reference: &Reference, top: usize) -> QueryScore {
    let expected = top.min(reference.fused.len());
    let mut seen = HashSet::new();
    let page: Vec<i64> = returned
        .iter()
        .copied()
        .filter(|id| seen.insert(*id))
        .take(top)
        .collect();
    if expected == 0 {
        let perfect = if page.is_empty() { 1.0 } else { 0.0 };
        return QueryScore {
            recall: perfect,
            strict_recall: perfect,
            precision: perfect,
        };
    }
    let cutoff = reference.fused[expected - 1].1;
    let floor = cutoff - cutoff.abs() * TIE_TOLERANCE;
    let ref_score: HashMap<i64, f64> = reference.fused.iter().copied().collect();
    let hits = page
        .iter()
        .filter(|id| ref_score.get(id).is_some_and(|s| *s >= floor))
        .count()
        .min(expected);
    let strict: HashSet<i64> = reference.fused[..expected]
        .iter()
        .map(|(id, _)| *id)
        .collect();
    let strict_hits = page.iter().filter(|id| strict.contains(id)).count();
    QueryScore {
        recall: hits as f64 / expected as f64,
        strict_recall: strict_hits as f64 / expected as f64,
        precision: if page.is_empty() {
            0.0
        } else {
            hits as f64 / page.len() as f64
        },
    }
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= b.abs() * TIE_TOLERANCE
}

/// Whether `score` is an RRF score at this `k` and `W`: `1/(k+a)` or
/// `1/(k+a) + 1/(k+b)` with `a, b` in `1..=W` (within [`TIE_TOLERANCE`]).
/// A range-and-lattice check: it refuses any non-RRF scoring (raw BM25,
/// linear, min-max — all outside the range or off the lattice), but it cannot
/// tell `k` from a nearby constant (`1/(60+a) = 1/(61+(a-1))`); the rank
/// constant itself is verified on the request (Redis) or the pipeline
/// read-back (OpenSearch). The lattice has ~W²/2 pair sums, so its power
/// falls as `W` grows: at W = 100, k = 60 the tolerance windows cover a few
/// percent of the range, at W ≈ 1000 they cover all of it and the check is
/// effectively range-only ([`lattice_coverage`] records which).
pub fn check_rrf_score(score: f64, spec: &RrfSpec) -> Result<(), String> {
    let k = f64::from(spec.rank_constant);
    let w = spec.window as f64;
    let max = 2.0 / (k + 1.0);
    let min = 1.0 / (k + w);
    let off = || {
        format!(
            "hybrid-rrf: returned score {score} is not a reciprocal-rank-fusion score at \
             rank_constant {} and window {} (range [{min}, {max}]) — the server did not fuse \
             with RRF as requested",
            spec.rank_constant, spec.window
        )
    };
    if !score.is_finite()
        || score > max * (1.0 + TIE_TOLERANCE)
        || score < min * (1.0 - TIE_TOLERANCE)
    {
        return Err(off());
    }
    for a in 1..=spec.window {
        let first = 1.0 / (k + a as f64);
        if close(score, first) {
            return Ok(());
        }
        let rest = score - first;
        if rest <= 0.0 {
            continue;
        }
        let b = (1.0 / rest - k).round();
        if (1.0..=w).contains(&b) && close(score, first + 1.0 / (k + b)) {
            return Ok(());
        }
    }
    Err(off())
}

/// Rough fraction of the RRF score range `[1/(k+W), 2/(k+1)]` covered by
/// the lattice's tolerance windows (single-list values plus pair sums,
/// overlaps ignored, capped at 1): how much [`check_rrf_score`] can refuse
/// beyond the range check at this `k` and `W`.
pub fn lattice_coverage(spec: &RrfSpec) -> f64 {
    let k = f64::from(spec.rank_constant);
    let w = spec.window;
    let (lo, hi) = (1.0 / (k + w as f64), 2.0 / (k + 1.0));
    let mut covered = 0.0;
    for a in 1..=w {
        let one = 1.0 / (k + a as f64);
        covered += 2.0 * TIE_TOLERANCE * one;
        for b in a..=w {
            covered += 2.0 * TIE_TOLERANCE * (one + 1.0 / (k + b as f64));
        }
    }
    (covered / (hi - lo)).min(1.0)
}

/// Check a whole fused page's scores: each an RRF score, non-increasing.
pub fn check_page_scores(scores: &[f64], spec: &RrfSpec) -> Result<(), String> {
    for (i, s) in scores.iter().enumerate() {
        check_rrf_score(*s, spec)?;
        if i > 0 && *s > scores[i - 1] * (1.0 + TIE_TOLERANCE) {
            return Err(format!(
                "hybrid-rrf: page is not ranked by descending fused score ({s} after {})",
                scores[i - 1]
            ));
        }
    }
    Ok(())
}

/// Check and score one fused page (`(row, fused score)` in rank order): every
/// score an RRF score at this `k`/`W`, non-increasing, at most `top` ids, no
/// duplicate; then [`score_query`] against the reference.
pub fn score_page(
    page: &[(i64, f64)],
    reference: &Reference,
    spec: &RrfSpec,
    top: usize,
) -> Result<QueryScore, String> {
    if page.len() > top {
        return Err(format!(
            "hybrid-rrf: {} results for a page of {top}",
            page.len()
        ));
    }
    let ids: Vec<i64> = page.iter().map(|(id, _)| *id).collect();
    if ids.iter().collect::<HashSet<_>>().len() != ids.len() {
        return Err("hybrid-rrf: duplicate id in a fused page".into());
    }
    let scores: Vec<f64> = page.iter().map(|(_, s)| *s).collect();
    check_page_scores(&scores, spec)?;
    Ok(score_query(&ids, reference, top))
}

/// The reference ids a `--dump-results` record is scored against: the top
/// `min(top, |reference|)` of the fused reference.
pub fn reference_top_ids(reference: &Reference, top: usize) -> Vec<i64> {
    reference
        .fused
        .iter()
        .take(top)
        .map(|(id, _)| *id)
        .collect()
}

// ── Redis (FT.HYBRID, RediSearch 8.4+) ────────────────────────────────────

/// Field the vector is indexed under on Redis (the engine's schema).
pub const REDIS_VECTOR_FIELD: &str = "vector";

/// Name of the query parameter that carries the vector blob.
pub const REDIS_VECTOR_PARAM: &str = "BLOB";

/// The BM25 arm's RediSearch query: `fulltext::redis_query`, with every term
/// quoted in parity mode. `FT.HYBRID`'s SEARCH subquery accepts only `SCORER`
/// and `YIELD_SCORE_AS` (no `VERBATIM`), and a quoted term is a verbatim
/// (never stem-expanded) node in DIALECT 2 — the same effect. The reference's
/// `FT.SEARCH` uses this exact string too.
pub fn redis_search_query(fields: &[FieldBoost], terms: &[String], analysis: Analysis) -> String {
    match analysis {
        Analysis::English => fulltext::redis_query(fields, terms),
        Analysis::Parity => {
            let quoted: Vec<String> = terms.iter().map(|t| format!("\"{t}\"")).collect();
            fulltext::redis_query(fields, &quoted)
        }
    }
}

/// Every `FT.HYBRID` argument after the command name, up to (not including)
/// the vector blob, which the caller appends as the value of
/// [`REDIS_VECTOR_PARAM`]:
///
/// `<idx> SEARCH <query> SCORER BM25STD VSIM @vector $BLOB KNN 4 K <W>
/// EF_RUNTIME <ef> COMBINE RRF 4 CONSTANT <k> WINDOW <W> LIMIT 0 <top>
/// TIMEOUT <ms> PARAMS 2 BLOB`
pub fn redis_hybrid_args(
    index: &str,
    query: &str,
    ef: i64,
    spec: &RrfSpec,
    top: usize,
    timeout_ms: i64,
) -> Vec<String> {
    let w = spec.window.to_string();
    [
        index,
        "SEARCH",
        query,
        "SCORER",
        fulltext::REDIS_SCORER,
        "VSIM",
        &format!("@{REDIS_VECTOR_FIELD}"),
        &format!("${REDIS_VECTOR_PARAM}"),
        "KNN",
        "4",
        "K",
        &w,
        "EF_RUNTIME",
        &ef.to_string(),
        "COMBINE",
        "RRF",
        "4",
        "CONSTANT",
        &spec.rank_constant.to_string(),
        "WINDOW",
        &w,
        "LIMIT",
        "0",
        &top.to_string(),
        "TIMEOUT",
        &timeout_ms.to_string(),
        "PARAMS",
        "2",
        REDIS_VECTOR_PARAM,
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

/// The reference's exact BM25 request: `FT.SEARCH` with the same query string
/// and scorer, `LIMIT 0 <W>` (parity adds `VERBATIM`, redundant with the
/// quoted terms).
pub fn redis_reference_args(
    index: &str,
    query: &str,
    spec: &RrfSpec,
    timeout_ms: i64,
    analysis: Analysis,
) -> Vec<String> {
    fulltext::redis_args_for(index, query, spec.window, timeout_ms, analysis)
}

/// A parsed fused page: `(engine id, fused score)` in rank order.
pub type FusedPage = Vec<(String, f64)>;

/// Parse an `FT.HYBRID` reply (RESP3 map, or the RESP2 flat array):
/// `total_results`, `results` (each with `__key` and `__score`), `warnings`,
/// `execution_time`. Any warning (a subquery timeout, OOM, …) is an error:
/// the page may be partial. At most `top` results, a complete page
/// (`min(total_results, top)`), no duplicate key.
pub fn parse_redis_hybrid_reply(value: &redis::Value, top: usize) -> Result<FusedPage, String> {
    use redis::Value as V;
    fn text(v: &V) -> Option<String> {
        match v {
            V::BulkString(b) => std::str::from_utf8(b).ok().map(str::to_string),
            V::SimpleString(s) => Some(s.clone()),
            V::VerbatimString { text, .. } => Some(text.clone()),
            _ => None,
        }
    }
    fn pairs(v: &V) -> Option<Vec<(String, &V)>> {
        match v {
            V::Map(m) => m.iter().map(|(k, v)| text(k).map(|k| (k, v))).collect(),
            V::Array(a) if a.len() % 2 == 0 => a
                .chunks(2)
                .map(|kv| text(&kv[0]).map(|k| (k, &kv[1])))
                .collect(),
            _ => None,
        }
    }
    fn get<'a>(p: &'a [(String, &'a V)], name: &str) -> Option<&'a V> {
        p.iter().find_map(|(k, v)| (k == name).then_some(*v))
    }
    let top_level = pairs(value).ok_or("hybrid-rrf: FT.HYBRID reply is not a map")?;
    match get(&top_level, "warnings") {
        Some(V::Array(w)) if w.is_empty() => {}
        Some(V::Array(w)) => {
            let msgs: Vec<String> = w.iter().filter_map(text).collect();
            return Err(format!("hybrid-rrf: FT.HYBRID warning(s): {msgs:?}"));
        }
        _ => return Err("hybrid-rrf: FT.HYBRID reply without `warnings`".into()),
    }
    let total = match get(&top_level, "total_results") {
        Some(V::Int(n)) if *n >= 0 => *n as usize,
        _ => return Err("hybrid-rrf: FT.HYBRID reply without `total_results`".into()),
    };
    let Some(V::Array(results)) = get(&top_level, "results") else {
        return Err("hybrid-rrf: FT.HYBRID reply without `results`".into());
    };
    if results.len() > top || results.len() != total.min(top) {
        return Err(format!(
            "hybrid-rrf: incomplete FT.HYBRID page: total_results={total}, returned={}, \
             limit={top}",
            results.len()
        ));
    }
    let mut seen = HashSet::new();
    let mut page = Vec::with_capacity(results.len());
    for r in results {
        let fields = pairs(r).ok_or("hybrid-rrf: FT.HYBRID result is not a map")?;
        let key = get(&fields, "__key")
            .and_then(text)
            .ok_or("hybrid-rrf: FT.HYBRID result without `__key`")?;
        let score = match get(&fields, "__score") {
            Some(V::Double(d)) => *d,
            Some(v) => text(v)
                .and_then(|s| s.parse::<f64>().ok())
                .ok_or("hybrid-rrf: unparseable FT.HYBRID `__score`")?,
            None => return Err("hybrid-rrf: FT.HYBRID result without `__score`".into()),
        };
        if !seen.insert(key.clone()) {
            return Err(format!("hybrid-rrf: duplicate FT.HYBRID result {key}"));
        }
        page.push((key, score));
    }
    Ok(page)
}

/// `cmdstat_<command>` `calls=` from an `INFO commandstats` reply
/// (case-insensitive). `None` when the command has no line yet.
pub fn commandstats_calls(info: &str, command: &str) -> Option<u64> {
    let want = format!("cmdstat_{}:", command.to_ascii_lowercase());
    info.lines().find_map(|line| {
        let line = line.trim();
        if !line.to_ascii_lowercase().starts_with(&want) {
            return None;
        }
        line[want.len()..]
            .split(',')
            .find_map(|kv| kv.strip_prefix("calls="))
            .and_then(|n| n.parse().ok())
    })
}

/// The server executed at least `expected` `FT.HYBRID` calls during the run
/// (`calls` delta of `INFO commandstats`, read after the untimed probe and
/// before the workers prime, so the delta also counts one prime call per
/// worker). `None` on EITHER side — the server would not say (managed
/// services may hide it) or a read failed — is recorded as unverified: a
/// one-sided read would compare the server's lifetime count.
pub fn check_redis_calls(
    before: Option<u64>,
    after: Option<u64>,
    expected: usize,
) -> Result<Value, String> {
    match (before, after) {
        (Some(b), Some(a)) => {
            let delta = a.saturating_sub(b);
            if delta < expected as u64 {
                return Err(format!(
                    "hybrid-rrf: INFO commandstats counts {delta} FT.HYBRID calls during a run \
                     that sent {expected} — the server did not run them as FT.HYBRID"
                ));
            }
            Ok(json!({"ft_hybrid_calls_delta": delta, "verified": true,
                      "note": "includes one untimed prime call per worker"}))
        }
        _ => Ok(json!({"ft_hybrid_calls_delta": null, "verified": false,
                       "note": "INFO commandstats did not report FT.HYBRID before and \
                                after the run"})),
    }
}

// ── OpenSearch (hybrid query + score-ranker-processor, 2.19+) ─────────────

/// Server-side RRF needs the `score-ranker-processor` and `pagination_depth`,
/// both OpenSearch 2.19+ (neural-search plugin).
pub fn check_opensearch_version(version: &str) -> Result<(), String> {
    let mut pieces = version.split('.');
    let major = pieces.next().and_then(|v| v.parse::<u64>().ok());
    let minor = pieces.next().and_then(|v| v.parse::<u64>().ok());
    match (major, minor) {
        (Some(major), Some(minor)) if major > 2 || (major == 2 && minor >= 19) => Ok(()),
        _ => Err(format!(
            "hybrid-rrf needs OpenSearch 2.19+ (score-ranker-processor with rrf, hybrid \
             pagination_depth); the server reports {version:?} — there is no client-side \
             fusion fallback"
        )),
    }
}

/// Processor type of OpenSearch's RRF search-phase-results processor.
pub const OPENSEARCH_PROCESSOR: &str = "score-ranker-processor";

/// Per-config search pipeline id: `vdbb-hybrid-rrf-<config>-<hash>-k<k>`,
/// the config name lowercased with anything outside `[a-z0-9_-]` mapped to
/// `_`, plus a 32-bit FNV-1a hash of the exact name.
/// Per engine config so two different configs on one cluster never fuse
/// through each other's pipeline; per `k` so search configs of one engine
/// config with different constants do not either. Two processes running the
/// SAME config against one cluster share the id (identical bodies): the
/// first to finish deletes it and the other fails loudly on the missing
/// pipeline, never silently.
pub fn opensearch_pipeline_name(config_name: &str, spec: &RrfSpec) -> String {
    let token: String = config_name
        .chars()
        .map(|c| {
            let c = c.to_ascii_lowercase();
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    // The mapping above is lossy (`a.b` and `a_b`, `A` and `a`), so the id
    // also carries an FNV-1a hash of the exact name: distinct configs never
    // share a pipeline.
    let hash = config_name.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    });
    format!(
        "vdbb-hybrid-rrf-{token}-{:08x}-k{}",
        hash as u32, spec.rank_constant
    )
}

/// The search pipeline body.
pub fn opensearch_pipeline_body(spec: &RrfSpec) -> Value {
    json!({
        "description": format!(
            "vector-db-benchmark hybrid-rrf ({SEMANTICS}): reciprocal rank fusion, rank_constant {}",
            spec.rank_constant
        ),
        "phase_results_processors": [{
            OPENSEARCH_PROCESSOR: {
                "combination": {"technique": "rrf", "rank_constant": spec.rank_constant}
            }
        }]
    })
}

/// Check the pipeline as the server reports it (`GET _search/pipeline/<id>`):
/// exactly one phase-results processor, `score-ranker-processor` combining
/// with `rrf` at this `rank_constant`, and no request/response processors.
pub fn check_opensearch_pipeline(
    read_back: &Value,
    name: &str,
    spec: &RrfSpec,
) -> Result<(), String> {
    let p = read_back
        .get(name)
        .ok_or_else(|| format!("hybrid-rrf: search pipeline {name} not found on read-back"))?;
    let bad = |why: &str| {
        Err(format!(
            "hybrid-rrf: search pipeline {name} is not the one this run needs ({why}): {p}"
        ))
    };
    for other in ["request_processors", "response_processors"] {
        if p.get(other)
            .and_then(Value::as_array)
            .is_some_and(|a| !a.is_empty())
        {
            return bad(&format!("it has {other}"));
        }
    }
    let Some([proc_]) = p["phase_results_processors"].as_array().map(Vec::as_slice) else {
        return bad("not exactly one phase_results_processor");
    };
    let combination = &proc_[OPENSEARCH_PROCESSOR]["combination"];
    if combination["technique"] != "rrf" {
        return bad("combination technique is not rrf");
    }
    let k = &combination["rank_constant"];
    let k = k
        .as_u64()
        .or_else(|| k.as_str().and_then(|s| s.parse().ok()));
    if k != Some(u64::from(spec.rank_constant)) {
        return bad(&format!("rank_constant is not {}", spec.rank_constant));
    }
    Ok(())
}

/// The hybrid search body: `size: top`, a `hybrid` query whose subqueries are
/// the fulltext workload's `multi_match` and a lucene `knn` with `k: W` and
/// `method_parameters.ef_search`, `pagination_depth: W` (each subquery keeps
/// its top W per shard). Sent with `?search_pipeline=<id>`.
pub fn opensearch_hybrid_body(
    fields: &[FieldBoost],
    terms: &[String],
    vector: &[f32],
    ef: i64,
    spec: &RrfSpec,
    top: usize,
) -> Value {
    let text = fulltext::opensearch_body(fields, terms, spec.window)["query"].clone();
    json!({
        "size": top,
        "_source": false,
        "track_total_hits": false,
        "query": {"hybrid": {
            "pagination_depth": spec.window,
            "queries": [
                text,
                {"knn": {"vector": {
                    "vector": vector,
                    "k": spec.window,
                    "method_parameters": {"ef_search": ef},
                }}},
            ],
        }},
    })
}

/// The reference's exact BM25 request: the fulltext workload's body at
/// `size: W`.
pub fn opensearch_reference_body(fields: &[FieldBoost], terms: &[String], spec: &RrfSpec) -> Value {
    fulltext::opensearch_body(fields, terms, spec.window)
}

// ── Result ────────────────────────────────────────────────────────────────

/// Aggregates over the reference of every eligible query.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ReferenceStats {
    pub queries: usize,
    pub mean_bm25_candidates: f64,
    pub mean_in_both: f64,
    /// Queries with fewer than `W` BM25 candidates (rare terms).
    pub queries_short_bm25: usize,
}

impl ReferenceStats {
    pub fn of<'a>(refs: impl IntoIterator<Item = &'a Reference>, window: usize) -> Self {
        let mut s = ReferenceStats::default();
        let (mut bm25, mut both) = (0usize, 0usize);
        for r in refs {
            s.queries += 1;
            bm25 += r.bm25_candidates;
            both += r.in_both;
            if r.bm25_candidates < window {
                s.queries_short_bm25 += 1;
            }
        }
        if s.queries > 0 {
            s.mean_bm25_candidates = bm25 as f64 / s.queries as f64;
            s.mean_in_both = both as f64 / s.queries as f64;
        }
        s
    }
}

/// The `hybrid_rrf` block of a result JSON. `text_meta` is the BM25 arm's
/// `fulltext::result_metadata_for` (fields, analyzers, stopwords, known
/// differences, index marker), `engine` the arm's own record (command or
/// pipeline read-back, server version, verification).
#[allow(clippy::too_many_arguments)]
pub fn result_metadata(
    spec: &RrfSpec,
    top: usize,
    ef: i64,
    text_meta: Value,
    engine: Value,
    stats: &ReferenceStats,
    mean_strict_recall: f64,
) -> Value {
    json!({
        "semantics": SEMANTICS,
        "fusion": "server-side reciprocal rank fusion (no client-side fallback)",
        "formula": "score(d) = sum over lists containing d of 1 / (rank_constant + rank_in_list), rank 1-based",
        "rank_constant": spec.rank_constant,
        "window": spec.window,
        "result_limit": top,
        "candidates": {
            "bm25": "top `window` of the fulltext workload's ranked BM25 query (bm25 block)",
            "knn": format!("top `window` of HNSW KNN on the vector field, ef = {ef}"),
        },
        "ef": ef,
        "bm25": text_meta,
        "reference": {
            "bm25": "the engine's exact BM25 top `window`: the same ranked query, sent untimed \
                     before the run (an inverted-index scan; BM25 scores are engine-defined)",
            "knn": "the dataset's brute-forced neighbours, top `window` (engine independent)",
            "fusion": "client-side RRF at the same rank_constant and window; ties broken by row id",
            "queries": stats.queries,
            "mean_bm25_candidates": stats.mean_bm25_candidates,
            "queries_with_fewer_than_window_bm25_candidates": stats.queries_short_bm25,
            "mean_ids_in_both_lists": stats.mean_in_both,
        },
        "quality": {
            "mean_recall": "tie-aware recall@top: a returned id is a hit when its reference RRF \
                            score is >= the reference score at position top (relative tolerance \
                            below); denominator min(top, |reference|)",
            "tie_tolerance_relative": TIE_TOLERANCE,
            "mean_strict_recall": mean_strict_recall,
            "strict_recall": "recall@top against the id-tie-broken reference list",
            "score_check": "every returned score is an RRF score at this rank_constant and \
                            window (range and 1/(k+a) [+ 1/(k+b)] lattice) and the page is \
                            non-increasing",
            "score_check_lattice_coverage": lattice_coverage(spec),
        },
        "engine": engine,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workload;

    fn spec(window: usize, k: u32) -> RrfSpec {
        RrfSpec {
            window,
            rank_constant: k,
        }
    }

    #[test]
    fn spec_requires_both_keys_and_bounds() {
        assert_eq!(
            parse_spec(Some(&json!({"window": 100, "rank_constant": 60}))).unwrap(),
            spec(100, 60)
        );
        assert!(parse_spec(None).is_err());
        assert!(parse_spec(Some(&json!({"window": 100}))).is_err());
        assert!(parse_spec(Some(&json!({"rank_constant": 60}))).is_err());
        assert!(parse_spec(Some(&json!({"window": 100, "rank_constant": 60, "x": 1}))).is_err());
        for bad in [
            json!({"window": 0, "rank_constant": 60}),
            json!({"window": 10001, "rank_constant": 60}),
            json!({"window": 10, "rank_constant": 0}),
            json!({"window": 10, "rank_constant": 10001}),
            json!({"window": 10, "rank_constant": 60.5}),
            json!({"window": 10, "rank_constant": -1}),
        ] {
            assert!(parse_spec(Some(&bad)).is_err(), "{bad}");
        }
    }

    #[test]
    fn rrf_block_only_on_hybrid_rrf() {
        let b = json!({"window": 10, "rank_constant": 60});
        assert!(validate_declared(Workload::HybridRrf, Some(&b)).is_ok());
        assert!(validate_declared(Workload::HybridRrf, None).is_err());
        for w in [
            Workload::Vector,
            Workload::Hybrid,
            Workload::Fulltext,
            Workload::FilterOnly,
        ] {
            assert!(validate_declared(w, Some(&b)).is_err(), "{w}");
            assert!(validate_declared(w, None).is_ok(), "{w}");
        }
    }

    #[test]
    fn text_arm_refuses_compound_and_vectorless_layouts() {
        assert!(parse_text_spec(None).is_ok());
        assert!(parse_text_spec(Some(&json!({"fields": {"title": 2.0}}))).is_ok());
        assert!(
            parse_text_spec(Some(&json!({"analysis": "parity", "fields": {"t": 1.0}}))).is_ok()
        );
        assert!(parse_text_spec(Some(&json!({"no_vector_index": true}))).is_err());
        let e = parse_text_spec(Some(&json!({"allow_multi_shard": true}))).unwrap_err();
        assert!(e.contains("one shard"), "{e}");
        assert!(parse_text_spec(Some(&json!({"fields": {"title": 1.0},
            "compound": {"suite": "msmarco-v1", "family": "segment"}})))
        .is_err());
    }

    #[test]
    fn params_require_top_and_ef_within_the_window() {
        let s = spec(100, 60);
        assert_eq!(
            check_params(Some(10), Some(100), &s, None, None).unwrap(),
            10
        );
        assert_eq!(
            check_params(Some(100), Some(500), &s, None, None).unwrap(),
            100
        );
        // top: required, >= 1, <= window
        assert!(check_params(None, Some(100), &s, None, None).is_err());
        assert!(check_params(Some(0), Some(100), &s, None, None).is_err());
        assert!(check_params(Some(101), Some(200), &s, None, None).is_err());
        // ef: required, >= window
        assert!(check_params(Some(10), None, &s, None, None).is_err());
        assert!(check_params(Some(10), Some(99), &s, None, None).is_err());
        assert!(check_params(Some(10), Some(i32::MAX as i64 + 1), &s, None, None).is_err());
        // query-count scheduling only
        assert!(check_params(Some(10), Some(100), &s, Some(50.0), None).is_err());
        assert!(check_params(Some(10), Some(100), &s, None, Some(5.0)).is_err());
    }

    #[test]
    fn redis_cluster_api_is_refused_for_hybrid_rrf_only() {
        let e = check_redis_deployment(true, &[Workload::Vector, Workload::HybridRrf]).unwrap_err();
        assert!(e.contains("Cluster API"), "{e}");
        assert!(check_redis_deployment(false, &[Workload::HybridRrf]).is_ok());
        assert!(check_redis_deployment(true, &[Workload::Vector, Workload::Fulltext]).is_ok());
    }

    #[test]
    fn fingerprint_is_exclusive_to_the_workload() {
        assert!(verify_fingerprint(Workload::HybridRrf, true).is_ok());
        assert!(verify_fingerprint(Workload::HybridRrf, false).is_err());
        for w in Workload::ALL {
            if w != Workload::HybridRrf {
                assert!(verify_fingerprint(w, true).is_err(), "{w}");
                assert!(verify_fingerprint(w, false).is_ok(), "{w}");
            }
        }
    }

    #[test]
    fn declared_conditions_are_refused_through_the_resolver() {
        let refuse = |c: Option<Value>| {
            crate::query_filter::resolve("x", 0, c.as_ref(), refuse_conditions).map(|_| ())
        };
        assert!(refuse(None).is_ok());
        assert!(refuse(Some(Value::Null)).is_ok());
        assert!(refuse(Some(json!({}))).is_ok());
        let e = refuse(Some(
            json!({"and": [{"color": {"match": {"value": "red"}}}]}),
        ))
        .unwrap_err();
        assert!(e.contains("no filter clause"), "{e}");
    }

    #[test]
    fn rrf_fuse_is_one_based_sums_and_breaks_ties_by_id() {
        let fused = rrf_fuse(&[&[7, 3, 5], &[3, 9]], 60);
        let s = |r: f64| 1.0 / (60.0 + r);
        // 3: rank 2 + rank 1; 7: rank 1; 9: rank 2 (ties 5? no: 5 is rank 3)
        assert_eq!(fused[0], (3, s(2.0) + s(1.0)));
        assert_eq!(fused[1], (7, s(1.0)));
        assert_eq!(fused[2], (9, s(2.0)));
        assert_eq!(fused[3], (5, s(3.0)));
        // Exact tie (BM25-only rank 1 vs KNN-only rank 1): lower id first.
        let tie = rrf_fuse(&[&[42], &[17]], 60);
        assert_eq!(tie, vec![(17, s(1.0)), (42, s(1.0))]);
        // A repeated id counts once, at its first rank; later ids keep their
        // distinct ranks.
        assert_eq!(rrf_fuse(&[&[1, 1, 2]], 60), vec![(1, s(1.0)), (2, s(2.0))]);
        // The constant matters.
        assert_eq!(rrf_fuse(&[&[1]], 20), vec![(1, 1.0 / 21.0)]);
    }

    #[test]
    fn query_set_pairs_texts_with_window_deep_truth() {
        let texts: Vec<String> = ["apple pie", "the of", "zebra"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let gt = vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, -1, 8, 9]];
        let set = build_query_set(&texts, &gt, &spec(3, 60)).unwrap();
        assert_eq!(set.total, 3);
        assert_eq!(set.skipped, vec![1]);
        assert_eq!(set.queries.len(), 2);
        assert_eq!(set.queries[0].terms, vec!["apple", "pie"]);
        assert_eq!(set.queries[0].knn_truth, vec![1, 2, 3]);
        // -1 padding is not a neighbour; the window takes the first W valid.
        assert_eq!(set.queries[1].query_id, 2);
        assert_eq!(set.queries[1].knn_truth, vec![7, 8, 9]);
        // Truth narrower than the window: refused, never truncated.
        let e = build_query_set(&texts, &gt, &spec(4, 60)).unwrap_err();
        assert!(e.contains("window"), "{e}");
        // Different query counts: not the same tests.jsonl.
        assert!(build_query_set(&texts, &gt[..2], &spec(3, 60)).is_err());
        // The fulltext view keeps ids, terms and skips.
        let view = set.text_view();
        assert_eq!(view.skipped, vec![1]);
        assert_eq!(view.queries[1].query_id, 2);
    }

    #[test]
    fn reference_fuses_exact_lists_and_refuses_overlong_bm25() {
        let s = spec(3, 60);
        let r = reference(&[10, 11], &[11, 12, 13, 14], &s).unwrap();
        assert_eq!(r.bm25_candidates, 2);
        assert_eq!(r.in_both, 1);
        // KNN truncated to W: 14 is not a candidate.
        assert!(r.fused.iter().all(|(id, _)| *id != 14));
        assert_eq!(r.fused[0].0, 11);
        assert!(reference(&[1, 2, 3, 4], &[1, 2, 3], &s).is_err());
        assert!(reference(&[1, 1], &[1, 2, 3], &s).is_err());
    }

    #[test]
    fn score_is_tie_aware_at_the_cutoff_and_strict_otherwise() {
        let s = spec(10, 60);
        // BM25 [1,2,3], KNN [4,5,6]: 1 and 4 tie at rank 1, 2 and 5 at 2, ...
        let r = reference(&[1, 2, 3], &[4, 5, 6], &s).unwrap();
        // Reference (ties by id): 1,4,2,5,3,6. top=3 cuts inside the (2,5) tie.
        assert_eq!(
            r.fused.iter().map(|x| x.0).collect::<Vec<_>>(),
            vec![1, 4, 2, 5, 3, 6]
        );
        // An engine that broke the tie the other way is perfect, tie-aware...
        let other = score_query(&[4, 1, 5], &r, 3);
        assert_eq!(other.recall, 1.0);
        assert_eq!(other.precision, 1.0);
        // ...and loses one strict hit.
        assert!((other.strict_recall - 2.0 / 3.0).abs() < 1e-12);
        // A document below the cut-off group is a miss in both.
        let miss = score_query(&[1, 4, 3], &r, 3);
        assert!((miss.recall - 2.0 / 3.0).abs() < 1e-12);
        assert!((miss.precision - 2.0 / 3.0).abs() < 1e-12);
        // Unknown ids, duplicates and a short page.
        let junk = score_query(&[99, 1, 1], &r, 3);
        assert!((junk.recall - 1.0 / 3.0).abs() < 1e-12);
        assert!((junk.precision - 0.5).abs() < 1e-12);
        assert_eq!(score_query(&[], &r, 3).recall, 0.0);
        // Hits are capped at the denominator: a whole tie group returned in
        // a larger page cannot push recall above 1.
        let r2 = reference(&[1], &[2], &s).unwrap();
        assert_eq!(score_query(&[1, 2], &r2, 1).recall, 1.0);
        // Denominator: min(top, |reference|).
        assert_eq!(score_query(&[1, 4, 2, 5, 3, 6], &r, 10).recall, 1.0);
    }

    #[test]
    fn rrf_score_check_accepts_the_lattice_and_refuses_other_scorers() {
        let s = spec(100, 60);
        let one = |r: f64| 1.0 / (60.0 + r);
        for ok in [
            one(1.0) * 2.0,
            one(1.0),
            one(100.0),
            one(3.0) + one(77.0),
            one(100.0) + one(99.0),
            // float32 rounding (OpenSearch)
            (one(5.0) as f32 + one(9.0) as f32) as f64,
        ] {
            check_rrf_score(ok, &s).unwrap_or_else(|e| panic!("{ok}: {e}"));
        }
        for bad in [
            12.7,  // raw BM25
            0.5,   // min-max normalized / linear
            1.0,   //
            0.0,   //
            -0.01, //
            f64::NAN,
            one(101.0) * 0.99, // below the range
            one(1.0) * 2.0 * 1.01,
        ] {
            assert!(check_rrf_score(bad, &s).is_err(), "{bad}");
        }
        // Off the lattice inside the range: the midpoint between two adjacent
        // single-list values, at a window too small to reach it by a pair.
        let small = spec(2, 60);
        let mid = (one(1.0) + one(2.0)) / 2.0;
        assert!(check_rrf_score(mid, &small).is_err());
        // A different constant's top score is out of range.
        assert!(check_rrf_score(2.0 / 21.0, &s).is_err());
        // Coverage: small at small W, saturating at large W.
        assert!(lattice_coverage(&small) < 0.01);
        assert!(lattice_coverage(&s) < 0.2, "{}", lattice_coverage(&s));
        assert_eq!(lattice_coverage(&spec(5000, 60)), 1.0);
        // Page order.
        assert!(check_page_scores(&[one(1.0) * 2.0, one(1.0), one(2.0)], &s).is_ok());
        assert!(check_page_scores(&[one(2.0), one(1.0)], &s).is_err());
    }

    #[test]
    fn score_page_checks_shape_before_scoring() {
        let s = spec(10, 60);
        let one = |r: f64| 1.0 / (60.0 + r);
        let r = reference(&[1, 2], &[1, 3], &s).unwrap();
        let ok = score_page(&[(1, 2.0 * one(1.0)), (2, one(2.0))], &r, &s, 2).unwrap();
        assert_eq!(ok.recall, 1.0);
        // Raw BM25 scores: refused, not scored.
        assert!(score_page(&[(1, 7.5), (2, 3.1)], &r, &s, 2).is_err());
        // Ascending: refused.
        assert!(score_page(&[(2, one(2.0)), (1, 2.0 * one(1.0))], &r, &s, 2).is_err());
        // Overlong and duplicate pages.
        assert!(score_page(&[(1, one(1.0)), (2, one(2.0)), (3, one(2.0))], &r, &s, 2).is_err());
        assert!(score_page(&[(1, one(1.0)), (1, one(1.0))], &r, &s, 2).is_err());
        assert_eq!(reference_top_ids(&r, 2), vec![1, 2]);
    }

    #[test]
    fn parity_quotes_every_term_and_english_is_the_fulltext_query() {
        let f = vec![
            FieldBoost {
                field: "title".into(),
                boost: 5.0,
            },
            FieldBoost {
                field: "segment".into(),
                boost: 1.0,
            },
        ];
        let terms = vec!["zebra".to_string(), "crossings".to_string()];
        assert_eq!(
            redis_search_query(&f, &terms, Analysis::English),
            fulltext::redis_query(&f, &terms)
        );
        assert_eq!(
            redis_search_query(&f, &terms, Analysis::Parity),
            "(@title:(\"zebra\"|\"crossings\")) => {$weight: 5.0;} | \
             (@segment:(\"zebra\"|\"crossings\")) => {$weight: 1.0;}"
        );
    }

    #[test]
    fn redis_command_carries_every_semantic_knob() {
        let args = redis_hybrid_args("idx:x", "(@t:(a))", 200, &spec(100, 60), 10, 5000);
        assert_eq!(
            args.join(" "),
            "idx:x SEARCH (@t:(a)) SCORER BM25STD VSIM @vector $BLOB KNN 4 K 100 EF_RUNTIME \
             200 COMBINE RRF 4 CONSTANT 60 WINDOW 100 LIMIT 0 10 TIMEOUT 5000 PARAMS 2 BLOB"
        );
        let r = redis_reference_args("idx:x", "q", &spec(100, 60), 5000, Analysis::English);
        assert_eq!(r[..2], ["idx:x".to_string(), "q".to_string()]);
        assert!(r.windows(3).any(|w| w == ["LIMIT", "0", "100"]), "{r:?}");
        assert!(r.windows(2).any(|w| w == ["SCORER", "BM25STD"]), "{r:?}");
        let p = redis_reference_args("i", "q", &spec(100, 60), 5000, Analysis::Parity);
        assert!(p.contains(&"VERBATIM".to_string()));
    }

    fn bulk(s: &str) -> redis::Value {
        redis::Value::BulkString(s.as_bytes().to_vec())
    }

    fn resp3_reply(results: Vec<(&str, &str)>, total: i64, warnings: Vec<&str>) -> redis::Value {
        use redis::Value as V;
        V::Map(vec![
            (bulk("total_results"), V::Int(total)),
            (
                bulk("results"),
                V::Array(
                    results
                        .into_iter()
                        .map(|(k, s)| {
                            V::Map(vec![(bulk("__key"), bulk(k)), (bulk("__score"), bulk(s))])
                        })
                        .collect(),
                ),
            ),
            (
                bulk("warnings"),
                V::Array(
                    warnings
                        .into_iter()
                        .map(|w| V::SimpleString(w.into()))
                        .collect(),
                ),
            ),
            (bulk("execution_time"), V::Double(0.3)),
        ])
    }

    #[test]
    fn redis_reply_parses_resp3_and_resp2_and_refuses_partial_pages() {
        use redis::Value as V;
        let ok = resp3_reply(vec![("p:1", "0.0327"), ("p:2", "0.0164")], 7, vec![]);
        assert_eq!(
            parse_redis_hybrid_reply(&ok, 2).unwrap(),
            vec![("p:1".to_string(), 0.0327), ("p:2".to_string(), 0.0164)]
        );
        // RESP2: flat key/value arrays.
        let resp2 = V::Array(vec![
            bulk("total_results"),
            V::Int(1),
            bulk("results"),
            V::Array(vec![V::Array(vec![
                bulk("__key"),
                bulk("p:9"),
                bulk("__score"),
                bulk("0.016"),
            ])]),
            bulk("warnings"),
            V::Array(vec![]),
            bulk("execution_time"),
            bulk("0.1"),
        ]);
        assert_eq!(
            parse_redis_hybrid_reply(&resp2, 10).unwrap(),
            vec![("p:9".to_string(), 0.016)]
        );
        // Any warning (e.g. a subquery timeout) fails the query.
        let warned = resp3_reply(
            vec![("p:1", "0.03")],
            1,
            vec!["Timeout limit was reached (VSIM)"],
        );
        assert!(parse_redis_hybrid_reply(&warned, 10)
            .unwrap_err()
            .contains("Timeout"));
        // Incomplete page: 7 total, limit 2, only 1 returned.
        let short = resp3_reply(vec![("p:1", "0.03")], 7, vec![]);
        assert!(parse_redis_hybrid_reply(&short, 2).is_err());
        // More than the limit.
        let long = resp3_reply(vec![("a", "0.03"), ("b", "0.02")], 2, vec![]);
        assert!(parse_redis_hybrid_reply(&long, 1).is_err());
        // Duplicate key, missing score, missing warnings.
        let dup = resp3_reply(vec![("a", "0.03"), ("a", "0.02")], 2, vec![]);
        assert!(parse_redis_hybrid_reply(&dup, 2).is_err());
        let no_score = V::Map(vec![
            (bulk("total_results"), V::Int(1)),
            (
                bulk("results"),
                V::Array(vec![V::Map(vec![(bulk("__key"), bulk("a"))])]),
            ),
            (bulk("warnings"), V::Array(vec![])),
        ]);
        assert!(parse_redis_hybrid_reply(&no_score, 1)
            .unwrap_err()
            .contains("__score"));
        let no_warnings = V::Map(vec![
            (bulk("total_results"), V::Int(0)),
            (bulk("results"), V::Array(vec![])),
        ]);
        assert!(parse_redis_hybrid_reply(&no_warnings, 1).is_err());
        assert!(parse_redis_hybrid_reply(&V::Int(3), 1).is_err());
    }

    #[test]
    fn commandstats_calls_delta_must_cover_the_run() {
        let info = "# Commandstats\r\ncmdstat_ft.search:calls=12,usec=5,usec_per_call=0.4,\
                    rejected_calls=0,failed_calls=0\r\ncmdstat_FT.HYBRID:calls=40,usec=9,\
                    usec_per_call=0.2,rejected_calls=0,failed_calls=0\r\n";
        assert_eq!(commandstats_calls(info, "FT.HYBRID"), Some(40));
        assert_eq!(commandstats_calls(info, "ft.search"), Some(12));
        assert_eq!(commandstats_calls(info, "FT.AGGREGATE"), None);
        assert!(check_redis_calls(Some(10), Some(40), 30).unwrap()["verified"] == true);
        // One-sided reads never verify (else: the lifetime count passes).
        assert!(check_redis_calls(None, Some(30), 30).unwrap()["verified"] == false);
        assert!(check_redis_calls(None, Some(1_000_000), 30).unwrap()["verified"] == false);
        assert!(check_redis_calls(Some(10), Some(39), 30).is_err());
        assert!(check_redis_calls(Some(10), None, 30).unwrap()["verified"] == false);
    }

    #[test]
    fn opensearch_version_gate_is_2_19() {
        for v in ["2.19.0", "2.19.2", "3.0.0", "3.7.0", "3.7.0-SNAPSHOT"] {
            assert!(check_opensearch_version(v).is_ok(), "{v}");
        }
        for v in ["2.18.9", "2.16.0", "1.3.0", "", "3", "x.y"] {
            assert!(check_opensearch_version(v).is_err(), "{v}");
        }
    }

    #[test]
    fn opensearch_pipeline_is_per_config_and_read_back_exactly() {
        let s = spec(100, 60);
        let name = opensearch_pipeline_name("OpenSearch MS:MARCO hybrid", &s);
        assert!(
            name.starts_with("vdbb-hybrid-rrf-opensearch_ms_marco_hybrid-")
                && name.ends_with("-k60"),
            "{name}"
        );
        assert_ne!(name, opensearch_pipeline_name("other", &s));
        // The lossy token alone would collide; the hash keeps them apart.
        assert_ne!(
            opensearch_pipeline_name("a.b", &s),
            opensearch_pipeline_name("a_b", &s)
        );
        assert_ne!(
            opensearch_pipeline_name("A", &s),
            opensearch_pipeline_name("a", &s)
        );
        assert_eq!(
            name,
            opensearch_pipeline_name("OpenSearch MS:MARCO hybrid", &s)
        );
        assert_ne!(
            name,
            opensearch_pipeline_name("OpenSearch MS:MARCO hybrid", &spec(100, 20))
        );
        let body = opensearch_pipeline_body(&s);
        let read_back = json!({ name.clone(): body.clone() });
        assert!(check_opensearch_pipeline(&read_back, &name, &s).is_ok());
        // Wrong constant, wrong technique, missing, extra processors.
        assert!(check_opensearch_pipeline(&read_back, &name, &spec(100, 61)).is_err());
        let mut lin = body.clone();
        lin["phase_results_processors"][0][OPENSEARCH_PROCESSOR]["combination"]["technique"] =
            json!("arithmetic_mean");
        assert!(check_opensearch_pipeline(&json!({ name.clone(): lin }), &name, &s).is_err());
        assert!(check_opensearch_pipeline(&json!({}), &name, &s).is_err());
        let mut extra = body.clone();
        extra["request_processors"] = json!([{"filter_query": {}}]);
        assert!(check_opensearch_pipeline(&json!({ name.clone(): extra }), &name, &s).is_err());
        let mut two = body;
        two["phase_results_processors"]
            .as_array_mut()
            .unwrap()
            .push(json!({"normalization-processor": {}}));
        assert!(check_opensearch_pipeline(&json!({ name.clone(): two }), &name, &s).is_err());
        // A string-typed constant (as some versions echo it) is accepted.
        let mut stringy = opensearch_pipeline_body(&s);
        stringy["phase_results_processors"][0][OPENSEARCH_PROCESSOR]["combination"]
            ["rank_constant"] = json!("60");
        assert!(check_opensearch_pipeline(&json!({ name.clone(): stringy }), &name, &s).is_ok());
    }

    #[test]
    fn opensearch_body_fuses_the_fulltext_query_and_a_window_deep_knn() {
        let f = vec![FieldBoost {
            field: "segment".into(),
            boost: 1.0,
        }];
        let terms = vec!["zebra".to_string()];
        let s = spec(100, 60);
        let b = opensearch_hybrid_body(&f, &terms, &[0.5, 0.25], 200, &s, 10);
        assert_eq!(b["size"], 10);
        let h = &b["query"]["hybrid"];
        assert_eq!(h["pagination_depth"], 100);
        let q = h["queries"].as_array().unwrap();
        assert_eq!(q.len(), 2);
        assert_eq!(q[0], fulltext::opensearch_body(&f, &terms, 10)["query"]);
        assert_eq!(q[1]["knn"]["vector"]["k"], 100);
        assert_eq!(q[1]["knn"]["vector"]["method_parameters"]["ef_search"], 200);
        assert_eq!(q[1]["knn"]["vector"]["vector"], json!([0.5, 0.25]));
        let r = opensearch_reference_body(&f, &terms, &s);
        assert_eq!(r["size"], 100);
        assert_eq!(r["query"], q[0]);
    }

    #[test]
    fn workload_is_planned_dispatched_and_captured_for_redis_and_opensearch_only() {
        for engine in ["redis", "opensearch"] {
            let p = workload::plan(engine, &[Some(Workload::HybridRrf)], false).unwrap();
            assert_eq!(p.layout, workload::IndexLayout::Vector);
            assert!(workload::dispatchable(engine, Workload::HybridRrf));
            assert!(workload::captures(engine, Workload::HybridRrf));
            assert!(workload::plan(engine, &[Some(Workload::HybridRrf)], true).is_err());
        }
        for engine in ["qdrant", "valkey", "elasticsearch"] {
            assert!(workload::plan(engine, &[Some(Workload::HybridRrf)], false).is_err());
            assert!(!workload::dispatchable(engine, Workload::HybridRrf));
        }
        assert_eq!(
            "hybrid-rrf".parse::<Workload>().unwrap(),
            Workload::HybridRrf
        );
        assert!(Workload::HybridRrf.has_ground_truth());
        assert!(Workload::HybridRrf.uses_text_index());
    }

    #[test]
    fn result_metadata_records_the_semantics() {
        let stats = ReferenceStats::of(
            &[
                reference(&[1, 2], &[2, 3, 4], &spec(3, 60)).unwrap(),
                reference(&[1, 2, 3], &[5, 6, 7], &spec(3, 60)).unwrap(),
            ],
            3,
        );
        assert_eq!(stats.queries, 2);
        assert_eq!(stats.queries_short_bm25, 1);
        assert!((stats.mean_bm25_candidates - 2.5).abs() < 1e-12);
        assert!((stats.mean_in_both - 0.5).abs() < 1e-12);
        let m = result_metadata(
            &spec(3, 60),
            2,
            64,
            json!({"analysis": "english"}),
            json!({"engine": "redis"}),
            &stats,
            0.9,
        );
        assert_eq!(m["semantics"], SEMANTICS);
        assert_eq!(m["rank_constant"], 60);
        assert_eq!(m["window"], 3);
        assert_eq!(m["result_limit"], 2);
        assert_eq!(m["ef"], 64);
        assert_eq!(m["bm25"]["analysis"], "english");
        assert_eq!(m["quality"]["mean_strict_recall"], 0.9);
        assert_eq!(m["reference"]["queries"], 2);
    }
}
