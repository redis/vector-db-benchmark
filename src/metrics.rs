//! Retrieval quality metrics: recall@K, precision@K, MRR, NDCG@K.
//!
//! # Three different denominators — read this before comparing numbers
//!
//! Let `hits = |deduped results[:K] ∩ valid ground-truth ids in expected[:K]|`.
//! Three quantities are computed from it and they are NOT interchangeable:
//!
//! | quantity     | formula                | emitted as                    |
//! |--------------|------------------------|-------------------------------|
//! | `precision`  | `hits / |results|`     | `mean_precision_at_returned`  |
//! | `recall`     | `hits / |valid truth|` | `mean_recall`                 |
//! | `recall_at_top` | `hits / K`          | (upstream's `mean_precisions`)|
//!
//! Upstream `qdrant/vector-db-benchmark` (`engine/base_client/search.py`) computes
//! `len(ids.intersection(query.expected_result[:top])) / top` — i.e. `recall_at_top`
//! — and emits it under the JSON key `mean_precisions`. Our `mean_precisions` key
//! used to carry `precision` instead: the same key name for a different formula.
//! That key no longer exists in our output (see #217); we emit
//! `mean_precision_at_returned` and `mean_recall`.
//!
//! Worked examples of how far apart they land:
//!
//! * Engine asked for `top = 10`, returns only 5 results, all correct, truth has
//!   10 ids: `precision = 5/5 = 1.00`, `recall = 5/10 = 0.50`,
//!   `recall_at_top = 5/10 = 0.50`. Upstream reports 0.50 under `mean_precisions`;
//!   we used to report 1.00 under the same key.
//! * Filtered query with a single true neighbour, `top = 100`, engine returns 100
//!   results including that neighbour: `precision = 1/100 = 0.01`,
//!   `recall = 1/1 = 1.00`, `recall_at_top = 1/100 = 0.01`. Our `recall` and
//!   upstream's `mean_precisions` diverge by 100x on identical data, because
//!   upstream always divides by `top` while we divide by the ground truth that
//!   actually exists.

use std::collections::HashSet;

/// Per-query retrieval quality metrics.
///
/// See the module docs for why `precision`, `recall` and `recall_at_top` are
/// three different numbers and which one upstream calls "precision".
#[derive(Debug, Clone, Default)]
pub struct QueryMetrics {
    /// recall@K: `hits / |valid, deduped ground-truth ids in expected[:K]|`.
    ///
    /// The denominator is the ground truth that actually exists, so a query with
    /// fewer than K true neighbours can still reach 1.0. This is the field
    /// emitted as `mean_recall`.
    pub recall: f64,
    /// precision@K: `hits / |deduped results kept|` (denominator = what the
    /// engine returned, at most K). Emitted as `mean_precision_at_returned`.
    pub precision: f64,
    /// recall@top with upstream's denominator: `hits / K`, exactly
    /// `len(ids & expected[:top]) / top` from upstream's `search.py`.
    ///
    /// Provided so the upstream definition has one canonical implementation here
    /// and so the divergence is pinned by tests. It is not aggregated into the
    /// results JSON today: every engine harness collects per-query samples into
    /// its own `Vec<f64>`s, so adding a fifth vector would have to touch all 15
    /// engines at once. `mean_recall` equals the mean of this field whenever every
    /// ground-truth row has at least K valid ids — which the emitted
    /// `metrics_schema.ground_truth` block reports per run.
    pub recall_at_top: f64,
    /// Mean Reciprocal Rank: 1/rank of the first relevant result
    pub mrr: f64,
    /// Normalized Discounted Cumulative Gain @ K
    pub ndcg: f64,
}

/// Compute all retrieval quality metrics for a single query.
///
/// - `result_ids_ordered`: engine results in ranked order (position 0 = best)
/// - `ground_truth`: true top-K neighbor IDs from the dataset
/// - `k`: the K value (top)
pub fn compute_metrics(result_ids_ordered: &[i64], ground_truth: &[i64], k: usize) -> QueryMetrics {
    if k == 0 {
        return QueryMetrics {
            recall: 1.0,
            precision: 1.0,
            recall_at_top: 1.0,
            mrr: 1.0,
            ndcg: 1.0,
        };
    }

    // Ground truth: drop sentinel/invalid ids (e.g. the `-1` padding HDF5
    // `neighbors` rows use when a query has fewer than K true neighbors), then
    // cap at K. The recall denominator is the number of *valid* truth ids, so a
    // query with fewer than K real neighbors can still reach recall 1.0 — this
    // matches the NDCG ideal-DCG convention below.
    let truth_set: HashSet<i64> = ground_truth
        .iter()
        .copied()
        .filter(|&id| id >= 0)
        .take(k)
        .collect();
    let truth_count = truth_set.len();

    // No valid ground truth (e.g. a filtered query with no matching points):
    // nothing to retrieve, so this query is not penalized. Note upstream would
    // score this query 0 (its denominator is `top` regardless of how much truth
    // exists); `recall_at_top` follows our not-penalized convention here so the
    // three fields stay mutually consistent within one run.
    if truth_count == 0 {
        return QueryMetrics {
            recall: 1.0,
            precision: 1.0,
            recall_at_top: 1.0,
            mrr: 1.0,
            ndcg: 1.0,
        };
    }

    // Engine results: dedup (preserving rank order) and keep only the top K, so
    // hits can't be double-counted and recall can't exceed 1.0.
    let mut seen = HashSet::new();
    let results_topk: Vec<i64> = result_ids_ordered
        .iter()
        .copied()
        .filter(|id| seen.insert(*id))
        .take(k)
        .collect();

    let hits = results_topk
        .iter()
        .filter(|id| truth_set.contains(id))
        .count();

    // Three denominators, three different numbers — see the module docs.
    //   recall:        the ground truth that actually exists (<= k)
    //   precision:     what the engine actually returned (<= k)
    //   recall_at_top: k itself — upstream's `mean_precisions`
    let recall = hits as f64 / truth_count as f64;
    let precision = if results_topk.is_empty() {
        0.0
    } else {
        hits as f64 / results_topk.len() as f64
    };
    let recall_at_top = hits as f64 / k as f64;

    // MRR: 1/rank of the first relevant result within the top K.
    let mrr = results_topk
        .iter()
        .enumerate()
        .find(|(_, id)| truth_set.contains(id))
        .map(|(rank, _)| 1.0 / (rank + 1) as f64)
        .unwrap_or(0.0);

    // NDCG@K over the deduped top-K results.
    let ndcg = {
        let dcg: f64 = results_topk
            .iter()
            .enumerate()
            .filter(|(_, id)| truth_set.contains(id))
            .map(|(i, _)| 1.0 / (i as f64 + 2.0).log2())
            .sum();

        let idcg: f64 = (0..truth_count)
            .map(|i| 1.0 / (i as f64 + 2.0).log2())
            .sum();

        if idcg > 0.0 {
            dcg / idcg
        } else {
            0.0
        }
    };

    QueryMetrics {
        recall,
        precision,
        recall_at_top,
        mrr,
        ndcg,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfect_retrieval() {
        let results = vec![1, 2, 3, 4, 5];
        let truth = vec![1, 2, 3, 4, 5];
        let m = compute_metrics(&results, &truth, 5);
        assert!((m.recall - 1.0).abs() < 1e-9);
        assert!((m.precision - 1.0).abs() < 1e-9);
        assert!((m.mrr - 1.0).abs() < 1e-9);
        assert!((m.ndcg - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_no_overlap() {
        let results = vec![6, 7, 8, 9, 10];
        let truth = vec![1, 2, 3, 4, 5];
        let m = compute_metrics(&results, &truth, 5);
        assert!((m.recall).abs() < 1e-9);
        assert!((m.precision).abs() < 1e-9);
        assert!((m.mrr).abs() < 1e-9);
        assert!((m.ndcg).abs() < 1e-9);
    }

    #[test]
    fn test_first_relevant_at_position_3() {
        let results = vec![10, 20, 3, 1, 5];
        let truth = vec![1, 2, 3, 4, 5];
        let m = compute_metrics(&results, &truth, 5);
        assert!((m.recall - 0.6).abs() < 1e-9);
        assert!((m.precision - 0.6).abs() < 1e-9);
        assert!((m.mrr - 1.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_fewer_results_than_k() {
        let results = vec![1, 2, 3];
        let truth = vec![1, 2, 3, 4, 5];
        let m = compute_metrics(&results, &truth, 5);
        assert!((m.recall - 0.6).abs() < 1e-9);
        assert!((m.precision - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_k_zero() {
        let m = compute_metrics(&[], &[], 0);
        assert!((m.recall - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_fewer_ground_truth_than_k_reaches_full_recall() {
        // Only 2 real neighbors but k=5: a perfect engine must score recall 1.0
        // (denominator = valid gt, not k). Previously this capped at 2/5 = 0.4.
        let results = vec![1, 2, 3, 4, 5];
        let truth = vec![1, 2];
        let m = compute_metrics(&results, &truth, 5);
        assert!((m.recall - 1.0).abs() < 1e-9, "recall={}", m.recall);
        assert!((m.ndcg - 1.0).abs() < 1e-9, "ndcg={}", m.ndcg);
    }

    #[test]
    fn test_sentinel_padding_ignored() {
        // HDF5-style -1 padding must not count as truth ids.
        let results = vec![1, 2, 9, 8, 7];
        let truth = vec![1, 2, -1, -1, -1];
        let m = compute_metrics(&results, &truth, 5);
        assert!((m.recall - 1.0).abs() < 1e-9, "recall={}", m.recall);
    }

    #[test]
    fn test_duplicate_results_not_double_counted() {
        // Engine returns duplicates; each relevant id counts once.
        let results = vec![1, 1, 2, 2, 3];
        let truth = vec![1, 2, 3, 4, 5];
        let m = compute_metrics(&results, &truth, 5);
        assert!((m.recall - 0.6).abs() < 1e-9, "recall={}", m.recall);
    }

    #[test]
    fn test_excess_results_truncated_to_k() {
        // More than k results: only the top-k count; recall cannot exceed 1.0.
        let results = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let truth = vec![1, 2, 3];
        let m = compute_metrics(&results, &truth, 3);
        assert!((m.recall - 1.0).abs() < 1e-9, "recall={}", m.recall);
        assert!(m.recall <= 1.0 + 1e-9);
    }

    #[test]
    fn test_empty_ground_truth_not_penalized() {
        let m = compute_metrics(&[9, 8, 7], &[], 5);
        assert!((m.recall - 1.0).abs() < 1e-9);
    }

    /// #217 canonical case: engine asked for 10, returns 5, all correct.
    ///
    /// Upstream `qdrant/vector-db-benchmark` reports 0.50 under the JSON key
    /// `mean_precisions` (`len(ids & expected[:top]) / top`). Our precision is
    /// 1.00 because its denominator is what came back. Both are correct numbers
    /// for their own definition; publishing them under one key name was the bug.
    #[test]
    fn precision_and_upstream_recall_diverge_on_short_result_set() {
        let results = vec![1, 2, 3, 4, 5]; // only 5 of the 10 requested came back
        let truth = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let m = compute_metrics(&results, &truth, 10);

        // ours, emitted as `mean_precision_at_returned`: 5 hits / 5 returned
        assert!(
            (m.precision - 1.0).abs() < 1e-9,
            "precision={}",
            m.precision
        );
        // ours, emitted as `mean_recall`: 5 hits / 10 valid truth ids
        assert!((m.recall - 0.5).abs() < 1e-9, "recall={}", m.recall);
        // upstream's `mean_precisions`: 5 hits / top=10
        assert!(
            (m.recall_at_top - 0.5).abs() < 1e-9,
            "recall_at_top={}",
            m.recall_at_top
        );
        // The precise 2x gap that made overlaid charts lie.
        assert!((m.precision / m.recall_at_top - 2.0).abs() < 1e-9);
    }

    /// #217 second case: ground truth narrower than `top` (filtered/sparse sets).
    ///
    /// One true neighbour, `top = 100`, engine returns a full page of 100 that
    /// contains it. Upstream caps at 1/100; our recall reaches 1.0; our precision
    /// is also 0.01 — which is why a `calibration_precision: 0.95` target is
    /// unreachable by construction on such a dataset.
    #[test]
    fn short_ground_truth_row_pins_all_three_denominators() {
        let mut results: Vec<i64> = (1000..1099).collect();
        results.push(7); // the single true neighbour, last on the page
        let truth = vec![7];
        let m = compute_metrics(&results, &truth, 100);

        assert!(
            (m.precision - 0.01).abs() < 1e-9,
            "precision={}",
            m.precision
        );
        assert!((m.recall - 1.0).abs() < 1e-9, "recall={}", m.recall);
        assert!(
            (m.recall_at_top - 0.01).abs() < 1e-9,
            "recall_at_top={}",
            m.recall_at_top
        );
        // 100x apart on identical data, on the same query.
        assert!((m.recall / m.recall_at_top - 100.0).abs() < 1e-6);
    }

    /// When the ground truth is full width (>= `top` valid ids) and the engine
    /// returns a full page, all three collapse onto the same number — which is
    /// why the two tools appeared to agree on standard HDF5 datasets.
    #[test]
    fn full_width_ground_truth_makes_all_three_agree() {
        let results = vec![1, 2, 3, 99, 98];
        let truth = vec![1, 2, 3, 4, 5];
        let m = compute_metrics(&results, &truth, 5);
        assert!((m.precision - 0.6).abs() < 1e-9);
        assert!((m.recall - 0.6).abs() < 1e-9);
        assert!((m.recall_at_top - 0.6).abs() < 1e-9);
    }

    /// Sentinel padding shrinks OUR denominator but never upstream's: upstream
    /// divides by `top` whether or not the row is padded with `-1`.
    #[test]
    fn sentinel_padding_shrinks_only_our_recall_denominator() {
        let results = vec![1, 2, 9, 8, 7];
        let truth = vec![1, 2, -1, -1, -1];
        let m = compute_metrics(&results, &truth, 5);
        assert!((m.recall - 1.0).abs() < 1e-9, "recall={}", m.recall);
        assert!(
            (m.recall_at_top - 0.4).abs() < 1e-9,
            "recall_at_top={}",
            m.recall_at_top
        );
    }
}
