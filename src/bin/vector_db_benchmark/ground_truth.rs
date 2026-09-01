//! Ground-truth width profiling.
//!
//! "Width" is how many *valid* (non-sentinel, deduped) neighbour ids a query
//! actually has in `expected[:top]`. On a standard HDF5 dataset every row is
//! 100 neighbours wide and width == top for any sane `top`; on filtered, sparse
//! and generated datasets rows are routinely shorter (the shipped
//! `h-and-m-2048-angular` `tests.jsonl` has 931 of 10 000 queries with fewer
//! than 25 `closest_ids`).
//!
//! Width is what makes the retrieval-quality denominators disagree (see
//! `crate::metrics`):
//!
//! * our `mean_recall` divides by the width (a query with one true neighbour can
//!   reach 1.0),
//! * upstream `qdrant/vector-db-benchmark` always divides by `top` (that same
//!   query caps at `1/top`),
//! * our `mean_precision_at_returned` divides by what the engine returned, which
//!   for a full page is `top` — so it is capped by the width too.
//!
//! Two consumers use this: the results JSON, which reports the profile so a
//! reader can tell whether our numbers are comparable to upstream's at all, and
//! the calibrator, which would otherwise binary-search for a target precision
//! that the ground truth makes unreachable.

use std::collections::HashSet;

use crate::dataset::Dataset;

/// Per-query ground-truth rows, kept only to derive width statistics.
pub struct GroundTruthProfile {
    rows: Vec<Vec<i64>>,
}

/// Width statistics evaluated at one specific `top`.
#[derive(Debug, Clone, PartialEq)]
pub struct GroundTruthStats {
    /// Number of ground-truth rows (queries) profiled.
    pub queries: usize,
    /// The `top` these statistics were evaluated at.
    pub top: usize,
    /// Mean number of valid truth ids per query, capped at `top`.
    pub mean_width: f64,
    /// Narrowest row (capped at `top`).
    pub min_width: usize,
    /// How many queries have fewer than `top` valid truth ids. Zero means our
    /// `mean_recall` and upstream's `mean_precisions` are the same quantity.
    pub queries_below_top: usize,
    /// `mean(min(width, top)) / top`: the largest value upstream's
    /// `mean_precisions` (recall@top) can take on this dataset, and — assuming
    /// the engine returns a full page of `top` results — the largest value our
    /// `mean_precision_at_returned` can take. 1.0 on full-width ground truth.
    pub recall_at_top_ceiling: f64,
}

impl GroundTruthProfile {
    /// Read the dataset's ground truth (dense, sparse or hybrid queries).
    ///
    /// This re-reads the query file the engine also reads; it is a few MB and is
    /// done once per experiment, not per search config.
    pub fn load(dataset: &Dataset) -> Result<Self, String> {
        let rows = if dataset.is_hybrid() {
            let (_dense, _sparse, neighbors) = dataset.read_hybrid_queries()?;
            neighbors
        } else if dataset.is_sparse() {
            let (_queries, neighbors) = dataset.read_sparse_queries()?;
            neighbors
        } else if dataset.is_multivector() {
            let (_queries, neighbors) = dataset.read_multivector_queries()?;
            neighbors
        } else {
            let (_queries, neighbors, _conditions) = dataset.read_queries()?;
            neighbors
        };
        Ok(Self::from_rows(rows))
    }

    pub fn from_rows(rows: Vec<Vec<i64>>) -> Self {
        Self { rows }
    }

    /// Width of the first row, used as the fallback `top` when no config pins one
    /// (the engines derive `top` the same way).
    pub fn first_row_len(&self) -> Option<usize> {
        self.rows.first().map(|r| r.len())
    }

    /// Width statistics at `top`. `None` when there is nothing to profile.
    pub fn stats(&self, top: usize) -> Option<GroundTruthStats> {
        if top == 0 || self.rows.is_empty() {
            return None;
        }
        let widths: Vec<usize> = self.rows.iter().map(|r| valid_width(r, top)).collect();
        let queries = widths.len();
        let sum: usize = widths.iter().sum();
        let min_width = widths.iter().copied().min().unwrap_or(0);
        let queries_below_top = widths.iter().filter(|&&w| w < top).count();
        let mean_width = sum as f64 / queries as f64;
        Some(GroundTruthStats {
            queries,
            top,
            mean_width,
            min_width,
            queries_below_top,
            recall_at_top_ceiling: mean_width / top as f64,
        })
    }
}

/// Number of valid truth ids a row contributes at `top`, mirroring exactly what
/// `crate::metrics::compute_metrics` counts: drop negative sentinels first, then
/// take the first `top`, then dedup.
fn valid_width(row: &[i64], top: usize) -> usize {
    let set: HashSet<i64> = row
        .iter()
        .copied()
        .filter(|&id| id >= 0)
        .take(top)
        .collect();
    set.len()
}

impl GroundTruthStats {
    /// Explain why a calibration target cannot be reached on this dataset, or
    /// `None` when the ceiling allows it.
    ///
    /// The calibrator binary-searches on `mean_precision_at_returned`, whose
    /// denominator is the number of results returned. An engine asked for `top`
    /// normally returns `top`, so the ground-truth width caps the achievable
    /// precision at `recall_at_top_ceiling` — a 0.95 target against rows that
    /// average 24 neighbours at `top: 100` is unreachable no matter how high `ef`
    /// goes, and the search silently converges on the maximum `ef`.
    ///
    /// The bound is an estimate in one direction only: a *filtered* query where
    /// the engine legitimately returns fewer than `top` results can score above
    /// the ceiling, so this is reported as a warning rather than treated as a
    /// hard failure.
    pub fn unreachable_target_note(&self, target: f64) -> Option<String> {
        // NaN compares false against everything, so it is excluded explicitly;
        // +inf is a legitimately unreachable target and must still be flagged.
        if target.is_nan() || target <= self.recall_at_top_ceiling + 1e-9 {
            return None;
        }
        Some(format!(
            "target precision {:.4} is above the ceiling this dataset allows: {} of {} queries \
             have fewer than top={} valid ground-truth neighbours (mean {:.2}, min {}), so \
             mean_precision_at_returned cannot exceed {:.4} for an engine that returns a full \
             page of {} results. Raising the calibrated parameter cannot close that gap — it \
             will converge on the maximum value tried. Lower calibration_precision to <= {:.4}, \
             lower `top`, or calibrate against a dataset with full-width ground truth. \
             (A filtered query that returns fewer than {} results can score above this ceiling, \
             so treat it as an estimate.)",
            target,
            self.queries_below_top,
            self.queries,
            self.top,
            self.mean_width,
            self.min_width,
            self.recall_at_top_ceiling,
            self.top,
            self.recall_at_top_ceiling,
            self.top,
        ))
    }

    /// Whether our `mean_recall` is numerically the same quantity as upstream's
    /// `mean_precisions` for this dataset/`top`: true when every row is at least
    /// `top` wide, so both denominators equal `top`.
    ///
    /// This assumes the two normal shapes — sentinel padding only ever trails the
    /// real ids, and the engine returns no more than `top` results. Two
    /// pathological shapes can still separate the numbers even at full width: a
    /// sentinel *interleaved* among real ids (upstream slices the raw row at
    /// `top` and keeps the sentinel, we filter first and reach one id further),
    /// and an engine returning MORE than `top` results (we truncate at `top`,
    /// upstream intersects everything it got). Neither occurs in the shipped
    /// datasets or engines.
    pub fn recall_matches_upstream(&self) -> bool {
        self.queries_below_top == 0
    }

    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "queries": self.queries,
            "top": self.top,
            "mean_valid_neighbours_per_query": self.mean_width,
            "min_valid_neighbours_per_query": self.min_width,
            "queries_with_fewer_than_top_neighbours": self.queries_below_top,
            "recall_at_top_ceiling": self.recall_at_top_ceiling,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(n: usize, width: usize) -> Vec<Vec<i64>> {
        (0..n)
            .map(|q| ((q * 1000) as i64..(q * 1000 + width) as i64).collect())
            .collect()
    }

    #[test]
    fn full_width_ground_truth_has_ceiling_one() {
        let p = GroundTruthProfile::from_rows(rows(10, 100));
        let s = p.stats(100).unwrap();
        assert_eq!(s.queries, 10);
        assert_eq!(s.min_width, 100);
        assert_eq!(s.queries_below_top, 0);
        assert!((s.recall_at_top_ceiling - 1.0).abs() < 1e-9);
        assert!(s.recall_matches_upstream());
        assert!(s.unreachable_target_note(0.95).is_none());
    }

    #[test]
    fn top_below_row_width_is_still_full_width() {
        // 100-neighbour HDF5 rows queried at top=10: still full width.
        let p = GroundTruthProfile::from_rows(rows(5, 100));
        let s = p.stats(10).unwrap();
        assert_eq!(s.min_width, 10);
        assert_eq!(s.queries_below_top, 0);
        assert!((s.recall_at_top_ceiling - 1.0).abs() < 1e-9);
    }

    /// The `h-and-m`-shaped case from #217, scaled down: most rows are wide, a
    /// tenth are single-neighbour, `top` is 100.
    #[test]
    fn short_rows_pull_the_ceiling_below_the_calibration_target() {
        let mut r = rows(9, 100);
        r.push(vec![42]);
        let p = GroundTruthProfile::from_rows(r);
        let s = p.stats(100).unwrap();
        assert_eq!(s.queries, 10);
        assert_eq!(s.queries_below_top, 1);
        assert_eq!(s.min_width, 1);
        // mean width = (9*100 + 1)/10 = 90.1 -> ceiling 0.901
        assert!((s.mean_width - 90.1).abs() < 1e-9, "{}", s.mean_width);
        assert!((s.recall_at_top_ceiling - 0.901).abs() < 1e-9);
        assert!(!s.recall_matches_upstream());
        let note = s
            .unreachable_target_note(0.95)
            .expect("0.95 > 0.901 must be flagged");
        assert!(note.contains("0.9010"), "{}", note);
        assert!(s.unreachable_target_note(0.90).is_none());
    }

    #[test]
    fn sentinel_padding_does_not_count_as_ground_truth() {
        let p = GroundTruthProfile::from_rows(vec![vec![1, 2, -1, -1, -1]]);
        let s = p.stats(5).unwrap();
        assert_eq!(s.min_width, 2);
        assert_eq!(s.queries_below_top, 1);
        assert!((s.recall_at_top_ceiling - 0.4).abs() < 1e-9);
    }

    #[test]
    fn duplicate_truth_ids_counted_once() {
        let p = GroundTruthProfile::from_rows(vec![vec![7, 7, 7, 8]]);
        let s = p.stats(4).unwrap();
        assert_eq!(s.min_width, 2);
        assert!((s.recall_at_top_ceiling - 0.5).abs() < 1e-9);
    }

    #[test]
    fn empty_or_zero_top_yields_no_stats() {
        assert!(GroundTruthProfile::from_rows(vec![]).stats(10).is_none());
        assert!(GroundTruthProfile::from_rows(rows(2, 10))
            .stats(0)
            .is_none());
    }

    #[test]
    fn nan_target_is_not_reported_unreachable_but_infinity_is() {
        let p = GroundTruthProfile::from_rows(rows(2, 10));
        let s = p.stats(10).unwrap();
        assert!(s.unreachable_target_note(f64::NAN).is_none());
        assert!(
            s.unreachable_target_note(f64::INFINITY).is_some(),
            "+inf is unreachable even against a ceiling of 1.0"
        );
    }
}
