//! Precision analysis, summary generation, and ASCII scatter plot.
//!
//! Mirrors Python v0/engine/base_client/client.py:
//! - analyze_precision_performance()
//! - _display_results_summary()
//! - _create_ascii_scatter_plot()

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use serde_json::json;

use crate::engine::SearchResults;

/// A single search result entry for precision analysis.
#[derive(Debug, Clone)]
pub struct SearchEntry {
    pub search_id: usize,
    pub ef: String,
    pub parallel: i64,
    pub results: SearchResults,
}

/// Precision analysis result for a single precision bucket.
#[derive(Debug, Clone)]
struct PrecisionBucket {
    qps: f64,
    p50_ms: f64,
    p95_ms: f64,
    precision: f64,
    recall: f64,
    mrr: f64,
    ndcg: f64,
    /// The run behind this bucket was flagged client-saturated.
    saturated: bool,
}

/// Format a precision value into a bucket key (matches Python v0 format_precision_key).
///
/// - For precision < 0.97: rounds to nearest 0.01
/// - For precision >= 0.97: rounds to nearest 0.0025
fn format_precision_key(precision: f64) -> String {
    if precision < 0.97 {
        format!("{:.2}", (precision * 100.0).round() / 100.0)
    } else {
        // Finer granularity near 1.0: 0.0025 steps
        let step = 0.0025;
        let rounded = (precision / step).round() * step;
        format!("{:.4}", rounded)
    }
}

/// Analyze search results and find best QPS at each precision level.
///
/// Groups results by formatted precision bucket and picks the entry
/// with the highest QPS for each bucket.
fn analyze_precision_performance(entries: &[SearchEntry]) -> Vec<PrecisionBucket> {
    let mut buckets: BTreeMap<String, PrecisionBucket> = BTreeMap::new();

    for entry in entries {
        let key = format_precision_key(entry.results.mean_precision_at_returned);

        let candidate = PrecisionBucket {
            qps: entry.results.rps,
            p50_ms: entry.results.p50_time * 1000.0,
            p95_ms: entry.results.p95_time * 1000.0,
            precision: entry.results.mean_precision_at_returned,
            recall: entry.results.mean_recall,
            mrr: entry.results.mean_mrr,
            ndcg: entry.results.mean_ndcg,
            saturated: entry.results.client_saturated,
        };

        // Keep the best entry for this precision bucket. Prefer a NON-saturated
        // point over a saturated one so the headline "best QPS" is never a
        // client-bound data point; only fall back to a saturated point when it is
        // the sole candidate. Within the same saturation class, keep higher QPS.
        let replace = match buckets.get(&key) {
            Some(existing) => match (candidate.saturated, existing.saturated) {
                (false, true) => true,
                (true, false) => false,
                _ => candidate.qps > existing.qps,
            },
            None => true,
        };
        if replace {
            buckets.insert(key, candidate);
        }
    }

    // Return sorted by precision descending (for display)
    let mut result: Vec<PrecisionBucket> = buckets.into_values().collect();
    result.sort_by(|a, b| b.precision.partial_cmp(&a.precision).unwrap());
    result
}

/// One concurrency level on a throughput-vs-concurrency curve.
#[derive(Debug, Clone)]
struct ConcurrencyPoint {
    parallel: i64,
    qps: f64,
    p50_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
    recall: f64,
    saturated: bool,
}

/// A throughput-vs-concurrency curve at ONE quality operating point (`ef`).
/// This is the saturation curve every external tool (VectorDBBench, VSB,
/// qdrant-bench) reports and we did not: sweeping the client concurrency while
/// holding the index/query config fixed reveals the throughput knee — the
/// concurrency at which QPS peaks and beyond which more clients stop buying
/// throughput (and latency inflates).
#[derive(Debug, Clone)]
struct ConcurrencyCurve {
    ef: String,
    points: Vec<ConcurrencyPoint>,
    /// Peak QPS across the swept levels.
    max_qps: f64,
    /// Concurrency at which the peak QPS occurs (smallest, on ties).
    max_qps_parallel: i64,
    /// Concurrency where throughput saturates: `Some(max_qps_parallel)` when the
    /// peak is BELOW the highest level tested (adding concurrency past it did not
    /// help). `None` when QPS was still climbing at the top of the swept range —
    /// the knee is beyond what was tested, so sweep higher to find it.
    knee_parallel: Option<i64>,
}

/// Build throughput-vs-concurrency curves from the run entries. Entries are
/// grouped by quality operating point (`ef`); within each group the distinct
/// `parallel` levels form the curve (on a repeated `(ef, parallel)` the better
/// point wins — a non-saturated one over a saturated one, else higher QPS). Only
/// groups with >= 2 distinct concurrency levels are returned (a single point is
/// not a sweep). Curves are ordered by `ef`, points by ascending concurrency.
fn concurrency_curves(entries: &[SearchEntry]) -> Vec<ConcurrencyCurve> {
    // ef -> (parallel -> best entry)
    let mut groups: BTreeMap<String, BTreeMap<i64, ConcurrencyPoint>> = BTreeMap::new();
    for e in entries {
        // Filter-only runs have no precision/recall curve to speak of; skip them.
        if e.results.mean_precision_at_returned < 0.0 {
            continue;
        }
        let point = ConcurrencyPoint {
            parallel: e.parallel,
            qps: e.results.rps,
            p50_ms: e.results.p50_time * 1000.0,
            p95_ms: e.results.p95_time * 1000.0,
            p99_ms: e.results.p99_time * 1000.0,
            recall: e.results.mean_recall,
            saturated: e.results.client_saturated,
        };
        let by_parallel = groups.entry(e.ef.clone()).or_default();
        let replace = match by_parallel.get(&e.parallel) {
            Some(existing) => match (point.saturated, existing.saturated) {
                (false, true) => true,
                (true, false) => false,
                _ => point.qps > existing.qps,
            },
            None => true,
        };
        if replace {
            by_parallel.insert(e.parallel, point);
        }
    }

    let mut curves = Vec::new();
    for (ef, by_parallel) in groups {
        if by_parallel.len() < 2 {
            continue; // not a sweep
        }
        let points: Vec<ConcurrencyPoint> = by_parallel.into_values().collect(); // BTreeMap → sorted by parallel
                                                                                 // Peak QPS and the smallest concurrency achieving it.
        let mut max_qps = f64::NEG_INFINITY;
        let mut max_qps_parallel = points[0].parallel;
        for p in &points {
            if p.qps > max_qps {
                max_qps = p.qps;
                max_qps_parallel = p.parallel;
            }
        }
        let highest = points.last().unwrap().parallel;
        let knee_parallel = if max_qps_parallel < highest {
            Some(max_qps_parallel)
        } else {
            None
        };
        curves.push(ConcurrencyCurve {
            ef,
            points,
            max_qps,
            max_qps_parallel,
            knee_parallel,
        });
    }
    curves
}

/// Render the concurrency curves as a compact text table (only when a sweep
/// exists). Shows QPS/latency per concurrency level and calls out the knee.
fn print_concurrency_curves(curves: &[ConcurrencyCurve]) {
    if curves.is_empty() {
        return;
    }
    println!("\nTHROUGHPUT vs CONCURRENCY (saturation curve):");
    for c in curves {
        println!(
            "  ef={} — peak {:.1} QPS @ parallel {}{}",
            c.ef,
            c.max_qps,
            c.max_qps_parallel,
            match c.knee_parallel {
                Some(k) =>
                    format!(" (throughput knee at parallel {k}; higher concurrency did not help)"),
                None => " (still climbing at top of swept range — sweep higher to find the knee)"
                    .to_string(),
            }
        );
        println!(
            "    {:>8}  {:>10}  {:>9}  {:>9}  {:>9}  {:>6}",
            "parallel", "QPS", "p50 ms", "p95 ms", "p99 ms", "sat?"
        );
        for p in &c.points {
            println!(
                "    {:>8}  {:>10.1}  {:>9.3}  {:>9.3}  {:>9.3}  {:>6}",
                p.parallel,
                p.qps,
                p.p50_ms,
                p.p95_ms,
                p.p99_ms,
                if p.saturated { "YES" } else { "-" }
            );
        }
    }
}

/// Detect and describe client/throughput-saturation problems across a set of
/// runs. Two independent signals:
///  1. any single run the client-CPU coverage flagged (`client_saturated`); and
///  2. throughput that stops scaling — grouped by `ef`, when raising `parallel`
///     gains <10% QPS while p95 rises, the higher-parallel point is not a clean
///     scaling data point (client or server saturated).
fn saturation_warnings(entries: &[SearchEntry]) -> Vec<String> {
    let mut warnings = Vec::new();

    let mut flagged: Vec<(&str, i64, &str)> = entries
        .iter()
        .filter(|e| e.results.client_saturated)
        .map(|e| {
            (
                e.ef.as_str(),
                e.parallel,
                e.results.saturation_reason.as_str(),
            )
        })
        .collect();
    flagged.sort_by_key(|f| f.1);
    for (ef, parallel, reason) in flagged {
        warnings.push(format!(
            "client-saturated: ef={} parallel={} ({}) — QPS/latency reflect the client, not the DB",
            ef, parallel, reason
        ));
    }

    // Throughput scaling collapse, per ef.
    let mut by_ef: BTreeMap<&str, Vec<(i64, f64, f64)>> = BTreeMap::new();
    for e in entries {
        by_ef.entry(e.ef.as_str()).or_default().push((
            e.parallel,
            e.results.rps,
            e.results.p95_time,
        ));
    }
    for (ef, mut pts) in by_ef {
        if pts.len() < 2 {
            continue;
        }
        pts.sort_by_key(|p| p.0);
        for w in pts.windows(2) {
            let (p_prev, q_prev, p95_prev) = w[0];
            let (p_cur, q_cur, p95_cur) = w[1];
            if p_cur > p_prev && q_prev > 0.0 && p95_prev > 0.0 {
                let gain = (q_cur - q_prev) / q_prev;
                if gain < 0.10 && p95_cur > p95_prev {
                    warnings.push(format!(
                        "throughput saturated: ef={} parallel {}→{} gained only {:.0}% QPS \
                         while p95 rose {:.0}% — higher concurrency is not paying off",
                        ef,
                        p_prev,
                        p_cur,
                        gain * 100.0,
                        (p95_cur / p95_prev - 1.0) * 100.0
                    ));
                }
            }
        }
    }

    warnings
}

/// Print saturation warnings, if any, under a clear header.
fn print_saturation_warnings(entries: &[SearchEntry]) {
    let warnings = saturation_warnings(entries);
    if warnings.is_empty() {
        return;
    }
    eprintln!("\n⚠ CONCURRENCY / CPU SATURATION WARNINGS (results below may be untrustworthy):");
    for w in &warnings {
        eprintln!("  - {}", w);
    }
    eprintln!();
}

/// Display a results summary table and ASCII scatter plot.
pub fn display_results_summary(engine_name: &str, dataset_name: &str, entries: &[SearchEntry]) {
    if entries.is_empty() {
        return;
    }

    print_saturation_warnings(entries);

    // Filter-only mode: precision is not applicable (mean_precision_at_returned == -1.0)
    let filter_only = entries
        .iter()
        .all(|e| e.results.mean_precision_at_returned < 0.0);

    if filter_only {
        println!("{}", "-".repeat(40));
        println!("{:<10} {:<12} {:<12}", "QPS", "P50 (ms)", "P95 (ms)");
        println!("{}", "-".repeat(40));

        for e in entries {
            println!(
                "{:<10.1} {:<12.3} {:<12.3}",
                e.results.rps,
                e.results.p50_time * 1000.0,
                e.results.p95_time * 1000.0,
            );
        }
        println!();
        return;
    }

    let buckets = analyze_precision_performance(entries);
    if buckets.is_empty() {
        return;
    }

    // Print table
    println!("{}", "-".repeat(82));
    println!(
        "{:<10} {:<10} {:<8} {:<8} {:<10} {:<12} {:<12}",
        "Recall", "Precision", "MRR", "NDCG", "QPS", "P50 (ms)", "P95 (ms)"
    );
    println!("{}", "-".repeat(82));

    for b in &buckets {
        println!(
            "{:<10.4} {:<10.4} {:<8.4} {:<8.4} {:<10.1} {:<12.3} {:<12.3}",
            b.recall, b.precision, b.mrr, b.ndcg, b.qps, b.p50_ms, b.p95_ms
        );
    }
    println!();

    // ASCII scatter plot
    create_ascii_scatter_plot(engine_name, dataset_name, &buckets);
}

/// Display a mixed benchmark summary table with one row per update-search ratio.
/// Only called when multiple phases (ratios) were used.
pub fn display_mixed_summary(entries: &[SearchEntry]) {
    // Group entries by ratio (None = pure search, Some("U:S") = mixed)
    let mut by_ratio: BTreeMap<String, Vec<&SearchEntry>> = BTreeMap::new();
    for entry in entries {
        let key = entry
            .results
            .update_search_ratio
            .clone()
            .unwrap_or_else(|| "search".to_string());
        by_ratio.entry(key).or_default().push(entry);
    }

    if by_ratio.len() < 2 {
        return;
    }

    println!("\n{}", "-".repeat(116));
    println!(
        "{:<14} {:<8} {:<8} {:<8} {:<8} {:<10} {:<12} {:<12} {:<10} {:<12}",
        "Ratio",
        "Recall",
        "Prec",
        "MRR",
        "NDCG",
        "QPS",
        "P50 (ms)",
        "P95 (ms)",
        "Upd QPS",
        "Upd P50 (ms)"
    );
    println!("{}", "-".repeat(116));

    // Sort: "search" first, then ratios by ascending numeric value
    let mut keys: Vec<String> = by_ratio.keys().cloned().collect();
    keys.sort_by(|a, b| {
        let ra = ratio_sort_key(a);
        let rb = ratio_sort_key(b);
        ra.partial_cmp(&rb).unwrap()
    });

    for key in &keys {
        let group = &by_ratio[key];
        // Pick the entry with the highest QPS in this group
        let best = group
            .iter()
            .max_by(|a, b| a.results.rps.partial_cmp(&b.results.rps).unwrap())
            .unwrap();

        let upd_qps = best
            .results
            .update_rps
            .map(|v| format!("{:.1}", v))
            .unwrap_or_else(|| "-".to_string());
        let upd_p50 = best
            .results
            .update_p50_time
            .map(|v| format!("{:.3}", v * 1000.0))
            .unwrap_or_else(|| "-".to_string());

        println!(
            "{:<14} {:<8.4} {:<8.4} {:<8.4} {:<8.4} {:<10.1} {:<12.3} {:<12.3} {:<10} {:<12}",
            key,
            best.results.mean_recall,
            best.results.mean_precision_at_returned,
            best.results.mean_mrr,
            best.results.mean_ndcg,
            best.results.rps,
            best.results.p50_time * 1000.0,
            best.results.p95_time * 1000.0,
            upd_qps,
            upd_p50,
        );
    }
    println!();
}

/// Return a sort key for ratio strings: "search" → -1, "U:S" → U/S.
fn ratio_sort_key(key: &str) -> f64 {
    if key == "search" {
        return -1.0;
    }
    let parts: Vec<&str> = key.split(':').collect();
    if parts.len() == 2 {
        if let (Ok(u), Ok(s)) = (parts[0].parse::<f64>(), parts[1].parse::<f64>()) {
            if s > 0.0 {
                return u / s;
            }
        }
    }
    0.0
}

/// Save summary JSON matching Python v0 format.
pub fn save_summary(
    engine_name: &str,
    dataset_name: &str,
    entries: &[SearchEntry],
    upload_json: Option<&serde_json::Value>,
    results_dir: &Path,
) -> Result<(), String> {
    let buckets = analyze_precision_performance(entries);

    // Build precision_summary matching Python v0 format
    let mut precision_summary = serde_json::Map::new();
    for b in &buckets {
        let key = format_precision_key(b.precision);
        precision_summary.insert(
            key,
            json!({
                "QPS": (b.qps * 10.0).round() / 10.0,
                "P50 (ms)": (b.p50_ms * 1000.0).round() / 1000.0,
                "P95 (ms)": (b.p95_ms * 1000.0).round() / 1000.0,
                "recall": (b.recall * 10000.0).round() / 10000.0,
                "mrr": (b.mrr * 10000.0).round() / 10000.0,
                "ndcg": (b.ndcg * 10000.0).round() / 10000.0,
            }),
        );
    }

    // Build search results array
    let search_results: Vec<serde_json::Value> = entries
        .iter()
        .map(|e| {
            json!({
                "search_id": e.search_id,
                "ef": e.ef,
                "parallel": e.parallel,
                // Not `mean_precisions`: upstream qdrant/vector-db-benchmark
                // publishes recall@top under that key, and this is precision with
                // "results returned" as its denominator (#217). See
                // `metrics_schema` in the per-search result files for the formulas.
                "mean_precision_at_returned": e.results.mean_precision_at_returned,
                "mean_recall": e.results.mean_recall,
                "recall_p10": e.results.recall_p10,
                "mean_mrr": e.results.mean_mrr,
                "mean_ndcg": e.results.mean_ndcg,
                "rps": e.results.rps,
                "mean_time": e.results.mean_time,
                "p50_time": e.results.p50_time,
                "p95_time": e.results.p95_time,
                "p99_time": e.results.p99_time,
                "client_saturated": e.results.client_saturated,
                "saturation_reason": e.results.saturation_reason,
                "oversubscribed": e.results.oversubscribed,
                "available_cores": e.results.available_cores,
                "client_cpu_cores_used": e.results.client_cpu_cores_used,
            })
        })
        .collect();

    // Throughput-vs-concurrency curves (only present when the config swept >= 2
    // parallel levels at some ef). Surfaces max_qps + the saturation knee.
    let curves = concurrency_curves(entries);
    let concurrency_curve: Vec<serde_json::Value> = curves
        .iter()
        .map(|c| {
            json!({
                "ef": c.ef,
                "max_qps": (c.max_qps * 10.0).round() / 10.0,
                "max_qps_parallel": c.max_qps_parallel,
                "knee_parallel": c.knee_parallel,
                "points": c.points.iter().map(|p| json!({
                    "parallel": p.parallel,
                    "qps": (p.qps * 10.0).round() / 10.0,
                    "p50_ms": (p.p50_ms * 1000.0).round() / 1000.0,
                    "p95_ms": (p.p95_ms * 1000.0).round() / 1000.0,
                    "p99_ms": (p.p99_ms * 1000.0).round() / 1000.0,
                    "recall": (p.recall * 10000.0).round() / 10000.0,
                    "client_saturated": p.saturated,
                })).collect::<Vec<_>>(),
            })
        })
        .collect();

    let mut summary = json!({
        "engine": engine_name,
        "dataset": dataset_name,
        "search_results": search_results,
        "precision_summary": precision_summary,
        "concurrency_curve": concurrency_curve,
    });

    if let Some(upload) = upload_json {
        summary
            .as_object_mut()
            .unwrap()
            .insert("upload".to_string(), upload.clone());
    }

    print_concurrency_curves(&curves);

    let filename = format!("{}-{}-summary.json", engine_name, dataset_name);
    let path = results_dir.join(&filename);
    fs::write(&path, serde_json::to_string_pretty(&summary).unwrap())
        .map_err(|e| format!("Failed to save summary: {}", e))?;

    println!(
        "\n{}\nResults saved to:  {}",
        "=".repeat(80),
        results_dir.display()
    );
    println!("Summary saved to:  {}", path.display());

    Ok(())
}

/// Create a 60x12 ASCII scatter plot of QPS vs Precision.
fn create_ascii_scatter_plot(engine_name: &str, dataset_name: &str, buckets: &[PrecisionBucket]) {
    if buckets.is_empty() {
        return;
    }

    let width: usize = 60;
    let height: usize = 12;

    // Get data ranges
    let precisions: Vec<f64> = buckets.iter().map(|b| b.precision).collect();
    let qps_values: Vec<f64> = buckets.iter().map(|b| b.qps).collect();

    let min_prec = precisions.iter().copied().fold(f64::INFINITY, f64::min);
    let max_prec = precisions.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let min_qps = 0.0_f64;
    let max_qps = qps_values.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    // Add padding to ranges
    let prec_range = if (max_prec - min_prec).abs() < 1e-9 {
        0.01
    } else {
        max_prec - min_prec
    };
    let qps_range = if max_qps.abs() < 1e-9 {
        1.0
    } else {
        max_qps - min_qps
    };

    // Initialize grid
    let mut grid = vec![vec![' '; width]; height];

    // Plot points
    for b in buckets {
        let x = ((b.precision - min_prec) / prec_range * (width - 1) as f64).round() as usize;
        let y = ((b.qps - min_qps) / qps_range * (height - 1) as f64).round() as usize;
        let x = x.min(width - 1);
        let y = y.min(height - 1);
        // Invert y (top = high QPS)
        grid[height - 1 - y][x] = '\u{25cf}'; // ●
    }

    println!(
        "\nQPS vs Precision Trade-off - {} - {} (up and to the right is better):\n",
        engine_name, dataset_name
    );

    // Print grid with Y-axis labels
    for (row_idx, row) in grid.iter().enumerate() {
        let qps_val = max_qps - (row_idx as f64 / (height - 1) as f64) * qps_range;
        let line: String = row.iter().collect();
        print!("{:>8.0} \u{2502}", qps_val);
        println!("{}", line);
    }

    // X-axis
    print!("         \u{2514}");
    println!("{}", "\u{2500}".repeat(width));

    // X-axis labels
    let label_count = 4;
    let label_spacing = width / label_count;
    print!("          ");
    for i in 0..label_count {
        let val = min_prec + (i as f64 / (label_count - 1) as f64) * prec_range;
        print!("{:<width$.3}", val, width = label_spacing);
    }
    println!();
    println!("{}Precision (0.0 = 0%, 1.0 = 100%)", " ".repeat(8));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_precision_key_low() {
        assert_eq!(format_precision_key(0.50), "0.50");
        assert_eq!(format_precision_key(0.9066), "0.91");
        assert_eq!(format_precision_key(0.96), "0.96");
    }

    #[test]
    fn test_format_precision_key_high() {
        // Near 1.0: finer 0.0025 granularity
        // 0.9986 / 0.0025 = 399.44 → rounds to 399 → 399 * 0.0025 = 0.9975
        assert_eq!(format_precision_key(0.9986), "0.9975");
        assert_eq!(format_precision_key(1.0), "1.0000");
        // 0.9988 / 0.0025 = 399.52 → rounds to 400 → 400 * 0.0025 = 1.0000
        assert_eq!(format_precision_key(0.9988), "1.0000");
        // 0.98 / 0.0025 = 392 → 392 * 0.0025 = 0.9800
        assert_eq!(format_precision_key(0.98), "0.9800");
    }

    /// Build a `SearchEntry` for saturation tests. `sat_reason` empty means the
    /// run was not flagged client-saturated.
    fn entry(ef: &str, parallel: i64, rps: f64, p95: f64, sat_reason: &str) -> SearchEntry {
        SearchEntry {
            search_id: 0,
            ef: ef.to_string(),
            parallel,
            results: SearchResults {
                rps,
                p95_time: p95,
                client_saturated: !sat_reason.is_empty(),
                saturation_reason: sat_reason.to_string(),
                ..Default::default()
            },
        }
    }

    #[test]
    fn saturation_warns_when_throughput_stops_scaling() {
        // Same ef, parallel 8→16: QPS rises only ~5% (< 10%) while p95 rises →
        // higher concurrency is not paying off, so a throughput warning fires.
        let entries = vec![
            entry("64", 8, 1000.0, 0.02, ""),
            entry("64", 16, 1050.0, 0.03, ""),
        ];
        let w = saturation_warnings(&entries);
        assert_eq!(w.len(), 1, "got {w:?}");
        assert!(
            w[0].starts_with("throughput saturated: ef=64 parallel 8→16"),
            "got {}",
            w[0]
        );
        // gain 50/1000 = 5%; p95 rose (0.03/0.02 - 1) = 50%.
        assert!(w[0].contains("gained only 5% QPS"), "got {}", w[0]);
        assert!(w[0].contains("p95 rose 50%"), "got {}", w[0]);
        assert!(
            w[0].ends_with("higher concurrency is not paying off"),
            "got {}",
            w[0]
        );
    }

    #[test]
    fn saturation_silent_when_throughput_scales_well() {
        // QPS doubles (1000→2000, gain 100% ≥ 10%) → clean scaling, no warning
        // even though p95 rose.
        let entries = vec![
            entry("64", 8, 1000.0, 0.02, ""),
            entry("64", 16, 2000.0, 0.03, ""),
        ];
        assert!(saturation_warnings(&entries).is_empty());
    }

    #[test]
    fn saturation_no_warn_when_p95_falls() {
        // Gain < 10% but p95 FALLS (0.03→0.02): the `p95_cur > p95_prev` guard is
        // not met, so no throughput warning is emitted.
        let entries = vec![
            entry("64", 8, 1000.0, 0.03, ""),
            entry("64", 16, 1050.0, 0.02, ""),
        ];
        assert!(saturation_warnings(&entries).is_empty());
    }

    #[test]
    fn saturation_warns_on_client_saturated_flag() {
        // A single client-saturated run emits a client-saturation warning
        // (and the throughput loop is skipped with < 2 points).
        let entries = vec![entry("128", 32, 5000.0, 0.05, "client CPU >90%")];
        let w = saturation_warnings(&entries);
        assert_eq!(w.len(), 1, "got {w:?}");
        assert_eq!(
            w[0],
            "client-saturated: ef=128 parallel=32 (client CPU >90%) — QPS/latency reflect the client, not the DB"
        );
    }

    #[test]
    fn saturation_client_warnings_ordered_by_parallel() {
        // Two client-saturated points at parallel 16 and 8 → sorted ascending by
        // parallel, so the parallel=8 warning is emitted first.
        let entries = vec![
            entry("64", 16, 4000.0, 0.04, "oversubscribed"),
            entry("64", 8, 3000.0, 0.03, "cpu"),
        ];
        let w = saturation_warnings(&entries);
        assert_eq!(w.len(), 2, "got {w:?}");
        assert!(w[0].contains("parallel=8"), "first: {}", w[0]);
        assert!(w[1].contains("parallel=16"), "second: {}", w[1]);
    }

    #[test]
    fn test_analyze_precision_performance_picks_best_qps() {
        let entries = vec![
            SearchEntry {
                search_id: 0,
                ef: "64".to_string(),
                parallel: 100,
                results: SearchResults {
                    mean_precision_at_returned: 0.90,
                    rps: 5000.0,
                    p50_time: 0.01,
                    p95_time: 0.02,
                    ..Default::default()
                },
            },
            SearchEntry {
                search_id: 1,
                ef: "128".to_string(),
                parallel: 100,
                results: SearchResults {
                    mean_precision_at_returned: 0.91,
                    rps: 4000.0,
                    p50_time: 0.012,
                    p95_time: 0.025,
                    ..Default::default()
                },
            },
        ];
        let buckets = analyze_precision_performance(&entries);
        // 0.90 rounds to "0.90", 0.91 rounds to "0.91" — separate buckets
        assert_eq!(buckets.len(), 2);
        assert!(buckets[0].precision > buckets[1].precision);
    }

    /// Build a `SearchEntry` with an explicit precision + saturation flag for the
    /// tie-break tests.
    fn prec_entry(precision: f64, rps: f64, saturated: bool) -> SearchEntry {
        SearchEntry {
            search_id: 0,
            ef: "64".to_string(),
            parallel: 1,
            results: SearchResults {
                mean_precision_at_returned: precision,
                rps,
                client_saturated: saturated,
                ..Default::default()
            },
        }
    }

    #[test]
    fn analyze_prefers_non_saturated_even_at_lower_qps() {
        // Same precision bucket (both 0.90). The saturated point has the higher
        // QPS, but the headline must NOT be a client-bound point: the
        // non-saturated (lower-QPS) point wins.
        for order in [false, true] {
            // Test both insertion orders so the tie-break isn't order-dependent.
            let entries = if order {
                vec![
                    prec_entry(0.90, 6000.0, true),
                    prec_entry(0.90, 5000.0, false),
                ]
            } else {
                vec![
                    prec_entry(0.90, 5000.0, false),
                    prec_entry(0.90, 6000.0, true),
                ]
            };
            let buckets = analyze_precision_performance(&entries);
            assert_eq!(buckets.len(), 1);
            assert!(!buckets[0].saturated, "order={order}");
            assert!(
                (buckets[0].qps - 5000.0).abs() < 1e-9,
                "order={order}, qps={}",
                buckets[0].qps
            );
        }
    }

    #[test]
    fn analyze_falls_back_to_saturated_when_sole_candidate() {
        // Only saturated points in the bucket → the higher-QPS one is kept.
        let entries = vec![
            prec_entry(0.90, 4000.0, true),
            prec_entry(0.90, 6000.0, true),
        ];
        let buckets = analyze_precision_performance(&entries);
        assert_eq!(buckets.len(), 1);
        assert!(buckets[0].saturated);
        assert!((buckets[0].qps - 6000.0).abs() < 1e-9);
    }

    #[test]
    fn analyze_same_saturation_class_keeps_higher_qps() {
        // Both non-saturated → higher QPS wins.
        let entries = vec![
            prec_entry(0.90, 3000.0, false),
            prec_entry(0.90, 7000.0, false),
        ];
        let buckets = analyze_precision_performance(&entries);
        assert_eq!(buckets.len(), 1);
        assert!((buckets[0].qps - 7000.0).abs() < 1e-9);
    }

    #[test]
    fn ratio_sort_key_branches() {
        // "search" is the sentinel and sorts first (before any real ratio).
        assert_eq!(ratio_sort_key("search"), -1.0);
        // "U:S" → U/S.
        assert_eq!(ratio_sort_key("2:1"), 2.0);
        assert_eq!(ratio_sort_key("1:2"), 0.5);
        // Denominator 0 → guarded, falls through to 0.0 (not inf/NaN).
        assert_eq!(ratio_sort_key("1:0"), 0.0);
        // Unparseable ratios → 0.0.
        assert_eq!(ratio_sort_key("garbage"), 0.0);
        assert_eq!(ratio_sort_key("a:b"), 0.0);
        // Wrong arity (not exactly two parts) → 0.0.
        assert_eq!(ratio_sort_key("1:2:3"), 0.0);
        // Result is always finite (NaN/inf guard holds).
        for k in ["search", "2:1", "1:0", "garbage", "a:b", "1:2:3"] {
            assert!(ratio_sort_key(k).is_finite(), "key={k}");
        }
    }

    #[test]
    fn ratio_sort_key_orders_search_first() {
        let mut keys = vec!["4:1", "search", "1:1"];
        keys.sort_by(|a, b| ratio_sort_key(a).partial_cmp(&ratio_sort_key(b)).unwrap());
        assert_eq!(keys, vec!["search", "1:1", "4:1"]);
    }

    // ── concurrency-curve (saturation) tests ────────────────────────────────

    #[test]
    fn concurrency_curve_finds_peak_and_knee() {
        // QPS rises then falls: peak at parallel 4, and 4 < highest (8) → the
        // throughput knee is at 4.
        let entries = vec![
            entry("64", 1, 100.0, 0.01, ""),
            entry("64", 2, 190.0, 0.02, ""),
            entry("64", 4, 300.0, 0.03, ""),
            entry("64", 8, 290.0, 0.06, ""),
        ];
        let curves = concurrency_curves(&entries);
        assert_eq!(curves.len(), 1);
        let c = &curves[0];
        assert_eq!(c.points.len(), 4);
        // sorted ascending by parallel
        assert_eq!(
            c.points.iter().map(|p| p.parallel).collect::<Vec<_>>(),
            vec![1, 2, 4, 8]
        );
        assert!((c.max_qps - 300.0).abs() < 1e-9);
        assert_eq!(c.max_qps_parallel, 4);
        assert_eq!(c.knee_parallel, Some(4));
    }

    #[test]
    fn concurrency_curve_still_climbing_has_no_knee() {
        // Monotonically rising QPS: peak is at the HIGHEST level tested, so the
        // knee is beyond the swept range → None.
        let entries = vec![
            entry("64", 1, 100.0, 0.01, ""),
            entry("64", 2, 200.0, 0.01, ""),
            entry("64", 4, 300.0, 0.01, ""),
            entry("64", 8, 400.0, 0.01, ""),
        ];
        let c = &concurrency_curves(&entries)[0];
        assert!((c.max_qps - 400.0).abs() < 1e-9);
        assert_eq!(c.max_qps_parallel, 8);
        assert_eq!(c.knee_parallel, None);
    }

    #[test]
    fn concurrency_curve_requires_two_levels() {
        // A single concurrency level is not a sweep → no curve.
        let entries = vec![entry("64", 4, 300.0, 0.03, "")];
        assert!(concurrency_curves(&entries).is_empty());
    }

    #[test]
    fn concurrency_curve_groups_by_ef() {
        // Two operating points (ef 64 and ef 128), each swept over 2 levels →
        // two independent curves.
        let entries = vec![
            entry("64", 1, 100.0, 0.01, ""),
            entry("64", 2, 150.0, 0.02, ""),
            entry("128", 1, 80.0, 0.01, ""),
            entry("128", 2, 120.0, 0.02, ""),
        ];
        let curves = concurrency_curves(&entries);
        assert_eq!(curves.len(), 2);
        assert_eq!(
            curves.iter().map(|c| c.ef.as_str()).collect::<Vec<_>>(),
            vec!["128", "64"]
        );
    }

    #[test]
    fn concurrency_curve_dedups_repeated_level_preferring_nonsaturated() {
        // Same (ef, parallel) twice: a saturated point with higher QPS and a
        // clean point with lower QPS. The clean one must win so the curve is not
        // built from a client-bound data point.
        let entries = vec![
            entry("64", 1, 100.0, 0.01, ""),
            entry("64", 2, 400.0, 0.02, "client cpu"), // saturated, higher qps
            entry("64", 2, 250.0, 0.02, ""),           // clean, lower qps
        ];
        let c = &concurrency_curves(&entries)[0];
        assert_eq!(c.points.len(), 2);
        let p2 = c.points.iter().find(|p| p.parallel == 2).unwrap();
        assert!(!p2.saturated, "clean point must win over saturated one");
        assert!((p2.qps - 250.0).abs() < 1e-9);
    }

    #[test]
    fn concurrency_curve_excludes_filter_only_runs() {
        // Filter-only runs carry mean_precision_at_returned == -1.0 (sentinel) and have no
        // recall/precision curve — they must not appear in the saturation curve.
        let mut a = entry("64", 1, 100.0, 0.01, "");
        let mut b = entry("64", 2, 200.0, 0.01, "");
        a.results.mean_precision_at_returned = -1.0;
        b.results.mean_precision_at_returned = -1.0;
        assert!(concurrency_curves(&[a, b]).is_empty());
    }
}
