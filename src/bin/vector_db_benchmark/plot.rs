//! Chart export: render a QPS-vs-precision trade-off plot (SVG) from the
//! `*-summary.json` result files, one colored series per engine.
//!
//! This mirrors the tradeoff plots the upstream Python tool publishes
//! (precision on X, throughput on Y — "up and to the right is better"), but
//! writes a self-contained SVG with no external plotting dependency.

use std::fs;
use std::path::PathBuf;

use serde_json::Value;

use crate::cli::Args;
use crate::config::{matches_pattern, project_root};

/// One (precision, qps) sample on an engine's trade-off curve.
#[derive(Clone, Copy)]
struct Point {
    precision: f64,
    qps: f64,
}

/// A named, colored trade-off curve (one engine, one dataset).
struct Series {
    engine: String,
    dataset: String,
    label: String,
    points: Vec<Point>,
}

/// Entry point for `--plot`: read matching summaries and write an SVG.
pub fn export_chart(args: &Args) -> Result<(), String> {
    let out_path = args
        .plot
        .as_ref()
        .ok_or_else(|| "internal: --plot not set".to_string())?;

    let results_dir = project_root().join("results");
    let pattern = results_dir
        .join("*-summary.json")
        .to_string_lossy()
        .to_string();
    let files: Vec<PathBuf> = glob::glob(&pattern)
        .map_err(|e| format!("bad results glob: {}", e))?
        .filter_map(Result::ok)
        .collect();

    if files.is_empty() {
        return Err(format!(
            "No *-summary.json files found in {}. Run a benchmark first.",
            results_dir.display()
        ));
    }

    // Parse each summary, filter by --engines/--datasets, build one series each.
    let mut series: Vec<Series> = Vec::new();
    let mut datasets_seen: Vec<String> = Vec::new();
    for f in &files {
        let Ok(text) = fs::read_to_string(f) else {
            continue;
        };
        let Ok(json) = serde_json::from_str::<Value>(&text) else {
            continue;
        };
        let engine = json.get("engine").and_then(|v| v.as_str()).unwrap_or("");
        let dataset = json.get("dataset").and_then(|v| v.as_str()).unwrap_or("");
        if engine.is_empty() || dataset.is_empty() {
            continue;
        }
        let engine_match = args.engines.iter().any(|p| matches_pattern(engine, p));
        let dataset_match = args.datasets.iter().any(|p| matches_pattern(dataset, p));
        if !engine_match || !dataset_match {
            continue;
        }

        let mut points = parse_points(&json);
        if points.is_empty() {
            continue;
        }
        // Sort by precision so the connecting line is monotone in X.
        points.sort_by(|a, b| a.precision.partial_cmp(&b.precision).unwrap());
        if !datasets_seen.contains(&dataset.to_string()) {
            datasets_seen.push(dataset.to_string());
        }
        series.push(Series {
            engine: engine.to_string(),
            dataset: dataset.to_string(),
            label: String::new(), // filled below once we know how many datasets
            points,
        });
    }

    if series.is_empty() {
        return Err(format!(
            "No summaries matched engines={:?} datasets={:?}.",
            args.engines, args.datasets
        ));
    }

    // Legend label: just the engine when a single dataset is plotted, otherwise
    // "engine (dataset)" so the series can be told apart.
    let multi = datasets_seen.len() > 1;
    for s in &mut series {
        s.label = if multi {
            format!("{} ({})", s.engine, s.dataset)
        } else {
            s.engine.clone()
        };
    }

    let title = if datasets_seen.len() == 1 {
        format!("QPS vs Precision — {}", datasets_seen[0])
    } else {
        format!("QPS vs Precision — {} datasets", datasets_seen.len())
    };

    let svg = render_svg(&title, &series);
    fs::write(out_path, svg).map_err(|e| format!("failed to write {}: {}", out_path, e))?;
    println!(
        "Wrote trade-off chart: {} ({} engine series across {} dataset(s))",
        out_path,
        series.len(),
        datasets_seen.len()
    );
    Ok(())
}

/// Extract (precision, qps) points from a summary's `precision_summary` map,
/// falling back to `search_results` if that's absent.
fn parse_points(json: &Value) -> Vec<Point> {
    // A summary without `metrics_schema_version` predates the #217 rename — or was
    // written by upstream qdrant/vector-db-benchmark, whose `precision_summary`
    // keys and `mean_precisions` values are recall@top rather than our
    // precision-at-returned. The two shapes are indistinguishable, so the X axis
    // cannot be honestly labelled for such a file; say so instead of charting it
    // silently under our label.
    if json.get("metrics_schema_version").is_none() {
        eprintln!(
            "WARNING: summary has no `metrics_schema_version` — its precision values are either \
             ours from before the #217 rename (precision = hits / results returned) or upstream's \
             (recall@top = hits / top). They are plotted on the same axis; re-run the benchmark to \
             get an unambiguous summary."
        );
    }
    if let Some(map) = json.get("precision_summary").and_then(|v| v.as_object()) {
        let mut pts: Vec<Point> = map
            .iter()
            .filter_map(|(k, v)| {
                let precision = k.parse::<f64>().ok()?;
                let qps = v.get("QPS").and_then(|q| q.as_f64())?;
                Some(Point { precision, qps })
            })
            .collect();
        pts.retain(|p| p.qps.is_finite() && p.precision.is_finite());
        if !pts.is_empty() {
            return pts;
        }
    }
    // Fallback: raw search_results entries.
    //
    // `mean_precisions` is read only for backwards compatibility with summary
    // files written before schema version 2 (#217), where our precision shipped
    // under that name. New files use `mean_precision_at_returned`. Note that a
    // summary produced by upstream qdrant/vector-db-benchmark also has a
    // `mean_precisions` field, but it holds recall@top — the two curves are not
    // interchangeable, which is exactly why the key was renamed.
    json.get("search_results")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|e| {
                    let precision = e
                        .get("mean_precision_at_returned")
                        .or_else(|| e.get("mean_precisions"))
                        .and_then(|v| v.as_f64())?;
                    let qps = e.get("rps").and_then(|v| v.as_f64())?;
                    Some(Point { precision, qps })
                })
                .filter(|p| p.qps.is_finite() && p.precision.is_finite())
                .collect()
        })
        .unwrap_or_default()
}

/// A fixed, visually distinct palette (color-blind friendly Okabe-Ito-ish).
const PALETTE: &[&str] = &[
    "#0072b2", "#d55e00", "#009e73", "#cc79a7", "#e69f00", "#56b4e9", "#f0e442", "#000000",
    "#8b4513", "#7f7f7f",
];

/// Render the trade-off plot as a standalone SVG document.
fn render_svg(title: &str, series: &[Series]) -> String {
    let (w, h) = (900.0_f64, 560.0_f64);
    let (ml, mr, mt, mb) = (70.0_f64, 210.0_f64, 50.0_f64, 60.0_f64);
    let plot_w = w - ml - mr;
    let plot_h = h - mt - mb;

    // Axis ranges. X = precision (pad a little below the min, cap at 1.0);
    // Y = QPS (0 to max, padded).
    let mut min_prec = f64::INFINITY;
    let mut max_prec = f64::NEG_INFINITY;
    let mut max_qps = 0.0_f64;
    for s in series {
        for p in &s.points {
            min_prec = min_prec.min(p.precision);
            max_prec = max_prec.max(p.precision);
            max_qps = max_qps.max(p.qps);
        }
    }
    if !min_prec.is_finite() {
        min_prec = 0.0;
        max_prec = 1.0;
    }
    let x_lo = (min_prec - 0.02).max(0.0);
    let x_hi = (max_prec + 0.01).min(1.0).max(x_lo + 1e-6);
    let y_hi = if max_qps > 0.0 { max_qps * 1.08 } else { 1.0 };

    let sx = |p: f64| ml + (p - x_lo) / (x_hi - x_lo) * plot_w;
    let sy = |q: f64| mt + plot_h - (q / y_hi) * plot_h;

    let mut svg = String::new();
    svg.push_str(&format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" font-family="sans-serif">"#
    ));
    svg.push_str(r#"<rect width="100%" height="100%" fill="white"/>"#);
    svg.push_str(&format!(
        r#"<text x="{}" y="28" font-size="18" font-weight="bold" text-anchor="middle">{}</text>"#,
        ml + plot_w / 2.0,
        xml_escape(title)
    ));

    // Gridlines + Y ticks (5) and X ticks (6).
    for i in 0..=5 {
        let q = y_hi * i as f64 / 5.0;
        let y = sy(q);
        svg.push_str(&format!(
            r##"<line x1="{ml}" y1="{y:.1}" x2="{}" y2="{y:.1}" stroke="#e6e6e6"/>"##,
            ml + plot_w
        ));
        svg.push_str(&format!(
            r##"<text x="{:.1}" y="{:.1}" font-size="11" text-anchor="end" fill="#555">{}</text>"##,
            ml - 8.0,
            y + 4.0,
            human_qps(q)
        ));
    }
    for i in 0..=6 {
        let p = x_lo + (x_hi - x_lo) * i as f64 / 6.0;
        let x = sx(p);
        svg.push_str(&format!(
            r##"<line x1="{x:.1}" y1="{mt}" x2="{x:.1}" y2="{:.1}" stroke="#f0f0f0"/>"##,
            mt + plot_h
        ));
        svg.push_str(&format!(
            r##"<text x="{x:.1}" y="{:.1}" font-size="11" text-anchor="middle" fill="#555">{:.3}</text>"##,
            mt + plot_h + 20.0,
            p
        ));
    }

    // Axis lines + labels.
    svg.push_str(&format!(
        r##"<line x1="{ml}" y1="{}" x2="{}" y2="{}" stroke="#333"/>"##,
        mt + plot_h,
        ml + plot_w,
        mt + plot_h
    ));
    svg.push_str(&format!(
        r##"<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{}" stroke="#333"/>"##,
        mt + plot_h
    ));
    svg.push_str(&format!(
        // The X axis is our precision-at-returned (hits / results returned), NOT
        // upstream's recall@top — the axis used to be labelled "recall@k", which
        // is a different quantity (#217).
        r#"<text x="{}" y="{}" font-size="13" text-anchor="middle">Precision@returned (hits / results returned)</text>"#,
        ml + plot_w / 2.0,
        h - 18.0
    ));
    svg.push_str(&format!(
        r#"<text x="18" y="{}" font-size="13" text-anchor="middle" transform="rotate(-90 18 {})">Queries / sec</text>"#,
        mt + plot_h / 2.0,
        mt + plot_h / 2.0
    ));

    // Series: polyline + points, plus a legend entry.
    let legend_x = ml + plot_w + 24.0;
    for (i, s) in series.iter().enumerate() {
        let color = PALETTE[i % PALETTE.len()];
        let pts: String = s
            .points
            .iter()
            .map(|p| format!("{:.1},{:.1}", sx(p.precision), sy(p.qps)))
            .collect::<Vec<_>>()
            .join(" ");
        svg.push_str(&format!(
            r#"<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2" opacity="0.9"/>"#
        ));
        for p in &s.points {
            svg.push_str(&format!(
                r#"<circle cx="{:.1}" cy="{:.1}" r="3.5" fill="{color}"/>"#,
                sx(p.precision),
                sy(p.qps)
            ));
        }
        let ly = mt + 6.0 + i as f64 * 20.0;
        svg.push_str(&format!(
            r#"<line x1="{legend_x}" y1="{ly}" x2="{}" y2="{ly}" stroke="{color}" stroke-width="3"/>"#,
            legend_x + 22.0
        ));
        svg.push_str(&format!(
            r#"<text x="{}" y="{:.1}" font-size="12">{}</text>"#,
            legend_x + 30.0,
            ly + 4.0,
            xml_escape(&s.label)
        ));
    }

    svg.push_str("</svg>\n");
    svg
}

fn human_qps(q: f64) -> String {
    if q >= 1000.0 {
        format!("{:.1}k", q / 1000.0)
    } else {
        format!("{:.0}", q)
    }
}

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_points_from_precision_summary() {
        let json = serde_json::json!({
            "engine": "e", "dataset": "d",
            "precision_summary": {
                "0.90": {"QPS": 5000.0, "P50 (ms)": 1.0},
                "0.99": {"QPS": 2000.0, "P50 (ms)": 2.0}
            }
        });
        let mut pts = parse_points(&json);
        pts.sort_by(|a, b| a.precision.partial_cmp(&b.precision).unwrap());
        assert_eq!(pts.len(), 2);
        assert!((pts[0].precision - 0.90).abs() < 1e-9 && (pts[0].qps - 5000.0).abs() < 1e-9);
        assert!((pts[1].precision - 0.99).abs() < 1e-9);
    }

    #[test]
    fn parse_points_falls_back_to_search_results() {
        let json = serde_json::json!({
            "metrics_schema_version": 2,
            "search_results": [
                {"mean_precision_at_returned": 0.8, "rps": 100.0},
                {"mean_precision_at_returned": 0.95, "rps": 60.0}
            ]
        });
        assert_eq!(parse_points(&json).len(), 2);
    }

    /// Summaries written before the #217 rename carry our precision under the
    /// legacy `mean_precisions` key; charting them must keep working.
    #[test]
    fn parse_points_reads_legacy_mean_precisions_key() {
        let json = serde_json::json!({
            "search_results": [
                {"mean_precisions": 0.8, "rps": 100.0},
                {"mean_precisions": 0.95, "rps": 60.0}
            ]
        });
        let mut pts = parse_points(&json);
        pts.sort_by(|a, b| a.precision.partial_cmp(&b.precision).unwrap());
        assert_eq!(pts.len(), 2);
        assert!((pts[0].precision - 0.8).abs() < 1e-9);
    }

    /// A file carrying both keys must prefer the new one (a hand-merged or
    /// re-written summary must not silently fall back to the legacy value).
    #[test]
    fn parse_points_prefers_new_key_over_legacy() {
        let json = serde_json::json!({
            "search_results": [
                {"mean_precision_at_returned": 0.42, "mean_precisions": 0.99, "rps": 10.0}
            ]
        });
        let pts = parse_points(&json);
        assert_eq!(pts.len(), 1);
        assert!(
            (pts[0].precision - 0.42).abs() < 1e-9,
            "{}",
            pts[0].precision
        );
    }

    #[test]
    fn render_svg_is_wellformed_and_contains_series() {
        let series = vec![Series {
            engine: "redis <x>".into(),
            dataset: "d".into(),
            label: "redis <x>".into(),
            points: vec![
                Point {
                    precision: 0.9,
                    qps: 5000.0,
                },
                Point {
                    precision: 0.99,
                    qps: 2000.0,
                },
            ],
        }];
        let svg = render_svg("QPS vs Precision — d", &series);
        assert!(svg.starts_with("<svg"));
        assert!(svg.trim_end().ends_with("</svg>"));
        assert!(svg.contains("<polyline"));
        // Legend label is XML-escaped.
        assert!(svg.contains("redis &lt;x&gt;"));
    }

    #[test]
    fn empty_series_still_renders() {
        let svg = render_svg("empty", &[]);
        assert!(svg.contains("</svg>"));
    }

    #[test]
    fn human_qps_k_suffix_and_plain() {
        // >= 1000 → one-decimal "k"; < 1000 → integer, no suffix.
        assert_eq!(human_qps(0.0), "0");
        assert_eq!(human_qps(500.0), "500");
        assert_eq!(human_qps(999.0), "999");
        assert_eq!(human_qps(999.4), "999"); // rounds down to integer
        assert_eq!(human_qps(1000.0), "1.0k");
        assert_eq!(human_qps(1500.0), "1.5k");
        assert_eq!(human_qps(12_345.0), "12.3k");
    }

    #[test]
    fn parse_points_filters_non_finite_precision_keys() {
        // Precision keys "NaN"/"inf" parse to non-finite f64 and must be dropped
        // by the `is_finite()` retain, leaving only the real 0.90 point.
        let json = serde_json::json!({
            "precision_summary": {
                "0.90": {"QPS": 5000.0},
                "NaN":  {"QPS": 100.0},
                "inf":  {"QPS": 200.0}
            }
        });
        let pts = parse_points(&json);
        assert_eq!(pts.len(), 1);
        assert!((pts[0].precision - 0.90).abs() < 1e-9);
        assert!(pts
            .iter()
            .all(|p| p.precision.is_finite() && p.qps.is_finite()));
    }

    #[test]
    fn parse_points_skips_entries_missing_qps() {
        // A precision_summary entry without a QPS field is dropped by filter_map;
        // if that empties the map, parse falls back to search_results.
        let json = serde_json::json!({
            "precision_summary": {
                "0.90": {"P50 (ms)": 1.0}
            },
            "search_results": [
                {"mean_precisions": 0.8, "rps": 100.0}
            ]
        });
        let pts = parse_points(&json);
        assert_eq!(pts.len(), 1);
        assert!((pts[0].precision - 0.8).abs() < 1e-9);
    }

    #[test]
    fn parse_points_empty_when_no_usable_data() {
        assert!(parse_points(&serde_json::json!({})).is_empty());
    }
}
