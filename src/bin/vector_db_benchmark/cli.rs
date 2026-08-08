//! CLI argument parsing for vector-db-benchmark.
//!
//! Mirrors the Python run.py CLI interface.

use clap::Parser;

/// Rust implementation of vector-db-benchmark.
/// Supports redis and vectorsets engines.
#[derive(Parser, Debug, Clone)]
#[command(name = "vector-db-benchmark")]
#[command(version, about = "Run vector database benchmarks", long_about = None)]
pub struct Args {
    /// Engine patterns to run (supports wildcards like "redis*", repeatable)
    #[arg(long, default_value = "*")]
    pub engines: Vec<String>,

    /// Path to JSON file containing engine configurations
    #[arg(long)]
    pub engines_file: Option<String>,

    /// Run even when some `experiments/configurations/*.json` file failed to
    /// load. Off by default: serde rejects a whole file on one bad entry, so
    /// under a wildcard `--engines` (the default is `*`) the sweep would just get
    /// smaller and still exit 0, publishing a truncated peak QPS and Pareto
    /// frontier. With this flag the run proceeds and records the offending files
    /// under `skipped_config_files` in every summary JSON it writes (#239).
    #[arg(long, default_value = "false")]
    pub allow_partial_configs: bool,

    /// Dataset patterns to run (supports wildcards, repeatable)
    #[arg(long, default_value = "*")]
    pub datasets: Vec<String>,

    /// Filter by parallel thread counts (comma-separated)
    #[arg(long, value_delimiter = ',')]
    pub parallels: Vec<i32>,

    /// Redis host
    #[arg(long, default_value = "localhost")]
    pub host: String,

    /// Skip upload phase.
    ///
    /// Means "the server already holds the corpus I want": the configure phase
    /// is NOT run (it is destructive on almost every engine — see #238) and the
    /// runner instead verifies, against the live server, that the corpus is
    /// present and complete before measuring anything.
    #[arg(long, default_value = "false")]
    pub skip_upload: bool,

    /// Downgrade the `--skip-upload` reuse precondition from a hard error to a
    /// warning (issues #238, #290).
    ///
    /// By default a `--skip-upload` run aborts when the engine holds fewer rows
    /// than the dataset declares — including zero, i.e. a missing index — because
    /// recall is scored against ground truth for the FULL corpus and a short
    /// corpus therefore publishes a wrong number under a config name that claims
    /// otherwise. It also aborts when the probe for the server-side count FAILS
    /// (restrictive ACLs, unreachable server), and when the dataset's own
    /// expected row count cannot be determined even after the corpus has been
    /// fetched — an unmeasurable layout with no `vector_count`, or a corpus that
    /// is neither on this machine nor downloadable (#290).
    ///
    /// It does NOT abort when an engine simply has no row-count probe wired up
    /// (Chroma, Milvus, Weaviate, Turbopuffer, Vertex): that prints a note and
    /// runs, with or without this flag. Set this to run anyway in the cases that
    /// do abort; the waiver is recorded in the result file under
    /// `params.corpus_reuse`.
    #[arg(long, default_value = "false")]
    pub allow_partial_corpus: bool,

    /// Skip search phase
    #[arg(long, default_value = "false")]
    pub skip_search: bool,

    /// Keep the configured index and uploaded data after the experiment
    #[arg(long, default_value = "false")]
    pub keep_data: bool,

    /// On a multi-config sweep with `--keep-data`, free each config's data before
    /// the next runs so only ONE config's corpus is resident at a time (keeping
    /// just the last config's). Prevents the per-config keyspaces from
    /// accumulating and OOMing a memory-bounded server (#184). Off by default, so
    /// `--keep-data` still keeps every config's data coexisting (the default,
    /// needed for `--skip-upload` reuse across configs).
    #[arg(long, default_value = "false")]
    pub reset_between_configs: bool,

    /// Skip if results already exist (accepts an optional value, e.g.
    /// `--skip-if-exists false`; bare `--skip-if-exists` means true)
    #[arg(
        long,
        num_args = 0..=1,
        default_value_t = true,
        default_missing_value = "true",
        action = clap::ArgAction::Set
    )]
    pub skip_if_exists: bool,

    /// Exit on first error (accepts an optional value, e.g.
    /// `--exit-on-error false`; bare `--exit-on-error` means true)
    #[arg(
        long,
        num_args = 0..=1,
        default_value_t = true,
        default_missing_value = "true",
        action = clap::ArgAction::Set
    )]
    pub exit_on_error: bool,

    /// Fail the run if any search query was dropped, instead of only warning.
    ///
    /// Dropped queries never reach the latency/recall vectors, so the reported
    /// figures cover the surviving subset only — and because a server sheds load
    /// exactly when it is busy, the survivors are the cheaper queries and recall
    /// is biased upward. `failed_queries` is always recorded and warned about
    /// (see `SearchResults::failed_queries`); this makes it fatal for runs whose
    /// numbers are going to be published, so a partial result cannot be quoted by
    /// accident. Off by default: a partial run still carries the strongest
    /// available overload signal, and discarding it loses that.
    ///
    /// Applies to every engine. Results files are still written before the run
    /// fails, so the evidence survives.
    #[arg(
        long,
        num_args = 0..=1,
        default_value_t = false,
        default_missing_value = "true",
        action = clap::ArgAction::Set
    )]
    pub fail_on_dropped_queries: bool,

    /// Overall wall-clock budget in seconds: stop launching new experiments once
    /// total elapsed exceeds this (any in-flight experiment finishes). 0 disables.
    #[arg(long, default_value = "86400.0")]
    pub timeout: f64,

    /// Per-search-point wall-clock watchdog in seconds. A single search/mixed
    /// call that runs longer than this (e.g. a proxy/connection-pool stall at
    /// high `parallel`) is aborted with a diagnostic instead of hanging the whole
    /// sweep silently. Progress is logged while a point is in flight. 0 disables
    /// (default) — behavior is then unchanged.
    #[arg(long, default_value = "0.0")]
    pub search_timeout: f64,

    /// Upload start index
    #[arg(long, default_value = "0")]
    pub upload_start_idx: usize,

    /// Upload end index (-1 means all)
    #[arg(long, default_value = "-1")]
    pub upload_end_idx: i64,

    /// Number of queries to run (-1 means all)
    #[arg(long, default_value = "-1")]
    pub queries: i64,

    /// Fixed offered query rate. 0 keeps the existing closed-loop behavior.
    /// Open-loop mode is currently supported by Redis and Vertex.
    #[arg(long, default_value = "0.0")]
    pub target_qps: f64,

    /// Measured search duration in seconds. With --target-qps this is open-loop;
    /// without it Redis and Vertex run unrestricted closed-loop for this long.
    #[arg(long, default_value = "0.0")]
    pub search_duration: f64,

    /// Warm-up duration before each measured open-loop search configuration.
    #[arg(long, default_value = "0.0")]
    pub warmup_seconds: f64,

    /// Drop an open-loop request when dispatch is this many milliseconds late.
    #[arg(long, default_value = "1000.0")]
    pub max_lateness_ms: f64,

    /// Repeat each measured search config this many times and report the
    /// best-RPS run (warm best-of, matching v0's REPETITIONS). The first run is
    /// often cold (OS page cache / index warm-up); best-of discards it. Set 1 to
    /// disable. Also honored via the REPETITIONS environment variable.
    #[arg(long, env = "REPETITIONS", default_value = "3")]
    pub repetitions: usize,

    /// Filter search experiments by ef runtime values (comma-separated)
    #[arg(long, value_delimiter = ',')]
    pub ef_runtime: Vec<i64>,

    /// Describe available options: 'datasets' or 'engines'
    #[arg(long)]
    pub describe: Option<String>,

    /// Instead of benchmarking, render a QPS-vs-precision trade-off chart (SVG)
    /// from existing `*-summary.json` files in results/, filtered by --engines
    /// and --datasets. One colored series per engine. Value is the output path.
    #[arg(long, value_name = "OUTPUT.svg")]
    pub plot: Option<String>,

    /// Show detailed information when using --describe
    #[arg(long, short)]
    pub verbose: bool,

    /// Mixed benchmark: update-to-search ratio (e.g., "1:10" = 1 update per 10 searches).
    /// Can be specified multiple times. "0:S" means pure search.
    #[arg(long)]
    pub update_search_ratio: Vec<String>,

    /// Skip vector indexing: upload vectors but don't index them, run filter-only queries.
    /// Collapses all M/EF variants of the same engine into a single "<engine>-no-vector" experiment.
    #[arg(long, default_value = "false")]
    pub skip_vector_index: bool,

    /// Also dump the FULL raw per-query arrays (precisions, recalls, mrrs, ndcgs,
    /// latencies, and mixed update_latencies) into each result file. Off by
    /// default: results carry only the compact HDR/quality digests, which shrink
    /// large-run files ~1000x while keeping every percentile re-derivable. Enable
    /// this only for full-fidelity archival of a specific run.
    #[arg(long, default_value = "false")]
    pub dump_raw_latencies: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(extra: &[&str]) -> Args {
        let mut argv = vec!["vector-db-benchmark", "--engines", "x", "--datasets", "y"];
        argv.extend_from_slice(extra);
        Args::try_parse_from(argv).expect("args should parse")
    }

    // Regression: the default-true bool flags must accept an explicit value
    // (`--skip-if-exists false`, as the mongodb integration test invokes it),
    // an `=` value, a bare flag, and default to true when omitted. Previously
    // they were SetTrue flags that rejected any value and could never be false.
    #[test]
    fn skip_if_exists_accepts_all_forms() {
        assert!(parse(&[]).skip_if_exists, "omitted → true");
        assert!(parse(&["--skip-if-exists"]).skip_if_exists, "bare → true");
        assert!(
            !parse(&["--skip-if-exists", "false"]).skip_if_exists,
            "space value → false"
        );
        assert!(
            !parse(&["--skip-if-exists=false"]).skip_if_exists,
            "= value → false"
        );
        assert!(parse(&["--skip-if-exists", "true"]).skip_if_exists);
    }

    #[test]
    fn exit_on_error_accepts_all_forms() {
        assert!(parse(&[]).exit_on_error, "omitted → true");
        assert!(parse(&["--exit-on-error"]).exit_on_error, "bare → true");
        assert!(!parse(&["--exit-on-error", "false"]).exit_on_error);
        assert!(!parse(&["--exit-on-error=false"]).exit_on_error);
    }

    #[test]
    fn fail_on_dropped_queries_defaults_off_and_accepts_all_forms() {
        // Default OFF is load-bearing: a partial run still carries the strongest
        // available overload signal, and flipping this default would turn every
        // load-shedding run into no data at all.
        assert!(
            !parse(&[]).fail_on_dropped_queries,
            "omitted → false (warn, don't fail)"
        );
        assert!(parse(&["--fail-on-dropped-queries"]).fail_on_dropped_queries);
        assert!(parse(&["--fail-on-dropped-queries", "true"]).fail_on_dropped_queries);
        assert!(!parse(&["--fail-on-dropped-queries=false"]).fail_on_dropped_queries);
    }

    #[test]
    fn parses_open_loop_options() {
        let args = parse(&[
            "--target-qps",
            "1500",
            "--search-duration",
            "300",
            "--warmup-seconds",
            "10",
            "--max-lateness-ms",
            "250",
        ]);
        assert_eq!(args.target_qps, 1500.0);
        assert_eq!(args.search_duration, 300.0);
        assert_eq!(args.warmup_seconds, 10.0);
        assert_eq!(args.max_lateness_ms, 250.0);
    }

    // Per-search watchdog (#151-5): opt-in, defaults to 0.0 (disabled) so the
    // unchanged behavior is preserved, and parses an explicit value.
    #[test]
    fn search_timeout_parses() {
        assert_eq!(parse(&[]).search_timeout, 0.0, "omitted → disabled");
        assert_eq!(parse(&["--search-timeout", "300"]).search_timeout, 300.0);
    }

    // Raw-array dump (#151-8): opt-in, defaults to false so result files carry
    // only the compact digests, and parses when explicitly requested.
    #[test]
    fn dump_raw_latencies_parses() {
        assert!(
            !parse(&[]).dump_raw_latencies,
            "omitted → false (digests only)"
        );
        assert!(parse(&["--dump-raw-latencies"]).dump_raw_latencies);
    }

    // `--describe datasets|engines` is what the docker-build smoke test exercises;
    // pin that it parses (and is absent by default) so the dispatch in main.rs
    // always receives the expected value.
    #[test]
    fn describe_option_parses() {
        assert_eq!(parse(&[]).describe, None, "omitted → None");
        assert_eq!(
            parse(&["--describe", "datasets"]).describe.as_deref(),
            Some("datasets")
        );
        assert_eq!(
            parse(&["--describe", "engines"]).describe.as_deref(),
            Some("engines")
        );
    }
}
