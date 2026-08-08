//! Experiment runner - orchestrates benchmark runs.
//!
//! Mirrors Python v0/engine/base_client/client.py run_experiment()

use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use chrono::Local;
use indicatif::{ProgressBar, ProgressStyle};
use serde_json::json;

use crate::cli::Args;
use crate::config::{
    self, matches_pattern, project_root, read_dataset_configs, InnerSearchParams, SearchParams,
};
use crate::dataset::Dataset;
use crate::engine::{create_engine, CorpusCount, Engine, UpdateSearchRatio};
use crate::summary::{self, SearchEntry};

/// Results directory
fn results_dir() -> PathBuf {
    let dir = project_root().join("results");
    fs::create_dir_all(&dir).ok();
    dir
}

/// Run all matching experiments
/// Run one search/mixed call under a per-point wall-clock watchdog (#151-5).
///
/// `f` executes on the CURRENT thread — so the timed measurement path and its
/// fidelity are untouched. A monitor thread only watches the clock: it logs
/// progress every 60s while the point is in flight and, if the point exceeds
/// `timeout_secs`, prints a diagnostic naming the stuck point and aborts the
/// process (rather than letting one hung search — e.g. connection-pool
/// exhaustion at high `parallel` — stall the whole sweep silently). A
/// `timeout_secs <= 0` disables the watchdog entirely (behavior unchanged).
fn run_with_search_watchdog<T>(timeout_secs: f64, label: &str, f: impl FnOnce() -> T) -> T {
    if !timeout_secs.is_finite() || timeout_secs <= 0.0 {
        return f();
    }
    let (tx, rx) = std::sync::mpsc::channel::<()>();
    let label = label.to_string();
    let watchdog = std::thread::spawn(move || {
        let start = Instant::now();
        loop {
            match rx.recv_timeout(Duration::from_secs(60)) {
                // `f` finished (tx dropped) → stop watching promptly.
                Ok(()) | Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    let secs = start.elapsed().as_secs_f64();
                    if secs >= timeout_secs {
                        eprintln!(
                            "\n✗ WATCHDOG: search point '{}' exceeded --search-timeout {:.0}s with no \
                             result — likely a proxy/connection-pool stall (parallel exceeding server \
                             capacity). Aborting; reduce parallel or raise --search-timeout.",
                            label, timeout_secs
                        );
                        std::process::exit(3);
                    }
                    eprintln!(
                        "\t⏳ WATCHDOG: '{}' still running after {:.0}s (limit {:.0}s, no result yet)",
                        label, secs, timeout_secs
                    );
                }
            }
        }
    });
    let result = f();
    drop(tx); // signals completion; the monitor wakes on Disconnected and exits
    let _ = watchdog.join();
    result
}

pub fn run(args: &Args) -> Result<(), String> {
    println!("vector-db-benchmark v{}", env!("CARGO_PKG_VERSION"));

    if args.target_qps != 0.0 {
        if !args.target_qps.is_finite() || args.target_qps <= 0.0 {
            return Err("--target-qps must be finite and greater than zero".to_string());
        }
        if !args.search_duration.is_finite() || args.search_duration <= 0.0 {
            return Err(
                "--search-duration is required and must be greater than zero with --target-qps"
                    .to_string(),
            );
        }
        if !args.warmup_seconds.is_finite() || args.warmup_seconds < 0.0 {
            return Err("--warmup-seconds must be finite and non-negative".to_string());
        }
        if !args.max_lateness_ms.is_finite() || args.max_lateness_ms < 0.0 {
            return Err("--max-lateness-ms must be finite and non-negative".to_string());
        }
        if args.skip_vector_index || !args.update_search_ratio.is_empty() {
            return Err(
                "--target-qps currently supports search-only vector benchmarks".to_string(),
            );
        }
    } else {
        if !args.search_duration.is_finite() || args.search_duration < 0.0 {
            return Err("--search-duration must be finite and non-negative".to_string());
        }
        // Warm-up is now allowed in closed-loop-duration mode too (natural
        // peak-throughput path: --search-duration without --target-qps), so the
        // measured window doesn't run against a cold Redis while Vertex primes.
        if args.warmup_seconds != 0.0 && args.search_duration <= 0.0 {
            return Err("--warmup-seconds requires --target-qps or --search-duration".to_string());
        }
    }

    let dataset_configs = read_dataset_configs()?;
    // A configuration file that fails to load takes EVERY configuration it
    // defines with it, and under a wildcard `--engines` that silently shrinks the
    // sweep rather than failing — so refuse to start unless the user says
    // otherwise, and if they do, carry the list into the artifacts (#239).
    let (engine_configs, skipped_configs) = if args.allow_partial_configs {
        let (configs, skipped) =
            config::read_engine_configs_reporting_skips(args.engines_file.as_deref())?;
        if !skipped.is_empty() {
            eprintln!(
                "Warning: --allow-partial-configs was passed, so the run continues with an \
                 INCOMPLETE configuration set.\n{}",
                config::describe_skipped_config_files(&skipped, config::SkipReport::PartialRun)
            );
        }
        (configs, skipped)
    } else {
        // Strict: an unloadable file aborts here rather than shrinking the sweep.
        (
            config::read_engine_configs(args.engines_file.as_deref())?,
            Vec::new(),
        )
    };
    // Carried into every summary this run writes, so a truncated sweep says so
    // in the artifact rather than only on a stderr line nobody keeps (#239).
    config::record_skipped_config_files(skipped_configs);

    // Filter datasets by pattern
    let datasets: Vec<_> = dataset_configs
        .iter()
        .filter(|(name, _)| args.datasets.iter().any(|p| matches_pattern(name, p)))
        .collect();

    if datasets.is_empty() {
        return Err(format!(
            "No datasets match pattern: '{}'",
            args.datasets.join(", ")
        ));
    }

    // Filter engines by pattern
    let supported_engines = [
        "redis",
        "vectorsets",
        "elasticsearch",
        "opensearch",
        "qdrant",
        "weaviate",
        "pgvector",
        "milvus",
        "mongodb",
        "valkey",
        "turbopuffer",
        "dragonfly",
        "kividb",
        "vertex",
        "chroma",
    ];
    let mut engines: Vec<_> = engine_configs
        .iter()
        .filter(|(name, config)| {
            let engine_type = config.engine.as_deref().unwrap_or("");
            supported_engines.contains(&engine_type)
                && args.engines.iter().any(|p| matches_pattern(name, p))
        })
        .collect();

    if engines.is_empty() {
        return Err(format!(
            "No engines match pattern: '{}'. Supported: {:?}.",
            args.engines.join(", "),
            supported_engines
        ));
    }

    if args.target_qps > 0.0 || args.search_duration > 0.0 {
        let unsupported: Vec<_> = engines
            .iter()
            .filter_map(|(name, config)| {
                let engine_type = config.engine.as_deref().unwrap_or("unknown");
                (!matches!(engine_type, "redis" | "vertex")).then_some(name.as_str())
            })
            .collect();
        if !unsupported.is_empty() {
            return Err(format!(
                "duration-bounded search currently supports Redis and Vertex only; unsupported: {}",
                unsupported.join(", ")
            ));
        }
    }

    // --skip-vector-index: deduplicate engine configs by engine type.
    // Multiple M/EF variants (e.g. redis-m-16-ef-64, redis-m-32-ef-128) collapse
    // into a single "<engine_type>-no-vector" experiment.
    if args.skip_vector_index {
        let mut seen_engine_types = std::collections::HashSet::new();
        engines.retain(|(_, config)| {
            let engine_type = config.engine.as_deref().unwrap_or("unknown");
            seen_engine_types.insert(engine_type.to_string())
        });
        println!(
            "--skip-vector-index: deduplicated to {} engine(s)",
            engines.len()
        );
    }

    // Collision guard (#151-4): among the selected configs, no two configs of the
    // same destructive Redis-wire engine (redis/valkey/dragonfly/kividb) may derive
    // the same index namespace, or a sweep would silently overwrite one config's graph
    // and keyspace with another's (the exact bug this fix closes). Also fires when
    // an `*_INDEX_NAME_EXACT` pin is set with >1 config for that engine (every
    // config then resolves to the same verbatim base). In --skip-vector-index mode
    // the dedup above already leaves one config per engine, so this is a no-op.
    {
        use crate::engine::index_naming::{derive_index_name, index_name_exact};
        let mut seen: std::collections::HashMap<(String, String), String> =
            std::collections::HashMap::new();
        for (_name, config) in &engines {
            let engine_type = config.engine.as_deref().unwrap_or("");
            let base_env = match engine_type {
                "redis" => "REDIS_INDEX_NAME",
                "valkey" => "VALKEY_INDEX_NAME",
                "dragonfly" => "DRAGONFLY_INDEX_NAME",
                "kividb" => "KIVIDB_INDEX_NAME",
                _ => continue,
            };
            let idx = derive_index_name(base_env, "idx", &config.name);
            if let Some(prev) =
                seen.insert((engine_type.to_string(), idx.clone()), config.name.clone())
            {
                let exact_hint = if index_name_exact(base_env) {
                    format!(" ({base_env}_EXACT is set — exact mode requires a single config per engine.)")
                } else {
                    String::new()
                };
                return Err(format!(
                    "Configs '{}' and '{}' derive the same index namespace '{}'; rename them — \
                     a sweep would silently overwrite one with the other (issue #151-4).{}",
                    prev, config.name, idx, exact_hint
                ));
            }
        }
    }

    println!(
        "Found {} datasets, {} engines",
        datasets.len(),
        engines.len()
    );

    // Run experiments
    let total_experiments = engines.len() * datasets.len();
    let pb = ProgressBar::new(total_experiments as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} Overall: [{elapsed_precise}] [{bar:30.cyan/blue}] {pos}/{len} experiments (ETA: {eta})",
            )
            .unwrap()
            .progress_chars("#>-"),
    );
    pb.enable_steady_tick(std::time::Duration::from_secs(1));

    let skip_vector_engines = ["redis", "valkey", "mongodb"];

    // Soft wall-clock budget: once total elapsed reaches --timeout, stop launching
    // further experiments and finish cleanly. This bounds the overall run without
    // interrupting an in-flight experiment (the Rust runner is blocking, so a hard
    // per-experiment abort as in the Python tool is not safe here).
    let run_start = Instant::now();
    let budget = if args.timeout.is_finite() && args.timeout > 0.0 {
        Some(Duration::from_secs_f64(args.timeout))
    } else {
        None
    };

    let num_engine_configs = engines.len();
    // `--keep-data` on a multi-config sweep keeps EVERY config's data coexisting
    // (the default — needed for `--skip-upload` reuse across configs). On a
    // memory-bounded server that accumulation OOMs (#184); `--reset-between-configs`
    // opts into freeing each config before the next so only the LAST config's data
    // is kept. Warn up-front when the user is on the accumulating path with >1
    // config so the peak-memory cost isn't a silent surprise.
    if args.keep_data && num_engine_configs > 1 && !args.reset_between_configs {
        eprintln!(
            "NOTE: --keep-data with {} configs keeps EVERY config's data resident \
             simultaneously (peak memory ≈ sum of all configs' datasets). On a \
             memory-bounded server pass --reset-between-configs to keep only one \
             config's data at a time, or drop --keep-data.",
            num_engine_configs
        );
    }
    'experiments: for (engine_idx, (_engine_name, engine_config)) in engines.iter().enumerate() {
        // With --reset-between-configs, only the LAST config keeps its data under
        // --keep-data; earlier configs tear down after finishing so at most one
        // config's corpus is resident (#184). (A full M×EF sweep runs every config
        // over the same datasets, so the last config is the last to touch each.)
        let is_last_config = engine_idx + 1 == num_engine_configs;
        // Apply --skip-vector-index: override name and set flag on config
        let mut engine_config = (*engine_config).clone();
        if args.skip_vector_index {
            let engine_type = engine_config.engine.as_deref().unwrap_or("unknown");
            if !skip_vector_engines.contains(&engine_type) {
                eprintln!(
                    "WARNING: --skip-vector-index not implemented for engine '{}', skipping",
                    engine_type
                );
                continue;
            }
            engine_config.name = format!("{}-no-vector", engine_type);
            engine_config.skip_vector_index = true;
        }

        for (dataset_name, dataset_config) in &datasets {
            // Stop before starting a new experiment if the time budget is exhausted.
            if let Some(budget) = budget {
                let elapsed = run_start.elapsed();
                if elapsed >= budget {
                    let remaining = total_experiments as u64 - pb.position();
                    pb.suspend(|| {
                        eprintln!(
                            "Reached --timeout budget ({:.0}s, elapsed {:.0}s); \
                             stopping with {} experiment(s) not started.",
                            budget.as_secs_f64(),
                            elapsed.as_secs_f64(),
                            remaining
                        );
                    });
                    break 'experiments;
                }
            }

            let experiment_num = pb.position() + 1;
            pb.suspend(|| {
                println!("\n{}", "=".repeat(60));
                println!(
                    "Running experiment ({}/{}): {} - {}",
                    experiment_num, total_experiments, engine_config.name, dataset_name
                );
                println!("{}", "=".repeat(60));
            });

            let dataset = Dataset::new((*dataset_config).clone());

            // Start this experiment's provenance recording BEFORE the engine is
            // built — engines resolve most of their environment knobs in `new()`,
            // and a sweep must not carry the previous configuration's knobs into
            // this one's artifacts (#212).
            // Consumed by `run_single_experiment` below. See `Recording`: the
            // move is what makes a commented-out, conditional or hoisted call a
            // compile error instead of a silently empty artifact.
            let recording = crate::effective_config::begin_experiment(
                &engine_config,
                invocation_provenance(args),
            );

            // Create engine
            let mut engine = create_engine(&engine_config, &args.host)?;

            // Shard count this run will be measured at, recorded in the result
            // files so it does not have to be inferred from the config name (#211).
            let number_of_shards = crate::engine::resolved_number_of_shards(&engine_config)?;
            if let Some(shards) = number_of_shards.as_ref() {
                crate::effective_config::record_effective("number_of_shards", shards.clone());
            }

            // Run experiment phases
            if let Err(e) = run_single_experiment(
                &mut *engine,
                &dataset,
                args,
                is_last_config,
                number_of_shards.as_ref(),
                recording,
            ) {
                // Name the sweep point. Under `--exit-on-error false` this line
                // is all the operator gets, and a bare engine-side message
                // (e.g. a #219 filter refusal) cannot otherwise be traced back
                // to the config/dataset pair that produced it.
                let e = format!(
                    "[config={} dataset={}] {}",
                    engine_config.name, dataset_name, e
                );
                eprintln!("Experiment failed: {}", e);
                if args.exit_on_error {
                    pb.finish_and_clear();
                    return Err(e);
                }
            }
            pb.inc(1);
        }
    }

    pb.finish_and_clear();
    Ok(())
}

/// The invocation facts that change what a run measures and live in neither the
/// configuration file nor the environment (#212).
///
/// `host` and `skip_upload` are the two that most often explain a number.
/// Without `host`, two runs against two different servers produced byte-identical
/// provenance; without `skip_upload`, a run that searched an index some earlier
/// config had built was distinguishable only by the absence of an upload file —
/// which is exactly the evidence somebody holding a published summary does not
/// have.
/// `host` is scrubbed by `effective_config::snapshot` on the way out — it is the
/// documented way to pass a Redis/Mongo password (`--host 'user:pw@node'`), and
/// it used to be published verbatim.
///
/// The boolean set here is CI-asserted against `Args` by
/// `every_measuring_flag_is_recorded_in_the_invocation`, so a new flag cannot be
/// added without either recording it or excusing it in writing. Every other
/// inventory in this module is bidirectional; this one was hand-maintained and
/// had already drifted by five flags.
pub(crate) fn invocation_provenance(args: &Args) -> serde_json::Value {
    json!({
        "host": args.host,
        "engines_file": args.engines_file,
        "allow_partial_configs": args.allow_partial_configs,
        // #271: suppresses the short-corpus refusal, so a run that measured a
        // partial corpus — and therefore a wrong recall under a config name that
        // claims otherwise — is exactly the run whose artifact must say so.
        // Caught by the guard below on the very first merge after it landed.
        "allow_partial_corpus": args.allow_partial_corpus,
        "dump_raw_latencies": args.dump_raw_latencies,
        "exit_on_error": args.exit_on_error,
        "fail_on_dropped_queries": args.fail_on_dropped_queries,
        "keep_data": args.keep_data,
        "reset_between_configs": args.reset_between_configs,
        "skip_if_exists": args.skip_if_exists,
        "skip_search": args.skip_search,
        "skip_upload": args.skip_upload,
        "skip_vector_index": args.skip_vector_index,
        // NON-BOOLEAN, so the guard below does NOT cover them: it enumerates
        // `pub <name>: bool,` only. Both change what is measured — `repetitions`
        // selects a best-of-N, `warmup_seconds` moves the measurement window —
        // and both are recorded by hand for that reason. Widening the guard past
        // booleans is #274.
        "repetitions": args.repetitions,
        "warmup_seconds": args.warmup_seconds,
        // Which directory the sweep globbed its configurations from, and where
        // the artifacts landed. `project_root()` resolves both from the process
        // cwd via `env::current_dir()`, the one raw environment read guard 1
        // waives — and this is the compensating record that waiver cites.
        "configurations_dir": crate::config::project_root()
            .join("experiments/configurations")
            .display()
            .to_string(),
        "results_dir": results_dir().display().to_string(),
    })
}

/// Parse "U:S" ratio string into UpdateSearchRatio.
fn parse_update_search_ratio(s: &str) -> Result<UpdateSearchRatio, String> {
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() != 2 {
        return Err(format!(
            "Invalid update-search-ratio format: '{}'. Expected 'U:S' (e.g., '1:10')",
            s
        ));
    }
    let updates: u64 = parts[0]
        .parse()
        .map_err(|_| format!("Invalid update count: '{}'", parts[0]))?;
    let searches: u64 = parts[1]
        .parse()
        .map_err(|_| format!("Invalid search count: '{}'", parts[1]))?;
    if searches == 0 {
        return Err("Search count must be > 0".to_string());
    }
    Ok(UpdateSearchRatio { updates, searches })
}

/// Outcome of the `--skip-upload` reuse precondition check, split out so it can
/// be unit-tested without a server (issue #238).
#[derive(Debug, PartialEq, Eq)]
pub enum ReusePrecondition {
    /// Server-side count matches what the dataset declares.
    Ok {
        actual: u64,
        expected: u64,
        approximate: bool,
    },
    /// Nothing to check against: the dataset's corpus size cannot be determined,
    /// or no server-side count is implemented for this engine. Proceed with a note.
    Unverifiable(String),
    /// The server holds MORE rows than the dataset declares. Warn and continue:
    /// leftovers from a bigger corpus change the reported recall, but so does
    /// refusing to run, and unlike a short corpus this is often deliberate
    /// (a shared prefix, a superset upload).
    Surplus {
        actual: u64,
        expected: u64,
        approximate: bool,
    },
    /// The server holds FEWER rows than the dataset declares — including zero,
    /// i.e. a missing index/collection. Fatal when the count is exact: every
    /// reported recall/precision figure is computed against ground truth for the
    /// FULL corpus, so a short corpus silently publishes the wrong number.
    /// `approximate` counts only ever warn — aborting on a number that is allowed
    /// to be wrong trades one silent failure for a noisy one.
    Short {
        actual: u64,
        expected: u64,
        approximate: bool,
    },
}

impl ReusePrecondition {
    /// Short machine-readable tag for the results JSON.
    fn status(&self) -> &'static str {
        match self {
            ReusePrecondition::Ok { .. } => "verified",
            ReusePrecondition::Unverifiable(_) => "unverified",
            ReusePrecondition::Surplus { .. } => "surplus",
            ReusePrecondition::Short { .. } => "short",
        }
    }
}

/// Classify a server-side corpus count against what the dataset declares.
pub fn classify_reuse_precondition(
    expected: Option<u64>,
    actual: Option<CorpusCount>,
    engine_name: &str,
) -> ReusePrecondition {
    let Some(expected) = expected else {
        return ReusePrecondition::Unverifiable(
            "the dataset's corpus size could not be determined".to_string(),
        );
    };
    let Some(actual) = actual else {
        // Deliberately phrased as a gap in this tool, not a limitation of the
        // database: Chroma, Milvus and Weaviate all expose a count, we simply
        // have not wired one up yet.
        return ReusePrecondition::Unverifiable(format!(
            "no server-side row count is implemented here for the engine behind config \
             '{engine_name}' yet"
        ));
    };
    let (a, e, approximate) = (actual.rows, expected, actual.approximate);
    if a < e {
        ReusePrecondition::Short {
            actual: a,
            expected: e,
            approximate,
        }
    } else if a > e {
        ReusePrecondition::Surplus {
            actual: a,
            expected: e,
            approximate,
        }
    } else {
        ReusePrecondition::Ok {
            actual: a,
            expected: e,
            approximate,
        }
    }
}

/// How many rows the reused corpus should hold, for the `--skip-upload` check.
///
/// Cost: this runs once per experiment and only on the `--skip-upload` path. For
/// npy/hdf5 corpora it is a header read. For the two `jsonl` datasets in
/// `datasets.json` it is a full-file line count that the search path did not
/// previously perform — cheap at their size, but not free, and worth knowing
/// before pointing `jsonl` at a very large corpus.
///
/// MEASURES the corpus on disk in preference to trusting `vector_count` in
/// `datasets.json`. `corpus_completeness_target()` is deliberately not used here:
/// it turns a declared-vs-measured DISAGREEMENT into an `Err`, and on this path an
/// `Err` degrades to "cannot verify" — so an under-declared `vector_count`
/// switched the guard off entirely, waving through exactly the half-empty corpus
/// it exists to catch (a `Recall: 0.0000` run, exit 0). The measurement is the
/// fact; a conflicting declaration is a `datasets.json` bug worth warning about,
/// not a reason to stop checking.
fn reuse_expected_rows(dataset: &Dataset) -> Result<Option<u64>, String> {
    if let Some(measured) = dataset.measured_vector_count()? {
        if let Some(declared) = dataset
            .config
            .vector_count
            .filter(|&n| n > 0)
            .map(|n| n as u64)
        {
            if declared != measured {
                eprintln!(
                    "WARNING: dataset '{}' declares vector_count {} but its corpus on disk holds \
                     {}. The --skip-upload reuse check uses the MEASURED {}; fix vector_count in \
                     datasets/datasets.json.",
                    dataset.config.name, declared, measured, measured
                );
            }
        }
        return Ok(Some(measured));
    }
    // Layouts with no cheap row count (sparse CSR, h5-multi) fall back to the
    // declared count, but only when their files are actually present.
    dataset.corpus_completeness_target()
}

/// Verify the promise `--skip-upload` makes: that the corpus it is told to reuse
/// is actually there, and whole (issue #238).
///
/// The check reads state back off the live server (`FT.INFO`, `GET
/// /collections/<n>`, `_count`, `countDocuments`, `reltuples`, `VCARD`) — "search
/// returned no error" is not evidence. Neither, in general, is recall: whether a
/// truncated corpus shows up in recall depends entirely on which rows went. On
/// `random-100` (9 queries, one ground-truth neighbour each, ids 0-8) deleting
/// the UPPER half of the collection leaves `mean_recall: 1.0` and deleting the
/// LOWER half gives `mean_recall: 0.0`. A metric that is a coin flip on the
/// deletion pattern cannot be the guard.
///
/// Repo policy on what is fatal: a short corpus changes the reported number, so
/// an exact count that comes up short is a hard error; everything softer warns.
///
/// Returns the verdict so it can be stamped into every result file this
/// experiment writes.
fn check_corpus_reuse_precondition(
    engine: &mut dyn Engine,
    dataset: &Dataset,
    args: &Args,
) -> Result<Option<serde_json::Value>, String> {
    // Nothing is measured, so nothing can be misreported.
    if args.skip_search {
        return Ok(None);
    }

    let expected = match reuse_expected_rows(dataset) {
        Ok(v) => v,
        Err(e) => {
            eprintln!(
                "\tNote: could not determine the dataset's corpus size ({e}); \
                 --skip-upload reuse left unverified"
            );
            None
        }
    };

    let actual = match engine.corpus_row_count() {
        Ok(v) => v,
        Err(e) => {
            let msg = format!(
                "--skip-upload: could not read the server-side corpus size for config '{}' ({}). \
                 This is a PROBE failure, not a verdict on the corpus — the data may well be \
                 intact. The flag promises to reuse an already-loaded corpus and that promise is \
                 now unverified. Fix the connection or the privilege, or pass \
                 --allow-partial-corpus to run anyway.",
                engine.name(),
                e
            );
            if args.allow_partial_corpus {
                eprintln!("WARNING: {msg}");
                return Ok(Some(json!({
                    "status": "probe_failed",
                    "detail": e,
                    "waived_by_allow_partial_corpus": true,
                })));
            }
            return Err(msg);
        }
    };

    let verdict = classify_reuse_precondition(expected, actual, engine.name());
    let approx_note = |approximate: bool| if approximate { "≈" } else { "" };
    let record = |waived: bool| {
        json!({
            "status": verdict.status(),
            "expected_rows": expected,
            "actual_rows": actual.map(|c| c.rows),
            "actual_is_estimate": actual.map(|c| c.approximate).unwrap_or(false),
            "waived_by_allow_partial_corpus": waived,
        })
    };

    match &verdict {
        ReusePrecondition::Ok {
            actual,
            expected,
            approximate,
        } => {
            println!(
                "Experiment stage: Reuse check — server holds {}{} of {} expected rows",
                approx_note(*approximate),
                actual,
                expected
            );
            Ok(Some(record(false)))
        }
        ReusePrecondition::Unverifiable(why) => {
            println!(
                "Experiment stage: Reuse check — SKIPPED ({why}); \
                 --skip-upload is running against an unverified corpus"
            );
            Ok(Some(record(false)))
        }
        ReusePrecondition::Surplus {
            actual,
            expected,
            approximate,
        } => {
            eprintln!(
                "WARNING: --skip-upload: config '{}' holds {}{} rows but dataset '{}' declares {} \
                 — the extra rows are searchable and can displace true neighbours, so recall may \
                 be understated.",
                engine.name(),
                approx_note(*approximate),
                actual,
                dataset.config.name,
                expected
            );
            Ok(Some(record(false)))
        }
        ReusePrecondition::Short {
            actual,
            expected,
            approximate,
        } => {
            let what = if *actual == 0 {
                "is empty or missing"
            } else {
                "is incomplete"
            };
            let msg = format!(
                "--skip-upload: the corpus you asked to reuse {what} — config '{}' holds {}{} of \
                 the {} rows dataset '{}' declares. Recall/precision are scored against ground \
                 truth for the FULL corpus, so continuing would publish a wrong number under a \
                 config name that claims otherwise. Re-upload this config (drop --skip-upload, \
                 add --keep-data). On a sweep, add --exit-on-error false so the configs whose \
                 corpus IS intact still publish their (correct) numbers while this one is \
                 skipped. --allow-partial-corpus measures the partial corpus deliberately — it \
                 publishes exactly the number this check exists to suppress, so reach for it \
                 last.",
                engine.name(),
                approx_note(*approximate),
                actual,
                expected,
                dataset.config.name
            );
            // An ESTIMATE may not abort a run: pgvector's count is a planner
            // figure, and a false hard error would be its own silent-wrong-result.
            if *approximate {
                eprintln!(
                    "WARNING: {msg}\n\t(the count above is an ESTIMATE, so this is a warning, \
                     not a rejection — verify it before quoting the run)"
                );
                return Ok(Some(record(false)));
            }
            if args.allow_partial_corpus {
                eprintln!("WARNING: {msg}");
                return Ok(Some(record(true)));
            }
            crate::summary::record_rejected_experiment(engine.name(), &dataset.config.name, &msg);
            Err(msg)
        }
    }
}

/// Why a run is not tearing its corpus down, for the "Keep data" line.
///
/// `--skip-upload` gets its own wording because it is not a user preference but
/// an invariant (#238): a run that did not upload the corpus never deletes it.
fn keep_data_reason(args: &Args) -> &'static str {
    if args.skip_upload {
        "--skip-upload did not create this corpus, so it does not delete it"
    } else {
        "cleanup skipped"
    }
}

/// Run a single experiment (configure, upload, search)
fn run_single_experiment(
    engine: &mut dyn Engine,
    dataset: &Dataset,
    args: &Args,
    is_last_config: bool,
    number_of_shards: Option<&serde_json::Value>,
    // Consumed, never read: holding it proves `begin_experiment` ran for THIS
    // (config, dataset) pair. See `effective_config::Recording`.
    _recording: crate::effective_config::Recording,
) -> Result<(), String> {
    // With --reset-between-configs, `--keep-data` only skips cleanup for the LAST
    // config in a sweep; earlier configs tear down so their (identical) corpus
    // copy doesn't accumulate and OOM the server (#184). Without the flag,
    // `--keep-data` keeps every config's data (the default coexistence behaviour,
    // needed for --skip-upload reuse). A single-config run is always the last.
    //
    // `--skip-upload` overrides all of that (#238). `configure()` is not the only
    // destructive call in an experiment: cleanup runs `engine.delete()`, and
    // `--keep-data` defaults to FALSE — so the plainest form of the flag
    // (`--skip-upload` alone) used to reuse a corpus, benchmark it, and then
    // delete it. Measured: qdrant 400 points -> collection MISSING, vectorsets
    // `VCARD idx` 400 -> 0, redis `DBSIZE` 400 -> 0, all exit 0. A run that did
    // not create the corpus must not tear it down, whatever `--keep-data` says.
    let keep_data =
        args.skip_upload || (args.keep_data && (is_last_config || !args.reset_between_configs));

    // Skip an incompatible (engine, dataset) pair BEFORE `get_path()`, which would
    // otherwise download the archive — hundreds of MB for the msmarco-sparse sets
    // — build an index at the fallback dimension, and only then fail inside the
    // reader. Mirrors upstream's IncompatibilityError skip.
    if (dataset.is_sparse() || dataset.is_hybrid()) && !engine.supports_sparse() {
        println!(
            "Skipping {} - {}: the dataset is {} and this engine has no sparse path",
            engine.name(),
            dataset.config.name,
            dataset.config.dataset_type.as_deref().unwrap_or("sparse")
        );
        return Ok(());
    }

    // Check if we should skip
    if args.skip_if_exists {
        let glob_pattern = format!("{}-{}-upload-*.json", engine.name(), dataset.config.name);
        let existing: Vec<_> = glob::glob(results_dir().join(&glob_pattern).to_str().unwrap())
            .map(|paths| paths.filter_map(|p| p.ok()).collect())
            .unwrap_or_default();

        if !existing.is_empty() && args.skip_upload {
            println!("Skipping (results exist): {}", glob_pattern);
            return Ok(());
        }
    }

    // Snapshot server metadata BEFORE any upload/search so results are
    // reproducible (server version, loaded modules incl. the search module,
    // full INFO/CONFIG, index state). None for non-Redis-wire engines.
    // Telemetry only — captured outside every timed window.
    let server_metadata_before = engine.server_metadata();

    // Configure phase
    let mut corpus_reuse: Option<serde_json::Value> = None;
    if !args.skip_upload {
        println!("Experiment stage: Configure");
        engine.configure(dataset)?;

        // Upload phase
        println!("Experiment stage: Upload");
        let mut upload_stats = engine.upload(dataset)?;

        // Collect memory usage after upload
        upload_stats.memory_usage = engine.get_memory_usage();

        // Save upload results
        save_upload_results(
            engine.name(),
            &dataset.config.name,
            &upload_stats,
            number_of_shards,
        )?;
    } else {
        // `--skip-upload` means: the server already holds the corpus I want —
        // do not create, drop, recreate or otherwise modify it. `configure()` is
        // destructive on 14 of the 15 engines (FT.DROPINDEX ... DD, SCAN+UNLINK,
        // collection.drop(), DROP TABLE, DELETE /collections/<n>, indices.delete,
        // DEL idx — only Vertex is not), so it must NOT run here under any flag
        // combination.
        //
        // This used to have an `else if args.skip_vector_index` arm that called
        // `configure()` "to create a schema-only index". It destroyed the corpus
        // the flags had just promised to reuse and then measured the empty index
        // without a word (issue #238) — verified live: Redis 400 -> 0 docs,
        // Valkey 400 -> 0 keys, MongoDB 400 -> 0 documents, each still printing a
        // QPS number and exiting 0. The arm is also unnecessary: the prior
        // `--skip-vector-index --keep-data` upload runs under the SAME rewritten
        // config name (`<engine>-no-vector`), so it left exactly the schema-only
        // index this run needs.
        //
        // A related shape survives the fix and the guard below is what catches
        // it: if phase 1 was a NORMAL upload, phase 2 does not destroy anything —
        // it publishes a QPS number from an empty `<engine>-no-vector` index while
        // the real corpus sits untouched under the original config name.
        corpus_reuse = check_corpus_reuse_precondition(engine, dataset, args)?;
    }

    // Build ordered search phases: pure search first, then mixed ratios ascending
    let search_phases: Vec<Option<UpdateSearchRatio>> = if args.update_search_ratio.is_empty() {
        vec![None]
    } else {
        let mut phases = Vec::new();
        let mut ratios: Vec<UpdateSearchRatio> = Vec::new();

        for s in &args.update_search_ratio {
            let ratio = parse_update_search_ratio(s)?;
            if ratio.updates == 0 {
                // 0:S means pure search
                if !phases.contains(&None) {
                    phases.push(None);
                }
            } else {
                ratios.push(ratio);
            }
        }

        // Sort mixed ratios ascending by updates/searches
        ratios.sort_by(|a, b| {
            let ra = a.updates as f64 / a.searches as f64;
            let rb = b.updates as f64 / b.searches as f64;
            ra.partial_cmp(&rb).unwrap()
        });

        for r in ratios {
            phases.push(Some(r));
        }
        phases
    };

    // Search phase
    let mut search_entries: Vec<SearchEntry> = Vec::new();
    // Search-result files are written after the whole search phase so the
    // AFTER server-metadata snapshot (taken once all reps complete) can be
    // embedded alongside the BEFORE snapshot in every file.
    let mut pending_saves: Vec<(
        usize,
        SearchParams,
        crate::engine::SearchResults,
        Option<serde_json::Value>,
    )> = Vec::new();
    // Ground-truth width profile, read once for the whole experiment. It decides
    // (a) whether our recall is the same quantity upstream calls `mean_precisions`
    // — reported in every result file — and (b) whether a calibration target is
    // reachable at all on this dataset (#217). Best-effort: a dataset whose query
    // file cannot be profiled simply omits the block.
    //
    // Skipped when nothing will consume it. `--skip-search` writes no search
    // results and runs no calibration, and filter-only runs have no ground truth
    // to speak of — reading the query file anyway costs a full parse of every
    // query vector just to drop it (~22 s warm / ~137 s cold per config on the
    // compound h-and-m dataset, paid once per config in a sweep).
    let ground_truth: Option<crate::ground_truth::GroundTruthProfile> =
        if args.skip_vector_index || args.skip_search {
            None
        } else {
            match crate::ground_truth::GroundTruthProfile::load(dataset) {
                Ok(gt) => Some(gt),
                Err(e) => {
                    eprintln!(
                    "\tNote: could not profile ground-truth widths ({}); result files will omit \
                     metrics_schema.ground_truth",
                    e
                );
                    None
                }
            }
        };
    // Configs that lost queries, collected so --fail-on-dropped-queries can fail
    // the run *after* every result file is written rather than instead of it.
    let mut dropped_query_failures: Vec<String> = Vec::new();
    // A search point whose every repetition failed. Recorded rather than
    // returned on the spot: an early `return` here would drop `pending_saves`
    // on the floor, throwing away every point that already succeeded. The
    // repo's stated policy for --fail-on-dropped-queries — "results files are
    // still written before the run fails, so the evidence survives" — applies
    // just as much to a hard search failure, and matters more now that a
    // short-staffed worker pool is one of them (#214).
    let mut fatal_search_error: Option<String> = None;
    let skip_vector_index = args.skip_vector_index;

    if !args.skip_search {
        // --skip-vector-index + no query conditions = nothing to search for
        if skip_vector_index {
            let has_schema = dataset
                .config
                .schema
                .as_ref()
                .and_then(|s| s.as_object())
                .map(|o| !o.is_empty())
                .unwrap_or(false);
            if !has_schema {
                println!(
                    "WARNING: --skip-vector-index with no schema fields on dataset '{}' — \
                     skipping search (no filter conditions possible)",
                    dataset.config.name
                );
                if keep_data {
                    println!("Experiment stage: Keep data ({})", keep_data_reason(args));
                } else {
                    if args.keep_data {
                        println!(
                            "Experiment stage: Cleanup (multi-config --keep-data: freeing this \
                             config's data before the next; only the last config's data is kept)"
                        );
                    } else {
                        println!("Experiment stage: Cleanup (deleting index and data)");
                    }
                    engine.delete()?;
                }
                println!("Experiment stage: Done");
                return Ok(());
            }
        }

        // Clone search params to avoid borrow conflict
        let all_search_params: Vec<_> = engine.search_params().to_vec();

        // --skip-vector-index: dedup search params by parallel level only
        // (ef values are irrelevant for filter-only queries)
        let effective_search_params: Vec<(usize, SearchParams)> = if skip_vector_index {
            let mut seen_parallels = std::collections::HashSet::new();
            all_search_params
                .into_iter()
                .enumerate()
                .filter(|(_, sp)| {
                    let p = sp.parallel.unwrap_or(1);
                    seen_parallels.insert(p)
                })
                .collect()
        } else {
            all_search_params.into_iter().enumerate().collect()
        };

        'search_phase: for phase in &search_phases {
            // --skip-vector-index: skip mixed phases (no vector updates to benchmark)
            if skip_vector_index && phase.is_some() {
                continue;
            }

            match phase {
                Some(ratio) => println!(
                    "Experiment stage: Mixed Search+Update (ratio {}:{})",
                    ratio.updates, ratio.searches
                ),
                None => {
                    if skip_vector_index {
                        println!("Experiment stage: Filter-only Search (no vector index)");
                    } else {
                        println!("Experiment stage: Search");
                    }
                }
            }

            for (search_id, search_params) in &effective_search_params {
                // Filter by parallel if specified
                if !args.parallels.is_empty() {
                    let parallel = search_params.parallel.unwrap_or(1) as i32;
                    if !args.parallels.contains(&parallel) {
                        continue;
                    }
                }

                // Filter by ef_runtime if specified (irrelevant for skip_vector_index)
                if !skip_vector_index && !args.ef_runtime.is_empty() {
                    if let Some(ref inner) = search_params.search_params {
                        if let Some(ef) = inner.ef {
                            if !args.ef_runtime.contains(&ef) {
                                continue;
                            }
                        }
                    }
                }

                let parallel = search_params.parallel.unwrap_or(1);

                // Calibration is skipped for filter-only mode (no vector search to tune)
                let mut calibration_json: Option<serde_json::Value> = None;
                let calibrated_params = if skip_vector_index {
                    None
                } else if let (Some(cal_param), Some(cal_precision)) = (
                    &search_params.calibration_param,
                    search_params.calibration_precision,
                ) {
                    println!(
                        "\tCalibrating {}: target mean_precision_at_returned={:.4}, parallel={}",
                        cal_param, cal_precision, parallel
                    );
                    match calibrate(
                        engine,
                        dataset,
                        search_params,
                        cal_param,
                        cal_precision,
                        args.queries,
                        ground_truth.as_ref(),
                    ) {
                        Ok(outcome) => {
                            let value = outcome.value;
                            println!(
                                "\tCalibrated {}={} → precision_at_returned={:.4} (target {:.4}{})",
                                cal_param,
                                value,
                                outcome.achieved,
                                outcome.target,
                                if outcome.reached_target {
                                    " reached"
                                } else {
                                    " NOT reached"
                                },
                            );
                            calibration_json = Some(outcome.to_json());
                            // Create a new SearchParams with calibrated value
                            let mut calibrated = search_params.clone();
                            let inner =
                                calibrated
                                    .search_params
                                    .get_or_insert_with(|| InnerSearchParams {
                                        ef: None,
                                        extra: None,
                                    });
                            // "EF" is the SAME field as "ef" (serde alias), so it
                            // must land on the typed field rather than in an
                            // `extra` map no engine reads.
                            let cal_key = SearchParams::canonical_knob_name(cal_param);
                            if cal_key == "ef" {
                                inner.ef = Some(value);
                            } else {
                                let extras = inner.extra.get_or_insert_with(Default::default);
                                extras.insert(cal_key.to_string(), serde_json::json!(value));
                            }
                            Some(calibrated)
                        }
                        Err(e) => {
                            eprintln!("\tCalibration failed: {}", e);
                            None
                        }
                    }
                } else {
                    None
                };

                let base_params = calibrated_params.as_ref().unwrap_or(search_params);
                let mut runtime_params = base_params.clone();
                if args.search_duration > 0.0 {
                    runtime_params.duration_seconds = Some(args.search_duration);
                }
                if args.target_qps > 0.0 {
                    runtime_params.target_qps = Some(args.target_qps);
                    runtime_params.max_lateness_ms = Some(args.max_lateness_ms);
                }
                let effective_params = &runtime_params;
                let effective_ef = if skip_vector_index {
                    "n/a".to_string()
                } else {
                    effective_params
                        .search_params
                        .as_ref()
                        .and_then(|p| p.ef)
                        .map(|e| e.to_string())
                        .unwrap_or_else(|| "default".to_string())
                };

                if skip_vector_index {
                    println!(
                        "\tRunning filter-only search {}: parallel={}",
                        search_id, parallel
                    );
                } else {
                    println!(
                        "\tRunning search {}: ef={}, parallel={}",
                        search_id, effective_ef, parallel
                    );
                }

                if args.target_qps > 0.0 && args.warmup_seconds > 0.0 {
                    let mut warmup_params = effective_params.clone();
                    warmup_params.duration_seconds = Some(args.warmup_seconds);
                    println!(
                        "\tOpen-loop warm-up: {:.1} QPS for {:.1}s",
                        args.target_qps, args.warmup_seconds
                    );
                    run_with_search_watchdog(args.search_timeout, "open-loop warm-up", || {
                        engine.search(dataset, &warmup_params, args.queries)
                    })
                    .map_err(|e| format!("open-loop warm-up failed: {}", e))?;
                } else if args.search_duration > 0.0 && args.warmup_seconds > 0.0 {
                    // Closed-loop-duration warm-up: a discarded search phase so the
                    // measured window sees a warm server for BOTH engines (Vertex
                    // primes per-connection; this warms Redis caches). No target_qps
                    // here, so this is a closed-loop run bounded by warmup_seconds.
                    let mut warmup_params = effective_params.clone();
                    warmup_params.duration_seconds = Some(args.warmup_seconds);
                    warmup_params.target_qps = None;
                    println!("\tClosed-loop warm-up: {:.1}s", args.warmup_seconds);
                    run_with_search_watchdog(args.search_timeout, "closed-loop warm-up", || {
                        engine.search(dataset, &warmup_params, args.queries)
                    })
                    .map_err(|e| format!("closed-loop warm-up failed: {}", e))?;
                }

                // Run the measured search `repetitions` times and keep the
                // best-RPS run. Restores v0's REPETITIONS behavior: the first run
                // is often cold (OS page cache / index warm-up), and best-of
                // discards it, so published QPS is a warm figure comparable to
                // the Python tool. --repetitions 1 disables it.
                // Best-of-N is meaningless in --search-duration (closed/open-loop
                // timed) mode — it just triples runtime for an upward-biased max.
                // Force a single rep there, warning if the user set >1 (#151).
                let repetitions = if args.search_duration > 0.0 {
                    if args.repetitions > 1 {
                        eprintln!(
                            "note: --repetitions {} ignored (using 1) in --search-duration mode",
                            args.repetitions
                        );
                    }
                    1
                } else {
                    args.repetitions.max(1)
                };
                let mut best: Option<crate::engine::SearchResults> = None;
                let mut last_err: Option<String> = None;

                for rep in 0..repetitions {
                    // Sample client CPU around the search so we can flag runs where
                    // the benchmark client — not the database — was the bottleneck.
                    //
                    // CAVEAT: this bracket wraps the WHOLE engine.search() call —
                    // read_queries(), connection setup, the per-connection prime,
                    // and the barrier/scheduling waits — not just the steady-state
                    // measured loop. The setup phase is dominated by idle waits
                    // (barrier + open-loop sleeps), which DILUTE the CPU fraction
                    // downward, so `client_cpu_cores_used` here is a CONSERVATIVE
                    // lower bound on steady-state client CPU: it can under-report
                    // saturation but will not falsely flag a run as client-bound.
                    // A window-scoped sample would require the Engine trait to
                    // return the measured-loop CPU, which is deliberately avoided.
                    let cpu_before = crate::proc_cpu::sample();
                    let wd_label = format!(
                        "{}/{} {}[parallel={}]",
                        engine.name(),
                        dataset.config.name,
                        if phase.is_some() { "mixed" } else { "search" },
                        effective_params.parallel.unwrap_or(1),
                    );
                    let search_result =
                        run_with_search_watchdog(args.search_timeout, &wd_label, || match phase {
                            Some(ratio) => {
                                engine.search_mixed(dataset, effective_params, args.queries, ratio)
                            }
                            None => engine.search(dataset, effective_params, args.queries),
                        });
                    let cpu_after = crate::proc_cpu::sample();

                    match search_result {
                        Ok(mut results) => {
                            // Attach CPU / oversubscription / saturation coverage.
                            let sat = crate::proc_cpu::compute(
                                cpu_before,
                                cpu_after,
                                results.parallel,
                                crate::proc_cpu::available_cores(),
                            );
                            results.available_cores = sat.available_cores;
                            results.oversubscribed = sat.oversubscribed;
                            results.client_cpu_cores_used = sat.client_cpu_cores_used;
                            results.system_cpu_pct = sat.system_cpu_pct;
                            results.client_saturated = sat.client_saturated;
                            results.saturation_reason = sat.saturation_reason;
                            if repetitions > 1 {
                                println!(
                                    "\t  rep {}/{}: QPS {:.1}",
                                    rep + 1,
                                    repetitions,
                                    results.rps
                                );
                            }
                            // Representative-rep selection. In closed-loop mode the
                            // best-RPS run is the warm figure we want. In open-loop
                            // mode rps is pinned to target_qps, so max-rps is noise;
                            // instead keep the rep that shed the FEWEST requests,
                            // breaking ties by the lower tail (end-to-end p95).
                            let is_open_loop = args.target_qps > 0.0;
                            let better = match best.as_ref() {
                                None => true,
                                Some(b) if is_open_loop => {
                                    (
                                        results.dropped_queries,
                                        results.end_to_end_p95_time.unwrap_or(f64::INFINITY),
                                    ) < (
                                        b.dropped_queries,
                                        b.end_to_end_p95_time.unwrap_or(f64::INFINITY),
                                    )
                                }
                                Some(b) => results.rps > b.rps,
                            };
                            if better {
                                best = Some(results);
                            }
                        }
                        Err(e) => {
                            eprintln!("\tSearch failed (rep {}/{}): {}", rep + 1, repetitions, e);
                            last_err = Some(e);
                        }
                    }
                }

                match best {
                    Some(results) => {
                        if skip_vector_index {
                            println!("\t→ QPS: {:.1} (filter-only, no precision)", results.rps);
                        } else {
                            println!(
                                "\t→ QPS: {:.1}, Recall: {:.4}, Precision@returned: {:.4}, MRR: {:.4}, NDCG: {:.4}{}",
                                results.rps, results.mean_recall, results.mean_precision_at_returned, results.mean_mrr, results.mean_ndcg,
                                if repetitions > 1 { " (best of reps)" } else { "" }
                            );
                        }
                        // Surface dropped queries loudly: latency percentiles and
                        // recall above cover only the successful subset, so a
                        // nonzero count means the numbers are not over the full
                        // requested workload.
                        if results.failed_queries > 0 {
                            eprintln!(
                                "\t⚠ WARNING: {}/{} queries FAILED (only {} succeeded); \
                                 latency/recall/QPS above reflect the successful subset only",
                                results.failed_queries,
                                results.requested_queries,
                                results.num_queries,
                            );
                            // Under --fail-on-dropped-queries the run must not end
                            // green, but the abort is deferred to after the results
                            // are written below. Returning here would discard every
                            // config already in `pending_saves` — a whole sweep lost
                            // to one shed query in its last config, which is the
                            // cascade this guard is supposed to prevent, not cause.
                            dropped_query_failures.push(format!(
                                "search #{}: {}/{} queries dropped",
                                search_id, results.failed_queries, results.requested_queries
                            ));
                        }
                        if let Some(target_qps) = results.target_qps {
                            println!(
                                "\t  offered {:.1} QPS; dropped {}; late {}; schedule p95 {:.3} ms; end-to-end p95 {:.3} ms",
                                target_qps,
                                results.dropped_queries,
                                results.late_queries,
                                results.schedule_delay_p95_time.unwrap_or_default() * 1000.0,
                                results.end_to_end_p95_time.unwrap_or_default() * 1000.0,
                            );
                            if results.dropped_queries > 0 {
                                eprintln!(
                                    "\t⚠ WARNING: {} offered requests were dropped after exceeding the dispatch-lateness limit",
                                    results.dropped_queries
                                );
                            }
                        }
                        // Flag client-side saturation: when the benchmark client is
                        // the bottleneck the QPS/latency above are not clean
                        // server-side measurements.
                        if results.client_saturated {
                            let cpu = results
                                .client_cpu_cores_used
                                .map(|c| format!("{:.1} cores", c))
                                .unwrap_or_else(|| "cpu n/a".to_string());
                            eprintln!(
                                "\t⚠ WARNING: CLIENT LIKELY SATURATED ({}) — client used {}; \
                                 QPS/latency may reflect the client, not the database",
                                results.saturation_reason, cpu,
                            );
                        }
                        // Defer the file write until the AFTER snapshot exists.
                        pending_saves.push((
                            *search_id,
                            effective_params.clone(),
                            results.clone(),
                            calibration_json.clone(),
                        ));

                        search_entries.push(SearchEntry {
                            search_id: *search_id,
                            ef: effective_ef.clone(),
                            parallel,
                            results,
                            calibration: calibration_json.clone(),
                        });
                    }
                    None => {
                        // Every rep of this search config failed. Under
                        // exit_on_error (the default) this must abort loudly rather
                        // than be swallowed with a zero exit — otherwise a hard
                        // error (e.g. the #151-4 "index not found" guard on a
                        // --skip-upload run against a config that was never
                        // uploaded) silently writes nothing and the process exits
                        // 0, masking wrong/absent results.
                        let msg = format!(
                            "search failed (all {} repetition(s)){}",
                            repetitions,
                            last_err
                                .as_ref()
                                .map(|e| format!(": {}", e))
                                .unwrap_or_default()
                        );
                        eprintln!("\t{}", msg);
                        if args.exit_on_error {
                            fatal_search_error = Some(msg);
                            break 'search_phase;
                        }
                    }
                }
            }
        }
    }

    // Snapshot server metadata AFTER all search reps complete (index still
    // present) and write the deferred search-result files, embedding both the
    // before and after snapshots. For non-Redis engines both are None and the
    // `server_metadata` key is omitted from the result JSON.
    let server_metadata_after = engine.server_metadata();
    for (search_id, params, results, calibration) in &pending_saves {
        save_search_results(
            engine.name(),
            &dataset.config.name,
            *search_id,
            params,
            results,
            server_metadata_before.as_ref(),
            server_metadata_after.as_ref(),
            args.dump_raw_latencies,
            number_of_shards,
            ground_truth.as_ref(),
            calibration.as_ref(),
            corpus_reuse.as_ref(),
        )?;
    }

    // Only now that the completed points are on disk may a hard search failure
    // end the run.
    if let Some(msg) = fatal_search_error {
        if !pending_saves.is_empty() {
            eprintln!(
                "\t↳ wrote {} completed search point(s) before failing, so the evidence survives",
                pending_saves.len()
            );
        }
        return Err(msg);
    }

    // Now that the evidence is on disk, honour --fail-on-dropped-queries. Every
    // engine drops a failed query from the latency/recall vectors, so this gate is
    // engine-agnostic by construction: it reads the `failed_queries` that
    // `compute_search_stats` derives for all of them.
    if args.fail_on_dropped_queries && !dropped_query_failures.is_empty() {
        return Err(format!(
            "{} search config(s) lost queries and --fail-on-dropped-queries is set: {}. \
             Recall/latency for these cover the surviving subset only, which biases recall \
             upward. Results were still written; re-run with less load, or drop the flag to \
             accept a partial run.",
            dropped_query_failures.len(),
            dropped_query_failures.join("; ")
        ));
    }

    // Repeat any unreached calibration target at the END of the run. The warning
    // the sweep printed up front is ~10 iterations of progress output away by now,
    // and a sweep that "calibrated" against an unreachable target otherwise exits 0
    // with nothing on screen to say the `ef` it settled on is meaningless (#217).
    let uncalibrated: Vec<&SearchEntry> = search_entries
        .iter()
        .filter(|e| {
            e.calibration
                .as_ref()
                .and_then(|c| c.get("reached_target"))
                .and_then(|v| v.as_bool())
                == Some(false)
        })
        .collect();
    if !uncalibrated.is_empty() {
        eprintln!(
            "\n⚠ WARNING: {} search config(s) did NOT reach their calibration target; the swept \
             parameter for those points is the closest value found, not a calibrated one. See \
             `uncalibrated_configs` in the summary JSON.",
            uncalibrated.len()
        );
        for e in &uncalibrated {
            if let Some(cal) = &e.calibration {
                eprintln!(
                    "\t  search #{}: {}={} → {:.4} (target {:.4})",
                    e.search_id,
                    cal.get("param").and_then(|v| v.as_str()).unwrap_or("?"),
                    cal.get("value").and_then(|v| v.as_i64()).unwrap_or(-1),
                    cal.get("achieved")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(f64::NAN),
                    cal.get("target")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(f64::NAN),
                );
            }
        }
    }

    // Display precision summary and save summary JSON
    if !search_entries.is_empty() {
        summary::display_results_summary(engine.name(), &dataset.config.name, &search_entries);
        if search_phases.len() > 1 {
            summary::display_mixed_summary(&search_entries);
        }
        summary::save_summary(
            engine.name(),
            &dataset.config.name,
            &search_entries,
            None,
            &results_dir(),
            &{
                crate::effective_config::set_phase("run");
                crate::effective_config::snapshot()
            },
        )?;
    }

    // Cleanup unless the caller wants to reuse the populated index. On a
    // multi-config sweep `--keep-data` keeps only the LAST config's data; earlier
    // configs tear down here so their corpus copy doesn't accumulate (#184).
    if keep_data {
        println!("Experiment stage: Keep data ({})", keep_data_reason(args));
    } else {
        if args.keep_data {
            println!(
                "Experiment stage: Cleanup (multi-config --keep-data: freeing this config's data \
                 before the next; only the last config's data is kept)"
            );
        } else {
            println!("Experiment stage: Cleanup (deleting index and data)");
        }
        engine.delete()?;
    }

    println!("Experiment stage: Done");
    Ok(())
}

/// Outcome of a calibration sweep, including whether it actually got there.
struct CalibrationOutcome {
    param: String,
    value: i64,
    target: f64,
    /// `mean_precision_at_returned` achieved at `value`.
    achieved: f64,
    reached_target: bool,
    /// Ceiling implied by the dataset's ground-truth widths at this `top`, when
    /// the ground truth could be profiled.
    ceiling: Option<f64>,
    /// Why the target was not reached, when it was not.
    note: Option<String>,
}

impl CalibrationOutcome {
    fn to_json(&self) -> serde_json::Value {
        json!({
            "param": self.param,
            "value": self.value,
            "metric": "mean_precision_at_returned",
            "target": self.target,
            "achieved": self.achieved,
            "reached_target": self.reached_target,
            "ground_truth_ceiling": self.ceiling,
            "note": self.note,
        })
    }
}

/// Binary search calibration matching Python v0.
///
/// Searches for the value of `calibration_param` (e.g., "ef") that achieves the
/// target `mean_precision_at_returned`. Uses binary search between `min_value`
/// (from `top` in search params, default 10) and `max_value` (1000).
///
/// # The target can be unreachable by construction (#217)
///
/// The metric being chased has "results returned" as its denominator, so on a
/// dataset whose ground-truth rows are narrower than `top` it is capped well
/// below 1.0 — with `top: 100` (as shipped in `cohere-calibration.json` and
/// `dbpedia-calibration.json`) and a 3-neighbour ground-truth row, no `ef`
/// reaches 0.95. The sweep used to converge on `ef = 1000` and report it as a
/// success. It now bounds the target against the ground-truth ceiling before
/// spending a single search, and reports `reached_target` either way — both on
/// stderr and in the result file's `params.calibration` block.
fn calibrate(
    engine: &mut dyn Engine,
    dataset: &Dataset,
    search_params: &SearchParams,
    calibration_param: &str,
    target_precision: f64,
    num_queries: i64,
    ground_truth: Option<&crate::ground_truth::GroundTruthProfile>,
) -> Result<CalibrationOutcome, String> {
    // "EF" (upstream's redis spelling) is the same typed field as "ef"; without
    // this the loop would sweep extra["EF"], which no engine reads.
    let calibration_param = SearchParams::canonical_knob_name(calibration_param);
    let min_value = search_params.top.unwrap_or(10);
    let max_value: i64 = 1000;

    // Bound the target against what the dataset's ground truth allows, before
    // burning ~10 full search runs chasing a number that cannot exist.
    let gt_stats = ground_truth.and_then(|gt| {
        let top = search_params
            .top
            .and_then(|t| usize::try_from(t).ok())
            .or_else(|| gt.first_row_len())?;
        gt.stats(top)
    });
    let ceiling = gt_stats.as_ref().map(|s| s.recall_at_top_ceiling);
    let unreachable_note = gt_stats
        .as_ref()
        .and_then(|s| s.unreachable_target_note(target_precision));
    if let Some(note) = &unreachable_note {
        eprintln!("\t⚠ WARNING: calibration target is unreachable — {}", note);
    }

    let mut lower_bound = min_value;
    let mut upper_bound = max_value;
    let mut lower_visited = false;
    let mut upper_visited = false;
    let mut current = (lower_bound + upper_bound) / 2;
    let mut previous = current;
    let mut previous_precision = 0.0_f64;

    loop {
        // Create search params with current calibration value
        let mut test_params = search_params.clone();
        let inner = test_params
            .search_params
            .get_or_insert_with(|| InnerSearchParams {
                ef: None,
                extra: None,
            });
        if calibration_param == "ef" {
            inner.ef = Some(current);
        } else {
            let extras = inner.extra.get_or_insert_with(Default::default);
            extras.insert(calibration_param.to_string(), serde_json::json!(current));
        }

        let results = engine.search(dataset, &test_params, num_queries)?;
        let current_precision = results.mean_precision_at_returned;

        println!(
            "\t  calibration: {}={} → precision_at_returned={:.4}",
            calibration_param, current, current_precision
        );

        if (current_precision - target_precision).abs() < 1e-9 {
            return Ok(finish_calibration(
                calibration_param,
                current,
                current_precision,
                target_precision,
                ceiling,
                unreachable_note,
                min_value,
                max_value,
            ));
        } else if current_precision > target_precision {
            upper_bound = current;
            upper_visited = true;
        } else {
            lower_bound = current;
            lower_visited = true;
        }

        let next_value = (lower_bound + upper_bound) / 2;

        // Check convergence: if next step would revisit a bound, pick the closer result
        if (lower_visited && next_value == lower_bound)
            || (upper_visited && next_value == upper_bound)
        {
            let (value, achieved) = if (previous_precision - target_precision).abs()
                < (current_precision - target_precision).abs()
            {
                (previous, previous_precision)
            } else {
                (current, current_precision)
            };
            return Ok(finish_calibration(
                calibration_param,
                value,
                achieved,
                target_precision,
                ceiling,
                unreachable_note,
                min_value,
                max_value,
            ));
        }

        previous = current;
        previous_precision = current_precision;
        current = next_value;
    }
}

/// Build the calibration outcome and say plainly whether the target was met.
///
/// Before #217 the sweep returned the closest value it found with no signal at
/// all, so a target the dataset makes unreachable came out looking like a
/// successful calibration at the maximum swept value.
#[allow(clippy::too_many_arguments)]
fn finish_calibration(
    param: &str,
    value: i64,
    achieved: f64,
    target: f64,
    ceiling: Option<f64>,
    unreachable_note: Option<String>,
    min_value: i64,
    max_value: i64,
) -> CalibrationOutcome {
    let reached_target = achieved + 1e-9 >= target;
    let note = if reached_target {
        None
    } else if let Some(n) = unreachable_note {
        Some(n)
    } else {
        Some(format!(
            "the sweep of `{}` over [{}, {}] topped out at mean_precision_at_returned={:.4}, \
             below the target {:.4}{}. The reported value is the closest point found, not a \
             value that meets the target.",
            param,
            min_value,
            max_value,
            achieved,
            target,
            if value >= max_value {
                format!(
                    ", with `{}` pinned at the maximum swept value ({})",
                    param, max_value
                )
            } else {
                String::new()
            },
        ))
    };
    if let Some(n) = &note {
        eprintln!(
            "\t⚠ WARNING: calibration did NOT reach its target ({}={} → {:.4} < {:.4}) — {}",
            param, value, achieved, target, n
        );
    }
    CalibrationOutcome {
        param: param.to_string(),
        value,
        target,
        achieved,
        reached_target,
        ceiling,
        note,
    }
}

/// Retrieval-quality key definitions, emitted into every result file.
///
/// This block exists because the same key name meant two different things in the
/// two benchmarks (#217). Upstream `qdrant/vector-db-benchmark`
/// (`engine/base_client/search.py`) computes
/// `len(ids.intersection(query.expected_result[:top])) / top` — recall@top — and
/// publishes it as `mean_precisions`. We published *precision* (denominator =
/// results returned) under that identical key, so overlaying the two tools'
/// result files silently plotted precision against recall. We no longer emit a
/// `mean_precisions` key at all; the definitions travel with the data instead of
/// living only in a commit message.
fn metrics_schema(gt: Option<&crate::ground_truth::GroundTruthStats>) -> serde_json::Value {
    let comparable = gt
        .filter(|s| s.recall_matches_upstream())
        .map(|_| "mean_recall");
    let mut schema = json!({
        "version": 2,
        "mean_precision_at_returned":
            "hits / |deduped results kept (<= top)|, where hits = |results ∩ valid, deduped \
             ground-truth ids in expected[:top]|. Renamed from `mean_precisions` in schema \
             version 2 (#217): that key name is taken by a DIFFERENT formula upstream. \
             -1.0 is the filter-only sentinel (no vector search was run).",
        "mean_recall":
            "hits / |valid, deduped ground-truth ids in expected[:top]|. The denominator is the \
             ground truth that actually exists, so a query with 3 true neighbours can reach 1.0.",
        "recall_p10": "10th percentile of the per-query recall defined above.",
        "upstream_qdrant_mean_precisions":
            "len(ids & expected[:top]) / top — recall@top. This is what upstream \
             qdrant/vector-db-benchmark emits under the key `mean_precisions`. We deliberately \
             do NOT emit that key name. It equals our `mean_recall` only when every ground-truth \
             row has at least `top` valid ids (see `ground_truth` below), and it equals our \
             `mean_precision_at_returned` only when the engine returns a full page of `top` \
             results.",
        // Names the field in THIS file that IS upstream's `mean_precisions`
        // number, or null when no field is. Not "similar to": when this says
        // `mean_recall`, our mean_recall and upstream's mean_precisions are the
        // same quantity computed the same way, and may be overlaid directly.
        "comparable_to_upstream_mean_precisions": comparable,
    });
    if let Some(stats) = gt {
        schema["ground_truth"] = stats.to_json();
        if comparable.is_none() {
            schema["comparability_note"] = json!(format!(
                "{} of {} queries have fewer than top={} valid ground-truth neighbours, so no \
                 field in this file is directly comparable to upstream's `mean_precisions`; \
                 upstream's value is capped at {:.4} on this dataset while our `mean_recall` \
                 can still reach 1.0.",
                stats.queries_below_top, stats.queries, stats.top, stats.recall_at_top_ceiling
            ));
        }
    }
    schema
}

/// Save search results to JSON file (matches Python v0 format)
#[allow(clippy::too_many_arguments)]
fn save_search_results(
    engine_name: &str,
    dataset_name: &str,
    search_id: usize,
    search_params: &crate::config::SearchParams,
    results: &crate::engine::SearchResults,
    server_metadata_before: Option<&serde_json::Value>,
    server_metadata_after: Option<&serde_json::Value>,
    dump_raw_latencies: bool,
    number_of_shards: Option<&serde_json::Value>,
    ground_truth: Option<&crate::ground_truth::GroundTruthProfile>,
    calibration: Option<&serde_json::Value>,
    corpus_reuse: Option<&serde_json::Value>,
) -> Result<(), String> {
    let timestamp = Local::now().format("%Y-%m-%d-%H-%M-%S");
    let pid = std::process::id();
    let mixed_tag = results
        .update_search_ratio
        .as_ref()
        .map(|r| format!("-mixed-{}", r.replace(':', "x")))
        .unwrap_or_default();
    let filename = format!(
        "{}-{}-search-{}{}-{}-{}.json",
        engine_name, dataset_name, search_id, mixed_tag, pid, timestamp
    );

    let result = build_search_result_json(
        engine_name,
        dataset_name,
        search_params,
        results,
        server_metadata_before,
        server_metadata_after,
        dump_raw_latencies,
        number_of_shards,
        ground_truth,
        calibration,
        corpus_reuse,
        // Snapshotted here rather than inside the builder so the builder stays a
        // pure function of its arguments and its tests do not race the
        // process-wide recorder.
        &{
            crate::effective_config::set_phase("search");
            crate::effective_config::snapshot()
        },
    );

    let path = results_dir().join(&filename);
    fs::write(&path, serde_json::to_string_pretty(&result).unwrap())
        .map_err(|e| format!("Failed to save results: {}", e))?;

    println!("\tResults saved to: {:?}", path);
    Ok(())
}

/// Build the search-result JSON document.
///
/// Split out of `save_search_results` so the emitted key set — in particular the
/// absence of `mean_precisions` and the presence of `metrics_schema` (#217) — is
/// unit-testable without writing a file.
#[allow(clippy::too_many_arguments)]
fn build_search_result_json(
    engine_name: &str,
    dataset_name: &str,
    search_params: &crate::config::SearchParams,
    results: &crate::engine::SearchResults,
    server_metadata_before: Option<&serde_json::Value>,
    server_metadata_after: Option<&serde_json::Value>,
    dump_raw_latencies: bool,
    number_of_shards: Option<&serde_json::Value>,
    ground_truth: Option<&crate::ground_truth::GroundTruthProfile>,
    calibration: Option<&serde_json::Value>,
    corpus_reuse: Option<&serde_json::Value>,
    engine_params: &serde_json::Value,
) -> serde_json::Value {
    let mut result = json!({
        "params": {
            "dataset": dataset_name,
            "experiment": engine_name,
            "parallel": search_params.parallel.unwrap_or(1),
            "top": results.top,
            "search_params": search_params.search_params,
            "target_qps": search_params.target_qps,
            "duration_seconds": search_params.duration_seconds,
            "max_lateness_ms": search_params.max_lateness_ms,
        },
        "results": {
            "total_time": results.total_time,
            "mean_time": results.mean_time,
            // Query accounting: rps and the latency percentiles are computed over
            // succeeded_queries only, while total_time (the rps denominator) spans
            // the whole run. A nonzero failed_queries means the reported latency
            // distribution covers a partial set — typically the regime where a
            // saturated client or an overloaded server sheds timeouts.
            "requested_queries": results.requested_queries,
            "succeeded_queries": results.num_queries,
            "failed_queries": results.failed_queries,
            // Client CPU / concurrency-saturation coverage: client_saturated=true
            // means the run was likely client-bound and the numbers below should
            // not be read as clean server-side measurements.
            "parallel": results.parallel,
            "available_cores": results.available_cores,
            "oversubscribed": results.oversubscribed,
            "client_cpu_cores_used": results.client_cpu_cores_used,
            "system_cpu_pct": results.system_cpu_pct,
            "client_saturated": results.client_saturated,
            "saturation_reason": results.saturation_reason,
            "target_qps": results.target_qps,
            "offered_queries": results.offered_queries,
            "dropped_queries": results.dropped_queries,
            "late_queries": results.late_queries,
            "schedule_delay_p50_time": results.schedule_delay_p50_time,
            "schedule_delay_p95_time": results.schedule_delay_p95_time,
            "schedule_delay_p99_time": results.schedule_delay_p99_time,
            "end_to_end_p50_time": results.end_to_end_p50_time,
            "end_to_end_p95_time": results.end_to_end_p95_time,
            "end_to_end_p99_time": results.end_to_end_p99_time,
            // Retrieval quality. THREE denominators are in play and only two of
            // them are ours; see the `metrics_schema` block below (and
            // src/metrics.rs) for the formulas. `mean_precisions` — upstream's
            // key for recall@top — is intentionally absent: we used to emit our
            // precision under that name, which made cross-tool overlays compare
            // precision against recall (#217).
            "mean_precision_at_returned": results.mean_precision_at_returned,
            "mean_recall": results.mean_recall,
            "recall_p10": results.recall_p10,
            "mean_mrr": results.mean_mrr,
            "mean_ndcg": results.mean_ndcg,
            "std_time": results.std_time,
            "min_time": results.min_time,
            "max_time": results.max_time,
            "rps": results.rps,
            "p50_time": results.p50_time,
            "p95_time": results.p95_time,
            "p99_time": results.p99_time,
            // Compact re-derivable digests replace the full per-query arrays
            // (which reached ~80 MB each on a 10M-query run). The top-level
            // p50/p95/p99_time seconds fields above are unchanged for back-compat.
            // Raw arrays are additionally dumped only under --dump-raw-latencies.
            "latency_hdr": crate::latency_digest::latency_hdr(&results.latencies),
            "precision_at_returned_dist":
                crate::latency_digest::quality_dist(&results.precisions_at_returned),
            "recall_dist": crate::latency_digest::quality_dist(&results.recalls),
            "mrr_dist": crate::latency_digest::quality_dist(&results.mrrs),
            "ndcg_dist": crate::latency_digest::quality_dist(&results.ndcgs),
        }
    });

    // Metric definitions travel with the numbers (#217). The ground-truth width
    // profile is evaluated at the `top` this run actually used, so a reader can
    // see whether `mean_recall` is the same quantity upstream calls
    // `mean_precisions` for this dataset — or by how much they cannot be.
    let gt_stats = ground_truth.and_then(|gt| gt.stats(results.top));
    result["metrics_schema"] = metrics_schema(gt_stats.as_ref());

    // How the swept parameter was chosen, and — crucially — whether the target
    // was actually reached. A calibration that converged on the maximum value
    // without reaching its target is not a calibration.
    if let Some(cal) = calibration {
        result["params"]["calibration"] = cal.clone();
    }

    // Whether this run's corpus was the one it claims to have measured (#238).
    // A verified run, an unverified-note run and an --allow-partial-corpus run
    // over a half-deleted corpus used to produce structurally identical
    // artifacts; the verdict travels with the numbers so it cannot be lost.
    if let Some(reuse) = corpus_reuse {
        result["params"]["corpus_reuse"] = reuse.clone();
    }

    // Shard count the index was built with. Recall and QPS both move with it, so
    // a published search result carries it instead of leaving it to be inferred
    // from the `experiment` (config) name (#211). Emitted only for the engines
    // where it means something; it also appears inside `engine_params.effective`
    // below, kept here at its established path for existing consumers.
    if let Some(shards) = number_of_shards {
        result["params"]["number_of_shards"] = shards.clone();
    }

    // What this run resolved (#212): `collection_params`/`upload_params` as the
    // configuration file spells them under `declared`, the environment-derived
    // knobs under `effective`, which variables were consulted at all under
    // `env`, the invocation flags that live in neither, and every
    // declared-but-overridden value under `overridden`. Without this block two
    // runs of the same config name against different hosts, or with different
    // retry budgets, produce byte-identical `params` and different numbers.
    //
    // Read `effective_config`'s module docs before treating a missing key as
    // proof of anything: `env` is complete for variables consulted so far, while
    // `effective` covers only the knobs resolved through the recording helpers.
    result["params"]["engine_params"] = engine_params.clone();

    // Opt-in full-fidelity archival: additionally emit the raw per-query arrays
    // exactly as before. Off by default so large runs stay ~1000x smaller.
    if dump_raw_latencies {
        let results_obj = result["results"].as_object_mut().unwrap();
        // `precisions` is another upstream key name carrying upstream's per-query
        // recall@top values, so our array ships under an unambiguous name too.
        results_obj.insert(
            "precisions_at_returned".to_string(),
            json!(results.precisions_at_returned),
        );
        results_obj.insert("recalls".to_string(), json!(results.recalls));
        results_obj.insert("mrrs".to_string(), json!(results.mrrs));
        results_obj.insert("ndcgs".to_string(), json!(results.ndcgs));
        results_obj.insert("latencies".to_string(), json!(results.latencies));
    }

    // Add update metrics when present (mixed benchmark mode)
    if let Some(ref ratio) = results.update_search_ratio {
        let results_obj = result["results"].as_object_mut().unwrap();
        results_obj.insert("update_search_ratio".to_string(), json!(ratio));
        if let Some(count) = results.update_count {
            results_obj.insert("update_count".to_string(), json!(count));
        }
        if let Some(rps) = results.update_rps {
            results_obj.insert("update_rps".to_string(), json!(rps));
        }
        if let Some(t) = results.update_mean_time {
            results_obj.insert("update_mean_time".to_string(), json!(t));
        }
        if let Some(t) = results.update_p50_time {
            results_obj.insert("update_p50_time".to_string(), json!(t));
        }
        if let Some(t) = results.update_p95_time {
            results_obj.insert("update_p95_time".to_string(), json!(t));
        }
        if let Some(t) = results.update_p99_time {
            results_obj.insert("update_p99_time".to_string(), json!(t));
        }
        if let Some(ref lats) = results.update_latencies {
            results_obj.insert(
                "update_latency_hdr".to_string(),
                crate::latency_digest::latency_hdr(lats),
            );
            if dump_raw_latencies {
                results_obj.insert("update_latencies".to_string(), json!(lats));
            }
        }
    }

    // Embed server reproducibility metadata (Redis-wire engines only). Both
    // before and after are stored; a None side is serialized as null. Omitted
    // entirely when the engine reports no metadata (non-Redis engines).
    if server_metadata_before.is_some() || server_metadata_after.is_some() {
        if let Some(obj) = result.as_object_mut() {
            obj.insert(
                "server_metadata".to_string(),
                json!({
                    "before": server_metadata_before,
                    "after": server_metadata_after,
                }),
            );
        }
    }

    result
}

/// Save upload results to JSON file
fn save_upload_results(
    engine_name: &str,
    dataset_name: &str,
    stats: &crate::engine::UploadStats,
    number_of_shards: Option<&serde_json::Value>,
) -> Result<(), String> {
    let timestamp = Local::now().format("%Y-%m-%d-%H-%M-%S");
    let filename = format!(
        "{}-{}-upload-0-{}-{}.json",
        engine_name, dataset_name, stats.upload_count, timestamp
    );

    crate::effective_config::set_phase("upload");
    let result = build_upload_result_json(
        engine_name,
        dataset_name,
        stats,
        number_of_shards,
        &crate::effective_config::snapshot(),
    );

    let path = results_dir().join(&filename);
    fs::write(&path, serde_json::to_string_pretty(&result).unwrap())
        .map_err(|e| format!("Failed to save results: {}", e))?;

    println!("Results saved to: {:?}", path);
    Ok(())
}

/// Build the upload-result JSON document. Split out for the same reason as
/// [`build_search_result_json`]: the emitted key set is then unit-testable
/// without writing a file, and deleting the provenance block from it fails a
/// test rather than passing silently.
fn build_upload_result_json(
    engine_name: &str,
    dataset_name: &str,
    stats: &crate::engine::UploadStats,
    number_of_shards: Option<&serde_json::Value>,
    engine_params: &serde_json::Value,
) -> serde_json::Value {
    let mut result = json!({
        "params": {
            "experiment": engine_name,
            "dataset": dataset_name,
            "parallel": stats.parallel,
            "batch_size": stats.batch_size,
        },
        "results": {
            "upload_time": stats.upload_time,
            "total_time": stats.total_time,
            "upload_count": stats.upload_count,
            "memory_usage": stats.memory_usage,
        }
    });
    // Shard count is the one collection param that changes indexing throughput
    // outright, so it is recorded rather than left to be inferred from the config
    // name (#211). Emitted only for the engines where it means something; it also
    // appears inside `engine_params.effective` below.
    if let Some(shards) = number_of_shards {
        result["params"]["number_of_shards"] = shards.clone();
    }

    // Same provenance block as the search files (#212), tagged `phase: "upload"`.
    // It is scoped to what had been resolved when this file was written, so a
    // knob the engine only resolves during search is absent here. The phase tag
    // is what keeps that readable as sequencing: without it the block's own
    // semantics ("a variable never read does not appear") would make the upload
    // file's silence assert that the run never consulted the variable, which is
    // false at run level. Compare the search file, or the summary, for the
    // whole-run view.
    result["params"]["engine_params"] = engine_params.clone();

    result
}

#[cfg(test)]
mod tests {
    use super::parse_update_search_ratio;
    use super::run_with_search_watchdog;
    use crate::engine::UpdateSearchRatio;

    // Watchdog disabled (timeout <= 0, non-finite): must run `f` inline on the
    // current thread and return its value verbatim — the default, unchanged path.
    #[test]
    fn watchdog_disabled_runs_inline_and_returns_value() {
        assert_eq!(run_with_search_watchdog(0.0, "off", || 42), 42);
        assert_eq!(run_with_search_watchdog(-1.0, "neg", || 7), 7);
        assert_eq!(run_with_search_watchdog(f64::NAN, "nan", || 5), 5);
        assert_eq!(
            run_with_search_watchdog(f64::INFINITY, "inf", || 9),
            9,
            "infinite (non-finite) timeout disables rather than never-firing"
        );
    }

    // Watchdog enabled but `f` completes well within the limit: the monitor
    // thread must observe completion (tx drop → Disconnected) and let the call
    // return the closure's value without aborting.
    #[test]
    fn watchdog_enabled_fast_completion_returns_value() {
        let out = run_with_search_watchdog(30.0, "fast", || {
            let mut acc = 0u64;
            for i in 0..1000 {
                acc += i;
            }
            acc
        });
        assert_eq!(out, 499_500);
    }

    #[test]
    fn parses_valid_ratio() {
        assert_eq!(
            parse_update_search_ratio("1:10"),
            Ok(UpdateSearchRatio {
                updates: 1,
                searches: 10,
            })
        );
    }

    #[test]
    fn allows_zero_updates() {
        // Zero updates is valid (search-heavy phase); only searches must be > 0.
        assert_eq!(
            parse_update_search_ratio("0:5"),
            Ok(UpdateSearchRatio {
                updates: 0,
                searches: 5,
            })
        );
    }

    #[test]
    fn rejects_wrong_arity() {
        let err = parse_update_search_ratio("1:2:3").unwrap_err();
        assert_eq!(
            err,
            "Invalid update-search-ratio format: '1:2:3'. Expected 'U:S' (e.g., '1:10')"
        );
    }

    #[test]
    fn rejects_invalid_update_count() {
        let err = parse_update_search_ratio("x:2").unwrap_err();
        assert_eq!(err, "Invalid update count: 'x'");
    }

    #[test]
    fn rejects_invalid_search_count() {
        let err = parse_update_search_ratio("1:y").unwrap_err();
        assert_eq!(err, "Invalid search count: 'y'");
    }

    #[test]
    fn rejects_zero_searches() {
        // searches == 0 would divide-by-zero later, so it is rejected up front.
        let err = parse_update_search_ratio("1:0").unwrap_err();
        assert_eq!(err, "Search count must be > 0");
    }

    mod metrics_keys {
        use super::super::{finish_calibration, metrics_schema};
        use crate::ground_truth::GroundTruthProfile;

        fn rows(n: usize, width: usize) -> Vec<Vec<i64>> {
            (0..n)
                .map(|q| ((q * 1000) as i64..(q * 1000 + width) as i64).collect())
                .collect()
        }

        /// The whole point of #217: we must never emit a key named
        /// `mean_precisions`, because upstream publishes recall@top under it.
        #[test]
        fn schema_never_reuses_the_upstream_key_name() {
            let schema = metrics_schema(None);
            let obj = schema.as_object().unwrap();
            assert!(
                !obj.contains_key("mean_precisions"),
                "`mean_precisions` must not be an emitted key: {:?}",
                obj.keys().collect::<Vec<_>>()
            );
            assert!(obj.contains_key("mean_precision_at_returned"));
            assert!(obj.contains_key("upstream_qdrant_mean_precisions"));
            assert_eq!(schema["version"], 2);
        }

        /// Full-width ground truth: our `mean_recall` IS upstream's
        /// `mean_precisions`, and the file says so.
        #[test]
        fn full_width_ground_truth_declares_mean_recall_comparable() {
            let gt = GroundTruthProfile::from_rows(rows(100, 100));
            let stats = gt.stats(100).unwrap();
            let schema = metrics_schema(Some(&stats));
            assert_eq!(
                schema["comparable_to_upstream_mean_precisions"],
                "mean_recall"
            );
            assert!(schema.get("comparability_note").is_none());
            assert_eq!(
                schema["ground_truth"]["queries_with_fewer_than_top_neighbours"],
                0
            );
            assert_eq!(schema["ground_truth"]["recall_at_top_ceiling"], 1.0);
        }

        /// Short ground-truth rows: nothing in the file is comparable to
        /// upstream's `mean_precisions`, and the file says that too rather than
        /// inviting the overlay.
        #[test]
        fn short_ground_truth_declares_nothing_comparable() {
            let mut r = rows(9, 100);
            r.push(vec![42]); // one query with a single true neighbour
            let gt = GroundTruthProfile::from_rows(r);
            let stats = gt.stats(100).unwrap();
            let schema = metrics_schema(Some(&stats));
            assert!(schema["comparable_to_upstream_mean_precisions"].is_null());
            let note = schema["comparability_note"].as_str().unwrap();
            assert!(note.contains("1 of 10"), "{}", note);
            assert!(note.contains("0.9010"), "{}", note);
        }

        #[test]
        fn calibration_that_reaches_its_target_is_reported_clean() {
            let out = finish_calibration("ef", 128, 0.9612, 0.95, Some(1.0), None, 10, 1000);
            assert!(out.reached_target);
            assert!(out.note.is_none());
            let j = out.to_json();
            assert_eq!(j["metric"], "mean_precision_at_returned");
            assert_eq!(j["reached_target"], true);
            assert_eq!(j["value"], 128);
        }

        /// The #217 calibrator failure mode: ground truth narrower than `top`
        /// caps precision, the sweep pins the knob at its maximum, and the run
        /// used to print that as a successful calibration.
        #[test]
        fn unreachable_target_is_reported_not_silently_accepted() {
            let gt = GroundTruthProfile::from_rows(vec![vec![1, 2, 3]; 8]);
            let stats = gt.stats(100).unwrap();
            let note = stats.unreachable_target_note(0.95);
            assert!(note.is_some(), "0.95 vs a 0.03 ceiling must be flagged");
            let out = finish_calibration(
                "ef",
                1000,
                0.03,
                0.95,
                Some(stats.recall_at_top_ceiling),
                note,
                100,
                1000,
            );
            assert!(!out.reached_target);
            let j = out.to_json();
            assert_eq!(j["reached_target"], false);
            assert!((j["ground_truth_ceiling"].as_f64().unwrap() - 0.03).abs() < 1e-9);
            let note = j["note"].as_str().unwrap();
            assert!(note.contains("ceiling"), "{}", note);
        }

        /// End-to-end over the document actually written to `results/`: the
        /// colliding key names must be gone, the unambiguous ones present, and the
        /// definitions must ship with the numbers.
        #[test]
        fn emitted_document_has_no_colliding_key_names() {
            use super::super::build_search_result_json;
            use crate::config::SearchParams;
            use crate::engine::SearchResults;

            let params: SearchParams =
                serde_json::from_value(serde_json::json!({"parallel": 1, "top": 10})).unwrap();
            let results = SearchResults {
                top: 10,
                mean_precision_at_returned: 1.0,
                mean_recall: 0.5,
                precisions_at_returned: vec![1.0, 1.0],
                recalls: vec![0.5, 0.5],
                mrrs: vec![1.0, 1.0],
                ndcgs: vec![1.0, 1.0],
                latencies: vec![0.001, 0.002],
                ..Default::default()
            };
            let gt = GroundTruthProfile::from_rows(rows(2, 10));
            let doc = build_search_result_json(
                "redis-test",
                "glove-25-angular",
                &params,
                &results,
                None,
                None,
                true, // --dump-raw-latencies: exercises the raw-array key too
                None,
                Some(&gt),
                None,
                None,
                &serde_json::json!({}),
            );

            let res = doc["results"].as_object().unwrap();
            // The 5-of-10 case from #217: our precision is 1.0 while upstream
            // would publish 0.5 under `mean_precisions`. Neither the aggregate nor
            // the per-query array may reuse an upstream key name.
            assert!(
                !res.contains_key("mean_precisions"),
                "emitted `mean_precisions`: {:?}",
                res.keys().collect::<Vec<_>>()
            );
            assert!(!res.contains_key("precisions"));
            assert!(!res.contains_key("precision_dist"));
            assert_eq!(res["mean_precision_at_returned"], 1.0);
            assert_eq!(res["mean_recall"], 0.5);
            assert_eq!(res["precisions_at_returned"], serde_json::json!([1.0, 1.0]));
            assert!(res.contains_key("precision_at_returned_dist"));

            // Definitions travel with the data, not just with the commit message.
            let schema = doc["metrics_schema"].as_object().unwrap();
            assert_eq!(schema["version"], 2);
            assert_eq!(doc["metrics_schema"]["ground_truth"]["top"], 10);
            assert_eq!(
                doc["metrics_schema"]["comparable_to_upstream_mean_precisions"],
                "mean_recall"
            );
        }

        /// A calibrated run records how the knob was chosen, including a target it
        /// failed to reach.
        #[test]
        fn emitted_document_carries_the_calibration_outcome() {
            use super::super::{build_search_result_json, finish_calibration};
            use crate::config::SearchParams;
            use crate::engine::SearchResults;

            let outcome = finish_calibration("ef", 1000, 0.24, 0.95, Some(0.25), None, 100, 1000);
            let doc = build_search_result_json(
                "redis-test",
                "h-and-m-2048-angular",
                &serde_json::from_value::<SearchParams>(
                    serde_json::json!({"parallel": 1, "top": 100}),
                )
                .unwrap(),
                &SearchResults {
                    top: 100,
                    ..Default::default()
                },
                None,
                None,
                false,
                None,
                None,
                Some(&outcome.to_json()),
                None,
                &serde_json::json!({}),
            );
            let cal = &doc["params"]["calibration"];
            assert_eq!(cal["reached_target"], false);
            assert_eq!(cal["metric"], "mean_precision_at_returned");
            assert_eq!(cal["value"], 1000);
            assert!(cal["note"].as_str().unwrap().contains("topped out"));
        }

        /// Target below the ceiling but still not reached by the sweep (a genuinely
        /// weak engine): still reported, with the saturated-knob detail.
        #[test]
        fn target_missed_without_a_ceiling_explanation_still_warns() {
            let out = finish_calibration("ef", 1000, 0.80, 0.95, Some(1.0), None, 10, 1000);
            assert!(!out.reached_target);
            let note = out.note.unwrap();
            assert!(
                note.contains("pinned at the maximum swept value"),
                "{}",
                note
            );
        }
    }

    /// The capability #212 asks for: an artifact you can attribute to the
    /// settings that produced it.
    mod engine_params {
        use super::super::build_search_result_json;
        use crate::config::SearchParams;
        use crate::engine::SearchResults;

        fn doc(engine_params: &serde_json::Value) -> serde_json::Value {
            build_search_result_json(
                "redis-m-16-ef-128",
                "glove-100-angular",
                &serde_json::from_value::<SearchParams>(
                    serde_json::json!({"parallel": 8, "top": 10}),
                )
                .unwrap(),
                &SearchResults {
                    top: 10,
                    ..Default::default()
                },
                None,
                None,
                false,
                None,
                None,
                None,
                None,
                engine_params,
            )
        }

        /// **This is the regression test for #212.**
        ///
        /// Two runs of ONE committed configuration, differing only in an
        /// environment variable, must produce artifacts that can be told apart.
        ///
        /// Before this change they could not be: `params` carried only the
        /// configuration *name*, the dataset, and the search knobs, and none of
        /// those move when you export a variable. Delete the `engine_params`
        /// assignment from `build_search_result_json` and this test fails with
        /// two byte-identical documents — which is exactly the state that made
        /// two historical claims this cycle provable only from git history and
        /// not from the artifacts themselves.
        ///
        /// The knob is driven through [`crate::engine::build_redis_url`], real
        /// engine code on the real resolution path, not a stand-in.
        #[test]
        fn same_config_differing_only_by_an_env_knob_yields_distinguishable_artifacts() {
            let _l = crate::effective_config::test_lock();
            let prev = std::env::var("REDIS_PORT").ok();

            let run = |port: Option<&str>| {
                match port {
                    Some(p) => std::env::set_var("REDIS_PORT", p),
                    None => std::env::remove_var("REDIS_PORT"),
                }
                crate::effective_config::reset();
                let url = crate::engine::build_redis_url("bench-host");
                (doc(&crate::effective_config::snapshot()), url)
            };

            let (default_port, default_url) = run(None);
            let (tuned_port, tuned_url) = run(Some("7777"));

            // Anchor the recorded value to the value the run USED. Without this
            // both sides of the inequality below come from the same recorder
            // call, and the test passes even if the recorder reports a port the
            // engine never dialled — a provenance test that cannot detect the
            // provenance being wrong.
            assert!(
                default_url.ends_with(":6379/"),
                "default url was {default_url}"
            );
            assert!(tuned_url.ends_with(":7777/"), "tuned url was {tuned_url}");

            match &prev {
                Some(v) => std::env::set_var("REDIS_PORT", v),
                None => std::env::remove_var("REDIS_PORT"),
            }

            assert_ne!(
                default_port, tuned_port,
                "two runs that used different settings produced identical artifacts"
            );
            assert_eq!(
                default_port["params"]["engine_params"]["effective"]["REDIS_PORT"],
                6379
            );
            assert_eq!(
                tuned_port["params"]["engine_params"]["effective"]["REDIS_PORT"],
                7777
            );
            // Everything a consumer already reads is untouched; the difference is
            // confined to the provenance block.
            assert_eq!(default_port["results"], tuned_port["results"]);
            assert_eq!(
                default_port["params"]["experiment"],
                tuned_port["params"]["experiment"]
            );
        }

        /// `Args` booleans deliberately absent from [`super::super::invocation_provenance`],
        /// with the reason. Asserted to still exist, so a renamed or deleted flag
        /// fails rather than leaving a stale excuse behind.
        ///
        /// Lives inside this test module rather than beside the function: a bare
        /// `#[cfg(test)] const` at item level has no brace, and the guard's
        /// `strip_test_modules` used to brace-count from the next `{` it found —
        /// silently deleting the production code in between.
        const INVOCATION_EXCLUDED_FLAGS: &[(&str, &str)] = &[(
            "verbose",
            "console output only: changes what is printed, never what is measured \
         or which experiments run",
        )];

        /// Every `Args` BOOLEAN that changes what is measured must be recorded.
        ///
        /// Booleans only: the scan reads `pub <name>: bool,`, so a new
        /// non-boolean measuring flag is invisible to it. `repetitions` and
        /// `warmup_seconds` are recorded by hand for that reason (#274).
        ///
        /// `invocation` was the one hand-maintained inventory in this PR while
        /// every other (`KNOWN_UNREAD`, `KNOWN_UNRECORDED`) is bidirectionally
        /// CI-asserted — and it had already drifted by five flags, including
        /// `skip_vector_index`, which decides whether a vector index exists at
        /// all. This is why the next flag would have drifted in silently too.
        #[test]
        fn every_measuring_flag_is_recorded_in_the_invocation() {
            use clap::Parser;
            let src = std::fs::read_to_string(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("src/bin/vector_db_benchmark/cli.rs"),
            )
            .expect("cli.rs");
            let declared: Vec<String> = src
                .lines()
                .filter_map(|l| {
                    let t = l.trim();
                    t.strip_prefix("pub ")
                        .and_then(|r| r.strip_suffix(": bool,"))
                        .map(str::to_string)
                })
                .collect();
            assert!(
                declared.len() >= 10,
                "the `pub <name>: bool,` scan found only {declared:?} — cli.rs \
                 formatting changed and this guard is no longer reading it"
            );

            let recorded =
                super::super::invocation_provenance(&crate::cli::Args::parse_from(["vdbb"]));
            let recorded = recorded.as_object().unwrap();
            let excused: Vec<&str> = INVOCATION_EXCLUDED_FLAGS.iter().map(|(f, _)| *f).collect();

            let missing: Vec<&String> = declared
                .iter()
                .filter(|f| !recorded.contains_key(*f) && !excused.contains(&f.as_str()))
                .collect();
            assert!(
                missing.is_empty(),
                "these Args flags change what a run does but reach no artifact: \
                 {missing:?}. Record them in `invocation_provenance`, or excuse \
                 them in INVOCATION_EXCLUDED_FLAGS with a reason."
            );
            let stale: Vec<&&str> = excused
                .iter()
                .filter(|f| !declared.contains(&f.to_string()))
                .collect();
            assert!(
                stale.is_empty(),
                "INVOCATION_EXCLUDED_FLAGS names flags that no longer exist: {stale:?}"
            );

            // Reverse leg. Without it a fabricated key — say a renamed flag
            // leaving `skip_vector_index_LEGACY` behind — stayed green forever,
            // which made this inventory weaker than `KNOWN_UNRECORDED`.
            // `NON_BOOL_INVOCATION_KEYS` are the deliberately hand-maintained
            // non-boolean entries.
            // REQUIRED, not merely permitted. Listing them in
            // NON_BOOL_INVOCATION_KEYS made them look guarded while deleting
            // either from `invocation_provenance` stayed green — the same
            // unguarded-claim shape as the original defect.
            for required in [
                "host",
                "repetitions",
                "warmup_seconds",
                "configurations_dir",
                "results_dir",
            ] {
                assert!(
                    recorded.contains_key(required),
                    "`invocation.{required}` is documented as recorded and is not"
                );
            }

            const NON_BOOL_INVOCATION_KEYS: &[&str] = &[
                "host",
                "engines_file",
                "repetitions",
                "warmup_seconds",
                "configurations_dir",
                "results_dir",
            ];
            let fabricated: Vec<&String> = recorded
                .keys()
                .filter(|k| {
                    !NON_BOOL_INVOCATION_KEYS.contains(&k.as_str()) && !declared.contains(k)
                })
                .collect();
            assert!(
                fabricated.is_empty(),
                "`invocation` publishes keys that map to no `Args` field: \
                 {fabricated:?}. A reader would take these for real flags."
            );
        }

        /// Drives the SAME entry point `experiment::run` uses, so deleting the
        /// reset — or the whole call — fails here.
        ///
        /// An earlier version called `effective_config::reset()` directly and a
        /// mutation campaign showed it: deleting the production `reset()` left
        /// 754/754 green while a sweep's later configurations inherited the
        /// earlier ones' knobs. A test that reaches past the wiring it is meant
        /// to protect cannot detect the wiring being removed.
        fn begin(cfg_name: &str, args: &crate::cli::Args) {
            let raw = serde_json::json!({"name": cfg_name, "engine": "redis"});
            let mut cfg: crate::config::EngineConfig = serde_json::from_value(raw.clone()).unwrap();
            cfg.raw = Some(raw);
            let _recording = crate::effective_config::begin_experiment(
                &cfg,
                super::super::invocation_provenance(args),
            );
        }

        #[test]
        fn a_sweep_does_not_inherit_the_previous_configs_knobs() {
            use clap::Parser;
            let _l = crate::effective_config::test_lock();
            let prev = std::env::var("REDIS_PORT").ok();
            let args = crate::cli::Args::parse_from(["vdbb", "--host", "db-alpha"]);

            std::env::set_var("REDIS_PORT", "7777");
            begin("config-a", &args);
            let _ = crate::engine::build_redis_url("h");
            let config_a = doc(&crate::effective_config::snapshot());

            // Second configuration of the same sweep, in the same process.
            std::env::remove_var("REDIS_PORT");
            begin("config-b", &args);
            let _ = crate::engine::build_redis_url("h");
            let config_b = doc(&crate::effective_config::snapshot());

            match &prev {
                Some(v) => std::env::set_var("REDIS_PORT", v),
                None => std::env::remove_var("REDIS_PORT"),
            }

            let ep = |d: &serde_json::Value| d["params"]["engine_params"].clone();
            assert_eq!(ep(&config_a)["effective"]["REDIS_PORT"], 7777);
            assert_eq!(
                ep(&config_b)["effective"]["REDIS_PORT"],
                6379,
                "configuration B inherited configuration A's port — the per-experiment \
                 reset is not wired"
            );
            assert_eq!(
                ep(&config_b)["env"]["REDIS_PORT"],
                serde_json::Value::Null,
                "configuration A's observation survived into configuration B"
            );
        }

        /// The invocation facts that live in neither the config file nor the
        /// environment, asserted THROUGH the wiring: two runs against two
        /// different servers used to produce byte-identical provenance.
        #[test]
        fn invocation_reaches_the_artifact_with_host_and_skip_upload() {
            use clap::Parser;
            let _l = crate::effective_config::test_lock();

            begin(
                "cfg",
                &crate::cli::Args::parse_from(["vdbb", "--host", "db-alpha"]),
            );
            let alpha = doc(&crate::effective_config::snapshot());

            begin(
                "cfg",
                &crate::cli::Args::parse_from(["vdbb", "--host", "db-beta", "--skip-upload"]),
            );
            let beta = doc(&crate::effective_config::snapshot());

            let inv = |d: &serde_json::Value| d["params"]["engine_params"]["invocation"].clone();
            assert_eq!(inv(&alpha)["host"], "db-alpha");
            assert_eq!(inv(&beta)["host"], "db-beta");
            assert_eq!(inv(&alpha)["skip_upload"], false);
            assert_eq!(
                inv(&beta)["skip_upload"],
                true,
                "a run that searched an index it did not build must say so"
            );
            assert_ne!(
                alpha, beta,
                "same config, different server: the artifacts must differ"
            );
        }

        /// The declared block reaches the artifact through the same entry point.
        #[test]
        fn declared_config_reaches_the_artifact() {
            use clap::Parser;
            let _l = crate::effective_config::test_lock();
            begin("cfg", &crate::cli::Args::parse_from(["vdbb"]));
            let d = doc(&crate::effective_config::snapshot());
            assert!(d["params"]["engine_params"]["declared"].is_object());
        }

        /// The upload file carries it too, tagged with its phase — and #212's
        /// indistinguishability has to be closed there as well, since
        /// `upload_time` is one of the numbers the issue names.
        #[test]
        fn upload_result_carries_the_provenance_block_tagged_upload() {
            use super::super::build_upload_result_json;
            use crate::engine::UploadStats;

            let stats = UploadStats {
                parallel: 8,
                batch_size: 64,
                ..Default::default()
            };
            let ep = serde_json::json!({
                "schema_version": 1,
                "phase": "upload",
                "effective": {"upload_parallel": 8},
            });
            let d = build_upload_result_json("redis-x", "glove", &stats, None, &ep);
            assert_eq!(d["params"]["engine_params"]["phase"], "upload");
            assert_eq!(
                d["params"]["engine_params"]["effective"]["upload_parallel"],
                8
            );
            // Existing keys keep their place.
            assert_eq!(d["params"]["parallel"], 8);
            assert_eq!(d["results"]["upload_count"], 0);

            // Two uploads differing only in a knob are distinguishable.
            let other = build_upload_result_json(
                "redis-x",
                "glove",
                &stats,
                None,
                &serde_json::json!({"schema_version": 1, "phase": "upload",
                                    "effective": {"upload_parallel": 32}}),
            );
            assert_ne!(d, other);
            assert_eq!(d["results"], other["results"]);
        }

        /// Every connection-string shape, asserted against the BYTES ON DISK.
        ///
        /// The previous version of this test asserted on the serialized document
        /// in memory, justified by a comment claiming "every client rejects
        /// `?password=` as an unknown option before a file is written". That was
        /// false — `REDIS_URI=redis://127.0.0.1:6411/?password=X` completes with
        /// rc=0 and writes three files — and building the test on that premise
        /// is precisely what let seven further shapes survive a round. So this
        /// writes real files through the real serializer and greps them.
        #[test]
        fn no_shape_reaches_the_files_on_disk() {
            let _l = crate::effective_config::test_lock();
            let dir = tempfile::tempdir().unwrap();

            for (label, host) in crate::effective_config::CONNECTION_SHAPE_CORPUS {
                crate::effective_config::reset();
                crate::effective_config::set_invocation(serde_json::json!({"host": host}));
                let snap = crate::effective_config::snapshot();

                // Both artifact kinds, through the same serializer the save
                // functions use.
                let search = serde_json::to_string_pretty(&doc(&snap)).unwrap();
                let upload = serde_json::to_string_pretty(&super::super::build_upload_result_json(
                    "e",
                    "d",
                    &crate::engine::UploadStats::default(),
                    None,
                    &snap,
                ))
                .unwrap();

                for (kind, bytes) in [("search", &search), ("upload", &upload)] {
                    let path = dir.path().join(format!("{kind}.json"));
                    std::fs::write(&path, bytes).unwrap();
                    let on_disk = std::fs::read_to_string(&path).unwrap();
                    assert!(
                        !on_disk.contains("CANARY"),
                        "[{label}] {kind} file on disk contains the credential:\n{on_disk}"
                    );
                }
            }

            // And the summary, which writes its own file for real.
            crate::effective_config::reset();
            crate::effective_config::set_invocation(serde_json::json!({
                "host": "mongodb://h/db?w=majority&password=CANARY"
            }));
            crate::summary::save_summary(
                "e",
                "d",
                &[],
                None,
                dir.path(),
                &crate::effective_config::snapshot(),
            )
            .unwrap();
            let summary = std::fs::read_to_string(dir.path().join("e-d-summary.json")).unwrap();
            assert!(!summary.contains("CANARY"), "{summary}");
        }

        /// The block is present, and shaped, on every search result.
        #[test]
        fn search_result_carries_the_provenance_block() {
            let d = doc(&serde_json::json!({
                "schema_version": 1,
                "declared": {"collection_params": {"hnsw_config": {"M": 16}}},
                "effective": {"m": 16},
                "env": {"REDIS_PORT": null},
                "overridden": [],
                "ignored_declared_keys": [],
            }));
            let ep = &d["params"]["engine_params"];
            assert_eq!(ep["schema_version"], 1);
            assert_eq!(ep["declared"]["collection_params"]["hnsw_config"]["M"], 16);
            assert_eq!(ep["effective"]["m"], 16);
            // Pre-existing keys still sit where consumers expect them.
            assert_eq!(d["params"]["parallel"], 8);
            assert_eq!(d["params"]["top"], 10);
        }
    }
}

#[cfg(test)]
mod reuse_precondition_tests {
    use super::{classify_reuse_precondition, ReusePrecondition};
    use crate::engine::CorpusCount;

    // A corpus shorter than the dataset declares is the case that publishes a
    // wrong recall under a config name claiming otherwise (issue #238).
    #[test]
    fn short_corpus_is_short() {
        assert_eq!(
            classify_reuse_precondition(Some(400), Some(CorpusCount::exact(200)), "redis-x"),
            ReusePrecondition::Short {
                actual: 200,
                expected: 400,
                approximate: false
            }
        );
    }

    // A missing index/collection reports 0 rows. Same verdict as a truncated
    // one: the corpus the flags promised to reuse is not there.
    #[test]
    fn missing_corpus_is_short_not_ok() {
        assert_eq!(
            classify_reuse_precondition(Some(400), Some(CorpusCount::exact(0)), "redis-x"),
            ReusePrecondition::Short {
                actual: 0,
                expected: 400,
                approximate: false
            }
        );
    }

    #[test]
    fn exact_match_is_ok() {
        assert_eq!(
            classify_reuse_precondition(Some(400), Some(CorpusCount::exact(400)), "redis-x"),
            ReusePrecondition::Ok {
                actual: 400,
                expected: 400,
                approximate: false
            }
        );
    }

    // Extra rows affect the number too, but unlike a shortfall they are often
    // deliberate (a shared prefix, a superset upload) — warn, do not abort.
    #[test]
    fn surplus_is_classified_as_surplus() {
        assert_eq!(
            classify_reuse_precondition(Some(400), Some(CorpusCount::exact(401)), "redis-x"),
            ReusePrecondition::Surplus {
                actual: 401,
                expected: 400,
                approximate: false
            }
        );
    }

    // An ESTIMATE carries through to the verdict, which is what lets the handler
    // downgrade a short estimate to a warning instead of aborting on a number
    // that is allowed to be wrong.
    #[test]
    fn approximate_flag_survives_classification() {
        assert_eq!(
            classify_reuse_precondition(Some(400), Some(CorpusCount::estimated(200)), "pg-x"),
            ReusePrecondition::Short {
                actual: 200,
                expected: 400,
                approximate: true
            }
        );
        assert_eq!(
            classify_reuse_precondition(Some(400), Some(CorpusCount::estimated(400)), "pg-x"),
            ReusePrecondition::Ok {
                actual: 400,
                expected: 400,
                approximate: true
            }
        );
    }

    // Nothing to compare against must never be silently reported as "Ok" — the
    // runner has to say the reuse went unverified.
    #[test]
    fn unknown_expected_or_actual_is_unverifiable() {
        assert!(matches!(
            classify_reuse_precondition(None, Some(CorpusCount::exact(400)), "redis-x"),
            ReusePrecondition::Unverifiable(_)
        ));
        let why = match classify_reuse_precondition(Some(400), None, "chroma-y") {
            ReusePrecondition::Unverifiable(w) => w,
            other => panic!("expected Unverifiable, got {other:?}"),
        };
        assert!(
            why.contains("chroma-y"),
            "the note must name the config it could not count: {why}"
        );
        // The wording must blame this tool, not the database: Chroma, Milvus and
        // Weaviate all expose a count, we simply have not wired one up.
        assert!(
            why.contains("implemented"),
            "the note must read as a gap in this tool, not a database limitation: {why}"
        );
    }

    // Zero expected rows cannot make a zero-row server look short.
    #[test]
    fn zero_expected_never_fires() {
        assert_eq!(
            classify_reuse_precondition(Some(0), Some(CorpusCount::exact(0)), "redis-x"),
            ReusePrecondition::Ok {
                actual: 0,
                expected: 0,
                approximate: false
            }
        );
    }
}
