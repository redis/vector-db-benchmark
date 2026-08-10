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
    // same destructive engine (redis/valkey/dragonfly/kividb/vectorsets/mongodb)
    // may derive the same index namespace, or a sweep would silently overwrite one config's graph
    // and keyspace with another's (the exact bug this fix closes). Also fires when
    // an `*_EXACT` pin is set with >1 config for that engine (every
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
                // #236: VectorSets' "index" is the single Redis key holding the
                // vector set, derived by the same helper, so it collides the same
                // way and belongs in the same guard.
                "vectorsets" => "VECTORSETS_INDEX_NAME",
                // #306: MongoDB's destructible namespace is the COLLECTION, not
                // the search index — `configure()` calls `collection.drop()`, and
                // the search index lives inside the collection it drops. So the
                // collection is what two configs must not share, and
                // `MONGODB_COLLECTION` (not `MONGODB_INDEX_NAME`) is the base
                // whose `_EXACT` pin can re-collapse a sweep.
                "mongodb" => "MONGODB_COLLECTION",
                _ => continue,
            };
            // MongoDB's name is bounded to fit the 255-byte `<db>.<collection>`
            // namespace, so it must come from the engine rather than from a bare
            // `derive_index_name` here — otherwise the guard would compare names
            // the engine never uses.
            let idx = if engine_type == "mongodb" {
                crate::engine::mongodb_collection_name(&config.name)
            } else {
                derive_index_name(base_env, "idx", &config.name)
            };
            if let Some(prev) =
                seen.insert((engine_type.to_string(), idx.clone()), config.name.clone())
            {
                let exact_hint = if index_name_exact(base_env) {
                    format!(" ({base_env}_EXACT is set — exact mode requires a single config per engine.)")
                } else {
                    String::new()
                };
                let object = if engine_type == "mongodb" {
                    "collection"
                } else {
                    "index"
                };
                return Err(format!(
                    "Configs '{}' and '{}' derive the same index namespace '{}' (the {} {}); \
                     rename them — a sweep would silently overwrite one with the other \
                     (issue #151-4).{}",
                    prev, config.name, idx, engine_type, object, exact_hint
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
        "configurations_dir": tildeify(
            &crate::config::project_root().join("experiments/configurations"),
        ),
        "results_dir": tildeify(&results_dir()),
    })
}

/// Replace the invoking user's home directory with `~`.
///
/// These paths land in every artifact, and an absolute one publishes the local
/// username to anyone the file is shared with. The part that carries provenance
/// — which subtree the configs and results came from — survives.
///
/// A path OUTSIDE `$HOME` is still published absolute (`/srv/bench/...`), which
/// can name an internal mount. Deliberate: the directory the sweep globbed is
/// the fact that decides which configs ran, and a digest would make it useless.
fn tildeify(path: &std::path::Path) -> String {
    let shown = path.display().to_string();
    match std::env::var_os("HOME").map(std::path::PathBuf::from) {
        Some(home) if !home.as_os_str().is_empty() => {
            let home = home.display().to_string();
            match shown.strip_prefix(&home) {
                Some(rest) => format!("~{rest}"),
                None => shown,
            }
        }
        _ => shown,
    }
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
    /// No server-side count came back: `corpus_row_count` returned `Ok(None)`,
    /// which happens either because no probe is wired up for this engine or
    /// because the engine's reply carried no count (Qdrant with no
    /// `points_count`). NOT a probe failure — that is an `Err` and is handled
    /// before classification. Proceed with a note: an engine we have not wired
    /// up must not become unusable under `--skip-upload`.
    ///
    /// Returned whenever the server side is missing, whatever the dataset side
    /// says. With neither side available there is nothing to compare in either
    /// direction, so a missing expected count adds no information and must not
    /// escalate this to an abort (#290 review).
    NoServerCount(String),
    /// The server side IS available but the DATASET side is not: the expected
    /// row count could not be determined. Fatal unless `--allow-partial-corpus`
    /// (issue #290). This is the same outcome as `Short` with LESS information,
    /// not a milder one: the server may hold the whole corpus, part of it, or
    /// none of it, and recall is scored against ground truth for the FULL
    /// corpus either way.
    CorpusSizeUnknown(String),
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
            ReusePrecondition::NoServerCount(_) => "unverified",
            ReusePrecondition::CorpusSizeUnknown(_) => "corpus_size_unknown",
            ReusePrecondition::Surplus { .. } => "surplus",
            ReusePrecondition::Short { .. } => "short",
        }
    }
}

/// Classify a server-side corpus count against what the dataset declares.
///
/// `expected` is a `Result`, not an `Option`, so the REASON the dataset side is
/// unavailable survives into the verdict instead of being flattened to "unknown"
/// at the call site (issue #290).
pub fn classify_reuse_precondition(
    expected: Result<u64, String>,
    actual: Option<CorpusCount>,
    engine_name: &str,
) -> ReusePrecondition {
    // The SERVER side is tested first, deliberately. If it is missing there is
    // nothing to compare against in either direction, so an unavailable expected
    // count adds no information — escalating to a fatal there would remove
    // `--skip-upload` from exactly the engines that have no probe, protecting no
    // number in exchange (#290 review).
    let Some(actual) = actual else {
        // Phrased as a gap on THIS side of the wire, not a limitation of the
        // database: Chroma, Milvus and Weaviate all expose a count we have not
        // wired up, and Qdrant's probe is wired up but can reply without one.
        // Deliberately distinct from a probe FAILURE, which is an `Err` handled
        // before classification and says "probe failure" out loud.
        return ReusePrecondition::NoServerCount(format!(
            "this tool read no server-side row count for config '{engine_name}' (either none \
             is implemented for its engine yet, or the engine replied without one)"
        ));
    };
    let expected = match expected {
        Ok(v) => v,
        Err(why) => return ReusePrecondition::CorpusSizeUnknown(why),
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
///
/// Returns `Result<u64, _>` rather than `Result<Option<u64>, _>` (issue #290):
/// "there is no count here" is not a success. The three ways it can fail — a
/// read that blew up, a corpus that is not on this machine, and an unmeasurable
/// layout with no declared count — carry different messages, where before all
/// three collapsed into one `None` at the call site.
///
/// For `sparse` (CSR) and `h5-multi` the declared `vector_count` is used
/// WITHOUT requiring every corpus file to be present, which is where this
/// deliberately parts company with `corpus_completeness_target()`. That
/// function's file-presence gate exists to stop an UPLOAD being skipped over a
/// half-present corpus (#188/#224); `--skip-upload` never uploads, so the gate
/// protects nothing here and costs a great deal — `laion-…-1Billion` is
/// `h5-multi` with 100 data parts (~3 TB) whose queries live in a separate
/// file, so requiring all of them would make the check unpassable on exactly
/// the upload-once/benchmark-later workflow it is meant to serve.
fn reuse_expected_rows(dataset: &Dataset) -> Result<u64, String> {
    if let Some(measured) = dataset
        .measured_vector_count()
        .map_err(|e| format!("measuring the corpus on disk failed: {e}"))?
    {
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
        return Ok(measured);
    }
    let declared = dataset
        .config
        .vector_count
        .filter(|&n| n > 0)
        .map(|n| n as u64);
    let dataset_type = dataset.config.dataset_type.as_deref().unwrap_or("<none>");
    // Render the path as a plain string; `config.path` is a serde_json::Value,
    // whose Display adds JSON quotes an operator has to look past.
    let path = dataset
        .config
        .path
        .as_str()
        .map(|s| s.to_string())
        .unwrap_or_else(|| dataset.config.path.to_string());
    if dataset.may_fall_back_to_declared_count() {
        // Nothing to measure by design, so datasets.json IS the answer here.
        return declared.ok_or_else(|| {
            format!(
                "dataset '{}' uses the '{}' layout, which carries no cheap row count to measure, \
                 and datasets.json declares no vector_count for it — so there is no number to \
                 check the server against. Add \"vector_count\" to this dataset's entry in \
                 datasets/datasets.json.",
                dataset.config.name, dataset_type,
            )
        });
    }
    // Two different situations, and conflating them sends the operator to the
    // wrong remedy. `get_path()` returns early when the path already exists, so
    // a dataset directory that is here but has lost its corpus file is NEVER
    // re-fetched, however valid its link — saying "could not be fetched" there
    // would describe an attempt that was never made.
    if dataset.corpus_path_exists() {
        // The remedy is gated on there actually BEING something to re-fetch.
        // 8 of the 57 shipped datasets have no `link` — the locally generated
        // ones, `random-100`, and laion — and for those "delete the directory"
        // is not merely useless but destructive: it also removes `tests.jsonl`
        // and `payloads.jsonl`, which nothing can restore. Advising it would
        // repeat the mistake this branch exists to fix (a message describing an
        // attempt that cannot be made), with data loss attached.
        let remedy = if dataset.config.link.is_some() {
            format!(
                "restore the corpus file — or delete '{path}' to make the next run fetch it \
                 fresh from the dataset's link"
            )
        } else {
            "restore the corpus file. This dataset has no download link, so deleting the \
             directory would NOT re-fetch it — and would destroy the query and payload files \
             sitting beside the missing corpus"
                .to_string()
        };
        return Err(format!(
            "dataset '{}' is present at '{}' but its corpus file is not in it (dataset_type \
             '{}'), so there is no number to check the server against. A dataset whose path \
             already exists is never re-downloaded, so {}",
            dataset.config.name, path, dataset_type, remedy,
        ));
    }
    Err(format!(
        "dataset '{}' has no corpus at '{}' on this machine (dataset_type '{}') and resolving it \
         produced none, so there is no number to check the server against",
        dataset.config.name, path, dataset_type,
    ))
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
/// What is fatal: a short corpus changes the reported number, so an exact count
/// that comes up short is a hard error; everything softer warns. Being unable to
/// determine the dataset's expected count is fatal too (#290), because it is the
/// same outcome as a short corpus with less information, not a milder one — but
/// only once the tool has had its own chance to fetch the corpus (see below) and
/// only when a server-side count IS available to have compared it against.
/// `--allow-partial-corpus` waives both.
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
    // The other way a run measures nothing: `--skip-vector-index` on a dataset
    // with no schema fields has no filter conditions to search for, so the
    // search phase returns before `read_queries()`. Same rationale as
    // `--skip-search` above, and checking it here also keeps the fetch below
    // from downloading a corpus for a run that will publish nothing.
    if args.skip_vector_index
        && !dataset
            .config
            .schema
            .as_ref()
            .and_then(|s| s.as_object())
            .map(|o| !o.is_empty())
            .unwrap_or(false)
    {
        return Ok(None);
    }

    // The SERVER side is probed FIRST — before the dataset is fetched — for the
    // same reason `classify_reuse_precondition` tests it first: if it gives us
    // nothing to compare against, fetching a corpus to measure buys nothing.
    //
    // This ordering is load-bearing, not tidiness. The fetch below can be
    // hundreds of GB, and probing afterwards meant an unreachable or mis-ACL'd
    // server — the case `--help` names — paid for the whole download before
    // aborting, where master aborted having touched nothing.
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

    // Only now, with a server-side number in hand that is worth comparing
    // against, give the tool its own chance to make the corpus measurable before
    // judging that it cannot be (#290 review). `read_queries()` opens with this
    // same `get_path()` call, which fetches a string-path dataset that is not on
    // disk — so measuring first turned "not downloaded yet" into a hard error
    // for every fetchable dataset on a fresh benchmark client.
    //
    // Note what this does NOT claim: that the download is free. A run whose
    // verdict is fatal (`Short`, `CorpusSizeUnknown`) now pays for the fetch
    // before it aborts. That is the price of measuring rather than trusting, and
    // it is bounded to the cases where a measurement could change the verdict.
    // The resolve error is KEPT, not dropped. When the fetch is what failed —
    // a 404, a dead mirror — that status code is the whole answer, and it used
    // to survive only as `Download attempt N/15 failed:` lines on stderr, with
    // the final attempt's error printed nowhere at all.
    let resolve_error: Option<String> = if actual.is_some() {
        dataset.get_path().err()
    } else {
        None
    };

    // Kept as a Result: an Err here is fatal below (#290), so degrading it to
    // "unknown" would be the very thing this check exists to prevent.
    let expected = reuse_expected_rows(dataset).map_err(|why| match &resolve_error {
        Some(e) => format!("{why}; resolving the dataset failed with: {e}"),
        None => why,
    });
    let expected_rows: Option<u64> = expected.as_ref().ok().copied();

    let verdict = classify_reuse_precondition(expected, actual, engine.name());
    let approx_note = |approximate: bool| if approximate { "≈" } else { "" };
    let record = |waived: bool| {
        let mut v = json!({
            "status": verdict.status(),
            "expected_rows": expected_rows,
            "actual_rows": actual.map(|c| c.rows),
            "actual_is_estimate": actual.map(|c| c.approximate).unwrap_or(false),
            "waived_by_allow_partial_corpus": waived,
        });
        // A waived `corpus_size_unknown` is a PUBLISHED result whose count was
        // never checked. Without the reason, the file says the count was unknown
        // but not why — and that is the artifact someone quotes months later.
        // `probe_failed` has carried a `detail` since #238; this matches it.
        if let ReusePrecondition::CorpusSizeUnknown(why) = &verdict {
            v["detail"] = json!(why);
        }
        v
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
        ReusePrecondition::NoServerCount(why) => {
            println!(
                "Experiment stage: Reuse check — SKIPPED ({why}); \
                 --skip-upload is running against an unverified corpus"
            );
            Ok(Some(record(false)))
        }
        ReusePrecondition::CorpusSizeUnknown(why) => {
            // Same policy as Short, for the same reason: this changes the
            // reported number. The difference is only that we cannot say by how
            // much (#290). `actual` is always Some in this arm — a missing
            // server-side count classifies as NoServerCount before we get here.
            // The `None` arm is unreachable today: a missing server-side count
            // classifies as `NoServerCount` above. It is a wording fallback
            // rather than an `unwrap()` so that a future reordering degrades
            // this message instead of panicking in the middle of a sweep.
            let held = actual
                .map(|c| format!("{}{}", if c.approximate { "≈" } else { "" }, c.rows))
                .unwrap_or_else(|| "an unreported number of".to_string());
            let msg = format!(
                "--skip-upload: the reuse check for config '{}' has nothing to compare against — \
                 the expected row count for dataset '{}' could not be determined ({}). The server \
                 holds {} rows, and this run cannot tell whether that is the whole corpus. \
                 Unverified is not the same as intact: recall/precision are scored against ground \
                 truth for the FULL corpus, so if the corpus behind this config is empty or \
                 partial, continuing publishes a wrong number (an empty corpus scores recall 0.0 \
                 with zero failed queries) under a config name that claims otherwise. The remedy \
                 is whatever the reason above asks for — give this machine the dataset's corpus, \
                 or declare its vector_count in datasets/datasets.json. On a sweep, add \
                 --exit-on-error false so the configs that DO verify still publish their (correct) \
                 numbers while this one is skipped. --allow-partial-corpus measures whatever the \
                 server holds without checking; the waiver is recorded in the result file.",
                engine.name(),
                dataset.config.name,
                why,
                held,
            );
            if args.allow_partial_corpus {
                eprintln!("WARNING: {msg}");
                return Ok(Some(record(true)));
            }
            crate::summary::record_rejected_experiment(engine.name(), &dataset.config.name, &msg);
            Err(msg)
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

/// Decide whether a mixed run's unattributed updates are fatal (issue #293).
///
/// The engine half only measures: `finalize_update_stats` records how many
/// writes the server accepted without attributing them to a row that already
/// existed. This is the policy half — the same split as `failed_queries`
/// (measured by `compute_search_stats`) vs `--fail-on-dropped-queries`
/// (enforced by the runner).
///
/// Fatal by default, because on the normal upload → mixed path every update
/// rewrites a vector the same run just uploaded, so the count should be 0 and a
/// nonzero one means `update_count`/`update_rps` describe writes that `recall`
/// cannot see.
///
/// Applied per REPETITION, before the point can reach `best`/`pending_saves`,
/// so a rejected point writes no result file. Note the narrow gap that leaves:
/// with `--repetitions N > 1`, only the reps that trip the gate are discarded,
/// and a run in which some rep came back clean still publishes that rep. A
/// genuine #293 defect fails every rep, so this matters only for a transient.
///
/// Two conditions waive it, because both mean the corpus was already permitted
/// to be short — and updates addressed to the rows that are missing then
/// legitimately create instead of overwrite. Aborting a run the operator
/// explicitly waived into would be the same class of over-strict gate as #295.
///
/// 1. `--allow-partial-corpus`, which declares exactly that.
/// 2. A corpus size the reuse check could only ESTIMATE. That check downgrades
///    its own `Short` verdict to a warning when the count is approximate,
///    because a false abort on a planner figure would be its own wrong result;
///    this gate has to honour the same downgrade or a run that warned there
///    would hard-abort here with no waiver available.
///
///    Two honest caveats. First, this waiver is WIDER than "honours the same
///    downgrade": `actual_is_estimate` is set on the `Ok` verdict too, so an
///    approximate-but-*sufficient* count — where the reuse check was fully
///    satisfied and warned about nothing — also waives this gate. Narrowing it
///    would mean reading the verdict rather than the estimate flag, which is
///    not worth doing for a path nothing can reach. Second, unreachable is why:
///    `CorpusCount::estimated` is built only by pgvector, which has no
///    `search_mixed`. This is a latch for the day a mixed-capable engine reports
///    an estimate.
///
///    COVERAGE: `an_estimated_corpus_size_waives_the_rejection_like_the_reuse_
///    check_does` pins the decision inside this function. The WIRING — that the
///    caller reads `actual_is_estimate` out of `corpus_reuse` and passes it —
///    is unpinned: replacing that argument with a literal `false` kills no test,
///    because no reachable configuration produces an estimate to notice.
///
/// A waived run still publishes `update_unattributed`, so the artifact records
/// how many of its updates did not overwrite an existing row.
fn gate_update_attribution(
    results: crate::engine::SearchResults,
    allow_partial_corpus: bool,
    corpus_size_was_estimated: bool,
    engine_name: &str,
    dataset_name: &str,
) -> Result<crate::engine::SearchResults, String> {
    let unattributed = results.update_unattributed.unwrap_or(0);
    if unattributed == 0 {
        return Ok(results);
    }
    let applied = results.update_count.unwrap_or(0);
    let failed = results.update_failures.unwrap_or(0);
    let dispatched = applied + unattributed + failed;
    let detail = results
        .update_attribution_detail
        .as_deref()
        .unwrap_or("no detail recorded");
    // The message states the SERVER SIGNAL and stops there. The server reports
    // only that the row was not present; it cannot tell us why, and several
    // distinct causes produce the identical signal. Naming one of them would be
    // a shipped error message asserting a mechanism the code cannot know.
    let msg = format!(
        "mixed workload: {unattributed} of {dispatched} dispatched updates were accepted by the \
         server, which reported that the row each one addressed did not already exist. \
         Signal read: {detail}. Those writes therefore did not OVERWRITE a row already in the \
         corpus this run searched, and `recall` cannot show it — the rows the queries score \
         against are there either way (issue #293). \
         The signal does not say WHY the rows were absent; all of these produce it: the update \
         half addressing a different key/collection than the search half; a reused corpus \
         (`--skip-upload`) that is shorter than the dataset or was written with a different \
         document shape; or rows lost mid-run to eviction or failover. Re-upload this config, or \
         pass --allow-partial-corpus to measure a deliberately partial corpus anyway."
    );
    if allow_partial_corpus || corpus_size_was_estimated {
        let why = if allow_partial_corpus {
            "--allow-partial-corpus is set"
        } else {
            "the corpus size could only be estimated, so the reuse check warned rather than \
             rejecting, and this gate honours the same downgrade"
        };
        eprintln!("\t⚠ WARNING: {msg}");
        eprintln!(
            "\t  (continuing because {why}; `update_unattributed` is recorded in the result file)"
        );
        return Ok(results);
    }
    // Same bookkeeping as the reuse check five functions up: without this a
    // sweep run with --exit-on-error false drops the point silently and its
    // summary is indistinguishable from a complete one.
    crate::summary::record_rejected_experiment(engine_name, dataset_name, &msg);
    Err(msg)
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
    // `VCARD idx:<config>` 400 -> 0, redis `DBSIZE` 400 -> 0, all exit 0. A run that did
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
        // DEL <key> — only Vertex is not), so it must NOT run here under any flag
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
                    // Bound before the watchdog closure takes `engine` mutably.
                    let engine_label = engine.name().to_string();
                    let search_result =
                        run_with_search_watchdog(args.search_timeout, &wd_label, || match phase {
                            Some(ratio) => {
                                engine.search_mixed(dataset, effective_params, args.queries, ratio)
                            }
                            None => engine.search(dataset, effective_params, args.queries),
                        });
                    let cpu_after = crate::proc_cpu::sample();

                    // #293. Applied BEFORE the point can reach `best`/`pending_saves`,
                    // so a rejected mixed point publishes no result file at all; a
                    // waived one carries the count into the file it writes.
                    // The reuse check downgrades a `Short` verdict to a warning when the
                    // server-side count is only an estimate; this gate must honour the
                    // same downgrade rather than hard-aborting a run that was allowed
                    // to continue there.
                    let corpus_size_was_estimated = corpus_reuse
                        .as_ref()
                        .and_then(|v| v.get("actual_is_estimate"))
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    let search_result = search_result.and_then(|r| {
                        gate_update_attribution(
                            r,
                            args.allow_partial_corpus,
                            corpus_size_was_estimated,
                            &engine_label,
                            &dataset.config.name,
                        )
                    });

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
        // What `update_count` is a count OF, published per run because it is not
        // uniform across engines (#293). Absent on files written before #293.
        if let Some(ref attribution) = results.update_attribution {
            results_obj.insert("update_attribution".to_string(), json!(attribution));
        }
        if let Some(failures) = results.update_failures {
            results_obj.insert("update_failures".to_string(), json!(failures));
        }
        // The exact server signal behind `update_attribution`, on every mixed
        // run. The tier is a three-word grade and two engines can share one while
        // reading materially different things, so the artifact carries the
        // mechanism as well (#293 cross-engine review).
        if let Some(ref detail) = results.update_attribution_detail {
            results_obj.insert("update_attribution_detail".to_string(), json!(detail));
        }
        // Absent entirely under `ack_only`, where there is no signal to count
        // with: a published 0 there would be indistinguishable from a
        // corpus_row engine's verified zero. Otherwise only ever nonzero on a run
        // waived by --allow-partial-corpus, since `gate_update_attribution`
        // rejects the point and writes no file.
        if let Some(unattributed) = results.update_unattributed {
            results_obj.insert("update_unattributed".to_string(), json!(unattributed));
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

/// Policy half of the #293 guard: which mixed runs may publish their update
/// metrics. The measurement half is unit-tested in
/// `engine::update_accounting_tests`.
#[cfg(test)]
mod update_attribution_gate_tests {
    use super::gate_update_attribution;
    use crate::engine::SearchResults;

    /// `gate(results, allow_partial_corpus, corpus_size_was_estimated)`.
    fn gate(
        r: SearchResults,
        allow_partial_corpus: bool,
        estimated: bool,
    ) -> Result<SearchResults, String> {
        gate_update_attribution(r, allow_partial_corpus, estimated, "eng", "ds")
    }

    fn results(applied: usize, unattributed: usize, failed: usize) -> SearchResults {
        SearchResults {
            update_count: Some(applied),
            update_failures: Some(failed),
            update_unattributed: Some(unattributed),
            update_attribution_detail: Some(
                "VADD replies 1 when it adds a new element and 0 when it overwrites one"
                    .to_string(),
            ),
            ..Default::default()
        }
    }

    /// The #293 condition: reject, and say how many of how many.
    #[test]
    fn unattributed_updates_reject_the_point_by_default() {
        let err = gate(results(0, 399, 0), false, false).unwrap_err();
        assert!(err.contains("399 of 399"), "{err}");
        // The signal is quoted, introduced as a signal rather than as a cause.
        assert!(
            err.contains("Signal read: VADD replies 1 when it adds a new element"),
            "{err}"
        );
        // The message must offer the escape hatch it actually honours.
        assert!(err.contains("--allow-partial-corpus"), "{err}");
    }

    /// The dispatched total is applied + unattributed + failed, so the ratio in
    /// the message is not silently computed off a partial denominator.
    #[test]
    fn the_rejection_counts_every_dispatched_write_in_its_denominator() {
        let err = gate(results(90, 7, 3), false, false).unwrap_err();
        assert!(err.contains("7 of 100"), "{err}");
    }

    /// BLOCKER CLASS (#295): `--skip-upload --allow-partial-corpus` deliberately
    /// measures a corpus that is SHORT, and updates addressed to the rows that
    /// are missing then legitimately create instead of overwrite. Aborting a run
    /// the operator explicitly waived into would be an over-strict gate, so the
    /// waiver must let the point through — still carrying its count.
    #[test]
    fn allow_partial_corpus_waives_the_rejection_and_keeps_the_count() {
        let r = gate(results(10, 5, 0), true, false)
            .expect("--allow-partial-corpus must not abort the run");
        assert_eq!(
            r.update_unattributed,
            Some(5),
            "the waived run must still record how many updates missed the corpus"
        );
    }

    /// POSITIVE CONTROL: a clean run passes untouched under BOTH settings — the
    /// tests above cannot be satisfied by a gate that rejects (or waives)
    /// everything.
    #[test]
    fn a_clean_run_passes_with_and_without_the_waiver() {
        for waived in [false, true] {
            let r = gate(results(1000, 0, 0), waived, false)
                .expect("a clean mixed run must never be rejected");
            assert_eq!(r.update_unattributed, Some(0));
            assert_eq!(r.update_count, Some(1000));
        }
    }

    /// The reuse check downgrades its own `Short` verdict to a warning when the
    /// server-side count is only an ESTIMATE. This gate has to honour the same
    /// downgrade, or a run that was allowed to continue there would hard-abort
    /// here with no waiver available. Unreachable today (only pgvector reports
    /// an estimate and it has no `search_mixed`), so this unit test is the only
    /// coverage of the DECISION, and the wiring that feeds it — the caller
    /// reading `actual_is_estimate` out of `corpus_reuse` — is not covered at
    /// all: replacing that argument with a literal `false` kills no test.
    #[test]
    fn an_estimated_corpus_size_waives_the_rejection_like_the_reuse_check_does() {
        let r = gate(results(10, 5, 0), false, true)
            .expect("an approximate corpus count must not abort the run here");
        assert_eq!(r.update_unattributed, Some(5));
        // ...and it is the estimate doing the waiving, not a blanket pass:
        // the same input with an exact count is still rejected.
        assert!(gate(results(10, 5, 0), false, false).is_err());
    }

    /// The JSON site, not the fold (#293 cross-engine review).
    ///
    /// `VERTEX_UPDATE_ATTRIBUTION` pins the tier on `SearchResults`, but nothing
    /// asserted what `build_search_result_json` WRITES for a non-`corpus_row`
    /// engine — and because all four live mixed engines are `corpus_row` and the
    /// Vertex integration test self-skips, hardcoding `json!("corpus_row")` at
    /// that line was behaviour-preserving for the entire suite. Same exposure one
    /// line over for `update_failures`, which every mixed test only ever asserts
    /// is 0.
    ///
    /// So: drive the emitter directly with a non-default `ack_only` result and
    /// read the emitted object.
    #[test]
    fn the_json_emitter_writes_the_engines_own_tier_and_counts_not_constants() {
        use super::build_search_result_json;
        use crate::config::SearchParams;
        use crate::engine::SearchResults;

        let params: SearchParams =
            serde_json::from_value(serde_json::json!({"parallel": 1, "top": 10})).unwrap();
        let results = SearchResults {
            top: 10,
            latencies: vec![0.001],
            update_search_ratio: Some("1:5".to_string()),
            update_count: Some(7),
            update_rps: Some(3.5),
            update_failures: Some(4),
            update_attribution: Some("ack_only".to_string()),
            // Omitted under ack_only — a published 0 would read like a
            // corpus_row engine's verified zero.
            update_unattributed: None,
            update_attribution_detail: Some("upsertDatapoints returns an empty body".to_string()),
            ..Default::default()
        };
        let doc = build_search_result_json(
            "vertex-test",
            "random-100",
            &params,
            &results,
            None,
            None,
            false,
            None,
            None,
            None,
            None,
            &serde_json::json!({}),
        );
        let res = doc["results"].as_object().unwrap();

        assert_eq!(
            res["update_attribution"], "ack_only",
            "the emitter must write the engine's own tier, not a constant"
        );
        assert_eq!(
            res["update_failures"], 4,
            "the emitter must write the engine's own failure count, not a constant"
        );
        assert_eq!(
            res["update_attribution_detail"],
            "upsertDatapoints returns an empty body"
        );
        assert!(
            !res.contains_key("update_unattributed"),
            "under ack_only the field must be ABSENT, not 0: a 0 here is \
             indistinguishable from a corpus_row engine's verified zero. Got: {:?}",
            res.get("update_unattributed")
        );
        // Control: the same emitter DOES write the field when the engine measured
        // it, so the absence above is the tier and not a broken emitter.
        let measured = SearchResults {
            update_unattributed: Some(0),
            update_attribution: Some("corpus_row".to_string()),
            ..results.clone()
        };
        let doc2 = build_search_result_json(
            "redis-test",
            "random-100",
            &params,
            &measured,
            None,
            None,
            false,
            None,
            None,
            None,
            None,
            &serde_json::json!({}),
        );
        assert_eq!(doc2["results"]["update_unattributed"], 0);
        assert_eq!(doc2["results"]["update_attribution"], "corpus_row");
    }

    /// The gate must RECORD a rejection, not only return `Err`. Without this a
    /// sweep run with `--exit-on-error false` drops the point and its summary is
    /// indistinguishable from a complete one. Mirrors
    /// `a_short_exact_count_still_aborts_and_is_recorded_as_rejected`, which
    /// pins the same behaviour at the reuse gate.
    #[test]
    fn a_rejected_mixed_point_is_recorded_for_the_summary() {
        gate_update_attribution(results(0, 100, 0), false, false, "m9bad", "ds9bad")
            .expect_err("must reject");
        let rejected = crate::summary::rejected_experiments();
        assert!(
            rejected.iter().any(|r| r["engine"] == "m9bad"
                && r["dataset"] == "ds9bad"
                && r["reason"]
                    .as_str()
                    .is_some_and(|s| s.contains("dispatched updates"))),
            "the rejected mixed point must reach the summary: {rejected:?}"
        );
    }

    /// An `ack_only` engine publishes `update_unattributed: None` rather than a
    /// `0` that would look like a verified zero. The gate must read that absence
    /// as "nothing to adjudicate", not as a violation, and must not invent a
    /// count for it.
    #[test]
    fn an_ack_only_run_has_nothing_to_adjudicate_and_passes() {
        let mut r = results(10, 0, 0);
        r.update_unattributed = None;
        r.update_attribution = Some("ack_only".to_string());
        let out = gate(r, false, false).expect("ack_only runs carry no violation to reject");
        assert!(out.update_unattributed.is_none());
    }

    /// Search-only runs have no update fields at all; the gate must be inert
    /// rather than reading `None` as a violation.
    #[test]
    fn a_search_only_run_is_untouched_by_the_gate() {
        let r = gate(SearchResults::default(), false, false)
            .expect("search-only runs must pass the mixed-update gate");
        assert!(r.update_unattributed.is_none());
    }
}

#[cfg(test)]
mod reuse_precondition_tests {
    use super::{classify_reuse_precondition, reuse_expected_rows, ReusePrecondition};
    use crate::config::DatasetConfig;
    use crate::dataset::Dataset;
    use crate::engine::CorpusCount;
    use vector_db_benchmark::readers::{write_npy_vectors, write_sparse_matrix, SparseVector};

    /// A dataset rooted at an ABSOLUTE temp path, so `datasets_dir().join(abs)`
    /// resolves back to `abs` and no download is ever attempted.
    fn dataset_at(
        path: &std::path::Path,
        dataset_type: &str,
        vector_count: Option<i64>,
    ) -> Dataset {
        Dataset::new(DatasetConfig {
            name: "reuse-unit".to_string(),
            dataset_type: Some(dataset_type.to_string()),
            path: serde_json::Value::String(path.to_str().unwrap().to_string()),
            distance: Some("l2".to_string()),
            vector_size: Some(3),
            vector_count,
            link: None,
            schema: None,
            description: None,
        })
    }

    // A corpus shorter than the dataset declares is the case that publishes a
    // wrong recall under a config name claiming otherwise (issue #238).
    #[test]
    fn short_corpus_is_short() {
        assert_eq!(
            classify_reuse_precondition(Ok(400), Some(CorpusCount::exact(200)), "redis-x"),
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
            classify_reuse_precondition(Ok(400), Some(CorpusCount::exact(0)), "redis-x"),
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
            classify_reuse_precondition(Ok(400), Some(CorpusCount::exact(400)), "redis-x"),
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
            classify_reuse_precondition(Ok(400), Some(CorpusCount::exact(401)), "redis-x"),
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
            classify_reuse_precondition(Ok(400), Some(CorpusCount::estimated(200)), "pg-x"),
            ReusePrecondition::Short {
                actual: 200,
                expected: 400,
                approximate: true
            }
        );
        assert_eq!(
            classify_reuse_precondition(Ok(400), Some(CorpusCount::estimated(400)), "pg-x"),
            ReusePrecondition::Ok {
                actual: 400,
                expected: 400,
                approximate: true
            }
        );
    }

    // An engine with no server-side row count wired up is the SOFT arm: the
    // runner says the reuse went unverified and proceeds, because refusing would
    // make --skip-upload unusable on that engine rather than protect a number.
    #[test]
    fn missing_server_side_count_is_the_soft_arm() {
        let verdict = classify_reuse_precondition(Ok(400), None, "chroma-y");
        let why = match &verdict {
            ReusePrecondition::NoServerCount(w) => w.clone(),
            other => panic!("expected NoServerCount, got {other:?}"),
        };
        // The tag that reaches the result file, so this pins the artifact too.
        assert_eq!(verdict.status(), "unverified");
        assert!(
            why.contains("chroma-y"),
            "the note must name the config it could not count: {why}"
        );
        // The wording must blame this side of the wire, not the database, AND
        // must not claim "not implemented" — Qdrant's probe IS implemented and
        // lands here when the reply carries no points_count.
        assert!(
            why.contains("this tool read no server-side row count"),
            "the note must read as a gap on this side of the wire: {why}"
        );
        assert!(
            why.contains("or the engine replied without one"),
            "the note must also cover the implemented-but-answered-nothing case \
             (Qdrant), not assert it is unimplemented: {why}"
        );
        // Positive control: with a count present the same call classifies.
        assert_eq!(
            classify_reuse_precondition(Ok(400), Some(CorpusCount::exact(400)), "chroma-y"),
            ReusePrecondition::Ok {
                actual: 400,
                expected: 400,
                approximate: false
            }
        );
    }

    // #290 review: with NEITHER side available there is nothing to compare in
    // either direction, so an unavailable expected count must not escalate the
    // soft arm into an abort — that would remove --skip-upload from exactly the
    // five engines with no probe while protecting no number.
    #[test]
    fn neither_side_available_stays_the_soft_arm() {
        let verdict = classify_reuse_precondition(
            Err("corpus not on this machine".to_string()),
            None,
            "wv-z",
        );
        assert!(
            matches!(verdict, ReusePrecondition::NoServerCount(_)),
            "expected NoServerCount, got {verdict:?}"
        );
        assert_eq!(verdict.status(), "unverified");
        // Positive control: the SAME unavailable expected count IS fatal once a
        // server-side count exists to have compared it against.
        let with_count = classify_reuse_precondition(
            Err("corpus not on this machine".to_string()),
            Some(CorpusCount::exact(0)),
            "wv-z",
        );
        assert!(
            matches!(with_count, ReusePrecondition::CorpusSizeUnknown(_)),
            "expected CorpusSizeUnknown, got {with_count:?}"
        );
        assert_eq!(with_count.status(), "corpus_size_unknown");
    }

    // Issue #290: an unavailable EXPECTED count is its own verdict, distinct
    // from the engine-side gap above, and it carries the reason through instead
    // of collapsing to a bare "unknown". The handler makes it fatal; keeping the
    // variant separate is what lets it.
    #[test]
    fn unknown_expected_count_is_its_own_verdict_and_keeps_the_reason() {
        let why = match classify_reuse_precondition(
            Err("measuring the corpus on disk failed: bad npy magic".to_string()),
            Some(CorpusCount::exact(400)),
            "redis-x",
        ) {
            ReusePrecondition::CorpusSizeUnknown(w) => w,
            other => panic!("expected CorpusSizeUnknown, got {other:?}"),
        };
        assert_eq!(
            why, "measuring the corpus on disk failed: bad npy magic",
            "the reason must survive into the verdict, not be flattened to 'unknown'"
        );
        // Positive control on the same input shape: a KNOWN expected count that
        // matches must still classify as Ok, so this cannot pass by rejecting
        // everything.
        assert_eq!(
            classify_reuse_precondition(Ok(400), Some(CorpusCount::exact(400)), "redis-x"),
            ReusePrecondition::Ok {
                actual: 400,
                expected: 400,
                approximate: false
            }
        );
    }

    // Issue #290, the widest real path to that verdict: a MEASURABLE layout
    // whose corpus file is not on this machine. `tests.jsonl` (queries + ground
    // truth) is still there, so the run would otherwise proceed and search — the
    // scenario --skip-upload exists for. Pre-fix this was `Ok(None)`.
    #[test]
    fn absent_corpus_file_yields_an_error_not_a_silent_none() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("tests.jsonl"), "{}\n").unwrap();
        let ds = dataset_at(dir.path(), "tar", Some(400));
        let err = reuse_expected_rows(&ds)
            .expect_err("a corpus that is not on disk has no expected row count");
        // The directory IS here — only the corpus file is gone — and `get_path()`
        // returns early on an existing path, so this is precisely the case that
        // a valid `link` will NOT repair. The message must say that rather than
        // imply a download was tried (#290 review).
        assert!(
            err.contains("but its corpus file is not in it") && err.contains("reuse-unit"),
            "the failure must name the dataset and the real remedy: {err}"
        );
        assert!(
            err.contains("never re-downloaded"),
            "it must warn that an existing directory is not re-fetched: {err}"
        );
        // `dataset_at` builds a LINK-LESS dataset, like 8 of the 57 shipped ones
        // (the generated fixtures, random-100, laion). "Delete the directory" is
        // not just useless for those, it destroys tests.jsonl/payloads.jsonl
        // that nothing can restore — so it must not be offered here.
        assert!(
            !err.contains("delete"),
            "a link-less dataset must never be told to delete its directory: {err}"
        );
        assert!(
            err.contains("no download link"),
            "it must say why deleting would not help: {err}"
        );
        // Positive control: the SAME state WITH a link does get the delete-and-
        // refetch remedy, so the branch above is a real distinction and not a
        // blanket removal.
        let mut linked_cfg = dataset_at(dir.path(), "tar", Some(400));
        linked_cfg.config.link = Some("https://example.invalid/ds.tgz".to_string());
        let linked_err = reuse_expected_rows(&linked_cfg).expect_err("still unmeasurable");
        assert!(
            linked_err.contains("delete") && linked_err.contains("fetch it fresh"),
            "a fetchable dataset should still be told it can be re-fetched: {linked_err}"
        );
        // The other shape: nothing at that path at all. Different message, and
        // it must not claim a directory is sitting there.
        let gone = dataset_at(&dir.path().join("does-not-exist"), "tar", Some(400));
        let gone_err = reuse_expected_rows(&gone).expect_err("no corpus, no count");
        assert!(
            gone_err.contains("has no corpus at") && !gone_err.contains("is present at"),
            "an absent dataset and a gutted one must not read alike: {gone_err}"
        );
        // Positive control: the SAME dataset, with the corpus present, measures.
        let vectors: Vec<Vec<f32>> = (0..7).map(|i| vec![i as f32, 0.0, 0.0]).collect();
        write_npy_vectors(dir.path().join("vectors.npy").to_str().unwrap(), &vectors).unwrap();
        assert_eq!(
            reuse_expected_rows(&dataset_at(dir.path(), "tar", Some(7))).unwrap(),
            7
        );
    }

    // The other #290 path: `measured_vector_count()` returning Err.
    //
    // Scope, stated so it is not overclaimed: this function ALREADY returned Err
    // here before the fix — the swallow was in `check_corpus_reuse_precondition`,
    // which turned that Err into `None`. So all this test pins is that the two
    // failure conditions now read differently to the operator. The end-to-end
    // coverage for the swallow itself is phase (d) of
    // `test_binary_redis_skip_upload_unverifiable_corpus_is_fatal`.
    #[test]
    fn a_failed_measurement_reads_as_a_failure_not_as_a_missing_count() {
        let dir = tempfile::tempdir().unwrap();
        // A vectors.npy that exists but is not a valid npy file: the header read
        // fails, which is the transient-read-failure shape of this bug.
        std::fs::write(dir.path().join("vectors.npy"), b"not an npy file at all").unwrap();
        let ds = dataset_at(dir.path(), "tar", Some(400));
        let err = reuse_expected_rows(&ds).expect_err("a failed header read must not become None");
        assert!(
            err.contains("measuring the corpus on disk failed"),
            "a read failure and 'this layout has no cheap count' must not read alike: {err}"
        );
        // Positive control: repair the same file and the same call measures it,
        // so this cannot pass by erroring on everything.
        let vectors: Vec<Vec<f32>> = (0..5).map(|i| vec![i as f32, 0.0, 0.0]).collect();
        write_npy_vectors(dir.path().join("vectors.npy").to_str().unwrap(), &vectors).unwrap();
        assert_eq!(
            reuse_expected_rows(&dataset_at(dir.path(), "tar", Some(5))).unwrap(),
            5
        );
    }

    // #290 review: when the corpus cannot be measured here, datasets.json's
    // vector_count IS the expected count — and it must be usable WITHOUT every
    // corpus file being present locally. `corpus_completeness_target()`'s
    // file-presence gate exists to stop an UPLOAD being skipped (#188/#224);
    // --skip-upload never uploads, and laion-1B is `h5-multi` with 100 parts
    // (~3 TB) whose queries are a separate file, so demanding them all would make
    // the check unpassable on the workflow it exists to serve.
    #[test]
    fn unmeasurable_layout_uses_the_declared_count_without_every_file() {
        let dir = tempfile::tempdir().unwrap();
        // A `sparse` dataset directory with the queries but NOT data.csr — the
        // shape of a client that searches a server-side corpus it never held.
        std::fs::write(dir.path().join("queries.csr"), b"\x00").unwrap();
        assert_eq!(
            reuse_expected_rows(&dataset_at(dir.path(), "sparse", Some(100_000))).unwrap(),
            100_000,
            "the declared count is the only available answer while data.csr is absent"
        );

        // Without a declared count there IS no answer — and the error must name
        // vector_count, not send the operator hunting for files.
        let err = reuse_expected_rows(&dataset_at(dir.path(), "sparse", None))
            .expect_err("no measurement and no declaration is not a number");
        assert!(
            err.contains("vector_count") && err.contains("datasets/datasets.json"),
            "the error must name the field to add, not the files: {err}"
        );
        assert!(
            !err.contains("no corpus at"),
            "an unmeasurable layout must not be reported as a missing corpus: {err}"
        );
    }

    // #290 review, SHOULD-FIX 1: `sparse` is no longer TRUSTED. Once `data.csr`
    // is on disk its 24-byte header is the authority, so a wrong `vector_count`
    // can no longer classify a correct corpus as `Short` (false abort) or a
    // short one as `Surplus` (warn, and publish the wrong number) — the failure
    // this whole issue is about, reached through the datasets.json door.
    #[test]
    fn a_present_csr_corpus_is_measured_not_trusted() {
        let dir = tempfile::tempdir().unwrap();
        let rows: Vec<SparseVector> = (0..150)
            .map(|i| SparseVector {
                indices: vec![i % 7],
                values: vec![1.0],
            })
            .collect();
        write_sparse_matrix(dir.path().join("data.csr").to_str().unwrap(), &rows).unwrap();

        // A declaration wildly higher than the corpus must NOT win.
        assert_eq!(
            reuse_expected_rows(&dataset_at(dir.path(), "sparse", Some(999_999))).unwrap(),
            150,
            "the measured corpus is the authority once data.csr is present"
        );
        // ... and neither must one lower than it, which is the direction that
        // would have published a wrong number as `Surplus`.
        assert_eq!(
            reuse_expected_rows(&dataset_at(dir.path(), "sparse", Some(10))).unwrap(),
            150
        );
        // Positive control: an agreeing declaration measures the same.
        assert_eq!(
            reuse_expected_rows(&dataset_at(dir.path(), "sparse", Some(150))).unwrap(),
            150
        );
        // And with no declaration at all it is still measurable, so the
        // "declare vector_count" error must NOT fire here.
        assert_eq!(
            reuse_expected_rows(&dataset_at(dir.path(), "sparse", None)).unwrap(),
            150
        );
    }

    // `queries.csr` is a matrix too. Counting it as the corpus would report 10
    // rows for a 150-row dataset and reject every reuse of it.
    #[test]
    fn the_query_matrix_is_never_mistaken_for_the_corpus() {
        let dir = tempfile::tempdir().unwrap();
        let queries: Vec<SparseVector> = (0..10)
            .map(|_| SparseVector {
                indices: vec![1],
                values: vec![1.0],
            })
            .collect();
        write_sparse_matrix(dir.path().join("queries.csr").to_str().unwrap(), &queries).unwrap();
        // Only queries.csr exists: unmeasurable, so the declaration stands.
        assert_eq!(
            reuse_expected_rows(&dataset_at(dir.path(), "sparse", Some(150))).unwrap(),
            150
        );
        // Now add the real corpus; the answer must be the corpus, not the queries.
        let rows: Vec<SparseVector> = (0..150)
            .map(|_| SparseVector {
                indices: vec![2],
                values: vec![1.0],
            })
            .collect();
        write_sparse_matrix(dir.path().join("data.csr").to_str().unwrap(), &rows).unwrap();
        assert_eq!(
            reuse_expected_rows(&dataset_at(dir.path(), "sparse", Some(150))).unwrap(),
            150
        );
    }

    // Zero expected rows cannot make a zero-row server look short.
    #[test]
    fn zero_expected_never_fires() {
        assert_eq!(
            classify_reuse_precondition(Ok(0), Some(CorpusCount::exact(0)), "redis-x"),
            ReusePrecondition::Ok {
                actual: 0,
                expected: 0,
                approximate: false
            }
        );
    }
}

/// Behavioural coverage for the `--skip-upload` reuse HANDLER, as opposed to the
/// classifier (issue #290 review).
///
/// A reviewer's mutation campaign found the classifier well pinned but the
/// handler bare: making the `NoServerCount` arm FATAL was killed by nothing —
/// 12 unit and 8 integration tests all stayed green. That arm is the one place
/// this PR deliberately lets #290's symptom through (five engines have no
/// row-count probe), so the decision is worth locking down rather than leaving
/// to the next person's judgement. Likewise, dropping
/// `record_rejected_experiment` from the fatal arm would silently erase a
/// rejected config from `summary.rejected_experiments` under
/// `--exit-on-error false`.
///
/// These drive `check_corpus_reuse_precondition` directly through a stub engine,
/// which is what lets a unit test force `corpus_row_count -> Ok(None)` — no
/// live server can be asked to have an unimplemented probe.
#[cfg(test)]
mod reuse_handler_tests {
    use super::*;
    use crate::config::DatasetConfig;
    use crate::dataset::Dataset;
    use crate::engine::{CorpusCount, SearchResults, UploadStats};
    use vector_db_benchmark::readers::write_npy_vectors;

    /// Minimal `Engine` whose only interesting behaviour is the row-count probe.
    /// Every other method is unreachable on this path and says so.
    struct StubEngine {
        name: String,
        count: Result<Option<CorpusCount>, String>,
        params: Vec<SearchParams>,
    }

    impl Engine for StubEngine {
        fn name(&self) -> &str {
            &self.name
        }
        fn configure(&mut self, _d: &Dataset) -> Result<(), String> {
            unreachable!("the reuse check never configures")
        }
        fn upload(&mut self, _d: &Dataset) -> Result<UploadStats, String> {
            unreachable!("the reuse check never uploads")
        }
        fn search(
            &mut self,
            _d: &Dataset,
            _p: &SearchParams,
            _n: i64,
        ) -> Result<SearchResults, String> {
            unreachable!("the reuse check never searches")
        }
        fn delete(&mut self) -> Result<(), String> {
            unreachable!("the reuse check never deletes")
        }
        fn search_params(&self) -> &[SearchParams] {
            &self.params
        }
        fn corpus_row_count(&mut self) -> Result<Option<CorpusCount>, String> {
            self.count.clone()
        }
    }

    fn stub(name: &str, count: Result<Option<CorpusCount>, String>) -> StubEngine {
        StubEngine {
            name: name.to_string(),
            count,
            params: Vec::new(),
        }
    }

    /// A dataset whose corpus really is on disk and really holds `rows` vectors,
    /// so the dataset side of the comparison is genuinely available.
    fn measurable_dataset(dir: &std::path::Path, rows: usize) -> Dataset {
        let vectors: Vec<Vec<f32>> = (0..rows).map(|i| vec![i as f32, 0.0, 0.0]).collect();
        write_npy_vectors(dir.join("vectors.npy").to_str().unwrap(), &vectors).unwrap();
        Dataset::new(DatasetConfig {
            name: "handler-unit".to_string(),
            dataset_type: Some("tar".to_string()),
            path: serde_json::Value::String(dir.to_str().unwrap().to_string()),
            distance: Some("l2".to_string()),
            vector_size: Some(3),
            vector_count: Some(rows as i64),
            link: None,
            schema: None,
            description: None,
        })
    }

    fn args() -> Args {
        use clap::Parser;
        Args::parse_from(["vector-db-benchmark", "--skip-upload"])
    }

    /// The deliberate soft arm. An engine with no row-count probe must WARN and
    /// continue, recording `unverified` — not abort. Making this fatal would
    /// remove `--skip-upload` from Chroma, Milvus, Weaviate, Turbopuffer and
    /// Vertex, which protects no number because neither side of the comparison
    /// is available.
    #[test]
    fn no_server_side_count_warns_and_continues() {
        let dir = tempfile::tempdir().unwrap();
        let ds = measurable_dataset(dir.path(), 400);
        let mut engine = stub("chroma-handler", Ok(None));

        let record = check_corpus_reuse_precondition(&mut engine, &ds, &args())
            .expect("an engine with no probe must not abort the run")
            .expect("the verdict must still be recorded in the artifact");

        assert_eq!(record["status"], "unverified");
        assert_eq!(record["actual_rows"], serde_json::Value::Null);
        assert_eq!(record["waived_by_allow_partial_corpus"], false);
        // The dataset side WAS available; that must not have turned into an
        // abort just because the server side was missing.
        assert_eq!(record["expected_rows"], 400);
    }

    /// Positive control for the test above: the same handler, same stub, one
    /// field different — a real count that comes up short — must abort. Without
    /// this, `no_server_side_count_warns_and_continues` would still pass if the
    /// handler never aborted at all.
    #[test]
    fn a_short_exact_count_still_aborts_and_is_recorded_as_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let ds = measurable_dataset(dir.path(), 400);
        let mut engine = stub("redis-handler", Ok(Some(CorpusCount::exact(1))));

        let err = check_corpus_reuse_precondition(&mut engine, &ds, &args())
            .expect_err("a short exact count must abort");
        assert!(
            err.contains("the corpus you asked to reuse is incomplete"),
            "{err}"
        );

        // A rejected config must reach the summary, or a sweep run with
        // --exit-on-error false loses all record of what it skipped.
        let rejected = crate::summary::rejected_experiments();
        assert!(
            rejected
                .iter()
                .any(|r| { r["engine"] == "redis-handler" && r["dataset"] == "handler-unit" }),
            "the rejected experiment must be recorded for the summary: {rejected:?}"
        );
    }

    /// `--skip-vector-index` on a schema-less dataset returns before
    /// `read_queries()`, so the run measures nothing — the check must return
    /// early rather than probe and fetch for a run that publishes nothing.
    /// The stub's probe would `unreachable!()`... it cannot, so instead it is an
    /// `Err` that would abort loudly if the early return were removed.
    #[test]
    fn a_run_that_will_not_search_is_not_checked_at_all() {
        let dir = tempfile::tempdir().unwrap();
        let ds = measurable_dataset(dir.path(), 400);
        let mut engine = stub("never-probed", Err("probe must not run".to_string()));

        use clap::Parser;
        let args = Args::parse_from([
            "vector-db-benchmark",
            "--skip-upload",
            "--skip-vector-index",
        ]);
        let out = check_corpus_reuse_precondition(&mut engine, &ds, &args)
            .expect("a run that searches nothing must not be rejected");
        assert!(
            out.is_none(),
            "no verdict should be recorded for a run that measures nothing: {out:?}"
        );
    }
}
