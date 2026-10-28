//! The per-search-config query family ("workload") and the rules that decide
//! whether an engine config may run it (#344, first half).
//!
//! # Contract for workload implementers (#339 fulltext, #340 filter-only, #341 hybrid)
//!
//! A search config (one entry of `search_params` in an engine configuration
//! file) may declare `"workload": "vector" | "hybrid" | "fulltext" |
//! "filter-only"`. Undeclared means `vector`, so every existing config keeps its
//! meaning. The runner resolves every config's workload ONCE, before any engine
//! is constructed, through [`plan`], and then sends every measured query of that
//! config through the single dispatch point `engine::search_workload` in the
//! binary. That function is the only place a [`Workload`] is turned into an
//! engine call:
//!
//! | workload      | `Engine` method        | ground truth | scored against the reference by |
//! |---------------|------------------------|--------------|---------------------------------|
//! | `vector`      | `search`               | yes          | recall@k, overlap@k, rank corr. |
//! | `hybrid`      | `search_hybrid`        | yes          | recall@k, overlap@k, rank corr. |
//! | `fulltext`    | `search_fulltext`      | no           | overlap@k, Jaccard@k, rank corr.|
//! | `filter-only` | `search_filter_only`   | no           | exact-count agreement           |
//!
//! To add an arm for an engine:
//!
//! 1. Implement the `Engine` method in the engine's module. Its default body is
//!    a hard error, never a fallback to `search`.
//! 2. Add the workload to that engine's row in [`implemented`]. [`plan`] refuses
//!    any config that declares a workload its engine's row does not list, before
//!    anything is downloaded or connected.
//! 3. When `SearchParams::capture_query_results` is set, push one
//!    [`crate::results_dump::RawQueryResult`] per timed query (failed ones
//!    included, with `error` set) into `SearchResults::query_records`. The runner
//!    refuses a run whose record count differs from `requested_queries`, and a
//!    run that captured nothing when asked to.
//!
//! Every workload must leave its fingerprint on the result: filter-only results
//! carry `filter_only` stats, the other three must not (see
//! [`verify_result_shape`]). That is what catches an arm that quietly ran the
//! vector path under another workload's name.
//!
//! # Index layout
//!
//! This step does NOT build the shared `full` schema (#344 second half). An
//! index is built in one [`IndexLayout`], so all search configs of one engine
//! config must need the same layout; mixing `filter-only` with any other
//! workload in one config is refused by [`plan`] until that schema exists.

use serde::{Deserialize, Serialize};

/// One query family. Serialized exactly as the config key spells it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Workload {
    /// Pure KNN (a dataset with per-query `conditions` adds them as a prefilter,
    /// exactly as today).
    Vector,
    /// Prefilter + KNN on derived metadata fields (#341).
    Hybrid,
    /// Ranked full-text (BM25) (#339).
    Fulltext,
    /// Exact-match tag + numeric predicates, no vector clause (#328, #340).
    FilterOnly,
}

impl Workload {
    pub const ALL: [Workload; 4] = [
        Workload::Vector,
        Workload::Hybrid,
        Workload::Fulltext,
        Workload::FilterOnly,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Workload::Vector => "vector",
            Workload::Hybrid => "hybrid",
            Workload::Fulltext => "fulltext",
            Workload::FilterOnly => "filter-only",
        }
    }

    /// Whether the dataset's brute-forced neighbours are the oracle for this
    /// workload. For the others the declared reference engine is.
    pub fn has_ground_truth(self) -> bool {
        matches!(self, Workload::Vector | Workload::Hybrid)
    }

    /// Whether result order is meaningful (ranked) — false for filter-only,
    /// whose contract is "up to `limit` ids in unspecified order".
    pub fn is_ranked(self) -> bool {
        !matches!(self, Workload::FilterOnly)
    }

    /// The index layout `configure()` must build for this workload.
    pub fn required_layout(self) -> IndexLayout {
        match self {
            Workload::FilterOnly => IndexLayout::NoVector,
            // Fulltext runs on the vector layout today because that is the one
            // whose schema indexes `text` fields (Redis `TEXT SORTABLE`); the
            // no-vector layout rejects them. #339 may revisit this.
            Workload::Vector | Workload::Hybrid | Workload::Fulltext => IndexLayout::Vector,
        }
    }
}

impl std::fmt::Display for Workload {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for Workload {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        Workload::ALL
            .into_iter()
            .find(|w| w.as_str() == s)
            .ok_or_else(|| {
                format!("unknown workload {s:?}; expected vector, hybrid, fulltext or filter-only")
            })
    }
}

/// What the configure phase builds. Maps onto the engines' existing
/// `skip_vector_index` switch: `NoVector` == `skip_vector_index = true`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexLayout {
    Vector,
    NoVector,
}

/// The workloads each engine has a contract-conforming code path for — what a
/// search config may DECLARE. An engine type not listed here implements
/// `vector` only. Checked by [`plan`] at startup and again by the binary's
/// `engine::search_workload` at dispatch.
///
/// Valkey and MongoDB are deliberately absent from the `filter-only` row: their
/// filter-only harnesses predate the #328 contract (no server-side layout check,
/// legacy result format, no `<engine>-no-vector` rename unless the flag is
/// used). They stay reachable exactly as on master — through
/// `--skip-vector-index` only — via [`legacy_filter_only`].
pub fn implemented(engine_type: &str) -> &'static [Workload] {
    match engine_type {
        "redis" | "opensearch" => &[Workload::Vector, Workload::FilterOnly],
        _ => &[Workload::Vector],
    }
}

/// Whether the engine's arm for `workload` records per-query results
/// (`results_dump::RawQueryResult`) when `--dump-results` asks it to. Checked
/// at startup for every planned (engine, workload) under `--dump-results`
/// ([`check_capture`]), so a run that could not produce its dump is refused
/// before any download or connection, not after the timed search. The
/// dispatcher still checks the records it actually got.
///
/// An arm added to [`implemented`] that captures must be added here too.
pub fn captures(engine_type: &str, workload: Workload) -> bool {
    matches!(
        (engine_type, workload),
        (
            "redis" | "opensearch",
            Workload::Vector | Workload::FilterOnly
        )
    )
}

/// Refuse `--dump-results` for a plan with any workload the engine cannot
/// capture (see [`captures`]).
pub fn check_capture(plan: &WorkloadPlan) -> Result<(), String> {
    match plan
        .workloads
        .iter()
        .find(|w| !captures(&plan.engine_type, **w))
    {
        Some(w) => Err(format!(
            "--dump-results: engine '{}' does not capture per-query results for workload \
             \"{w}\"; per-query capture exists for redis and opensearch (vector, filter-only)",
            plan.engine_type
        )),
        None => Ok(()),
    }
}

/// Refuse a search config that declares `calibration_param` for a workload
/// that calibration does not tune. Calibration drives the VECTOR search path,
/// so on any other workload the knob would be silently ignored or tuned
/// against the wrong query family. The one tolerated case is the legacy
/// `--skip-vector-index` flag (`skip_vector_index`), where configs written for
/// a vector sweep are reused as filter-only and their calibration keys were
/// always ignored.
///
/// `calibration_declared[i]` is whether search config `i` sets
/// `calibration_param`; indexed like `plan.workloads`.
pub fn check_calibration(
    plan: &WorkloadPlan,
    calibration_declared: &[bool],
    skip_vector_index: bool,
) -> Result<(), String> {
    for (i, (w, declared)) in plan.workloads.iter().zip(calibration_declared).enumerate() {
        if *declared && *w != Workload::Vector && !skip_vector_index {
            return Err(format!(
                "search config #{i} declares calibration_param, which only tunes the vector \
                 workload; it is \"{w}\" — remove calibration_param"
            ));
        }
    }
    Ok(())
}

/// Engines whose pre-#328 filter-only harness is reachable ONLY through the
/// process-wide `--skip-vector-index` flag. Their results keep the legacy shape:
/// no `filter_only` stats, `mean_precision_at_returned == -1.0` as the
/// filter-only sentinel.
pub fn legacy_filter_only(engine_type: &str) -> bool {
    matches!(engine_type, "valkey" | "mongodb")
}

/// Whether `--skip-vector-index` can run on this engine type at all.
pub fn supports_skip_vector_index(engine_type: &str) -> bool {
    implemented(engine_type).contains(&Workload::FilterOnly) || legacy_filter_only(engine_type)
}

/// Whether the dispatch point may route `workload` to this engine type: a
/// declared workload from [`implemented`], or the legacy flag-only filter-only
/// harness. [`plan`] has already refused a DECLARED legacy filter-only.
pub fn dispatchable(engine_type: &str, workload: Workload) -> bool {
    implemented(engine_type).contains(&workload)
        || (workload == Workload::FilterOnly && legacy_filter_only(engine_type))
}

/// Every search config's resolved workload, and the one layout they share.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkloadPlan {
    pub engine_type: String,
    pub layout: IndexLayout,
    /// Indexed like the config's `search_params` array.
    pub workloads: Vec<Workload>,
}

impl WorkloadPlan {
    pub fn workload(&self, search_id: usize) -> Result<Workload, String> {
        self.workloads.get(search_id).copied().ok_or_else(|| {
            format!(
                "search config #{search_id} has no resolved workload (plan covers {})",
                self.workloads.len()
            )
        })
    }
}

/// Resolve every search config's workload for one engine config.
///
/// `declared[i]` is search config `i`'s `workload` key. `skip_vector_index` is
/// the legacy process-wide `--skip-vector-index`, kept as an alias for "every
/// search config is `filter-only`"; a config that declares anything else under
/// it is refused rather than silently re-labelled.
///
/// Errors (all before any download or connection):
/// - a workload the engine has no arm for (see [`implemented`]);
/// - search configs needing different index layouts (needs #344's `full`
///   schema);
/// - `--skip-vector-index` with a declared non-`filter-only` workload.
pub fn plan(
    engine_type: &str,
    declared: &[Option<Workload>],
    skip_vector_index: bool,
) -> Result<WorkloadPlan, String> {
    let mut workloads = Vec::with_capacity(declared.len());
    for (i, d) in declared.iter().enumerate() {
        let w = match (skip_vector_index, d) {
            (true, Some(w)) if *w != Workload::FilterOnly => {
                return Err(format!(
                    "--skip-vector-index runs every search config as filter-only, but search \
                     config #{i} declares workload \"{w}\"; drop the flag or the declaration"
                ));
            }
            (true, _) => Workload::FilterOnly,
            (false, Some(w)) => *w,
            (false, None) => Workload::Vector,
        };
        let allowed = if skip_vector_index {
            supports_skip_vector_index(engine_type)
        } else {
            implemented(engine_type).contains(&w)
        };
        if !allowed {
            return Err(format!(
                "workload \"{w}\" is not implemented for engine '{engine_type}' (search config \
                 #{i}); implemented: {}",
                implemented(engine_type)
                    .iter()
                    .map(|w| w.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        workloads.push(w);
    }
    let layout = if skip_vector_index {
        IndexLayout::NoVector
    } else {
        let mut layouts = workloads.iter().map(|w| (w, w.required_layout()));
        match layouts.next() {
            None => IndexLayout::Vector,
            Some((first_w, first)) => {
                if let Some((w, _)) = layouts.find(|(_, l)| *l != first) {
                    return Err(format!(
                        "engine config mixes workloads \"{first_w}\" and \"{w}\", which need \
                         different index layouts; one index serving both needs the shared \
                         `full` schema (#344), which does not exist yet — split them into \
                         separate engine configs"
                    ));
                }
                first
            }
        }
    };
    Ok(WorkloadPlan {
        engine_type: engine_type.to_string(),
        layout,
        workloads,
    })
}

/// [`verify_result_shape`] for a specific engine type. Legacy filter-only
/// engines ([`legacy_filter_only`]) never publish `filter_only` stats, so their
/// fingerprint is the precision sentinel instead: a filter-only result must
/// carry `mean_precision_at_returned == -1.0` (`precision_sentinel`), and any
/// other workload must not.
pub fn verify_engine_result_shape(
    engine_type: &str,
    workload: Workload,
    filter_only_stats: bool,
    precision_sentinel: bool,
) -> Result<(), String> {
    if !legacy_filter_only(engine_type) {
        return verify_result_shape(workload, filter_only_stats);
    }
    match (workload, precision_sentinel) {
        (Workload::FilterOnly, false) => Err(format!(
            "workload filter-only on '{engine_type}' returned a result without the -1.0 \
             precision sentinel — the engine did not run its filter-only path"
        )),
        (w, true) if w != Workload::FilterOnly => Err(format!(
            "workload \"{w}\" on '{engine_type}' returned the filter-only precision sentinel \
             — the engine ran its filter-only path"
        )),
        _ => Ok(()),
    }
}

/// Refuse a result that does not carry its workload's fingerprint.
///
/// `filter_only_stats` is whether `SearchResults::filter_only` is set. A
/// filter-only result without it did not go through the filter-only path; any
/// other workload with it ran the filter-only path under the wrong name.
pub fn verify_result_shape(workload: Workload, filter_only_stats: bool) -> Result<(), String> {
    match (workload, filter_only_stats) {
        (Workload::FilterOnly, false) => Err(
            "workload filter-only returned a result without filter-only stats — the engine did \
             not run its filter-only path"
                .into(),
        ),
        (w, true) if w != Workload::FilterOnly => Err(format!(
            "workload \"{w}\" returned filter-only stats — the engine ran its filter-only path"
        )),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serde_spelling_matches_the_config_key() {
        for w in Workload::ALL {
            let json = serde_json::to_value(w).unwrap();
            assert_eq!(json, serde_json::json!(w.as_str()));
            assert_eq!(serde_json::from_value::<Workload>(json).unwrap(), w);
            assert_eq!(w.as_str().parse::<Workload>().unwrap(), w);
        }
        assert!(serde_json::from_value::<Workload>(serde_json::json!("filter_only")).is_err());
        assert!("knn".parse::<Workload>().is_err());
    }

    #[test]
    fn undeclared_is_vector_and_layout_is_vector() {
        let p = plan("redis", &[None, None], false).unwrap();
        assert_eq!(p.workloads, vec![Workload::Vector; 2]);
        assert_eq!(p.layout, IndexLayout::Vector);
        // A config with no search params still needs a layout.
        assert_eq!(
            plan("qdrant", &[], false).unwrap().layout,
            IndexLayout::Vector
        );
    }

    #[test]
    fn skip_vector_index_maps_onto_filter_only() {
        let p = plan("opensearch", &[None, Some(Workload::FilterOnly)], true).unwrap();
        assert_eq!(p.workloads, vec![Workload::FilterOnly; 2]);
        assert_eq!(p.layout, IndexLayout::NoVector);
    }

    #[test]
    fn declared_filter_only_selects_the_no_vector_layout() {
        let p = plan("redis", &[Some(Workload::FilterOnly)], false).unwrap();
        assert_eq!(p.layout, IndexLayout::NoVector);
    }

    #[test]
    fn skip_vector_index_refuses_a_declared_non_filter_workload() {
        let e = plan("redis", &[Some(Workload::Vector)], true).unwrap_err();
        assert!(e.contains("--skip-vector-index"), "{e}");
    }

    #[test]
    fn unimplemented_combinations_are_hard_errors() {
        // No engine has fulltext or hybrid yet: refused, not run as vector.
        for engine in ["redis", "opensearch", "qdrant"] {
            for w in [Workload::Fulltext, Workload::Hybrid] {
                let e = plan(engine, &[Some(w)], false).unwrap_err();
                assert!(e.contains("not implemented"), "{e}");
            }
        }
        // Filter-only only where an arm exists.
        assert!(plan("qdrant", &[Some(Workload::FilterOnly)], false).is_err());
        assert!(plan("qdrant", &[None], true).is_err());
    }

    /// Valkey/MongoDB filter-only stays exactly as reachable as on master:
    /// through `--skip-vector-index`, never through a declared workload.
    #[test]
    fn legacy_filter_only_engines_are_flag_only() {
        for engine in ["valkey", "mongodb"] {
            let p = plan(engine, &[None], true).unwrap();
            assert_eq!(p.workloads, vec![Workload::FilterOnly]);
            assert_eq!(p.layout, IndexLayout::NoVector);
            let e = plan(engine, &[Some(Workload::FilterOnly)], false).unwrap_err();
            assert!(e.contains("not implemented"), "{e}");
            assert!(supports_skip_vector_index(engine));
            assert!(dispatchable(engine, Workload::FilterOnly));
            assert!(!dispatchable(engine, Workload::Fulltext));
        }
        for engine in ["redis", "opensearch"] {
            assert!(plan(engine, &[Some(Workload::FilterOnly)], false).is_ok());
            assert!(!legacy_filter_only(engine));
        }
        assert!(!supports_skip_vector_index("qdrant"));
        assert!(!dispatchable("qdrant", Workload::FilterOnly));
    }

    #[test]
    fn capture_is_checked_per_engine_and_workload() {
        let ok = plan("redis", &[None, None], false).unwrap();
        assert!(check_capture(&ok).is_ok());
        let ok = plan("opensearch", &[Some(Workload::FilterOnly)], false).unwrap();
        assert!(check_capture(&ok).is_ok());
        // Implemented but not capturing: refused.
        let e = check_capture(&plan("qdrant", &[None], false).unwrap()).unwrap_err();
        assert!(e.contains("does not capture"), "{e}");
        for legacy in ["valkey", "mongodb"] {
            assert!(check_capture(&plan(legacy, &[None], true).unwrap()).is_err());
        }
        assert!(!captures("redis", Workload::Fulltext));
    }

    #[test]
    fn calibration_is_refused_off_the_vector_workload_except_under_the_flag() {
        let fo = plan("redis", &[Some(Workload::FilterOnly)], false).unwrap();
        assert!(check_calibration(&fo, &[true], false).is_err());
        assert!(check_calibration(&fo, &[false], false).is_ok());
        let flag = plan("redis", &[None], true).unwrap();
        assert!(
            check_calibration(&flag, &[true], true).is_ok(),
            "legacy flag tolerated"
        );
        let vec = plan("redis", &[None], false).unwrap();
        assert!(check_calibration(&vec, &[true], false).is_ok());
    }

    /// The CI failure this pins: legacy engines publish no `filter_only`
    /// stats, so their fingerprint is the -1.0 sentinel — and the check must
    /// still catch a misroute in either direction.
    #[test]
    fn legacy_engines_are_checked_by_the_precision_sentinel() {
        for engine in ["valkey", "mongodb"] {
            assert!(verify_engine_result_shape(engine, Workload::FilterOnly, false, true).is_ok());
            assert!(
                verify_engine_result_shape(engine, Workload::FilterOnly, false, false).is_err()
            );
            assert!(verify_engine_result_shape(engine, Workload::Vector, false, false).is_ok());
            assert!(verify_engine_result_shape(engine, Workload::Vector, false, true).is_err());
        }
        // Contract engines still need the stats; the sentinel alone is not enough.
        assert!(verify_engine_result_shape("redis", Workload::FilterOnly, false, true).is_err());
        assert!(verify_engine_result_shape("redis", Workload::FilterOnly, true, true).is_ok());
    }

    #[test]
    fn mixed_layouts_are_refused_until_the_full_schema_exists() {
        let e = plan("redis", &[None, Some(Workload::FilterOnly)], false).unwrap_err();
        assert!(e.contains("#344"), "{e}");
    }

    #[test]
    fn plan_lookup_is_bounds_checked() {
        let p = plan("redis", &[None], false).unwrap();
        assert_eq!(p.workload(0).unwrap(), Workload::Vector);
        assert!(p.workload(1).is_err());
    }

    #[test]
    fn result_shape_must_match_the_workload() {
        assert!(verify_result_shape(Workload::FilterOnly, true).is_ok());
        assert!(verify_result_shape(Workload::Vector, false).is_ok());
        assert!(verify_result_shape(Workload::FilterOnly, false).is_err());
        for w in [Workload::Vector, Workload::Hybrid, Workload::Fulltext] {
            assert!(verify_result_shape(w, true).is_err(), "{w}");
            assert!(verify_result_shape(w, false).is_ok(), "{w}");
        }
    }
}
