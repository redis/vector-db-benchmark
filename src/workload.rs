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
//! # Index layout: one ingest, many workloads (#344)
//!
//! All search configs of one engine config run against ONE index, built once
//! by `configure()` + `upload()`. [`plan`] picks the layout that index needs
//! from the workloads its search configs declare ([`combine_layouts`]):
//!
//! | workloads                                   | layout                    |
//! |---------------------------------------------|---------------------------|
//! | only vector / hybrid / fulltext             | [`IndexLayout::Vector`]   |
//! | only filter-only (or `--skip-vector-index`) | [`IndexLayout::NoVector`] |
//! | filter-only + any of the others             | [`IndexLayout::Full`]     |
//!
//! `Full` is the union: the vector field and the text fields exactly as in
//! `Vector`, and the scalar fields typed the way the filter-only contract
//! needs them (Redis `TAG CASESENSITIVE`, OpenSearch `double` for `float`). A
//! single-workload config keeps the layout it had before `Full` existed. A
//! filter-only search on a `Full` index still sends no vector clause: every
//! request is checked, as it will be sent, before the timed run
//! ([`crate::filter_only::check_opensearch_body`] on the request body,
//! [`crate::filter_only::check_redis_command`] on the FT.SEARCH arguments)
//! and the result
//! records `filter_only.index_layout`.
//!
//! Because `Full` types tags case-sensitively on Redis (and floats as `double`
//! on OpenSearch), a vector/hybrid filter on a `Full` index matches exactly
//! what the filter-only contract matches, which can differ from the same
//! config on a `Vector` index for mixed-case tags. Every search result
//! therefore records `params.index_layout`.

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
            // Fulltext runs on the vector layout: it is the one whose schema
            // indexes `text` fields (Redis `TEXT SORTABLE`, OpenSearch `text`),
            // while the no-vector layout refuses them (CASESENSITIVE tags,
            // scalar-only schema checks). So vector + fulltext share the
            // vector layout; filter-only + fulltext share the `full` layout
            // (#344, see `combine_layouts`). An engine
            // config with a fulltext search config builds its text fields with
            // the pinned analyzer (`crate::fulltext`), which then also governs
            // any text `match` filter its vector configs run.
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
/// `skip_vector_index` switch (`NoVector` == `skip_vector_index = true`) and
/// the runtime-only `full_index_layout` engine-config flag (`Full`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexLayout {
    /// Vector field + every declared schema field: what every
    /// vector/hybrid/fulltext config has always built.
    Vector,
    /// No indexed vector field; scalar fields typed for the filter-only
    /// contract (#328).
    NoVector,
    /// The union (#344): `Vector` with its scalar fields typed as in
    /// `NoVector`, so filter-only shares the index with the vector-index
    /// workloads. Built only when an engine config mixes them.
    Full,
}

impl IndexLayout {
    /// Spelling used in result files (`params.index_layout`,
    /// `filter_only.index_layout`).
    pub fn as_str(self) -> &'static str {
        match self {
            IndexLayout::Vector => "vector",
            IndexLayout::NoVector => "no-vector",
            IndexLayout::Full => "full",
        }
    }

    /// Whether `workload` can run on an index built in this layout.
    pub fn serves(self, workload: Workload) -> bool {
        self == IndexLayout::Full || workload.required_layout() == self
    }
}

impl std::fmt::Display for IndexLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The layout an engine config's two flags select: `skip_vector_index`
/// (`NoVector`) and the runtime-only `full_index_layout` (`Full`). Both at once
/// is contradictory — a full index HAS a vector field — and is refused rather
/// than letting one flag silently win.
pub fn layout_from_flags(
    skip_vector_index: bool,
    full_index_layout: bool,
) -> Result<IndexLayout, String> {
    match (skip_vector_index, full_index_layout) {
        (true, true) => Err(
            "engine config sets both skip_vector_index and the full index layout; the full \
             layout indexes the vector field (#344)"
                .into(),
        ),
        (true, false) => Ok(IndexLayout::NoVector),
        (false, true) => Ok(IndexLayout::Full),
        (false, false) => Ok(IndexLayout::Vector),
    }
}

/// Engines whose `configure()` can build [`IndexLayout::Full`]. An engine
/// added here must build the union schema, accept filter-only on it (checking
/// the realized schema, and that no request carries a vector clause), and
/// stamp `filter_only.index_layout`.
pub fn supports_full_layout(engine_type: &str) -> bool {
    matches!(engine_type, "redis" | "opensearch")
}

/// The one layout an index must have to serve every workload in `workloads`
/// (see the module docs). Mixing workloads that need the vector layout with
/// ones that need the no-vector layout needs [`IndexLayout::Full`], which only
/// [`supports_full_layout`] engines build; anywhere else the mix is refused
/// with the reason, before anything is downloaded or connected.
pub fn combine_layouts(engine_type: &str, workloads: &[Workload]) -> Result<IndexLayout, String> {
    let needing = |l: IndexLayout| workloads.iter().find(|w| w.required_layout() == l);
    match (needing(IndexLayout::Vector), needing(IndexLayout::NoVector)) {
        (Some(_), Some(_)) if supports_full_layout(engine_type) => Ok(IndexLayout::Full),
        (Some(v), Some(nv)) => Err(format!(
            "engine config mixes workloads \"{v}\" and \"{nv}\", which need a vector index and \
             a no-vector index; engine '{engine_type}' cannot build the shared `full` index \
             (#344), which exists for redis and opensearch only — split them into separate \
             engine configs"
        )),
        (None, Some(_)) => Ok(IndexLayout::NoVector),
        // No search configs at all still builds the vector layout.
        (_, None) => Ok(IndexLayout::Vector),
    }
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
        "redis" | "opensearch" => &[Workload::Vector, Workload::FilterOnly, Workload::Fulltext],
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
            Workload::Vector | Workload::FilterOnly | Workload::Fulltext
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
             \"{w}\"; per-query capture exists for redis and opensearch (vector, filter-only, fulltext)",
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

/// Whether `--ef-runtime` (`ef_filter`; empty = no filter) keeps a search
/// config. EF is a vector-index knob, so filter-only configs are never
/// dropped by it — on `no-vector` and on the shared `full` layout alike
/// (#344). Every other workload is kept unless it declares an `ef` that is
/// not in the filter.
pub fn ef_runtime_keeps(workload: Workload, ef: Option<i64>, ef_filter: &[i64]) -> bool {
    if ef_filter.is_empty() || workload == Workload::FilterOnly {
        return true;
    }
    ef.is_none_or(|ef| ef_filter.contains(&ef))
}

/// `--update-search-ratio` mixed phases interleave VECTOR searches with
/// updates; any other workload would be measured as something it is not. A
/// config that holds one must be refused at startup, before any download or
/// connection — not when the mixed phase reaches it, which returns before
/// the completed pure-search results are written (#344 review). This covers
/// the shared `full` layout (filter-only + vector) and vector + fulltext /
/// hybrid on the `vector` layout.
///
/// The `no-vector` layout is exempt: it holds only filter-only, and the run
/// keeps skipping its mixed phases as it always has (there are no vector
/// updates to benchmark there).
pub fn check_update_search_ratio(plan: &WorkloadPlan, has_ratio: bool) -> Result<(), String> {
    if !has_ratio || plan.layout == IndexLayout::NoVector {
        return Ok(());
    }
    match plan
        .workloads
        .iter()
        .enumerate()
        .find(|(_, w)| **w != Workload::Vector)
    {
        Some((i, w)) => Err(format!(
            "--update-search-ratio runs the vector workload in its mixed phases, but search \
             config #{i} is \"{w}\"; run the non-vector search configs in a separate engine \
             config without --update-search-ratio"
        )),
        None => Ok(()),
    }
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
/// - search configs needing different index layouts on an engine that cannot
///   build the shared `full` index ([`combine_layouts`]);
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
        combine_layouts(engine_type, &workloads)?
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

/// Fulltext fingerprint (#339): a fulltext result carries its
/// `crate::fulltext::result_metadata` block, no other workload's does.
/// Checked at the dispatch point next to [`verify_engine_result_shape`].
pub fn verify_fulltext_shape(workload: Workload, fulltext_metadata: bool) -> Result<(), String> {
    match (workload, fulltext_metadata) {
        (Workload::Fulltext, false) => Err(
            "workload fulltext returned a result without fulltext metadata — the engine did not \
             run its fulltext path"
                .into(),
        ),
        (w, true) if w != Workload::Fulltext => Err(format!(
            "workload \"{w}\" returned fulltext metadata — the engine ran its fulltext path"
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
        // No engine has hybrid yet: refused, not run as vector.
        for engine in ["redis", "opensearch", "qdrant"] {
            let e = plan(engine, &[Some(Workload::Hybrid)], false).unwrap_err();
            assert!(e.contains("not implemented"), "{e}");
        }
        // Fulltext exists on Redis and OpenSearch only (#339).
        for engine in ["valkey", "mongodb", "qdrant", "elasticsearch"] {
            let e = plan(engine, &[Some(Workload::Fulltext)], false).unwrap_err();
            assert!(e.contains("not implemented"), "{e}");
        }
        for engine in ["redis", "opensearch"] {
            assert!(
                implemented(engine).contains(&Workload::Fulltext),
                "{engine}"
            );
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

    /// #344 review: a non-vector search config next to `--update-search-ratio`
    /// is refused at plan time, on `full` (filter-only + vector) and on
    /// `vector` (vector + fulltext) alike; only-vector configs and the
    /// no-vector layout (which skips its mixed phases) are unaffected.
    #[test]
    fn update_search_ratio_refuses_non_vector_search_configs_up_front() {
        let full = plan("redis", &[None, Some(Workload::FilterOnly)], false).unwrap();
        assert_eq!(full.layout, IndexLayout::Full);
        let e = check_update_search_ratio(&full, true).unwrap_err();
        assert!(e.contains("search config #1 is \"filter-only\""), "{e}");
        check_update_search_ratio(&full, false).unwrap();
        let text = plan("opensearch", &[None, Some(Workload::Fulltext)], false).unwrap();
        assert_eq!(text.layout, IndexLayout::Vector);
        let e = check_update_search_ratio(&text, true).unwrap_err();
        assert!(e.contains("search config #1 is \"fulltext\""), "{e}");
        check_update_search_ratio(&plan("redis", &[None, None], false).unwrap(), true).unwrap();
        let no_vector = plan("redis", &[Some(Workload::FilterOnly)], false).unwrap();
        assert_eq!(no_vector.layout, IndexLayout::NoVector);
        check_update_search_ratio(&no_vector, true).unwrap();
        check_update_search_ratio(&plan("redis", &[None], true).unwrap(), true).unwrap();
    }

    /// #344 review: `--ef-runtime` never drops a filter-only config (EF is a
    /// vector-index knob), keyed on the workload so `full` behaves like
    /// `no-vector`; other workloads keep the old rule.
    #[test]
    fn ef_runtime_filter_is_keyed_on_the_workload() {
        let filter = [64, 128];
        for ef in [None, Some(64), Some(512)] {
            assert!(
                ef_runtime_keeps(Workload::FilterOnly, ef, &filter),
                "{ef:?}"
            );
            assert!(ef_runtime_keeps(Workload::Vector, ef, &[]), "{ef:?}");
        }
        for w in [Workload::Vector, Workload::Hybrid, Workload::Fulltext] {
            assert!(ef_runtime_keeps(w, Some(128), &filter));
            assert!(!ef_runtime_keeps(w, Some(512), &filter), "{w}");
            assert!(ef_runtime_keeps(w, None, &filter), "{w}");
        }
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
        // #339: fulltext captures on Redis and OpenSearch, hybrid nowhere.
        for engine in ["redis", "opensearch"] {
            assert!(captures(engine, Workload::Fulltext), "{engine}");
            assert!(!captures(engine, Workload::Hybrid), "{engine}");
            let ft = plan(engine, &[Some(Workload::Fulltext)], false).unwrap();
            assert!(check_capture(&ft).is_ok(), "{engine}");
        }
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
    fn mixing_filter_only_with_vector_index_workloads_builds_the_full_layout() {
        for engine in ["redis", "opensearch"] {
            let p = plan(engine, &[None, Some(Workload::FilterOnly)], false).unwrap();
            assert_eq!(p.layout, IndexLayout::Full, "{engine}");
            let p = plan(
                engine,
                &[
                    Some(Workload::Fulltext),
                    Some(Workload::FilterOnly),
                    Some(Workload::Vector),
                ],
                false,
            )
            .unwrap();
            assert_eq!(p.layout, IndexLayout::Full, "{engine}");
            assert_eq!(
                p.workloads,
                vec![Workload::Fulltext, Workload::FilterOnly, Workload::Vector]
            );
            // Single-layout configs keep their pre-#344 layout: `Full` is only
            // ever built for a mix.
            let p = plan(engine, &[None, Some(Workload::Fulltext)], false).unwrap();
            assert_eq!(p.layout, IndexLayout::Vector);
            assert_eq!(p.workloads, vec![Workload::Vector, Workload::Fulltext]);
            let p = plan(engine, &[Some(Workload::FilterOnly); 2], false).unwrap();
            assert_eq!(p.layout, IndexLayout::NoVector);
        }
        // The flag is still "every config is filter-only", never a mix.
        assert!(plan("redis", &[Some(Workload::Fulltext)], true).is_err());
    }

    #[test]
    fn combine_layouts_is_the_union_only_where_the_engine_builds_it() {
        use Workload::*;
        // Every workload needing the vector layout, mixed with filter-only.
        for w in [Vector, Hybrid, Fulltext] {
            for engine in ["redis", "opensearch"] {
                assert_eq!(
                    combine_layouts(engine, &[w, FilterOnly]).unwrap(),
                    IndexLayout::Full
                );
                assert_eq!(
                    combine_layouts(engine, &[FilterOnly, w]).unwrap(),
                    IndexLayout::Full
                );
            }
            // An engine that cannot build `full` gets a reasoned refusal.
            for engine in ["qdrant", "valkey", "mongodb", "elasticsearch"] {
                let e = combine_layouts(engine, &[w, FilterOnly]).unwrap_err();
                assert!(e.contains("#344") && e.contains(engine), "{e}");
                assert!(!supports_full_layout(engine));
            }
            assert_eq!(
                combine_layouts("qdrant", &[w, w]).unwrap(),
                IndexLayout::Vector
            );
        }
        assert_eq!(
            combine_layouts("qdrant", &[FilterOnly]).unwrap(),
            IndexLayout::NoVector
        );
        assert_eq!(combine_layouts("redis", &[]).unwrap(), IndexLayout::Vector);
    }

    #[test]
    fn engine_flags_select_one_layout_and_refuse_the_contradiction() {
        assert_eq!(
            layout_from_flags(false, false).unwrap(),
            IndexLayout::Vector
        );
        assert_eq!(
            layout_from_flags(true, false).unwrap(),
            IndexLayout::NoVector
        );
        assert_eq!(layout_from_flags(false, true).unwrap(), IndexLayout::Full);
        assert!(layout_from_flags(true, true).unwrap_err().contains("both"));
    }

    #[test]
    fn full_layout_serves_every_workload_and_the_others_only_their_own() {
        for w in Workload::ALL {
            assert!(IndexLayout::Full.serves(w), "{w}");
            assert_eq!(
                IndexLayout::NoVector.serves(w),
                w == Workload::FilterOnly,
                "{w}"
            );
            assert_eq!(
                IndexLayout::Vector.serves(w),
                w != Workload::FilterOnly,
                "{w}"
            );
        }
        assert_eq!(IndexLayout::Full.as_str(), "full");
        assert_eq!(IndexLayout::NoVector.to_string(), "no-vector");
        assert_eq!(IndexLayout::Vector.as_str(), "vector");
    }

    #[test]
    fn fulltext_results_must_carry_fulltext_metadata() {
        assert!(verify_fulltext_shape(Workload::Fulltext, true).is_ok());
        assert!(verify_fulltext_shape(Workload::Fulltext, false).is_err());
        for w in [Workload::Vector, Workload::Hybrid, Workload::FilterOnly] {
            assert!(verify_fulltext_shape(w, true).is_err(), "{w}");
            assert!(verify_fulltext_shape(w, false).is_ok(), "{w}");
        }
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
