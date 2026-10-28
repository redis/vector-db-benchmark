//! `compare-results`: score a candidate engine's per-query dumps against a
//! declared reference engine's (#343).
//!
//! Dumps are paired by (dataset, workload, parallel, k, id space); within a
//! pair, queries are grouped by shape (`QueryRecord::shape`, else the workload
//! name) and each query is compared on its first answer
//! ([`crate::results_dump::first_answers`]).
//!
//! Metrics, per (pair, shape):
//! - `overlap_at_k` = |R∩C| / |R| over the first k ids (R = reference). 1.0 when
//!   both are empty. Ranked workloads only.
//! - `jaccard_at_k` = |R∩C| / |R∪C| over the first k ids. Ranked workloads only.
//! - `rank_correlation` = Kendall tau-a of the ids both lists share, by their
//!   positions in each list; queries sharing fewer than 2 ids are skipped.
//!   Ranked workloads only.
//! - `count_agreement` (filter-only): fraction of queries whose exact totals are
//!   equal (`None` when no compared query carried totals), the largest absolute
//!   difference, and — where both totals fit in k,
//!   i.e. both pages are complete result sets — the fraction whose id sets are
//!   equal. Filter-only order is unspecified, so overlap/rank are not reported.
//! - `recall_at_k_{reference,candidate}` (vector, hybrid): against the
//!   ground truth carried in the dump; same denominator as `mean_recall`.
//!
//! Two things fail a comparison ([`Report::passed`] is false, the CLI exits
//! non-zero), together the `--fail-on-dropped-queries` policy (#213) applied to
//! the dumps — a comparison over the queries both sides happened to answer is
//! biased:
//! - a query answered by one side and not the other (failed on every attempt,
//!   or absent from the dump) — listed in `one_sided`;
//! - ANY failed attempt on either side, counted per side in
//!   `SideInfo::failed_attempts`. Under query cycling (`--queries N` above the
//!   query count, duration runs) a query can fail once and succeed on a later
//!   cycle; it is still compared on its first answer, but the failed attempt is
//!   exactly what `--fail-on-dropped-queries` would have counted, so it is not
//!   hidden by the later success.
//!
//! Directory mode pairs dumps by (dataset, workload, parallel, k, id space), so
//! each side may hold ONE dump per key — e.g. a single search config per
//! (parallel, k). An `ef` / `M` sweep in one directory collides and is refused;
//! pass the two files to compare explicitly.

use crate::results_dump::{first_answers, Dump, IdSpace, QueryRecord};
use crate::workload::Workload;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct SideInfo {
    pub engine: String,
    pub config: String,
    pub file: String,
    pub parallel: usize,
    pub queries_attempted: usize,
    pub queries_answered: usize,
    /// Records with `error` set, counting every attempt (a query cycled several
    /// times can contribute more than one). Nonzero fails the comparison.
    pub failed_attempts: usize,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct CountAgreement {
    pub queries: usize,
    pub exact_matches: usize,
    /// `exact_matches / queries`; `None` when no compared query carried totals.
    pub fraction: Option<f64>,
    pub max_abs_diff: u64,
    /// Queries where both totals are <= k, so both pages are the full result set.
    pub complete_pages: usize,
    /// Fraction of those whose id SETS are equal. `None` when there are none.
    pub complete_page_set_agreement: Option<f64>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ShapeReport {
    pub shape: String,
    /// Queries answered by both sides.
    pub queries: usize,
    pub overlap_at_k: Option<f64>,
    pub jaccard_at_k: Option<f64>,
    pub rank_correlation: Option<f64>,
    pub rank_correlation_queries: usize,
    pub count_agreement: Option<CountAgreement>,
    pub recall_at_k_reference: Option<f64>,
    pub recall_at_k_candidate: Option<f64>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct OneSided {
    pub query_id: usize,
    /// `"reference"` or `"candidate"`: the side that answered.
    pub answered_by: String,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct PairReport {
    pub dataset: String,
    pub workload: Workload,
    pub k: usize,
    pub reference: SideInfo,
    pub candidate: SideInfo,
    pub shapes: Vec<ShapeReport>,
    pub one_sided: Vec<OneSided>,
    /// Attempted by at least one side, answered by neither.
    pub unanswered_by_both: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Report {
    /// The engine declared on the command line as the source of truth.
    pub reference_engine: String,
    pub pairs: Vec<PairReport>,
    /// Dumps with no counterpart on the other side (file names).
    pub unpaired: Vec<String>,
    pub one_sided_total: usize,
    /// Sum of both sides' `failed_attempts` over every pair.
    pub failed_attempts_total: usize,
}

impl Report {
    /// Strict verdict: no one-sided query, no failed attempt, and no dump left
    /// without a counterpart (a missing file is the same gap as a missing
    /// query, one level up).
    pub fn passed(&self) -> bool {
        self.passed_allowing_unpaired(false)
    }

    /// [`Self::passed`], optionally tolerating unpaired dumps
    /// (`compare-results --allow-unpaired`).
    pub fn passed_allowing_unpaired(&self, allow_unpaired: bool) -> bool {
        self.one_sided_total == 0
            && self.failed_attempts_total == 0
            && (allow_unpaired || self.unpaired.is_empty())
    }
}

fn top_k(ids: &[String], k: usize) -> Vec<&str> {
    let mut seen = HashSet::new();
    ids.iter()
        .map(String::as_str)
        .filter(|id| seen.insert(*id))
        .take(k)
        .collect()
}

/// |R∩C| / |R| over the first k (deduplicated) ids.
pub fn overlap_at_k(reference: &[String], candidate: &[String], k: usize) -> f64 {
    let r = top_k(reference, k);
    let c: HashSet<&str> = top_k(candidate, k).into_iter().collect();
    if r.is_empty() {
        return if c.is_empty() { 1.0 } else { 0.0 };
    }
    r.iter().filter(|id| c.contains(*id)).count() as f64 / r.len() as f64
}

/// |R∩C| / |R∪C| over the first k (deduplicated) ids.
pub fn jaccard_at_k(reference: &[String], candidate: &[String], k: usize) -> f64 {
    let r: HashSet<&str> = top_k(reference, k).into_iter().collect();
    let c: HashSet<&str> = top_k(candidate, k).into_iter().collect();
    let union = r.union(&c).count();
    if union == 0 {
        return 1.0;
    }
    r.intersection(&c).count() as f64 / union as f64
}

/// Kendall tau-a over the ids both top-k lists share. `None` below 2 shared.
pub fn rank_correlation(reference: &[String], candidate: &[String], k: usize) -> Option<f64> {
    let r = top_k(reference, k);
    let c_pos: HashMap<&str, usize> = top_k(candidate, k)
        .into_iter()
        .enumerate()
        .map(|(i, id)| (id, i))
        .collect();
    // Candidate positions of the shared ids, in reference order.
    let pos: Vec<usize> = r.iter().filter_map(|id| c_pos.get(id).copied()).collect();
    let n = pos.len();
    if n < 2 {
        return None;
    }
    let mut score: i64 = 0;
    for i in 0..n {
        for j in i + 1..n {
            score += if pos[i] < pos[j] { 1 } else { -1 };
        }
    }
    Some(score as f64 / (n * (n - 1) / 2) as f64)
}

/// |ids[:k] ∩ expected| / |expected| (deduplicated). `None` for an empty
/// ground-truth row.
pub fn recall_at_k(ids: &[String], expected: &[String], k: usize) -> Option<f64> {
    let exp: HashSet<&str> = expected.iter().map(String::as_str).collect();
    if exp.is_empty() {
        return None;
    }
    let got = top_k(ids, k);
    Some(got.iter().filter(|id| exp.contains(*id)).count() as f64 / exp.len() as f64)
}

fn mean(v: &[f64]) -> Option<f64> {
    (!v.is_empty()).then(|| v.iter().sum::<f64>() / v.len() as f64)
}

fn side_info(file: &str, d: &Dump) -> SideInfo {
    let (answered, _) = first_answers(&d.records);
    SideInfo {
        engine: d.header.engine.clone(),
        config: d.header.config.clone(),
        file: file.to_string(),
        parallel: d.header.parallel,
        queries_attempted: d
            .records
            .iter()
            .map(|r| r.query_id)
            .collect::<BTreeSet<_>>()
            .len(),
        queries_answered: answered.len(),
        failed_attempts: d.records.iter().filter(|r| !r.answered()).count(),
    }
}

/// Compare one candidate dump against one reference dump.
pub fn compare_pair(
    reference: (&str, &Dump),
    candidate: (&str, &Dump),
) -> Result<PairReport, String> {
    let (rf, rd) = reference;
    let (cf, cd) = candidate;
    let (rh, ch) = (&rd.header, &cd.header);
    if rh.dataset != ch.dataset {
        return Err(format!(
            "{rf} and {cf} are different datasets ({} vs {})",
            rh.dataset, ch.dataset
        ));
    }
    if rh.workload != ch.workload {
        return Err(format!(
            "{rf} and {cf} are different workloads ({} vs {})",
            rh.workload, ch.workload
        ));
    }
    if rh.id_space != ch.id_space {
        return Err(format!(
            "{rf} and {cf} name documents in different id spaces ({:?} vs {:?})",
            rh.id_space, ch.id_space
        ));
    }
    if rh.top != ch.top {
        return Err(format!(
            "{rf} and {cf} asked for different k ({} vs {})",
            rh.top, ch.top
        ));
    }
    let k = rh.top;
    let workload = rh.workload;
    let (r_ans, _) = first_answers(&rd.records);
    let (c_ans, _) = first_answers(&cd.records);
    let all: BTreeSet<usize> = rd
        .records
        .iter()
        .chain(cd.records.iter())
        .map(|r| r.query_id)
        .collect();

    let mut one_sided = Vec::new();
    let mut both = Vec::new();
    let mut neither = Vec::new();
    for q in all {
        match (r_ans.get(&q), c_ans.get(&q)) {
            (Some(r), Some(c)) => both.push((q, *r, *c)),
            (Some(_), None) => one_sided.push(OneSided {
                query_id: q,
                answered_by: "reference".into(),
            }),
            (None, Some(_)) => one_sided.push(OneSided {
                query_id: q,
                answered_by: "candidate".into(),
            }),
            (None, None) => neither.push(q),
        }
    }

    let mut groups: BTreeMap<String, Vec<(&QueryRecord, &QueryRecord)>> = BTreeMap::new();
    for (q, r, c) in &both {
        match (&r.expected, &c.expected) {
            (Some(re), Some(ce)) if re != ce => {
                return Err(format!(
                    "query {q}: {rf} and {cf} carry different ground truth — the dumps were \
                     not produced from the same corpus"
                ));
            }
            (Some(_), None) | (None, Some(_)) => {
                return Err(format!(
                    "query {q}: only one of {rf} and {cf} carries ground truth; recall would be \
                     reported for one side only"
                ));
            }
            _ => {}
        }
        let shape = r
            .shape
            .clone()
            .or_else(|| c.shape.clone())
            .unwrap_or_else(|| workload.as_str().to_string());
        groups.entry(shape).or_default().push((*r, *c));
    }

    let shapes = groups
        .into_iter()
        .map(|(shape, qs)| shape_report(shape, workload, k, &qs))
        .collect();

    Ok(PairReport {
        dataset: rh.dataset.clone(),
        workload,
        k,
        reference: side_info(rf, rd),
        candidate: side_info(cf, cd),
        shapes,
        one_sided,
        unanswered_by_both: neither,
    })
}

fn shape_report(
    shape: String,
    workload: Workload,
    k: usize,
    qs: &[(&QueryRecord, &QueryRecord)],
) -> ShapeReport {
    let mut overlap = Vec::new();
    let mut jaccard = Vec::new();
    let mut tau = Vec::new();
    let mut rec_r = Vec::new();
    let mut rec_c = Vec::new();
    for (r, c) in qs {
        if workload.is_ranked() {
            overlap.push(overlap_at_k(&r.ids, &c.ids, k));
            jaccard.push(jaccard_at_k(&r.ids, &c.ids, k));
            if let Some(t) = rank_correlation(&r.ids, &c.ids, k) {
                tau.push(t);
            }
        }
        if workload.has_ground_truth() {
            if let Some(exp) = &r.expected {
                rec_r.extend(recall_at_k(&r.ids, exp, k));
            }
            if let Some(exp) = &c.expected {
                rec_c.extend(recall_at_k(&c.ids, exp, k));
            }
        }
    }
    let count_agreement = (!workload.is_ranked()).then(|| {
        let mut exact = 0;
        let mut max_diff = 0u64;
        let mut complete = 0;
        let mut complete_equal = 0;
        let mut with_totals = 0;
        for (r, c) in qs {
            let (Some(rt), Some(ct)) = (r.total, c.total) else {
                continue;
            };
            with_totals += 1;
            if rt == ct {
                exact += 1;
            }
            max_diff = max_diff.max(rt.abs_diff(ct));
            if rt <= k as u64 && ct <= k as u64 {
                complete += 1;
                let rs: HashSet<&String> = r.ids.iter().collect();
                let cs: HashSet<&String> = c.ids.iter().collect();
                if rs == cs {
                    complete_equal += 1;
                }
            }
        }
        CountAgreement {
            queries: with_totals,
            exact_matches: exact,
            fraction: (with_totals > 0).then(|| exact as f64 / with_totals as f64),
            max_abs_diff: max_diff,
            complete_pages: complete,
            complete_page_set_agreement: (complete > 0)
                .then(|| complete_equal as f64 / complete as f64),
        }
    });
    ShapeReport {
        shape,
        queries: qs.len(),
        overlap_at_k: mean(&overlap),
        jaccard_at_k: mean(&jaccard),
        rank_correlation: mean(&tau),
        rank_correlation_queries: tau.len(),
        count_agreement,
        recall_at_k_reference: mean(&rec_r),
        recall_at_k_candidate: mean(&rec_c),
    }
}

/// Which side a dump belongs to: does its header name the reference engine
/// (engine type or config name)?
fn names(d: &Dump, engine: &str) -> bool {
    d.header.engine == engine || d.header.config == engine
}

type PairKey = (String, Workload, usize, usize, String);

fn key(d: &Dump) -> PairKey {
    let space = match &d.header.id_space {
        IdSpace::Row => "row".to_string(),
        IdSpace::Field { field } => field.clone(),
    };
    (
        d.header.dataset.clone(),
        d.header.workload,
        d.header.parallel,
        d.header.top,
        space,
    )
}

fn index<'a>(
    side: &'a [(String, Dump)],
    label: &str,
) -> Result<BTreeMap<PairKey, &'a (String, Dump)>, String> {
    let mut out: BTreeMap<PairKey, &(String, Dump)> = BTreeMap::new();
    for entry in side {
        if let Some(prev) = out.insert(key(&entry.1), entry) {
            return Err(format!(
                "{label}: {} and {} have the same (dataset, workload, parallel, k, id space); \
                 directory mode holds one dump per key (a sweep of search configs collides); pass the \
                 two files to compare explicitly",
                prev.0, entry.0
            ));
        }
    }
    Ok(out)
}

/// Compare two sets of dumps. The set whose every header names
/// `reference_engine` is the reference; when both do (a self-comparison) the
/// first is.
pub fn compare(
    first: &[(String, Dump)],
    second: &[(String, Dump)],
    reference_engine: &str,
) -> Result<Report, String> {
    if first.is_empty() || second.is_empty() {
        return Err("compare-results needs at least one dump on each side".into());
    }
    let all_name = |s: &[(String, Dump)]| s.iter().all(|(_, d)| names(d, reference_engine));
    let (reference, candidate) = if all_name(first) {
        (first, second)
    } else if all_name(second) {
        (second, first)
    } else {
        return Err(format!(
            "neither side's dumps all name the reference engine {reference_engine:?} (engine \
             type or config name in the dump header)"
        ));
    };
    // Pair one-to-one when both sides are a single file; otherwise by key.
    let mut pairs = Vec::new();
    let mut unpaired = Vec::new();
    if reference.len() == 1 && candidate.len() == 1 {
        let (rf, rd) = &reference[0];
        let (cf, cd) = &candidate[0];
        pairs.push(compare_pair((rf, rd), (cf, cd))?);
    } else {
        let r = index(reference, "reference")?;
        let mut c = index(candidate, "candidate")?;
        for (k, (rf, rd)) in r {
            match c.remove(&k) {
                Some((cf, cd)) => pairs.push(compare_pair((rf, rd), (cf, cd))?),
                None => unpaired.push(rf.clone()),
            }
        }
        unpaired.extend(c.into_values().map(|(f, _)| f.clone()));
    }
    if pairs.is_empty() {
        return Err(format!(
            "no reference dump has a candidate counterpart (unpaired: {})",
            unpaired.join(", ")
        ));
    }
    let one_sided_total = pairs.iter().map(|p| p.one_sided.len()).sum();
    let failed_attempts_total = pairs
        .iter()
        .map(|p| p.reference.failed_attempts + p.candidate.failed_attempts)
        .sum();
    Ok(Report {
        reference_engine: reference_engine.to_string(),
        pairs,
        unpaired,
        one_sided_total,
        failed_attempts_total,
    })
}

fn f(v: Option<f64>) -> String {
    v.map(|x| format!("{x:.4}")).unwrap_or_else(|| "—".into())
}

pub fn render_markdown(r: &Report) -> String {
    let mut s = format!(
        "# compare-results\n\nReference engine: **{}**\n\n\
         | dataset | workload | shape | reference | candidate | parallel (ref/cand) | k | queries | \
         overlap@k | Jaccard@k | rank corr. | count agreement | recall@k ref | recall@k cand | one-sided |\n\
         |---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
        r.reference_engine
    );
    for p in &r.pairs {
        for sh in &p.shapes {
            let counts = sh
                .count_agreement
                .as_ref()
                .map(|c| {
                    format!(
                        "{}/{} ({}), max Δ {}",
                        c.exact_matches,
                        c.queries,
                        f(c.fraction),
                        c.max_abs_diff
                    )
                })
                .unwrap_or_else(|| "—".into());
            s.push_str(&format!(
                "| {} | {} | {} | {} | {} | {}/{} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
                p.dataset,
                p.workload,
                sh.shape,
                p.reference.config,
                p.candidate.config,
                p.reference.parallel,
                p.candidate.parallel,
                p.k,
                sh.queries,
                f(sh.overlap_at_k),
                f(sh.jaccard_at_k),
                f(sh.rank_correlation),
                counts,
                f(sh.recall_at_k_reference),
                f(sh.recall_at_k_candidate),
                p.one_sided.len(),
            ));
        }
    }
    if !r.unpaired.is_empty() {
        s.push_str(&format!(
            "\n**FAILED** (unless --allow-unpaired): {} dump(s) with no counterpart on the other \
             side:\n\n",
            r.unpaired.len()
        ));
        for u in &r.unpaired {
            s.push_str(&format!("- {u}\n"));
        }
    }
    if r.failed_attempts_total > 0 {
        s.push_str(&format!(
            "\n**FAILED**: {} failed query attempt(s) in the dumps (the \
             --fail-on-dropped-queries policy, #213):\n\n",
            r.failed_attempts_total
        ));
        for p in &r.pairs {
            for (side, info) in [("reference", &p.reference), ("candidate", &p.candidate)] {
                if info.failed_attempts > 0 {
                    s.push_str(&format!(
                        "- {} {} p{}: {} failed attempt(s) on the {side} ({})\n",
                        p.dataset, p.workload, info.parallel, info.failed_attempts, info.config
                    ));
                }
            }
        }
    }
    if r.one_sided_total > 0 {
        s.push_str(&format!(
            "\n**FAILED**: {} quer{} answered by one engine and not the other:\n\n",
            r.one_sided_total,
            if r.one_sided_total == 1 { "y" } else { "ies" }
        ));
        for p in &r.pairs {
            for o in p.one_sided.iter().take(20) {
                s.push_str(&format!(
                    "- {} {} p{}: query {} answered only by the {}\n",
                    p.dataset, p.workload, p.reference.parallel, o.query_id, o.answered_by
                ));
            }
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::results_dump::{DumpHeader, DUMP_SCHEMA_VERSION};

    fn ids(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    fn rec(q: usize, got: &[&str], total: Option<u64>, ok: bool) -> QueryRecord {
        QueryRecord {
            query_id: q,
            seq: q,
            shape: None,
            ids: ids(got),
            scores: None,
            total,
            latency: 0.001,
            error: (!ok).then(|| "failed".to_string()),
            expected: Some(ids(&["a", "b", "c", "d"])),
        }
    }

    fn dump(engine: &str, workload: Workload, parallel: usize, records: Vec<QueryRecord>) -> Dump {
        Dump {
            header: DumpHeader {
                schema_version: DUMP_SCHEMA_VERSION,
                engine: engine.into(),
                config: format!("{engine}-cfg"),
                dataset: "ds".into(),
                workload,
                parallel,
                top: 4,
                search_id: 0,
                id_space: IdSpace::Row,
            },
            records,
        }
    }

    fn vector_dump(engine: &str) -> Dump {
        dump(
            engine,
            Workload::Vector,
            1,
            vec![
                rec(0, &["a", "b", "c", "d"], None, true),
                rec(1, &["b", "a", "x", "c"], None, true),
                rec(2, &["d", "c", "b", "a"], None, true),
            ],
        )
    }

    fn one(r: &Report) -> &ShapeReport {
        assert_eq!(r.pairs.len(), 1);
        assert_eq!(r.pairs[0].shapes.len(), 1);
        &r.pairs[0].shapes[0]
    }

    /// Acceptance #1: a dump compared against itself agrees perfectly.
    #[test]
    fn self_comparison_is_perfect_agreement() {
        let a = vec![("a.jsonl".to_string(), vector_dump("opensearch"))];
        let r = compare(&a, &a, "opensearch").unwrap();
        let s = one(&r);
        assert_eq!(s.queries, 3);
        assert_eq!(s.overlap_at_k, Some(1.0));
        assert_eq!(s.jaccard_at_k, Some(1.0));
        assert_eq!(s.rank_correlation, Some(1.0));
        assert_eq!(s.recall_at_k_reference, s.recall_at_k_candidate);
        assert!(r.passed());
    }

    /// Acceptance #2 (mutation): reordering a dump's ids keeps overlap at 1.0
    /// but must lower the rank correlation.
    #[test]
    fn a_reordered_dump_lowers_rank_correlation_only() {
        let reference = vector_dump("opensearch");
        let mut reordered = vector_dump("redis");
        for r in &mut reordered.records {
            r.ids.reverse();
        }
        let r = compare(
            &[("ref".into(), reference)],
            &[("cand".into(), reordered)],
            "opensearch",
        )
        .unwrap();
        let s = one(&r);
        assert_eq!(s.overlap_at_k, Some(1.0));
        assert_eq!(s.jaccard_at_k, Some(1.0));
        assert!(
            s.rank_correlation.unwrap() < 1.0,
            "{:?}",
            s.rank_correlation
        );
        assert_eq!(s.rank_correlation, Some(-1.0));
    }

    #[test]
    fn metric_definitions() {
        let r = ids(&["a", "b", "c", "d"]);
        assert_eq!(overlap_at_k(&r, &ids(&["a", "b", "x", "y"]), 4), 0.5);
        assert!((jaccard_at_k(&r, &ids(&["a", "b", "x", "y"]), 4) - 2.0 / 6.0).abs() < 1e-12);
        // k truncates before comparing.
        assert_eq!(overlap_at_k(&r, &ids(&["a", "b", "x", "y"]), 2), 1.0);
        // Empty vs empty agrees; empty reference vs non-empty candidate does not.
        assert_eq!(overlap_at_k(&[], &[], 4), 1.0);
        assert_eq!(overlap_at_k(&[], &ids(&["a"]), 4), 0.0);
        assert_eq!(jaccard_at_k(&[], &[], 4), 1.0);
        // One swapped adjacent pair out of 6 pairs: tau = (5-1)/6.
        let t = rank_correlation(&r, &ids(&["b", "a", "c", "d"]), 4).unwrap();
        assert!((t - 4.0 / 6.0).abs() < 1e-12, "{t}");
        assert_eq!(rank_correlation(&r, &ids(&["a", "x"]), 4), None);
        assert_eq!(
            recall_at_k(&ids(&["a", "x"]), &ids(&["a", "b"]), 2),
            Some(0.5)
        );
        assert_eq!(recall_at_k(&ids(&["a"]), &[], 2), None);
    }

    #[test]
    fn one_sided_answers_fail_the_report() {
        let reference = vector_dump("opensearch");
        let mut cand = vector_dump("redis");
        cand.records[1].error = Some("timeout".into());
        cand.records[1].ids.clear();
        cand.records.remove(2); // never attempted at all
        let r = compare(
            &[("ref".into(), reference)],
            &[("cand".into(), cand)],
            "opensearch",
        )
        .unwrap();
        assert!(!r.passed());
        assert_eq!(r.one_sided_total, 2);
        assert!(r.pairs[0]
            .one_sided
            .iter()
            .all(|o| o.answered_by == "reference"));
        assert!(render_markdown(&r).contains("FAILED"));
        // A query both sides dropped is not one-sided, but the failed attempts
        // still fail the comparison.
        let mut a = vector_dump("opensearch");
        a.records[0].error = Some("x".into());
        let mut b = vector_dump("redis");
        b.records[0].error = Some("x".into());
        let r = compare(&[("a".into(), a)], &[("b".into(), b)], "opensearch").unwrap();
        assert!(r.pairs[0].one_sided.is_empty());
        assert_eq!(r.pairs[0].unanswered_by_both, vec![0]);
        assert_eq!(r.failed_attempts_total, 2);
        assert!(!r.passed());
    }

    /// #343 review: under query cycling a query can fail once and be answered
    /// on a later cycle. It is compared on that answer, but the failed attempt
    /// must still fail the comparison, as --fail-on-dropped-queries would.
    #[test]
    fn a_failed_attempt_answered_on_a_later_cycle_still_fails() {
        let reference = vector_dump("opensearch");
        let mut cand = vector_dump("redis");
        let mut retry = cand.records[0].clone();
        cand.records[0].error = Some("timeout".into());
        cand.records[0].ids.clear();
        retry.seq = 10; // the same query, answered on the next cycle
        cand.records.push(retry);
        let r = compare(
            &[("ref".into(), reference)],
            &[("cand".into(), cand)],
            "opensearch",
        )
        .unwrap();
        assert!(r.pairs[0].one_sided.is_empty(), "answered on the retry");
        assert_eq!(r.pairs[0].candidate.failed_attempts, 1);
        assert_eq!(r.pairs[0].reference.failed_attempts, 0);
        assert_eq!(one(&r).overlap_at_k, Some(1.0), "compared on the answer");
        assert!(!r.passed());
        assert!(render_markdown(&r).contains("1 failed attempt(s) on the candidate"));
    }

    #[test]
    fn count_agreement_without_totals_is_not_computed() {
        let mut a = dump(
            "opensearch",
            Workload::FilterOnly,
            1,
            vec![rec(0, &["a"], None, true)],
        );
        a.records[0].expected = None;
        let r = compare(&[("a".into(), a.clone())], &[("b".into(), a)], "opensearch").unwrap();
        let c = one(&r).count_agreement.as_ref().unwrap();
        assert_eq!((c.queries, c.fraction), (0, None));
        assert!(render_markdown(&r).contains("0/0 (—)"));
    }

    #[test]
    fn filter_only_reports_count_agreement_not_rank() {
        let a = dump(
            "opensearch",
            Workload::FilterOnly,
            2,
            vec![
                rec(0, &["a", "b"], Some(2), true),
                rec(1, &["a", "b", "c", "d"], Some(100), true),
            ],
        );
        let b = dump(
            "redis",
            Workload::FilterOnly,
            2,
            vec![
                rec(0, &["b", "a"], Some(2), true),
                rec(1, &["x", "y", "z", "w"], Some(99), true),
            ],
        );
        let r = compare(&[("b".into(), b)], &[("a".into(), a)], "opensearch").unwrap();
        // Reference is the side naming the engine, whichever position it is in.
        assert_eq!(r.pairs[0].reference.engine, "opensearch");
        let s = one(&r);
        assert_eq!(s.overlap_at_k, None);
        assert_eq!(s.rank_correlation, None);
        assert_eq!(s.recall_at_k_reference, None);
        let c = s.count_agreement.as_ref().unwrap();
        assert_eq!((c.queries, c.exact_matches, c.max_abs_diff), (2, 1, 1));
        assert_eq!(c.fraction, Some(0.5));
        assert_eq!(c.complete_pages, 1);
        assert_eq!(c.complete_page_set_agreement, Some(1.0));
    }

    #[test]
    fn incompatible_dumps_are_refused() {
        let a = vector_dump("opensearch");
        let mut b = vector_dump("redis");
        b.header.dataset = "other".into();
        assert!(compare_pair(("a", &a), ("b", &b)).is_err());
        let mut b = vector_dump("redis");
        b.header.top = 10;
        assert!(compare_pair(("a", &a), ("b", &b)).is_err());
        let mut b = vector_dump("redis");
        b.header.id_space = IdSpace::Field {
            field: "docid".into(),
        };
        assert!(compare_pair(("a", &a), ("b", &b)).is_err());
        let mut b = vector_dump("redis");
        b.records[0].expected = Some(ids(&["z"]));
        assert!(compare_pair(("a", &a), ("b", &b)).is_err());
        // Ground truth on one side only.
        let mut b = vector_dump("redis");
        b.records[0].expected = None;
        assert!(compare_pair(("a", &a), ("b", &b)).is_err());
        // Neither side names the reference.
        assert!(compare(
            &[("a".into(), a.clone())],
            &[("b".into(), vector_dump("redis"))],
            "qdrant"
        )
        .is_err());
    }

    #[test]
    fn directories_pair_by_parallel_and_report_the_rest() {
        let r1 = vector_dump("opensearch");
        let mut r2 = vector_dump("opensearch");
        r2.header.parallel = 8;
        let c1 = vector_dump("redis");
        let mut c3 = vector_dump("redis");
        c3.header.parallel = 16;
        let r = compare(
            &[("r1".into(), r1.clone()), ("r2".into(), r2)],
            &[("c1".into(), c1), ("c3".into(), c3)],
            "opensearch",
        )
        .unwrap();
        assert_eq!(r.pairs.len(), 1);
        assert_eq!(r.pairs[0].reference.parallel, 1);
        let mut unpaired = r.unpaired.clone();
        unpaired.sort();
        assert_eq!(unpaired, vec!["c3".to_string(), "r2".to_string()]);
        // A dump with no counterpart fails the comparison unless allowed.
        assert!(!r.passed());
        assert!(r.passed_allowing_unpaired(true));
        assert!(render_markdown(&r).contains("2 dump(s) with no counterpart"));
        // Two reference dumps with the same key are ambiguous.
        let e = compare(
            &[("x".into(), r1.clone()), ("y".into(), r1)],
            &[
                ("c".into(), vector_dump("redis")),
                ("d".into(), vector_dump("redis")),
            ],
            "opensearch",
        )
        .unwrap_err();
        assert!(e.contains("explicitly"), "{e}");
    }

    #[test]
    fn shapes_are_reported_separately() {
        let mut a = vector_dump("opensearch");
        a.records[0].shape = Some("sel-1pct".into());
        let r = compare(&[("a".into(), a.clone())], &[("b".into(), a)], "opensearch").unwrap();
        let shapes: Vec<_> = r.pairs[0].shapes.iter().map(|s| s.shape.as_str()).collect();
        assert_eq!(shapes, vec!["sel-1pct", "vector"]);
        let md = render_markdown(&r);
        assert!(md.contains("sel-1pct") && md.contains("Reference engine: **opensearch**"));
    }
}
