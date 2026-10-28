//! The hybrid (prefilter + KNN) workload (#341): its query-set file, the
//! selectivity buckets results are reported by, the checks that an engine
//! expressed every predicate it was given, and the per-bucket report.
//!
//! # Query set
//!
//! A compound dataset directory may carry [`HYBRID_QUERIES_FILE`] next to
//! `tests.jsonl`. Each line is one [`HybridQueryRow`]: a query vector BY
//! REFERENCE (`test_index` into `tests.jsonl`), a predicate in the repo's
//! `conditions` grammar, the exact number of corpus rows satisfying it, and
//! the brute-forced top-k over THOSE rows only (ids + cosine scores). For MS
//! MARCO the file is written by `prepare-msmarco` (`msmarco::hybrid`).
//!
//! # Contract the engine arms follow
//!
//! 1. Every predicate is compiled by the oracle's own evaluator
//!    (`msmarco::derived::Predicate::compile`) and checked against the
//!    dataset schema ([`validate_against_schema`]) before anything runs.
//! 2. After the engine's builder turned it into an engine filter, the filter
//!    must still contain every leaf ([`check_redis_filter`],
//!    [`check_opensearch_filter`]). A predicate an engine cannot express is a
//!    hard error — never an unfiltered KNN scored against filtered truth.
//! 3. Each bucket runs as its own timed pass, so its QPS is a measured
//!    throughput, not a share of a mixed stream; the run's headline numbers
//!    are the union of the passes ([`HybridReport`]).
//! 4. The result carries a [`HybridReport`]; a hybrid result without one, or
//!    any other workload with one, is refused ([`verify_fingerprint`]).

use crate::msmarco::derived::{Leaf, Predicate};
use crate::workload::Workload;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::io::BufRead;
use std::path::Path;

/// File name of the hybrid query set inside a compound dataset directory.
pub const HYBRID_QUERIES_FILE: &str = "hybrid_queries.jsonl";

/// Fewest corpus rows a hybrid predicate may match: below this, recall@10 is
/// computed over a ground truth narrower than the page and stops meaning
/// what its name says.
pub const MIN_MATCHES: u64 = 10;

/// Selectivity buckets, by REALIZED selectivity (matching rows / corpus).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Bucket {
    #[serde(rename = "50%")]
    S50,
    #[serde(rename = "10%")]
    S10,
    #[serde(rename = "1%")]
    S1,
    #[serde(rename = "0.1%")]
    S01,
}

impl Bucket {
    pub const ALL: [Bucket; 4] = [Bucket::S50, Bucket::S10, Bucket::S1, Bucket::S01];

    pub fn as_str(self) -> &'static str {
        match self {
            Bucket::S50 => "50%",
            Bucket::S10 => "10%",
            Bucket::S1 => "1%",
            Bucket::S01 => "0.1%",
        }
    }

    /// The selectivity the bucket is named after.
    pub fn target(self) -> f64 {
        match self {
            Bucket::S50 => 0.5,
            Bucket::S10 => 0.1,
            Bucket::S1 => 0.01,
            Bucket::S01 => 0.001,
        }
    }

    /// The bucket a realized selectivity falls in: boundaries at a factor of
    /// sqrt(10)-ish below each target (25%, 2.5%, 0.25%), so each bucket is
    /// "within ~4x of its name".
    pub fn of(matching: u64, corpus: u64) -> Result<Bucket, String> {
        if corpus == 0 || matching == 0 || matching > corpus {
            return Err(format!(
                "selectivity {matching}/{corpus} is not in (0, 1] — no bucket"
            ));
        }
        // Integer comparisons: no float rounding at a boundary.
        let m = matching as u128 * 1000;
        let c = corpus as u128;
        Ok(if m >= 250 * c {
            Bucket::S50
        } else if m * 10 >= 250 * c {
            Bucket::S10
        } else if m * 100 >= 250 * c {
            Bucket::S1
        } else {
            Bucket::S01
        })
    }
}

impl std::fmt::Display for Bucket {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One line of [`HYBRID_QUERIES_FILE`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HybridQueryRow {
    pub query_id: String,
    /// Line index into `tests.jsonl`: the query vector this row searches with.
    pub test_index: usize,
    /// The source query's own id (MS MARCO: the TREC topic id), for tracing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_query_id: Option<String>,
    pub predicate_id: String,
    /// What the predicate exercises (`numeric_range`, `text`, ...).
    pub predicate_shape: String,
    /// Bucket of the REALIZED selectivity; `Bucket::of(matching_count,
    /// corpus_size)` by construction (checked on read).
    pub bucket: Bucket,
    pub conditions: Value,
    /// Exact number of corpus rows satisfying `conditions`.
    pub matching_count: u64,
    pub corpus_size: u64,
    /// `matching_count / corpus_size`.
    pub selectivity: f64,
    /// Depth the ground truth was brute-forced to; `closest_ids` holds
    /// `min(gt_depth, matching_count)` ids.
    pub gt_depth: usize,
    /// Brute-forced nearest rows among those satisfying `conditions`, best
    /// first.
    pub closest_ids: Vec<i64>,
    /// Cosine of each id, aligned with `closest_ids`.
    pub closest_scores: Vec<f32>,
}

impl HybridQueryRow {
    /// Internal consistency, checked whenever a row is read.
    pub fn check(&self) -> Result<(), String> {
        let id = &self.query_id;
        let bucket =
            Bucket::of(self.matching_count, self.corpus_size).map_err(|e| format!("{id}: {e}"))?;
        if bucket != self.bucket {
            return Err(format!(
                "{id}: declared bucket {} but {}/{} rows is bucket {bucket}",
                self.bucket, self.matching_count, self.corpus_size
            ));
        }
        let s = self.matching_count as f64 / self.corpus_size as f64;
        if (s - self.selectivity).abs() > 1e-12 {
            return Err(format!(
                "{id}: selectivity {} != {}/{}",
                self.selectivity, self.matching_count, self.corpus_size
            ));
        }
        if self.matching_count < MIN_MATCHES {
            return Err(format!(
                "{id}: matches {} rows, fewer than the {MIN_MATCHES} a recall@k needs",
                self.matching_count
            ));
        }
        let want = (self.gt_depth as u64).min(self.matching_count) as usize;
        if self.closest_ids.len() != want || self.closest_scores.len() != want {
            return Err(format!(
                "{id}: ground truth holds {} ids / {} scores, expected min(gt_depth {}, matching \
                 {}) = {want}",
                self.closest_ids.len(),
                self.closest_scores.len(),
                self.gt_depth,
                self.matching_count
            ));
        }
        let mut seen = std::collections::HashSet::with_capacity(want);
        for (i, &r) in self.closest_ids.iter().enumerate() {
            if r < 0 || r as u64 >= self.corpus_size || !seen.insert(r) {
                return Err(format!(
                    "{id}: ground-truth id {r} is out of range or repeated"
                ));
            }
            if i > 0 && self.closest_scores[i] > self.closest_scores[i - 1] {
                return Err(format!(
                    "{id}: ground-truth scores are not descending at {i}"
                ));
            }
        }
        Predicate::compile(&self.conditions).map_err(|e| format!("{id}: {e}"))?;
        Ok(())
    }
}

/// Per-query metadata the runner and report need besides vector + truth.
#[derive(Debug, Clone, PartialEq)]
pub struct HybridQueryMeta {
    pub query_id: String,
    pub predicate_id: String,
    pub predicate_shape: String,
    pub bucket: Bucket,
    pub matching_count: u64,
    pub selectivity: f64,
}

/// A dataset's hybrid query set, aligned by index.
pub struct HybridQuerySet {
    pub queries: Vec<Vec<f32>>,
    pub neighbors: Vec<Vec<i64>>,
    /// The predicates, behind the #219 resolver door like every other filter.
    pub conditions: crate::query_filter::QueryConditions,
    /// Compiled by the oracle's evaluator; aligned with `queries`.
    pub predicates: Vec<Predicate>,
    /// The same predicates as JSON, for structural comparison against an
    /// engine's rendered filter ([`check_opensearch_filter`]).
    raw_conditions: Vec<Value>,
    pub meta: Vec<HybridQueryMeta>,
}

impl HybridQuerySet {
    /// Assemble a set from aligned parts; every predicate is compiled by the
    /// oracle's evaluator (an uncompilable one is an error here, not later).
    pub fn from_parts(
        queries: Vec<Vec<f32>>,
        neighbors: Vec<Vec<i64>>,
        conditions: Vec<Value>,
        meta: Vec<HybridQueryMeta>,
    ) -> Result<Self, String> {
        let n = queries.len();
        if neighbors.len() != n || conditions.len() != n || meta.len() != n {
            return Err(format!(
                "hybrid set parts disagree: {n} queries, {} truth rows, {} predicates, {} metas",
                neighbors.len(),
                conditions.len(),
                meta.len()
            ));
        }
        let predicates = conditions
            .iter()
            .zip(&meta)
            .map(|(c, m)| Predicate::compile(c).map_err(|e| format!("{}: {e}", m.query_id)))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            queries,
            neighbors,
            conditions: crate::query_filter::QueryConditions::new(
                conditions.iter().cloned().map(Some).collect(),
            ),
            predicates,
            raw_conditions: conditions,
            meta,
        })
    }

    /// Query positions per bucket, in file order. Empty buckets are omitted.
    pub fn by_bucket(&self) -> Vec<(Bucket, Vec<usize>)> {
        Bucket::ALL
            .into_iter()
            .map(|b| {
                (
                    b,
                    (0..self.meta.len())
                        .filter(|&i| self.meta[i].bucket == b)
                        .collect::<Vec<_>>(),
                )
            })
            .filter(|(_, v)| !v.is_empty())
            .collect()
    }
}

/// The raw `conditions` of query `i`, through the #219 resolver door (the
/// conditions value is otherwise not reachable from an engine).
pub fn conditions_of(set: &HybridQuerySet, i: usize) -> Result<&Value, String> {
    set.raw_conditions
        .get(i)
        .ok_or_else(|| format!("hybrid query {i} is outside the set"))
}

/// Read and check [`HYBRID_QUERIES_FILE`] rows.
pub fn read_rows(path: &Path) -> Result<Vec<HybridQueryRow>, String> {
    let f = std::fs::File::open(path).map_err(|e| {
        format!(
            "open {}: {e} — the dataset has no hybrid query set (MS MARCO: run `prepare-msmarco \
             --dataset <name> --add-hybrid-queries`)",
            path.display()
        )
    })?;
    let mut rows: Vec<HybridQueryRow> = Vec::new();
    for (i, line) in std::io::BufReader::new(f).lines().enumerate() {
        let line = line.map_err(|e| format!("read {}: {e}", path.display()))?;
        if line.trim().is_empty() {
            continue;
        }
        let row: HybridQueryRow = serde_json::from_str(&line)
            .map_err(|e| format!("{} line {}: {e}", path.display(), i + 1))?;
        row.check()
            .map_err(|e| format!("{} line {}: {e}", path.display(), i + 1))?;
        if let Some(first) = rows.first() {
            if first.corpus_size != row.corpus_size {
                return Err(format!(
                    "{} line {}: corpus_size {} differs from the first row's {}",
                    path.display(),
                    i + 1,
                    row.corpus_size,
                    first.corpus_size
                ));
            }
        }
        rows.push(row);
    }
    if rows.is_empty() {
        return Err(format!("{} holds no hybrid queries", path.display()));
    }
    Ok(rows)
}

/// Load a compound dataset's hybrid query set: query vectors from
/// `tests.jsonl` (normalized like the vector workload's), predicates and
/// filtered truth from [`HYBRID_QUERIES_FILE`].
pub fn read_query_set(dir: &Path, normalize: bool) -> Result<HybridQuerySet, String> {
    let rows = read_rows(&dir.join(HYBRID_QUERIES_FILE))?;
    let dir_str = dir.to_str().ok_or("dataset path is not UTF-8")?;
    let (tests, _neighbors, _vector_conditions) =
        crate::readers::read_compound_queries(dir_str, normalize)?;
    let (mut queries, mut neighbors, mut conditions, mut meta) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for r in rows {
        let q = tests.get(r.test_index).ok_or_else(|| {
            format!(
                "{}: test_index {} is outside the {} queries of tests.jsonl",
                r.query_id,
                r.test_index,
                tests.len()
            )
        })?;
        queries.push(q.clone());
        neighbors.push(r.closest_ids);
        conditions.push(r.conditions);
        meta.push(HybridQueryMeta {
            query_id: r.query_id,
            predicate_id: r.predicate_id,
            predicate_shape: r.predicate_shape,
            bucket: r.bucket,
            matching_count: r.matching_count,
            selectivity: r.selectivity,
        });
    }
    HybridQuerySet::from_parts(queries, neighbors, conditions, meta)
}

// ---------------------------------------------------------------------------
// Expressibility
// ---------------------------------------------------------------------------

/// Every leaf of a compiled predicate, as (field, leaf), in order.
pub fn leaves(p: &Predicate) -> Vec<(&str, &Leaf)> {
    let mut out = Vec::new();
    fn walk<'a>(p: &'a Predicate, out: &mut Vec<(&'a str, &'a Leaf)>) {
        match p {
            Predicate::And(c) | Predicate::Or(c) => c.iter().for_each(|c| walk(c, out)),
            Predicate::Leaf { field, leaf } => out.push((field.as_str(), leaf)),
        }
    }
    walk(p, &mut out);
    out
}

/// Whether any leaf is a full-text match (Redis must then run `VERBATIM`).
pub fn has_text(p: &Predicate) -> bool {
    leaves(p).iter().any(|(_, l)| matches!(l, Leaf::Text(_)))
}

/// Refuse a predicate whose leaves do not fit the field types the dataset
/// registered: text match only on `text` fields; exact match on `keyword`
/// (string operands) or `int` (integer operands); range only on `int`.
/// A leaf on an unregistered field is refused too — the engines would not
/// index it, so they would match nothing where the oracle matches nothing as
/// well, and the "agreement" would be vacuous.
pub fn validate_against_schema(p: &Predicate, schema: Option<&Value>) -> Result<(), String> {
    let schema = schema
        .and_then(Value::as_object)
        .ok_or("the hybrid workload needs a dataset `schema` to type its predicates")?;
    for (field, leaf) in leaves(p) {
        let ty = schema
            .get(field)
            .and_then(Value::as_str)
            .ok_or_else(|| format!("predicate field {field:?} is not in the dataset schema"))?;
        let ok = match leaf {
            Leaf::Text(_) => ty == "text",
            Leaf::Range { .. } => ty == "int",
            Leaf::Match(ops) => ops.iter().all(|o| match o {
                crate::msmarco::derived::Operand::Keyword(_) => ty == "keyword",
                crate::msmarco::derived::Operand::Int(_) => ty == "int",
            }),
        };
        if !ok {
            return Err(format!(
                "predicate leaf on {field:?} ({leaf:?}) does not fit its schema type {ty:?}"
            ));
        }
    }
    Ok(())
}

/// A Redis `PARAMS` value, as the engine bound it.
#[derive(Debug, Clone, PartialEq)]
pub enum RedisParam {
    Str(String),
    Num(f64),
}

/// The subset of RediSearch query syntax the Redis filter builder emits.
#[derive(Debug, Clone, PartialEq)]
enum RNode {
    And(Vec<RNode>),
    Or(Vec<RNode>),
    /// `@f:[lo hi]`; each side is `None` (infinite) or (param, exclusive).
    Range {
        field: String,
        lo: Option<(String, bool)>,
        hi: Option<(String, bool)>,
    },
    /// `@f:{$a | $b}`.
    Tag {
        field: String,
        params: Vec<String>,
    },
    /// `@f:($p)`.
    Text {
        field: String,
        param: String,
    },
}

struct RParser<'a> {
    s: &'a [u8],
    i: usize,
}

impl RParser<'_> {
    fn ws(&mut self) {
        while self.i < self.s.len() && self.s[self.i] == b' ' {
            self.i += 1;
        }
    }
    fn peek(&self) -> Option<u8> {
        self.s.get(self.i).copied()
    }
    fn err(&self, what: &str) -> String {
        format!(
            "Redis filter is not in the builder's grammar ({what} at byte {} of {:?})",
            self.i,
            String::from_utf8_lossy(self.s)
        )
    }
    /// alternatives := sequence ('|' sequence)*
    fn alternatives(&mut self) -> Result<RNode, String> {
        let mut alts = vec![self.sequence()?];
        loop {
            self.ws();
            if self.peek() == Some(b'|') {
                self.i += 1;
                alts.push(self.sequence()?);
            } else {
                break;
            }
        }
        Ok(if alts.len() == 1 {
            alts.pop().expect("one")
        } else {
            RNode::Or(alts)
        })
    }
    /// sequence := unit (' ' unit)* — juxtaposition is AND.
    fn sequence(&mut self) -> Result<RNode, String> {
        let mut units = Vec::new();
        loop {
            self.ws();
            match self.peek() {
                Some(b'(') | Some(b'@') => units.push(self.unit()?),
                Some(b'|') | Some(b')') | None => break,
                Some(_) => return Err(self.err("unexpected token")),
            }
        }
        match units.len() {
            0 => Err(self.err("empty clause")),
            1 => Ok(units.pop().expect("one")),
            _ => Ok(RNode::And(units)),
        }
    }
    fn unit(&mut self) -> Result<RNode, String> {
        if self.peek() == Some(b'(') {
            self.i += 1;
            let inner = self.alternatives()?;
            self.ws();
            if self.peek() != Some(b')') {
                return Err(self.err("unclosed group"));
            }
            self.i += 1;
            return Ok(inner);
        }
        self.i += 1; // '@'
        let start = self.i;
        while self.i < self.s.len() && self.s[self.i] != b':' {
            self.i += 1;
        }
        let field = String::from_utf8_lossy(&self.s[start..self.i]).into_owned();
        self.i += 1; // ':'
        let (open, close) = match self.peek() {
            Some(b'[') => (b'[', b']'),
            Some(b'{') => (b'{', b'}'),
            Some(b'(') => (b'(', b')'),
            _ => return Err(self.err("leaf without a value")),
        };
        self.i += 1;
        let body_start = self.i;
        while self.i < self.s.len() && self.s[self.i] != close {
            self.i += 1;
        }
        if self.i >= self.s.len() {
            return Err(self.err("unclosed leaf"));
        }
        let body = String::from_utf8_lossy(&self.s[body_start..self.i]).into_owned();
        self.i += 1;
        let param = |t: &str| -> Result<String, String> {
            t.strip_prefix('$')
                .filter(|p| !p.is_empty() && !p.contains(' '))
                .map(str::to_string)
                .ok_or_else(|| format!("Redis filter leaf on {field}: {t:?} is not a parameter"))
        };
        match open {
            b'[' => {
                let parts: Vec<&str> = body.split(' ').filter(|t| !t.is_empty()).collect();
                let [lo, hi] = parts[..] else {
                    return Err(format!("Redis range on {field} is not `[lo hi]`: [{body}]"));
                };
                let side = |t: &str, inf: &str| -> Result<Option<(String, bool)>, String> {
                    if t == inf {
                        return Ok(None);
                    }
                    match t.strip_prefix('(') {
                        Some(rest) => Ok(Some((param(rest)?, true))),
                        None => Ok(Some((param(t)?, false))),
                    }
                };
                Ok(RNode::Range {
                    lo: side(lo, "-inf")?,
                    hi: side(hi, "+inf")?,
                    field,
                })
            }
            b'{' => Ok(RNode::Tag {
                params: body
                    .split('|')
                    .map(|t| param(t.trim()))
                    .collect::<Result<_, _>>()?,
                field,
            }),
            _ => Ok(RNode::Text {
                param: param(body.trim())?,
                field,
            }),
        }
    }
}

fn parse_redis_filter(expr: &str) -> Result<RNode, String> {
    let mut p = RParser {
        s: expr.as_bytes(),
        i: 0,
    };
    let node = p.alternatives()?;
    p.ws();
    if p.i != p.s.len() {
        return Err(p.err("trailing input"));
    }
    Ok(node)
}

/// Collapse single-child groups and nested same-kind groups, so `(a)` == `a`
/// and `((a b) c)` == `(a b c)` — both mean the same in RediSearch.
fn flatten(n: RNode) -> RNode {
    match n {
        RNode::And(c) | RNode::Or(c) if c.len() == 1 => flatten(c.into_iter().next().expect("one")),
        RNode::And(c) => RNode::And(
            c.into_iter()
                .map(flatten)
                .flat_map(|k| match k {
                    RNode::And(g) => g,
                    other => vec![other],
                })
                .collect(),
        ),
        RNode::Or(c) => RNode::Or(
            c.into_iter()
                .map(flatten)
                .flat_map(|k| match k {
                    RNode::Or(g) => g,
                    other => vec![other],
                })
                .collect(),
        ),
        leaf => leaf,
    }
}

fn flatten_predicate(p: &Predicate) -> Predicate {
    match p {
        Predicate::And(c) | Predicate::Or(c) if c.len() == 1 => flatten_predicate(&c[0]),
        Predicate::And(c) => Predicate::And(
            c.iter()
                .map(flatten_predicate)
                .flat_map(|k| match k {
                    Predicate::And(g) => g,
                    other => vec![other],
                })
                .collect(),
        ),
        Predicate::Or(c) => Predicate::Or(
            c.iter()
                .map(flatten_predicate)
                .flat_map(|k| match k {
                    Predicate::Or(g) => g,
                    other => vec![other],
                })
                .collect(),
        ),
        leaf => leaf.clone(),
    }
}

/// Redis: the RediSearch expression the builder rendered must be the
/// predicate, structurally — parsed back (AND = juxtaposition, OR = `|`,
/// parenthesised groups) and compared node by node, with every `$param`
/// resolved through the bound `PARAMS`: `and`/`or` must keep their kind and
/// arity; a text leaf must be `@f:($term)`; a keyword match `@f:{$v | ...}`
/// with the same value set; an integer match one `[$v $v]` interval per value;
/// a range one interval with the same bounds, inclusive/exclusive as declared.
/// A dropped leaf, an `or` rendered as an implicit AND, or a wrong bound is an
/// error.
pub fn check_redis_filter(
    p: &Predicate,
    expr: &str,
    params: &std::collections::HashMap<String, RedisParam>,
) -> Result<(), String> {
    let node = flatten(parse_redis_filter(expr)?);
    let pred = flatten_predicate(p);
    let num = |name: &str| -> Result<f64, String> {
        match params.get(name) {
            Some(RedisParam::Num(n)) => Ok(*n),
            other => Err(format!(
                "Redis parameter ${name} is {other:?}, not a number"
            )),
        }
    };
    let text = |name: &str| -> Result<String, String> {
        match params.get(name) {
            Some(RedisParam::Str(s)) => Ok(s.clone()),
            other => Err(format!(
                "Redis parameter ${name} is {other:?}, not a string"
            )),
        }
    };
    fn walk(
        p: &Predicate,
        n: &RNode,
        num: &dyn Fn(&str) -> Result<f64, String>,
        text: &dyn Fn(&str) -> Result<String, String>,
    ) -> Result<(), String> {
        let mismatch = || format!("Redis filter node {n:?} does not express {p:?}");
        match (p, n) {
            (Predicate::And(pc), RNode::And(nc)) | (Predicate::Or(pc), RNode::Or(nc)) => {
                if pc.len() != nc.len() {
                    return Err(mismatch());
                }
                pc.iter()
                    .zip(nc)
                    .try_for_each(|(p, n)| walk(p, n, num, text))
            }
            (Predicate::Leaf { field, leaf }, n) => match (leaf, n) {
                (Leaf::Text(term), RNode::Text { field: f, param }) if f == field => {
                    (text(param)? == *term).then_some(()).ok_or_else(mismatch)
                }
                (Leaf::Range { gt, gte, lt, lte }, RNode::Range { field: f, lo, hi })
                    if f == field =>
                {
                    let want_lo = gt.map(|v| (v, true)).or(gte.map(|v| (v, false)));
                    let want_hi = lt.map(|v| (v, true)).or(lte.map(|v| (v, false)));
                    let got = |s: &Option<(String, bool)>| -> Result<Option<(f64, bool)>, String> {
                        s.as_ref().map(|(p, x)| Ok((num(p)?, *x))).transpose()
                    };
                    (got(lo)? == want_lo && got(hi)? == want_hi)
                        .then_some(())
                        .ok_or_else(mismatch)
                }
                (Leaf::Match(ops), n) => {
                    use crate::msmarco::derived::Operand;
                    let words: Vec<&str> = ops
                        .iter()
                        .filter_map(|o| match o {
                            Operand::Keyword(k) => Some(k.as_str()),
                            Operand::Int(_) => None,
                        })
                        .collect();
                    let ints: Vec<f64> = ops
                        .iter()
                        .filter_map(|o| match o {
                            Operand::Int(i) => Some(*i as f64),
                            Operand::Keyword(_) => None,
                        })
                        .collect();
                    if !words.is_empty() && ints.is_empty() {
                        let RNode::Tag { field: f, params } = n else {
                            return Err(mismatch());
                        };
                        let mut got = params
                            .iter()
                            .map(|p| text(p))
                            .collect::<Result<Vec<_>, _>>()?;
                        let mut want: Vec<String> = words.iter().map(|w| w.to_string()).collect();
                        got.sort_unstable();
                        want.sort_unstable();
                        return (f == field && got == want)
                            .then_some(())
                            .ok_or_else(mismatch);
                    }
                    // Integer IN-list: one point interval per value, OR-ed.
                    let points: Vec<&RNode> = match n {
                        RNode::Or(c) if ints.len() > 1 => c.iter().collect(),
                        leaf if ints.len() == 1 => vec![leaf],
                        _ => return Err(mismatch()),
                    };
                    let mut got = Vec::new();
                    for pt in points {
                        match pt {
                            RNode::Range {
                                field: f,
                                lo: Some((a, false)),
                                hi: Some((b, false)),
                            } if f == field && num(a)? == num(b)? => got.push(num(a)?),
                            _ => return Err(mismatch()),
                        }
                    }
                    let mut want = ints;
                    got.sort_by(f64::total_cmp);
                    want.sort_by(f64::total_cmp);
                    (got == want).then_some(()).ok_or_else(mismatch)
                }
                _ => Err(mismatch()),
            },
            _ => Err(mismatch()),
        }
    }
    walk(&pred, &node, &num, &text)
}

/// Resolve the Redis `HYBRID_POLICY` from the search config's `hybrid_policy`
/// knob and `REDIS_HYBRID_POLICY`. Both set and different is an error, never a
/// silent pick of one. Needs nothing from the server, so the runner calls it
/// at startup, before anything is downloaded or connected.
pub fn resolve_redis_policy(
    config: Option<&Value>,
    env: &str,
) -> Result<Option<&'static str>, String> {
    let from_config = config
        .map(|v| {
            v.as_str()
                .ok_or_else(|| format!("hybrid_policy must be a string, got {v}"))
        })
        .transpose()?;
    match (from_config, env.trim()) {
        (Some(c), e) if !e.is_empty() && !c.eq_ignore_ascii_case(e) => Err(format!(
            "search config sets hybrid_policy {c:?} but REDIS_HYBRID_POLICY is {e:?}; unset \
             one or make them agree"
        )),
        (Some(c), _) => redis_policy(Some(c)),
        (None, e) => redis_policy(Some(e)),
    }
}

/// The `hybrid_policy` search knob is read ONLY by the Redis hybrid arm.
/// Anywhere else — a Redis vector config, an OpenSearch hybrid config — it
/// would be parsed and silently ignored, and the result would carry a knob
/// that never took effect. Refused at startup instead.
pub fn check_policy_knob_allowed(
    engine_type: &str,
    workload: Workload,
    knob_present: bool,
) -> Result<(), String> {
    if knob_present && !(engine_type == "redis" && workload == Workload::Hybrid) {
        return Err(format!(
            "`hybrid_policy` is only read by the Redis hybrid workload; on engine \
             '{engine_type}' with workload \"{workload}\" it would be silently ignored — remove \
             it (Redis vector configs: use REDIS_HYBRID_POLICY)"
        ));
    }
    Ok(())
}

/// Startup check for a dataset a hybrid config will run on: the query set
/// exists, every row is internally consistent, every predicate fits the
/// schema, and no hybrid search config asks for more results (`tops`) than
/// the narrowest predicate matches — recall@k over a ground truth narrower
/// than k would be scored on fewer than k ids. Run after the dataset is on
/// disk and BEFORE any engine is built, so a corpus without
/// `hybrid_queries.jsonl` fails before configure/upload.
pub fn preflight(dir: &Path, schema: Option<&Value>, tops: &[usize]) -> Result<usize, String> {
    let rows = read_rows(&dir.join(HYBRID_QUERIES_FILE))?;
    for r in &rows {
        let p = Predicate::compile(&r.conditions).map_err(|e| format!("{}: {e}", r.query_id))?;
        validate_against_schema(&p, schema).map_err(|e| format!("{}: {e}", r.query_id))?;
    }
    check_bound_to_corpus(dir, &rows)?;
    check_top_against_truth(&rows, tops)?;
    Ok(rows.len())
}

/// The query set must belong to THIS corpus: every row's `corpus_size` equals
/// the row count `vectors.npy` declares, and every row's `test_index` names a
/// line of `tests.jsonl` whose `query_id` equals the row's `source_query_id`
/// (when both carry one). A `hybrid_queries.jsonl` copied next to another
/// variant's corpus — same file names, plausible ids — would otherwise score
/// recall against truth for a different corpus.
pub fn check_bound_to_corpus(dir: &Path, rows: &[HybridQueryRow]) -> Result<(), String> {
    let npy = dir.join("vectors.npy");
    let corpus =
        crate::readers::npy_row_count(npy.to_str().ok_or("vectors.npy path is not UTF-8")?)?;
    if let Some(r) = rows.iter().find(|r| r.corpus_size != corpus) {
        return Err(format!(
            "{}: hybrid_queries.jsonl was built for a {}-row corpus but {} declares {corpus} rows \
             — the query set belongs to another corpus",
            r.query_id,
            r.corpus_size,
            npy.display()
        ));
    }
    let tests = dir.join("tests.jsonl");
    let f = std::fs::File::open(&tests).map_err(|e| format!("open {}: {e}", tests.display()))?;
    let mut ids: Vec<Option<String>> = Vec::new();
    for (i, line) in std::io::BufReader::new(f).lines().enumerate() {
        let line = line.map_err(|e| format!("read {}: {e}", tests.display()))?;
        if line.trim().is_empty() {
            continue;
        }
        // Only the id is needed; the 1024-float vector is skipped unparsed.
        #[derive(serde::Deserialize)]
        struct Id {
            #[serde(default)]
            query_id: Option<Value>,
        }
        let id: Id = serde_json::from_str(&line)
            .map_err(|e| format!("{} line {}: {e}", tests.display(), i + 1))?;
        ids.push(id.query_id.map(|v| match v {
            Value::String(s) => s,
            other => other.to_string(),
        }));
    }
    for r in rows {
        let found = ids.get(r.test_index).ok_or_else(|| {
            format!(
                "{}: test_index {} is outside the {} queries of {}",
                r.query_id,
                r.test_index,
                ids.len(),
                tests.display()
            )
        })?;
        if let (Some(want), Some(got)) = (&r.source_query_id, found) {
            if want != got {
                return Err(format!(
                    "{}: built from query {want:?} but tests.jsonl line {} is query {got:?} — the \
                     query set belongs to another tests.jsonl",
                    r.query_id,
                    r.test_index + 1
                ));
            }
        }
    }
    Ok(())
}

/// Refuse a `top` wider than some query's filtered ground truth.
pub fn check_top_against_truth(rows: &[HybridQueryRow], tops: &[usize]) -> Result<(), String> {
    let Some(narrowest) = rows.iter().min_by_key(|r| r.closest_ids.len()) else {
        return Ok(());
    };
    if let Some(top) = tops
        .iter()
        .copied()
        .find(|t| *t > narrowest.closest_ids.len())
    {
        return Err(format!(
            "top {top} exceeds the filtered ground truth of {} ({} ids, {} matching rows); \
             recall@{top} there would be scored over fewer than {top} ids — use a top <= {}",
            narrowest.query_id,
            narrowest.closest_ids.len(),
            narrowest.matching_count,
            narrowest.closest_ids.len()
        ));
    }
    Ok(())
}

/// Rows of `<dir>/hybrid_queries.jsonl` whose predicate filters on any of
/// `text_fields` (any condition type). Used by the fulltext mix rule
/// (`fulltext::check_mixed_text_filters`): a fulltext search config pins the
/// index's text analyzer, which would change what a hybrid text predicate
/// matches — and the text-match agreement rules (`msmarco::text_match`) hold
/// only under each engine's default analysis. A dataset without the file has
/// no hybrid predicates.
pub fn count_text_field_rows(dir: &Path, text_fields: &[String]) -> Result<usize, String> {
    let path = dir.join(HYBRID_QUERIES_FILE);
    if !path.is_file() || text_fields.is_empty() {
        return Ok(0);
    }
    Ok(read_rows(&path)?
        .iter()
        .filter(|r| {
            crate::fulltext::filtered_fields(&r.conditions)
                .iter()
                .any(|f| text_fields.contains(f))
        })
        .count())
}

/// Queries profiled per predicate after each bucket's pass (Redis
/// `FT.PROFILE`). A fixed sample, not the whole bucket: profiling replays
/// queries serially, and a brute-force bucket on a large corpus would
/// otherwise spend tens of minutes there. Each bucket has 3 predicates.
pub const PROFILE_SAMPLE_PER_PREDICATE: usize = 4;

/// Positions (into `positions`) of the profiled sample: the first
/// `per_predicate` queries of each predicate, in pass order.
pub fn profile_sample(predicate_ids: &[&str], per_predicate: usize) -> Vec<usize> {
    let mut seen: BTreeMap<&str, usize> = BTreeMap::new();
    let mut out = Vec::new();
    for (i, p) in predicate_ids.iter().enumerate() {
        let n = seen.entry(p).or_insert(0);
        if *n < per_predicate {
            *n += 1;
            out.push(i);
        }
    }
    out
}

/// OpenSearch: the filter must be exactly the predicate's structure — `and`
/// as `bool.must`, `or` as `bool.should` with `minimum_should_match: 1`, and
/// each leaf as the one query the builder documents (`match` for value and
/// text, `terms` for any, `range` with the same bounds).
pub fn check_opensearch_filter(conditions: &Value, filter: &Value) -> Result<(), String> {
    fn node(c: &Value, f: &Value) -> Result<(), String> {
        let c_obj = c.as_object().ok_or("condition is not an object")?;
        let (key, body) = c_obj.iter().next().ok_or("empty condition")?;
        match key.as_str() {
            "and" | "or" => {
                let kids = body.as_array().ok_or("group is not an array")?;
                let b = f
                    .get("bool")
                    .and_then(Value::as_object)
                    .ok_or_else(|| format!("expected a bool query for `{key}`, got {f}"))?;
                let (slot, other) = if key == "and" {
                    ("must", "should")
                } else {
                    ("should", "must")
                };
                if b.contains_key(other) {
                    return Err(format!(
                        "`{key}` group rendered with a `{other}` clause: {f}"
                    ));
                }
                if key == "or" && b.get("minimum_should_match") != Some(&Value::from(1)) {
                    return Err(format!(
                        "`or` without minimum_should_match 1 matches all: {f}"
                    ));
                }
                let fk = b
                    .get(slot)
                    .and_then(Value::as_array)
                    .ok_or_else(|| format!("`{key}` group has no `{slot}` array: {f}"))?;
                if fk.len() != kids.len() {
                    return Err(format!(
                        "`{key}` group has {} entries but the filter has {} — a leaf was \
                         dropped: {f}",
                        kids.len(),
                        fk.len()
                    ));
                }
                kids.iter().zip(fk).try_for_each(|(c, f)| node(c, f))
            }
            field => {
                let (op, crit) = body
                    .as_object()
                    .and_then(|o| o.iter().next())
                    .ok_or("leaf has no operation")?;
                let expected = match op.as_str() {
                    "match" => {
                        if let Some(t) = crit.get("text") {
                            serde_json::json!({"match": {field: t}})
                        } else if let Some(any) = crit.get("any") {
                            serde_json::json!({"terms": {field: any}})
                        } else if let Some(v) = crit.get("value") {
                            serde_json::json!({"match": {field: v}})
                        } else {
                            return Err(format!("{field}: unsupported match {crit}"));
                        }
                    }
                    "range" => serde_json::json!({"range": {field: crit}}),
                    other => return Err(format!("{field}: unsupported operation {other}")),
                };
                if *f != expected {
                    return Err(format!(
                        "leaf {c} rendered as {f}, expected {expected} — the filter does not \
                         express the predicate"
                    ));
                }
                Ok(())
            }
        }
    }
    node(conditions, filter)
}

// ---------------------------------------------------------------------------
// Engine evidence
// ---------------------------------------------------------------------------

/// The `Vector search mode` values in a flattened `FT.PROFILE` reply, in
/// order (one per vector iterator).
pub fn redis_vector_search_modes(profile: &[String]) -> Vec<String> {
    profile
        .windows(2)
        .filter(|w| w[0] == "Vector search mode")
        .map(|w| w[1].clone())
        .collect()
}

/// The vector search mode of every shard in ONE `FT.PROFILE` reply.
///
/// A standalone server reports one profile; through a cluster coordinator
/// (OSS cluster, Redis Software / Cloud multi-shard) the reply carries one
/// `Shard ID …` profile per shard, each with its own `Vector search mode`
/// (verified on a 3-shard Redis 8.8 OSS cluster). Every shard must report
/// exactly one mode: `max(1, number of "Shard ID" entries)` modes, else the
/// reply is not one we understand and the check refuses it.
pub fn redis_profile_modes(profile: &[String]) -> Result<Vec<String>, String> {
    let modes = redis_vector_search_modes(profile);
    let shards = profile.iter().filter(|s| *s == "Shard ID").count().max(1);
    if modes.len() != shards {
        return Err(format!(
            "FT.PROFILE reply reports {} vector search mode(s) for {shards} shard(s); every \
             shard's profile must report exactly one",
            modes.len()
        ));
    }
    Ok(modes)
}

/// The explicit `HYBRID_POLICY` values the workload accepts.
pub const REDIS_POLICIES: [&str; 2] = ["ADHOC_BF", "BATCHES"];

/// Validate a configured policy (`None`/empty = let Redis choose).
pub fn redis_policy(configured: Option<&str>) -> Result<Option<&'static str>, String> {
    match configured.map(str::trim).filter(|s| !s.is_empty()) {
        None => Ok(None),
        Some(p) => REDIS_POLICIES
            .iter()
            .find(|k| k.eq_ignore_ascii_case(p))
            .copied()
            .map(Some)
            .ok_or_else(|| format!("HYBRID_POLICY {p:?} is not one of {REDIS_POLICIES:?}")),
    }
}

/// Every shard of every profiled query must have run the configured policy:
/// `ADHOC_BF` -> `HYBRID_ADHOC_BF`; `BATCHES` -> `HYBRID_BATCHES` (or its
/// documented in-query switch `HYBRID_BATCHES_TO_ADHOC_BF`). Unconfigured:
/// any hybrid mode, but each shard must report one — a shard that ran without
/// a hybrid iterator did not filter. `modes` tallies shard-level modes over
/// `profiled` queries on a `shards`-shard database, so it must sum to
/// exactly `profiled * shards`.
pub fn check_redis_modes(
    policy: Option<&str>,
    modes: &BTreeMap<String, u64>,
    profiled: u64,
    shards: u64,
) -> Result<(), String> {
    let total: u64 = modes.values().sum();
    if shards == 0 || total != profiled * shards {
        return Err(format!(
            "FT.PROFILE reported {total} shard-level vector search modes for {profiled} hybrid \
             queries on {shards} shard(s)"
        ));
    }
    for mode in modes.keys() {
        let ok = match policy {
            Some("ADHOC_BF") => mode == "HYBRID_ADHOC_BF",
            Some("BATCHES") => mode == "HYBRID_BATCHES" || mode == "HYBRID_BATCHES_TO_ADHOC_BF",
            _ => mode.starts_with("HYBRID_"),
        };
        if !ok {
            return Err(format!(
                "a hybrid query ran in vector search mode {mode:?} under HYBRID_POLICY {}",
                policy.unwrap_or("(unset)")
            ));
        }
    }
    Ok(())
}

/// Sum over nodes of `knn_query_with_filter_requests` in a
/// `GET /_plugins/_knn/stats` reply. A node without the counter is an error:
/// the evidence would silently read as zero.
pub fn opensearch_filter_requests(stats: &Value) -> Result<u64, String> {
    let nodes = stats
        .get("nodes")
        .and_then(Value::as_object)
        .filter(|n| !n.is_empty())
        .ok_or("k-NN stats reply has no nodes")?;
    nodes.iter().try_fold(0u64, |acc, (id, n)| {
        n.get("knn_query_with_filter_requests")
            .and_then(Value::as_u64)
            .map(|v| acc + v)
            .ok_or_else(|| format!("k-NN stats of node {id} has no knn_query_with_filter_requests"))
    })
}

/// OpenSearch: the k-NN plugin's `knn_query_with_filter_requests` counter
/// must have grown by at least one per ANSWERED query (`requested - failed`)
/// — i.e. the filter reached the k-NN query (efficient filtering), rather
/// than being dropped or applied as a post-filter outside it. A failed query
/// (rejected, or out of retries) may never have reached the plugin, so it is
/// not counted.
pub fn check_opensearch_filtered_requests(
    delta: u64,
    requested: u64,
    failed: u64,
) -> Result<(), String> {
    let answered = requested.saturating_sub(failed);
    if delta < answered {
        return Err(format!(
            "OpenSearch counted {delta} filtered k-NN requests for {answered} answered hybrid \
             queries ({requested} requested, {failed} failed and not counted) — the filter did \
             not reach the k-NN query"
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

/// Linear-interpolation percentile over a sorted slice (numpy's default).
fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let pos = q * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)
}

/// One bucket's measured pass.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct BucketReport {
    pub bucket: Bucket,
    /// Queries dispatched in this bucket's pass.
    pub requested_queries: usize,
    pub failed_queries: usize,
    pub mean_recall: f64,
    pub recall_p10: f64,
    /// Answered queries / the pass's wall-clock time.
    pub rps: f64,
    pub total_time: f64,
    pub mean_time: f64,
    pub p50_time: f64,
    pub p90_time: f64,
    pub p99_time: f64,
    pub min_selectivity: f64,
    pub max_selectivity: f64,
    /// Predicate ids exercised in this bucket.
    pub predicates: Vec<String>,
    /// Engine evidence for the pass (Redis: vector search modes; OpenSearch:
    /// filtered k-NN request counter).
    pub engine: Value,
}

/// Per-bucket statistics from one pass's successful-query samples.
#[allow(clippy::too_many_arguments)]
pub fn bucket_report(
    bucket: Bucket,
    latencies: &[f64],
    recalls: &[f64],
    total_time: f64,
    requested: usize,
    meta: &[&HybridQueryMeta],
    engine: Value,
) -> Result<BucketReport, String> {
    if latencies.len() != recalls.len() {
        return Err(format!(
            "bucket {bucket}: {} latencies but {} recalls",
            latencies.len(),
            recalls.len()
        ));
    }
    if latencies.is_empty() {
        return Err(format!("bucket {bucket}: no query succeeded"));
    }
    let mut lat = latencies.to_vec();
    lat.sort_by(f64::total_cmp);
    let mut rec = recalls.to_vec();
    rec.sort_by(f64::total_cmp);
    let n = lat.len() as f64;
    let mut predicates: Vec<String> = meta.iter().map(|m| m.predicate_id.clone()).collect();
    predicates.sort();
    predicates.dedup();
    let sel = meta.iter().map(|m| m.selectivity);
    Ok(BucketReport {
        bucket,
        requested_queries: requested,
        failed_queries: requested.saturating_sub(lat.len()),
        mean_recall: rec.iter().sum::<f64>() / n,
        recall_p10: percentile(&rec, 0.10),
        rps: if total_time > 0.0 {
            n / total_time
        } else {
            0.0
        },
        total_time,
        mean_time: lat.iter().sum::<f64>() / n,
        p50_time: percentile(&lat, 0.50),
        p90_time: percentile(&lat, 0.90),
        p99_time: percentile(&lat, 0.99),
        min_selectivity: sel.clone().fold(f64::INFINITY, f64::min),
        max_selectivity: sel.fold(f64::NEG_INFINITY, f64::max),
        predicates,
        engine,
    })
}

/// What a hybrid result publishes under `results.hybrid`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct HybridReport {
    pub query_file: &'static str,
    pub top: usize,
    pub queries: usize,
    /// How text predicates were sent (Redis `VERBATIM`, OpenSearch `match`)
    /// and the agreement rule they were generated under.
    pub text_match: Value,
    pub buckets: Vec<BucketReport>,
    /// Engine-wide notes (e.g. which policy was configured).
    pub engine: Value,
}

/// Refuse a result that does not carry the hybrid fingerprint, or carries it
/// under another workload's name.
pub fn verify_fingerprint(workload: Workload, has_report: bool) -> Result<(), String> {
    match (workload, has_report) {
        (Workload::Hybrid, false) => Err(
            "workload hybrid returned a result without a per-bucket hybrid report — the engine \
             did not run its hybrid path"
                .into(),
        ),
        (w, true) if w != Workload::Hybrid => Err(format!(
            "workload \"{w}\" returned a hybrid report — the engine ran its hybrid path"
        )),
        _ => Ok(()),
    }
}

/// Map one pass's captured records (query ids = positions within
/// `positions`, seq = dispatch ordinal within the pass) onto the whole query
/// set: query ids become `positions[id]`, seqs are offset by `seq_offset`.
pub fn remap_records(
    records: Vec<crate::results_dump::RawQueryResult>,
    positions: &[usize],
    seq_offset: usize,
) -> Result<Vec<crate::results_dump::RawQueryResult>, String> {
    records
        .into_iter()
        .map(|mut r| {
            r.query_id = *positions.get(r.query_id).ok_or_else(|| {
                format!(
                    "captured query {} is outside the {}-query pass",
                    r.query_id,
                    positions.len()
                )
            })?;
            r.seq += seq_offset;
            Ok(r)
        })
        .collect()
}

/// The search config of a hybrid run must pin `top` (a ground truth up to
/// 1000 wide would otherwise set k), and may not use the open-loop or
/// duration modes (their per-bucket accounting is not defined). `--queries N`
/// is refused too: each bucket is its own pass, so N would silently mean N
/// PER BUCKET and the run's `requested_queries` would no longer match the
/// flag. A hybrid run always runs the whole query set.
pub fn check_params(
    top: Option<i64>,
    target_qps: Option<f64>,
    duration: Option<f64>,
    num_queries: i64,
) -> Result<usize, String> {
    if num_queries > 0 {
        return Err(format!(
            "--queries {num_queries} is not supported by the hybrid workload: every bucket is \
             its own timed pass over the whole query set, so a cap would silently apply per \
             bucket; drop --queries for hybrid configs"
        ));
    }
    if target_qps.is_some() || duration.is_some() {
        return Err(
            "the hybrid workload runs one closed-loop pass per selectivity bucket; target_qps / \
             duration_seconds are not supported"
                .into(),
        );
    }
    match top {
        Some(t) if t > 0 => Ok(t as usize),
        _ => Err(
            "the hybrid workload needs an explicit positive `top`: its ground truth is up to \
             1000 wide, and deriving k from it would run every query at that width"
                .into(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn buckets_split_at_quarter_of_each_decade() {
        let b = |m, c| Bucket::of(m, c).unwrap();
        assert_eq!(b(50, 100), Bucket::S50);
        assert_eq!(b(25, 100), Bucket::S50);
        assert_eq!(b(249, 1000), Bucket::S10);
        assert_eq!(b(25, 1000), Bucket::S10);
        assert_eq!(b(249, 10_000), Bucket::S1);
        assert_eq!(b(25, 10_000), Bucket::S1);
        assert_eq!(b(249, 100_000), Bucket::S01);
        assert_eq!(b(1, 1_000_000), Bucket::S01);
        for bad in [(0, 10), (11, 10), (1, 0)] {
            assert!(Bucket::of(bad.0, bad.1).is_err(), "{bad:?}");
        }
        for x in Bucket::ALL {
            assert_eq!(serde_json::to_value(x).unwrap(), json!(x.as_str()));
            assert_eq!(Bucket::of((x.target() * 1e6) as u64, 1_000_000).unwrap(), x);
        }
    }

    fn row() -> HybridQueryRow {
        HybridQueryRow {
            query_id: "q0".into(),
            test_index: 0,
            source_query_id: None,
            predicate_id: "p0".into(),
            predicate_shape: "numeric_range".into(),
            bucket: Bucket::S1,
            conditions: json!({"and":[{"tier":{"range":{"lt":1}}}]}),
            matching_count: 12,
            corpus_size: 1000,
            selectivity: 0.012,
            gt_depth: 5,
            closest_ids: vec![3, 1, 2, 9, 0],
            closest_scores: vec![0.9, 0.8, 0.8, 0.5, 0.1],
        }
    }

    #[test]
    fn a_row_must_be_internally_consistent() {
        assert!(row().check().is_ok());
        let mut bad = vec![];
        let mut r = row();
        r.bucket = Bucket::S10;
        bad.push(r);
        let mut r = row();
        r.selectivity = 0.5;
        bad.push(r);
        let mut r = row();
        r.matching_count = 9;
        r.selectivity = 0.009;
        bad.push(r);
        let mut r = row();
        r.closest_ids.pop();
        bad.push(r);
        let mut r = row();
        r.closest_ids[1] = 3;
        bad.push(r);
        let mut r = row();
        r.closest_ids[1] = 1000;
        bad.push(r);
        let mut r = row();
        r.closest_scores[4] = 0.95;
        bad.push(r);
        let mut r = row();
        r.conditions = json!({"tier":{"range":{"lt":1}}});
        bad.push(r);
        for r in bad {
            assert!(r.check().is_err(), "{r:?}");
        }
        // Fewer matches than the depth: the truth is exactly that wide.
        let mut r = row();
        r.gt_depth = 1000;
        assert!(r.check().is_err());
        r.matching_count = 10;
        r.selectivity = 0.01;
        r.closest_ids = (0..10).collect();
        r.closest_scores = vec![0.5; 10];
        assert!(r.check().is_ok());
    }

    #[test]
    fn rows_reject_unknown_keys() {
        let mut v = serde_json::to_value(row()).unwrap();
        v["shape"] = json!("x");
        assert!(serde_json::from_value::<HybridQueryRow>(v).is_err());
    }

    #[test]
    fn schema_types_bound_each_leaf() {
        let schema = json!({"title":"text","category":"keyword","tier":"int"});
        let ok = |c: Value| {
            validate_against_schema(&Predicate::compile(&c).unwrap(), Some(&schema)).is_ok()
        };
        assert!(ok(json!({"and":[{"title":{"match":{"text":"quora"}}}]})));
        assert!(ok(
            json!({"or":[{"category":{"match":{"any":["cat00","cat01"]}}},{"tier":{"range":{"lte":4}}}]})
        ));
        assert!(!ok(
            json!({"and":[{"category":{"match":{"text":"quora"}}}]})
        ));
        assert!(!ok(json!({"and":[{"title":{"match":{"value":"quora"}}}]})));
        assert!(!ok(json!({"and":[{"title":{"range":{"lt":3}}}]})));
        assert!(!ok(json!({"and":[{"tier":{"match":{"value":"3"}}}]})));
        assert!(!ok(json!({"and":[{"missing":{"match":{"value":3}}}]})));
        let p = Predicate::compile(&json!({"and":[{"tier":{"range":{"lt":3}}}]})).unwrap();
        assert!(validate_against_schema(&p, None).is_err());
    }

    fn params(kv: &[(&str, RedisParam)]) -> std::collections::HashMap<String, RedisParam> {
        kv.iter().map(|(k, v)| (k.to_string(), v.clone())).collect()
    }

    #[test]
    fn redis_filter_must_be_the_predicate_structurally() {
        use RedisParam::{Num, Str};
        let p = Predicate::compile(&json!({"or":[
            {"rand_key":{"range":{"lt":10}}},
            {"and":[{"rand_key":{"range":{"gte":90}}},{"category":{"match":{"any":["cat00","cat01"]}}}]},
            {"title":{"match":{"text":"quora"}}}
        ]}))
        .unwrap();
        let ps = params(&[
            ("a", Num(10.0)),
            ("b", Num(90.0)),
            ("c", Str("quora".into())),
            ("t0", Str("cat00".into())),
            ("t1", Str("cat01".into())),
        ]);
        let good =
            "((@rand_key:[-inf ($a] | (@rand_key:[$b +inf] @category:{$t1 | $t0}) | @title:($c)))";
        check_redis_filter(&p, good, &ps).unwrap();
        for (bad, why) in [
            ("(@rand_key:[-inf ($a] | @title:($c))", "dropped leaf"),
            ("(@rand_key:[-inf ($a] (@rand_key:[$b +inf] @category:{$t1 | $t0}) @title:($c))", "or rendered as and"),
            ("(@rand_key:[-inf $a] | (@rand_key:[$b +inf] @category:{$t1 | $t0}) | @title:($c))", "inclusive upper bound"),
            ("(@rand_key:[-inf ($b] | (@rand_key:[$b +inf] @category:{$t1 | $t0}) | @title:($c))", "wrong bound value"),
            ("(@rand_key:[-inf ($a] | (@rand_key:[$b +inf] @category:{$t1}) | @title:($c))", "tag value dropped"),
            ("(@rand_key:[-inf ($a] | (@rand_key:[$b +inf] | @category:{$t1 | $t0}) | @title:($c))", "inner and as or"),
            ("(@rand_key:[-inf ($a] | (@rand_key:[$b +inf] @category:{$t1 | $t0}) | @title:($t0))", "wrong text"),
            ("*", "match all"),
            ("(@rand_key:[-inf ($a] | -@title:($c))", "negation"),
        ] {
            assert!(check_redis_filter(&p, bad, &ps).is_err(), "{why}: {bad}");
        }
        // hybrid-p02's shape: `title:RARE OR tier<=4` (~50%) rendered as an
        // implicit AND (~0.05%) keeps the arity — only the group kind differs.
        let p = Predicate::compile(&json!({"or":[
            {"title":{"match":{"text":"ancient"}}},{"tier":{"range":{"lte":4}}}
        ]}))
        .unwrap();
        let ps = params(&[("c", Str("ancient".into())), ("d", Num(4.0))]);
        check_redis_filter(&p, "(@title:($c) | @tier:[-inf $d])", &ps).unwrap();
        assert!(check_redis_filter(&p, "(@title:($c) @tier:[-inf $d])", &ps).is_err());
        // Integer IN-list: one point interval per value.
        let p = Predicate::compile(&json!({"and":[{"tier":{"match":{"any":[1,3]}}}]})).unwrap();
        let ps = params(&[("x", Num(1.0)), ("y", Num(3.0))]);
        check_redis_filter(&p, "((@tier:[$x $x] | @tier:[$y $y]))", &ps).unwrap();
        assert!(check_redis_filter(&p, "((@tier:[$x $x] @tier:[$y $y]))", &ps).is_err());
        assert!(check_redis_filter(&p, "((@tier:[$x $y]))", &ps).is_err());
        assert!(has_text(
            &Predicate::compile(&json!({"and":[{"title":{"match":{"text":"quora"}}}]})).unwrap()
        ));
    }

    #[test]
    fn profile_sample_takes_a_fixed_number_per_predicate() {
        let ids = ["a", "b", "c", "a", "b", "c", "a", "b", "c", "a"];
        assert_eq!(profile_sample(&ids, 2), vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(profile_sample(&ids, 4), (0..10).collect::<Vec<_>>());
        assert_eq!(profile_sample(&ids[..1], 4), vec![0]);
        let big: Vec<&str> = (0..420).map(|i| ["p0", "p1", "p2"][i % 3]).collect();
        assert_eq!(profile_sample(&big, PROFILE_SAMPLE_PER_PREDICATE).len(), 12);
    }

    #[test]
    fn top_wider_than_the_narrowest_truth_is_refused() {
        let mut narrow = row();
        narrow.query_id = "narrow".into();
        let rows = vec![row(), narrow];
        // row() holds 5 truth ids.
        assert!(check_top_against_truth(&rows, &[5]).is_ok());
        let e = check_top_against_truth(&rows, &[5, 6]).unwrap_err();
        assert!(e.contains("top 6"), "{e}");
        assert!(check_top_against_truth(&[], &[1000]).is_ok());
    }

    #[test]
    fn the_policy_knob_is_refused_where_nothing_reads_it() {
        assert!(check_policy_knob_allowed("redis", Workload::Hybrid, true).is_ok());
        for (engine, w) in [
            ("redis", Workload::Vector),
            ("redis", Workload::FilterOnly),
            ("redis", Workload::Fulltext),
            ("opensearch", Workload::Hybrid),
            ("opensearch", Workload::Vector),
            ("valkey", Workload::Vector),
        ] {
            let e = check_policy_knob_allowed(engine, w, true).unwrap_err();
            assert!(e.contains("silently ignored"), "{engine} {w}: {e}");
            assert!(check_policy_knob_allowed(engine, w, false).is_ok());
        }
    }

    #[test]
    fn redis_policy_conflicts_are_errors() {
        assert_eq!(resolve_redis_policy(None, "").unwrap(), None);
        assert_eq!(
            resolve_redis_policy(None, "batches").unwrap(),
            Some("BATCHES")
        );
        assert_eq!(
            resolve_redis_policy(Some(&json!("ADHOC_BF")), "").unwrap(),
            Some("ADHOC_BF")
        );
        assert_eq!(
            resolve_redis_policy(Some(&json!("ADHOC_BF")), "adhoc_bf").unwrap(),
            Some("ADHOC_BF")
        );
        let e = resolve_redis_policy(Some(&json!("ADHOC_BF")), "BATCHES").unwrap_err();
        assert!(e.contains("REDIS_HYBRID_POLICY"), "{e}");
        assert!(resolve_redis_policy(Some(&json!(3)), "").is_err());
        assert!(resolve_redis_policy(Some(&json!("NONE")), "").is_err());
    }

    #[test]
    fn opensearch_filter_must_mirror_the_predicate() {
        let c = json!({"or":[
            {"title":{"match":{"text":"quora"}}},
            {"and":[{"category":{"match":{"any":["cat00"]}}},{"tier":{"range":{"gte":3,"lte":3}}}]}
        ]});
        let good = json!({"bool":{"should":[
            {"match":{"title":"quora"}},
            {"bool":{"must":[{"terms":{"category":["cat00"]}},{"range":{"tier":{"gte":3,"lte":3}}}]}}
        ],"minimum_should_match":1}});
        assert!(check_opensearch_filter(&c, &good).is_ok());
        let mut dropped = good.clone();
        dropped["bool"]["should"][1]["bool"]["must"]
            .as_array_mut()
            .unwrap()
            .pop();
        assert!(check_opensearch_filter(&c, &dropped).is_err());
        let mut widened = good.clone();
        widened["bool"]
            .as_object_mut()
            .unwrap()
            .remove("minimum_should_match");
        assert!(check_opensearch_filter(&c, &widened).is_err());
        let mut changed = good.clone();
        changed["bool"]["should"][1]["bool"]["must"][1]["range"]["tier"]["lte"] = json!(4);
        assert!(check_opensearch_filter(&c, &changed).is_err());
        let as_and = json!({"bool":{"must":good["bool"]["should"].clone()}});
        assert!(check_opensearch_filter(&c, &as_and).is_err());
    }

    #[test]
    fn redis_modes_must_match_the_policy() {
        let modes = |m: &[(&str, u64)]| {
            m.iter()
                .map(|(k, v)| (k.to_string(), *v))
                .collect::<BTreeMap<_, _>>()
        };
        let adhoc = modes(&[("HYBRID_ADHOC_BF", 4)]);
        let batches = modes(&[("HYBRID_BATCHES", 3), ("HYBRID_BATCHES_TO_ADHOC_BF", 1)]);
        assert!(check_redis_modes(Some("ADHOC_BF"), &adhoc, 4, 1).is_ok());
        assert!(check_redis_modes(Some("ADHOC_BF"), &batches, 4, 1).is_err());
        assert!(check_redis_modes(Some("BATCHES"), &batches, 4, 1).is_ok());
        assert!(check_redis_modes(Some("BATCHES"), &adhoc, 4, 1).is_err());
        assert!(check_redis_modes(None, &batches, 4, 1).is_ok());
        assert!(check_redis_modes(None, &modes(&[("STANDARD_KNN", 4)]), 4, 1).is_err());
        // A query whose profile carried no mode is a failure, not a skip.
        assert!(check_redis_modes(None, &adhoc, 5, 1).is_err());
        // Three shards: 4 queries x 3 shard-level modes.
        let sharded = modes(&[("HYBRID_ADHOC_BF", 12)]);
        assert!(check_redis_modes(Some("ADHOC_BF"), &sharded, 4, 3).is_ok());
        assert!(check_redis_modes(Some("ADHOC_BF"), &sharded, 4, 1).is_err());
        assert!(check_redis_modes(Some("ADHOC_BF"), &adhoc, 4, 3).is_err());
        assert!(check_redis_modes(Some("ADHOC_BF"), &adhoc, 4, 0).is_err());
        // One shard of three ran another mode under an explicit policy.
        let mixed = modes(&[("HYBRID_ADHOC_BF", 11), ("HYBRID_BATCHES", 1)]);
        assert!(check_redis_modes(Some("ADHOC_BF"), &mixed, 4, 3).is_err());
        assert_eq!(redis_policy(Some("adhoc_bf")).unwrap(), Some("ADHOC_BF"));
        assert_eq!(redis_policy(Some(" ")).unwrap(), None);
        assert!(redis_policy(Some("NONE")).is_err());
        let flat: Vec<String> = ["Type", "VECTOR", "Vector search mode", "HYBRID_BATCHES"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(redis_vector_search_modes(&flat), vec!["HYBRID_BATCHES"]);
    }

    #[test]
    fn opensearch_stats_counter_is_summed_and_required() {
        let stats = json!({"nodes": {
            "a": {"knn_query_with_filter_requests": 3},
            "b": {"knn_query_with_filter_requests": 4}
        }});
        assert_eq!(opensearch_filter_requests(&stats).unwrap(), 7);
        assert!(opensearch_filter_requests(&json!({"nodes": {}})).is_err());
        assert!(opensearch_filter_requests(&json!({"nodes": {"a": {}}})).is_err());
    }

    #[test]
    fn opensearch_filter_counter_must_cover_every_answered_query() {
        assert!(check_opensearch_filtered_requests(10, 10, 0).is_ok());
        assert!(check_opensearch_filtered_requests(12, 10, 0).is_ok());
        assert!(check_opensearch_filtered_requests(9, 10, 0).is_err());
        // Two queries failed (e.g. out of retries): 8 answered, 8 counted.
        assert!(check_opensearch_filtered_requests(8, 10, 2).is_ok());
        let e = check_opensearch_filtered_requests(7, 10, 2).unwrap_err();
        assert!(e.contains("8 answered") && e.contains("2 failed"), "{e}");
    }

    /// Flattened FT.PROFILE replies captured from Redis 8.8: a standalone
    /// server (one profile) and a 3-shard OSS cluster through its coordinator
    /// (one `Shard ID` profile per shard). Trimmed to the keys that matter.
    fn profile_fixture(shard_modes: &[&str]) -> Vec<String> {
        let mut out: Vec<String> = ["2", "d:{13}x13", "d:{23}x23", "Shards"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let profile = |mode: &str| -> Vec<String> {
            [
                "Total profile time",
                "0.36",
                "Warning",
                "None",
                "Iterators profile",
                "Type",
                "VECTOR",
                "Time",
                "0.0099",
                "Number of reading operations",
                "2",
                "Vector search mode",
                mode,
                "Child iterator",
                "Type",
                "NUMERIC",
                "Term",
                "0 - 9",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect()
        };
        if let [only] = shard_modes {
            out.extend(profile(only));
        } else {
            for (i, m) in shard_modes.iter().enumerate() {
                out.push("Shard ID".into());
                out.push(format!("{i:040x}"));
                out.extend(profile(m));
            }
            out.extend(["Coordinator", "Total Coordinator time", "1.04"].map(String::from));
        }
        out
    }

    #[test]
    fn profile_modes_are_read_per_shard() {
        let one = profile_fixture(&["HYBRID_ADHOC_BF"]);
        assert_eq!(redis_profile_modes(&one).unwrap(), vec!["HYBRID_ADHOC_BF"]);
        let three = profile_fixture(&["HYBRID_BATCHES", "HYBRID_BATCHES", "HYBRID_ADHOC_BF"]);
        assert_eq!(
            redis_profile_modes(&three).unwrap(),
            vec!["HYBRID_BATCHES", "HYBRID_BATCHES", "HYBRID_ADHOC_BF"]
        );
        // A shard profile without a mode, or a reply with no mode at all.
        let mut missing = three.clone();
        let pos = missing
            .iter()
            .rposition(|s| s == "Vector search mode")
            .unwrap();
        missing.drain(pos..pos + 2);
        assert!(redis_profile_modes(&missing).is_err());
        let mut none = one.clone();
        none.retain(|s| s != "Vector search mode" && s != "HYBRID_ADHOC_BF");
        assert!(redis_profile_modes(&none).is_err());
    }

    #[test]
    fn bucket_report_computes_measured_stats() {
        let m = HybridQueryMeta {
            query_id: "q".into(),
            predicate_id: "p1".into(),
            predicate_shape: "text".into(),
            bucket: Bucket::S1,
            matching_count: 10,
            selectivity: 0.01,
        };
        let lat: Vec<f64> = (1..=100).map(|i| i as f64 / 1000.0).collect();
        let rec = vec![1.0; 100];
        let r = bucket_report(Bucket::S1, &lat, &rec, 2.0, 101, &[&m, &m], json!({})).unwrap();
        assert_eq!(r.failed_queries, 1);
        assert!((r.rps - 50.0).abs() < 1e-9);
        assert!((r.p50_time - 0.0505).abs() < 1e-9);
        assert!((r.p90_time - 0.0901).abs() < 1e-9);
        assert!((r.p99_time - 0.09901).abs() < 1e-9);
        assert_eq!(r.predicates, vec!["p1"]);
        assert_eq!(r.mean_recall, 1.0);
        assert!(bucket_report(Bucket::S1, &[], &[], 1.0, 1, &[], json!({})).is_err());
        assert!(bucket_report(Bucket::S1, &[0.1], &[], 1.0, 1, &[], json!({})).is_err());
    }

    #[test]
    fn records_are_remapped_to_query_set_positions() {
        use crate::results_dump::RawQueryResult;
        let raw = vec![
            RawQueryResult::ok(1, 0, vec![], None, 0.1),
            RawQueryResult::failed(0, 1, 0.1, "x"),
        ];
        let out = remap_records(raw, &[7, 9], 100).unwrap();
        assert_eq!((out[0].query_id, out[0].seq), (9, 100));
        assert_eq!((out[1].query_id, out[1].seq), (7, 101));
        let bad = vec![RawQueryResult::ok(2, 0, vec![], None, 0.1)];
        assert!(remap_records(bad, &[7, 9], 0).is_err());
    }

    #[test]
    fn fingerprint_and_params() {
        assert!(verify_fingerprint(Workload::Hybrid, true).is_ok());
        assert!(verify_fingerprint(Workload::Hybrid, false).is_err());
        assert!(verify_fingerprint(Workload::Vector, true).is_err());
        assert!(verify_fingerprint(Workload::Vector, false).is_ok());
        assert_eq!(check_params(Some(10), None, None, -1).unwrap(), 10);
        assert_eq!(check_params(Some(10), None, None, 0).unwrap(), 10);
        assert!(check_params(None, None, None, -1).is_err());
        assert!(check_params(Some(0), None, None, -1).is_err());
        assert!(check_params(Some(10), Some(5.0), None, -1).is_err());
        assert!(check_params(Some(10), None, Some(5.0), -1).is_err());
        let e = check_params(Some(10), None, None, 100).unwrap_err();
        assert!(e.contains("--queries"), "{e}");
    }
}
