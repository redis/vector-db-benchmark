//! The MS MARCO hybrid (prefilter + KNN) query set and its brute-forced
//! FILTERED ground truth (#341). Written by `prepare-msmarco` next to the
//! corpus as `hybrid_queries.jsonl`; the file format and the workload's
//! contract live in [`crate::hybrid`].
//!
//! # What is generated
//!
//! Twelve predicates, three per selectivity bucket (50% / 10% / 1% / 0.1%),
//! over the derived exact-match fields of [`super::derived`] and full-text
//! matches on `title` ([`super::text_match`]):
//!
//! | bucket | predicates |
//! |---|---|
//! | 50%  | `rand_key` range; `category` in {cat00..cat03}; `title` has RARE OR `tier` <= 4 |
//! | 10%  | `rand_key` range; `category` = cat01; `title` has COMMON OR `rand_key` range |
//! | 1%   | `rand_key` range; `title` has COMMON; `category` = cat00 AND `rand_key` range |
//! | 0.1% | `rand_key` range; `title` has RARE; `title` has COMMON AND `tier` = 3 |
//!
//! COMMON and RARE are title terms chosen FROM THE CORPUS: the terms whose
//! document frequency is nearest 1% and 0.1% of the corpus among the terms on
//! which Redis and OpenSearch provably agree on every document
//! ([`super::text_match::choose_term`]). They therefore differ between the
//! 100K, 1M and 10M variants; `PREPARED.json` records them.
//!
//! Every query vector of `tests.jsonl` is used once: query `i` is paired with
//! predicate `i % 12`. The bucket of a row is the bucket of its REALIZED
//! selectivity, and preparation fails if a predicate lands outside the bucket
//! it was designed for or matches fewer than [`crate::hybrid::MIN_MATCHES`]
//! rows.
//!
//! # The oracle
//!
//! The filtered top-k is computed by [`super::TopK::add_block_where`] — the
//! unfiltered oracle's own scoring/insertion/tie-break, with non-matching rows
//! skipped — over the normalized vectors read back from `vectors.npy`. It is
//! then cross-checked against the unfiltered ground truth in `tests.jsonl`:
//! the unfiltered top-k restricted to matching rows must be, id for id and in
//! order, the head of the filtered list. That one assertion covers the
//! predicate masks, the row alignment and the TopK reuse.

use super::derived::{self, DerivedFields, FieldValue, Fields, Predicate};
use super::text_match::{self, TextMatch};
use crate::hybrid::{Bucket, HybridQueryRow, MIN_MATCHES};
use rayon::prelude::*;
use serde_json::{json, Value};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

/// Field the text predicates search.
pub const TEXT_FIELD: &str = "title";

/// Ground-truth depth of every hybrid row (same as the unfiltered oracle).
pub const GT_DEPTH: usize = super::NEIGHBOURS;

/// Rows per scored block (same as the unfiltered oracle's pass).
const BLOCK_ROWS: usize = 8192;

/// One passage as a hybrid predicate sees it: the derived fields plus the
/// title text.
#[derive(Debug, Clone, PartialEq)]
pub struct TitledRow {
    pub derived: DerivedFields,
    pub title: Option<String>,
}

impl Fields for TitledRow {
    fn field(&self, name: &str) -> Option<FieldValue<'_>> {
        if name == TEXT_FIELD {
            self.title.as_deref().map(FieldValue::Keyword)
        } else {
            self.derived.field(name)
        }
    }
}

/// Every passage's predicate-visible fields, in corpus-id order.
#[derive(Debug, Default, Clone)]
pub struct HybridCorpus {
    rows: Vec<TitledRow>,
}

impl HybridCorpus {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a stored payload (the next corpus id). Its derived fields must
    /// equal the derivation; a non-string `title` is an error.
    pub fn push_payload(&mut self, payload: &Value) -> Result<(), String> {
        let d = derived::check_payload(payload)?;
        let title = match payload.get(TEXT_FIELD) {
            None | Some(Value::Null) => None,
            Some(Value::String(s)) => Some(s.clone()),
            Some(other) => return Err(format!("payload {TEXT_FIELD} is not a string: {other}")),
        };
        self.rows.push(TitledRow { derived: d, title });
        Ok(())
    }

    /// Stream a `payloads.jsonl`.
    pub fn from_payloads(path: &Path) -> Result<Self, String> {
        let f = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let mut c = Self::new();
        for (i, line) in BufReader::with_capacity(1 << 22, f).lines().enumerate() {
            let line = line.map_err(|e| format!("read {}: {e}", path.display()))?;
            let v: Value = serde_json::from_str(&line)
                .map_err(|e| format!("{} line {}: {e}", path.display(), i + 1))?;
            c.push_payload(&v)
                .map_err(|e| format!("{} line {}: {e}", path.display(), i + 1))?;
        }
        Ok(c)
    }

    pub fn rows(&self) -> &[TitledRow] {
        &self.rows
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    fn title(&self, i: usize) -> &str {
        self.rows[i].title.as_deref().unwrap_or("")
    }

    /// (documents `Yes`, documents `Ambiguous`) for `term` over every title.
    pub fn scan_term(&self, term: &str) -> (u64, u64) {
        (0..self.rows.len())
            .into_par_iter()
            .map(|i| match text_match::classify(self.title(i), term) {
                TextMatch::Yes => (1, 0),
                TextMatch::Ambiguous => (0, 1),
                TextMatch::No => (0, 0),
            })
            .reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1))
    }

    /// The COMMON (~1%) and RARE (~0.1%) title terms. See the module docs.
    pub fn choose_terms(&self) -> Result<TextTerms, String> {
        let n = self.rows.len() as u64;
        let mut counter = text_match::TermCounter::new();
        for i in 0..self.rows.len() {
            counter.add_doc(self.title(i));
        }
        let common =
            text_match::choose_term(&counter, n.div_ceil(100), &[], |t| self.scan_term(t))?;
        let rare = text_match::choose_term(&counter, n.div_ceil(1000), &[&common.0], |t| {
            self.scan_term(t)
        })?;
        Ok(TextTerms { common, rare })
    }
}

/// The corpus-chosen text terms, with their document frequencies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextTerms {
    pub common: (String, u64),
    pub rare: (String, u64),
}

/// One predicate of the set, before evaluation.
#[derive(Debug, Clone, PartialEq)]
pub struct HybridTemplate {
    pub predicate_id: String,
    pub shape: &'static str,
    /// The bucket the predicate is designed to land in.
    pub design: Bucket,
    pub conditions: Value,
}

fn rand_key_below(per_million: u64) -> Value {
    json!({"rand_key": {"range": {"lt": derived::RAND_KEY_SPAN * per_million / 1_000_000}}})
}

fn has_text(term: &str) -> Value {
    json!({TEXT_FIELD: {"match": {"text": term}}})
}

/// The twelve predicates (see the module docs' table).
pub fn templates(terms: &TextTerms) -> Vec<HybridTemplate> {
    let (common, rare) = (terms.common.0.as_str(), terms.rare.0.as_str());
    let rows: Vec<(&'static str, Bucket, Value)> = vec![
        (
            "numeric_range",
            Bucket::S50,
            json!({"and": [rand_key_below(500_000)]}),
        ),
        (
            "tag_any",
            Bucket::S50,
            json!({"and": [{"category": {"match": {"any": ["cat00", "cat01", "cat02", "cat03"]}}}]}),
        ),
        (
            "text_or_range",
            Bucket::S50,
            json!({"or": [has_text(rare), {"tier": {"range": {"lte": 4}}}]}),
        ),
        (
            "numeric_range",
            Bucket::S10,
            json!({"and": [rand_key_below(100_000)]}),
        ),
        (
            "tag_eq",
            Bucket::S10,
            json!({"and": [{"category": {"match": {"value": "cat01"}}}]}),
        ),
        (
            "text_or_range",
            Bucket::S10,
            json!({"or": [has_text(common), rand_key_below(90_000)]}),
        ),
        (
            "numeric_range",
            Bucket::S1,
            json!({"and": [rand_key_below(10_000)]}),
        ),
        ("text", Bucket::S1, json!({"and": [has_text(common)]})),
        (
            "tag_and_range",
            Bucket::S1,
            json!({"and": [{"category": {"match": {"value": "cat00"}}}, rand_key_below(40_000)]}),
        ),
        (
            "numeric_range",
            Bucket::S01,
            json!({"and": [rand_key_below(1_000)]}),
        ),
        ("text", Bucket::S01, json!({"and": [has_text(rare)]})),
        (
            "text_and_range",
            Bucket::S01,
            json!({"and": [has_text(common), {"tier": {"range": {"gte": 3, "lte": 3}}}]}),
        ),
    ];
    rows.into_iter()
        .enumerate()
        .map(|(i, (shape, design, conditions))| HybridTemplate {
            predicate_id: format!("hybrid-p{i:02}"),
            shape,
            design,
            conditions,
        })
        .collect()
}

/// The schema the predicates are typed against: the derived fields plus the
/// text field (as registered in `datasets.json`).
pub fn predicate_schema() -> Value {
    let mut s = derived::derived_schema();
    s[TEXT_FIELD] = json!("text");
    s
}

/// The templates, compiled and evaluated over a corpus.
#[derive(Debug, Clone)]
pub struct EvaluatedPredicates {
    pub templates: Vec<HybridTemplate>,
    pub predicates: Vec<Predicate>,
    pub counts: Vec<u64>,
    /// `masks[p][row]`: whether corpus row `row` satisfies predicate `p`.
    pub masks: Vec<Vec<bool>>,
}

/// Compile, type-check and evaluate every template over the corpus. Errors
/// when a predicate is outside the grammar or schema, matches fewer than
/// [`MIN_MATCHES`] rows, or lands outside its designed bucket.
pub fn evaluate(
    corpus: &HybridCorpus,
    templates: Vec<HybridTemplate>,
) -> Result<EvaluatedPredicates, String> {
    let n = corpus.len() as u64;
    let schema = predicate_schema();
    let mut out = EvaluatedPredicates {
        templates: Vec::new(),
        predicates: Vec::new(),
        counts: Vec::new(),
        masks: Vec::new(),
    };
    for t in templates {
        let id = &t.predicate_id;
        let p = Predicate::compile(&t.conditions).map_err(|e| format!("{id}: {e}"))?;
        crate::hybrid::validate_against_schema(&p, Some(&schema))
            .map_err(|e| format!("{id}: {e}"))?;
        let mask: Vec<bool> = corpus
            .rows()
            .par_iter()
            .map(|r| p.matches(r))
            .collect::<Result<_, _>>()
            .map_err(|e| format!("{id}: {e}"))?;
        let count = mask.iter().filter(|&&m| m).count() as u64;
        if count < MIN_MATCHES {
            return Err(format!(
                "{id} ({}) matches {count} of {n} rows, fewer than {MIN_MATCHES}",
                t.shape
            ));
        }
        let realized = Bucket::of(count, n).map_err(|e| format!("{id}: {e}"))?;
        if realized != t.design {
            return Err(format!(
                "{id} ({}) is designed for the {} bucket but matches {count} of {n} rows \
                 ({realized})",
                t.shape, t.design
            ));
        }
        out.predicates.push(p);
        out.counts.push(count);
        out.masks.push(mask);
        out.templates.push(t);
    }
    Ok(out)
}

/// The query vectors and unfiltered truth read from `tests.jsonl`.
#[derive(Debug, Clone)]
pub struct TestQueries {
    pub vectors: Vec<Vec<f32>>,
    pub unfiltered: Vec<Vec<i64>>,
    pub ids: Vec<Option<String>>,
}

/// Read `tests.jsonl` exactly as stored (no renormalization: the stored
/// vectors ARE what the unfiltered oracle scored, and the cross-check compares
/// against that ranking id for id). A vector that is not unit-length is an
/// error rather than a silent renormalization.
pub fn read_test_queries(path: &Path) -> Result<TestQueries, String> {
    let f = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut out = TestQueries {
        vectors: Vec::new(),
        unfiltered: Vec::new(),
        ids: Vec::new(),
    };
    for (i, line) in BufReader::with_capacity(1 << 20, f).lines().enumerate() {
        let line = line.map_err(|e| format!("read {}: {e}", path.display()))?;
        if line.trim().is_empty() {
            continue;
        }
        let row: Value = serde_json::from_str(&line)
            .map_err(|e| format!("{} line {}: {e}", path.display(), i + 1))?;
        let q: Vec<f32> = row["query"]
            .as_array()
            .ok_or_else(|| format!("{} line {}: no query", path.display(), i + 1))?
            .iter()
            .map(|v| v.as_f64().map(|x| x as f32))
            .collect::<Option<_>>()
            .ok_or_else(|| format!("{} line {}: non-numeric query", path.display(), i + 1))?;
        let norm = q.iter().map(|x| x * x).sum::<f32>().sqrt();
        if (norm - 1.0).abs() > 1e-3 {
            return Err(format!(
                "{} line {}: query vector has norm {norm}, not 1 — the oracle scores unit \
                 vectors",
                path.display(),
                i + 1
            ));
        }
        let ids: Vec<i64> = row["closest_ids"]
            .as_array()
            .ok_or_else(|| format!("{} line {}: no closest_ids", path.display(), i + 1))?
            .iter()
            .map(Value::as_i64)
            .collect::<Option<_>>()
            .ok_or_else(|| format!("{} line {}: non-integer closest_ids", path.display(), i + 1))?;
        out.vectors.push(q);
        out.unfiltered.push(ids);
        out.ids.push(row["query_id"].as_str().map(str::to_string));
    }
    Ok(out)
}

/// Open a C-order little-endian `float32` 2-D NPY; returns (reader positioned
/// at the data, rows, cols).
pub fn open_f32_npy(path: &Path) -> Result<(BufReader<File>, u64, usize), String> {
    let mut f = BufReader::with_capacity(
        1 << 22,
        File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?,
    );
    let mut pre = [0u8; 8];
    f.read_exact(&mut pre)
        .map_err(|e| format!("read {}: {e}", path.display()))?;
    if &pre[..6] != b"\x93NUMPY" {
        return Err(format!("{} is not an NPY file", path.display()));
    }
    let hlen = match pre[6] {
        1 => {
            let mut b = [0u8; 2];
            f.read_exact(&mut b).map_err(|e| e.to_string())?;
            u16::from_le_bytes(b) as usize
        }
        2 | 3 => {
            let mut b = [0u8; 4];
            f.read_exact(&mut b).map_err(|e| e.to_string())?;
            u32::from_le_bytes(b) as usize
        }
        v => return Err(format!("{}: unsupported NPY version {v}", path.display())),
    };
    if hlen > 1 << 16 {
        return Err(format!("{}: NPY header of {hlen} bytes", path.display()));
    }
    let mut h = vec![0u8; hlen];
    f.read_exact(&mut h)
        .map_err(|e| format!("read {} header: {e}", path.display()))?;
    let h = String::from_utf8_lossy(&h);
    if !h.contains("'descr': '<f4'") || !h.contains("'fortran_order': False") {
        return Err(format!(
            "{}: expected a C-order '<f4' array, header is {h}",
            path.display()
        ));
    }
    let shape = h
        .split("'shape': (")
        .nth(1)
        .and_then(|s| s.split(')').next())
        .ok_or_else(|| format!("{}: no shape in header {h}", path.display()))?;
    let dims: Vec<u64> = shape
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<u64>())
        .collect::<Result<_, _>>()
        .map_err(|e| format!("{}: shape {shape:?}: {e}", path.display()))?;
    match dims.as_slice() {
        [rows, cols] if *cols > 0 => Ok((f, *rows, *cols as usize)),
        _ => Err(format!(
            "{}: expected a 2-D array, shape is ({shape})",
            path.display()
        )),
    }
}

/// Brute-force each query's top-`depth` over the rows its predicate admits.
/// `assign[q]` is query `q`'s predicate index into `masks`.
pub fn filtered_ground_truth(
    vectors: &Path,
    queries: &[Vec<f32>],
    assign: &[usize],
    masks: &[Vec<bool>],
    depth: usize,
) -> Result<Vec<Vec<(f32, i64)>>, String> {
    let (mut reader, rows, dim) = open_f32_npy(vectors)?;
    if assign.len() != queries.len() {
        return Err("one predicate assignment per query is required".into());
    }
    for m in masks {
        if m.len() as u64 != rows {
            return Err(format!(
                "a predicate mask covers {} rows but {} holds {rows}",
                m.len(),
                vectors.display()
            ));
        }
    }
    if let Some(q) = queries.iter().find(|q| q.len() != dim) {
        return Err(format!(
            "query of dimension {} against {dim}-dimensional vectors",
            q.len()
        ));
    }
    let flat: Vec<f32> = queries.iter().flatten().copied().collect();
    let mut acc = super::TopK::new(flat, dim, depth)?;
    let mut raw = vec![0u8; BLOCK_ROWS * dim * 4];
    let mut done = 0u64;
    while done < rows {
        let n = ((rows - done) as usize).min(BLOCK_ROWS);
        let want = n * dim * 4;
        reader
            .read_exact(&mut raw[..want])
            .map_err(|e| format!("read {} at row {done}: {e}", vectors.display()))?;
        let mut block: Vec<f32> = raw[..want]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        super::ensure_finite(&block, done as i64, dim)?;
        // Same normalization as the unfiltered oracle (cosine on unit vectors).
        for row in block.chunks_exact_mut(dim) {
            super::normalize_in_place(row);
        }
        acc.add_block_where(&block, done as i64, |qi, id| masks[assign[qi]][id as usize])?;
        done += n as u64;
    }
    let mut trailing = [0u8; 1];
    if reader.read(&mut trailing).map_err(|e| e.to_string())? != 0 {
        return Err(format!(
            "{} holds more data than its header's {rows} rows",
            vectors.display()
        ));
    }
    Ok(acc.finish())
}

/// The unfiltered top-k restricted to matching rows must be, in order, the
/// head of the filtered top-k. Returns how many positions were compared.
pub fn cross_check(
    filtered: &[(f32, i64)],
    unfiltered: &[i64],
    mask: &[bool],
) -> Result<usize, String> {
    let head: Vec<i64> = unfiltered
        .iter()
        .copied()
        .filter(|&id| mask.get(id as usize).copied().unwrap_or(false))
        .collect();
    if head.len() > filtered.len() {
        return Err(format!(
            "{} matching rows in the unfiltered top-k but only {} in the filtered list",
            head.len(),
            filtered.len()
        ));
    }
    for (i, (want, (_, got))) in head.iter().zip(filtered).enumerate() {
        if want != got {
            return Err(format!(
                "filtered ground truth disagrees with the unfiltered oracle at rank {i}: \
                 expected row {want}, got row {got}"
            ));
        }
    }
    Ok(head.len())
}

/// Input files (paths, so a full build can read its STAGED outputs back).
#[derive(Debug, Clone)]
pub struct HybridInputs {
    pub payloads: PathBuf,
    pub vectors: PathBuf,
    pub tests: PathBuf,
}

impl HybridInputs {
    pub fn in_dir(dir: &Path) -> Self {
        Self {
            payloads: dir.join("payloads.jsonl"),
            vectors: dir.join("vectors.npy"),
            tests: dir.join("tests.jsonl"),
        }
    }
}

/// The generated query set plus its `PREPARED.json` entry.
#[derive(Debug, Clone)]
pub struct PreparedHybrid {
    pub rows: Vec<HybridQueryRow>,
    pub manifest: Value,
}

/// Build the hybrid query set for a prepared corpus. `expect_rows` is the
/// registered passage count, when known.
pub fn prepare(inputs: &HybridInputs, expect_rows: Option<u64>) -> Result<PreparedHybrid, String> {
    let corpus = HybridCorpus::from_payloads(&inputs.payloads)?;
    let n = corpus.len() as u64;
    if let Some(want) = expect_rows {
        if n != want {
            return Err(format!(
                "{} holds {n} payloads but {want} are expected",
                inputs.payloads.display()
            ));
        }
    }
    let tests = read_test_queries(&inputs.tests)?;
    if tests.vectors.is_empty() {
        return Err(format!("{} holds no queries", inputs.tests.display()));
    }
    let terms = corpus.choose_terms()?;
    let ev = evaluate(&corpus, templates(&terms))?;
    let p = ev.templates.len();
    let assign: Vec<usize> = (0..tests.vectors.len()).map(|i| i % p).collect();
    let gt = filtered_ground_truth(
        &inputs.vectors,
        &tests.vectors,
        &assign,
        &ev.masks,
        GT_DEPTH,
    )?;
    let mut compared = 0usize;
    let mut rows = Vec::with_capacity(gt.len());
    for (i, ranked) in gt.into_iter().enumerate() {
        let pi = assign[i];
        let t = &ev.templates[pi];
        compared += cross_check(&ranked, &tests.unfiltered[i], &ev.masks[pi])
            .map_err(|e| format!("query {i} / {}: {e}", t.predicate_id))?;
        let count = ev.counts[pi];
        let row = HybridQueryRow {
            query_id: format!("msmarco-hybrid-{i:04}"),
            test_index: i,
            source_query_id: tests.ids[i].clone(),
            predicate_id: t.predicate_id.clone(),
            predicate_shape: t.shape.to_string(),
            bucket: t.design,
            conditions: t.conditions.clone(),
            matching_count: count,
            corpus_size: n,
            selectivity: count as f64 / n as f64,
            gt_depth: GT_DEPTH,
            closest_ids: ranked.iter().map(|(_, id)| *id).collect(),
            closest_scores: ranked.iter().map(|(s, _)| *s).collect(),
        };
        row.check()?;
        rows.push(row);
    }
    let manifest = json!({
        "hybrid_queries": {
            "file": crate::hybrid::HYBRID_QUERIES_FILE,
            "queries": rows.len(),
            "pairing": "query i of tests.jsonl is paired with predicate i % 12",
            "gt_depth": GT_DEPTH,
            "ground_truth": "brute force over the rows satisfying the predicate, with the \
                             unfiltered oracle's TopK (add_block_where), on the normalized float32 \
                             vectors in vectors.npy",
            "cross_checked_against_unfiltered": {"queries": rows.len(), "positions": compared},
            "text_match": {
                "rule": text_match::RULE_VERSION,
                "field": TEXT_FIELD,
                "common_term": {"term": terms.common.0, "documents": terms.common.1},
                "rare_term": {"term": terms.rare.0, "documents": terms.rare.1},
                "note": "terms chosen from the corpus: nearest 1% / 0.1% document frequency \
                         among terms with zero documents on which Redis and OpenSearch could \
                         tokenize differently",
            },
            "predicates": ev.templates.iter().zip(&ev.counts).map(|(t, c)| json!({
                "predicate_id": t.predicate_id,
                "shape": t.shape,
                "bucket": t.design,
                "matching_count": c,
                "conditions": t.conditions,
            })).collect::<Vec<_>>(),
        }
    });
    Ok(PreparedHybrid { rows, manifest })
}

pub fn write_rows(path: &Path, rows: &[HybridQueryRow]) -> Result<(), String> {
    let mut out =
        BufWriter::new(File::create(path).map_err(|e| format!("create {}: {e}", path.display()))?);
    for r in rows {
        serde_json::to_writer(&mut out, r).map_err(|e| format!("write {}: {e}", path.display()))?;
        out.write_all(b"\n")
            .map_err(|e| format!("write {}: {e}", path.display()))?;
    }
    out.flush()
        .map_err(|e| format!("flush {}: {e}", path.display()))
}

/// What [`check_rows`] confirmed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HybridSummary {
    pub queries: usize,
    pub per_bucket: Vec<(Bucket, usize)>,
}

/// Compare an on-disk query set against the one the corpus regenerates.
/// Exact, row by row; a changed predicate, count or ground truth is reported
/// as such.
pub fn check_rows(
    expected: &[HybridQueryRow],
    found: &[HybridQueryRow],
) -> Result<HybridSummary, String> {
    if expected.len() != found.len() {
        return Err(format!(
            "hybrid_queries.jsonl holds {} rows; the corpus regenerates {}",
            found.len(),
            expected.len()
        ));
    }
    for (e, f) in expected.iter().zip(found) {
        let id = &e.query_id;
        if (&e.query_id, e.test_index, &e.predicate_id, &e.conditions)
            != (&f.query_id, f.test_index, &f.predicate_id, &f.conditions)
        {
            return Err(format!(
                "hybrid_queries.jsonl {id}: predicate or pairing differs from what the corpus \
                 regenerates ({} / {})",
                f.conditions, e.conditions
            ));
        }
        if e.matching_count != f.matching_count || e.corpus_size != f.corpus_size {
            return Err(format!(
                "hybrid_queries.jsonl {id}: matching_count {} of {} but {} of {} rows satisfy \
                 the predicate",
                f.matching_count, f.corpus_size, e.matching_count, e.corpus_size
            ));
        }
        if e.closest_ids != f.closest_ids {
            return Err(format!(
                "hybrid_queries.jsonl {id}: filtered ground truth differs from the brute force"
            ));
        }
        // JSON floats do not round-trip bit-exactly through serde_json's
        // default parser (it can be 1 ulp off), so the two float fields compare
        // within a tolerance far below any meaningful difference; ids, counts
        // and predicates above compare exactly.
        let mut f = f.clone();
        if (f.selectivity - e.selectivity).abs() <= 1e-12 {
            f.selectivity = e.selectivity;
        }
        if f.closest_scores.len() == e.closest_scores.len()
            && f.closest_scores
                .iter()
                .zip(&e.closest_scores)
                .all(|(a, b)| (a - b).abs() <= 1e-6)
        {
            f.closest_scores = e.closest_scores.clone();
        }
        if *e != f {
            let (ev, fv) = (
                serde_json::to_value(e).map_err(|x| x.to_string())?,
                serde_json::to_value(&f).map_err(|x| x.to_string())?,
            );
            let fields: Vec<&String> = ev
                .as_object()
                .into_iter()
                .flatten()
                .filter(|(k, v)| fv.get(k.as_str()) != Some(*v))
                .map(|(k, _)| k)
                .collect();
            return Err(format!(
                "hybrid_queries.jsonl {id}: {fields:?} differ from the regenerated row"
            ));
        }
    }
    Ok(HybridSummary {
        queries: expected.len(),
        per_bucket: Bucket::ALL
            .iter()
            .map(|b| (*b, expected.iter().filter(|r| r.bucket == *b).count()))
            .collect(),
    })
}

/// A small, deterministic MS MARCO-shaped corpus for tests and CI fixtures:
/// `n` passages with real derived fields, titles drawn so that the COMMON and
/// RARE terms exist, `dim`-dimensional vectors, and `queries` unit-length
/// query vectors whose unfiltered top-[`GT_DEPTH`] is brute-forced with the
/// production oracle. Writes `vectors.npy`, `payloads.jsonl`, `tests.jsonl`.
pub fn write_fixture_corpus(
    dir: &Path,
    n: usize,
    dim: usize,
    queries: usize,
) -> Result<(), String> {
    use rand::{Rng, SeedableRng};
    if queries == 0 || n == 0 || dim == 0 {
        return Err("fixture needs rows, queries and a dimension".into());
    }
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let mut rng = rand::rngs::StdRng::seed_from_u64(0x341);
    // Title vocabulary: each word is present in a known fraction of titles.
    // `quoras` / `ancients` stem to the chosen terms in Redis: a Redis
    // query sent WITHOUT `VERBATIM` matches them too, and recall drops.
    const WORDS: [(&str, f64); 10] = [
        ("guide", 0.30),
        ("review", 0.10),
        ("history", 0.03),
        ("quora", 0.011),
        ("quoras", 0.004),
        ("recipes", 0.006),
        ("ancient", 0.0011),
        ("ancients", 0.002),
        ("glacier", 0.0006),
        ("zebra", 0.0002),
    ];
    let unit = |v: &mut Vec<f32>| super::normalize_in_place(v);
    let mut vectors = Vec::with_capacity(n);
    let mut payloads = String::new();
    for i in 0..n {
        let mut v: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
        unit(&mut v);
        vectors.push(v);
        let docid = format!("fixture_doc_{:02}_{i}#{}_{}", i % 7, i % 5, i);
        let mut words: Vec<String> = WORDS
            .iter()
            .filter(|(_, p)| rng.gen_bool(*p))
            .map(|(w, _)| w.to_string())
            .collect();
        // Every title also carries a filler word so none is empty, and some
        // carry contexts the engines tokenize differently (never on a
        // vocabulary word, so the chosen terms stay unambiguous).
        words.push(format!("item{}", i % 97));
        if i % 11 == 0 {
            words.push("Nature's".into());
        }
        if i % 13 == 0 {
            words.push("example.com".into());
        }
        let mut p = serde_json::Map::new();
        p.insert("docid".into(), json!(docid));
        p.insert("title".into(), json!(words.join(" ")));
        DerivedFields::from_docid(&docid).insert_into(&mut p)?;
        payloads.push_str(&Value::Object(p).to_string());
        payloads.push('\n');
    }
    crate::readers::write_npy_vectors(dir.join("vectors.npy").to_str().ok_or("path")?, &vectors)?;
    std::fs::write(dir.join("payloads.jsonl"), payloads).map_err(|e| e.to_string())?;
    let qs: Vec<Vec<f32>> = (0..queries)
        .map(|_| {
            let mut v: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
            unit(&mut v);
            v
        })
        .collect();
    // Unfiltered truth through the SAME pass the filtered oracle uses (read
    // back from vectors.npy, normalized), as `prepare-msmarco` does.
    let ranked = filtered_ground_truth(
        &dir.join("vectors.npy"),
        &qs,
        &vec![0; qs.len()],
        &[vec![true; n]],
        GT_DEPTH,
    )?;
    let mut tests = String::new();
    for (i, ranked) in ranked.iter().enumerate() {
        tests.push_str(
            &super::test_row(&qs[i], &format!("fixture-q{i}"), "fixture", None, ranked).to_string(),
        );
        tests.push('\n');
    }
    std::fs::write(dir.join("tests.jsonl"), tests).map_err(|e| e.to_string())
}

// ---------------------------------------------------------------------------
// Invalidation by `--add-derived-fields`
// ---------------------------------------------------------------------------

/// The `PREPARED.json` keys that describe [`crate::hybrid::HYBRID_QUERIES_FILE`].
pub const MANIFEST_KEYS: [&str; 2] = ["hybrid_queries", "hybrid_queries_added_at"];

/// One step of publishing `prepare-msmarco --add-derived-fields`' staged
/// outputs: rename a staged file over its published name, or remove a
/// published file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PublishStep {
    Rename(&'static str),
    Remove(&'static str),
}

/// How `--add-derived-fields` publishes, in order. Rewriting `payloads.jsonl`
/// invalidates `hybrid_queries.jsonl`: its predicates, matching counts and
/// filtered truth were computed from the old payloads, and the startup
/// check (`hybrid::check_bound_to_corpus`) only binds it to `vectors.npy`
/// and `tests.jsonl`, so a stale set would still pass. So the set is removed
/// in the same publish, and the binary's `run()` and the benchmark's
/// hybrid preflight then refuse until `--add-hybrid-queries` regenerates it.
///
/// The order keeps one invariant at every point a run could die: a
/// published `hybrid_queries.jsonl` was computed from the published
/// `payloads.jsonl`.
/// 1. `PREPARED.json` first. The new manifest no longer lists the hybrid
///    set ([`drop_from_manifest`]). If the run dies here, the old set is
///    still valid for the old payloads.
/// 2. Remove `hybrid_queries.jsonl`, BEFORE the payloads change.
/// 3. `payloads.jsonl`.
/// 4. `filter_queries.jsonl` LAST: its presence marks the upgrade as done
///    (`run()` and `derived::check_prepared_dir` key on it).
pub const DERIVED_UPGRADE_PUBLISH: [PublishStep; 4] = [
    PublishStep::Rename("PREPARED.json"),
    PublishStep::Remove(crate::hybrid::HYBRID_QUERIES_FILE),
    PublishStep::Rename("payloads.jsonl"),
    PublishStep::Rename(derived::FILTER_QUERIES_FILE),
];

/// Remove the hybrid query set's entries ([`MANIFEST_KEYS`]) from a
/// `PREPARED.json` value, for a publish that removes the set itself.
pub fn drop_from_manifest(manifest: &mut Value) -> Result<(), String> {
    let obj = manifest
        .as_object_mut()
        .ok_or("PREPARED.json must be a JSON object")?;
    for key in MANIFEST_KEYS {
        obj.remove(key);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Review of #357: `--add-derived-fields` rewrites the payloads, so it
    /// must remove the hybrid set computed from the old ones. The removal
    /// lands after the new manifest (which drops the set) and before the
    /// payloads, and `filter_queries.jsonl` is still the last step.
    #[test]
    fn derived_upgrade_removes_the_hybrid_set_before_the_payloads_change() {
        use PublishStep::*;
        let at = |step: PublishStep| {
            DERIVED_UPGRADE_PUBLISH
                .iter()
                .position(|s| *s == step)
                .unwrap_or_else(|| panic!("{step:?} is not in the derived-upgrade publish"))
        };
        let remove = at(Remove(crate::hybrid::HYBRID_QUERIES_FILE));
        assert!(at(Rename("PREPARED.json")) < remove);
        assert!(remove < at(Rename("payloads.jsonl")));
        assert_eq!(
            DERIVED_UPGRADE_PUBLISH.last(),
            Some(&Rename(derived::FILTER_QUERIES_FILE))
        );

        let mut manifest = json!({
            "corpus": "x",
            "hybrid_queries": {"queries": 3},
            "hybrid_queries_added_at": "t",
            "filter_queries": {"queries": 31},
        });
        drop_from_manifest(&mut manifest).unwrap();
        assert_eq!(
            manifest,
            json!({"corpus": "x", "filter_queries": {"queries": 31}})
        );
        assert!(drop_from_manifest(&mut json!([])).is_err());
    }

    const N: usize = 20_000;

    fn fixture() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        write_fixture_corpus(dir.path(), N, 8, 48).unwrap();
        dir
    }

    #[test]
    fn fixture_set_covers_every_bucket_and_cross_checks() {
        let dir = fixture();
        let prepared = prepare(&HybridInputs::in_dir(dir.path()), Some(N as u64)).unwrap();
        let rows = &prepared.rows;
        assert_eq!(rows.len(), 48);
        for b in Bucket::ALL {
            assert_eq!(rows.iter().filter(|r| r.bucket == b).count(), 12, "{b}");
        }
        let m = &prepared.manifest["hybrid_queries"];
        assert_eq!(m["text_match"]["common_term"]["term"], "quora");
        assert_eq!(m["text_match"]["rare_term"]["term"], "ancient");
        assert!(
            m["cross_checked_against_unfiltered"]["positions"]
                .as_u64()
                .unwrap()
                > 0
        );
        // Every row's truth is exactly the brute force over the matching rows.
        let corpus = HybridCorpus::from_payloads(&dir.path().join("payloads.jsonl")).unwrap();
        let tests = read_test_queries(&dir.path().join("tests.jsonl")).unwrap();
        let (_, mut vectors) = crate::readers::read_npy_vectors(
            dir.path().join("vectors.npy").to_str().unwrap(),
            false,
        )
        .unwrap();
        vectors
            .iter_mut()
            .for_each(|v| super::super::normalize_in_place(v));
        for r in rows.iter().step_by(5) {
            let p = Predicate::compile(&r.conditions).unwrap();
            let q = &tests.vectors[r.test_index];
            let mut scored: Vec<(f32, i64)> = (0..N)
                .filter(|&i| p.matches(&corpus.rows()[i]).unwrap())
                .map(|i| (super::super::dot(q, &vectors[i]), i as i64))
                .collect();
            assert_eq!(scored.len() as u64, r.matching_count);
            scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            let want: Vec<i64> = scored.iter().take(GT_DEPTH).map(|(_, i)| *i).collect();
            assert_eq!(r.closest_ids, want, "{}", r.query_id);
        }
        // Round trip through the file, and the verify comparison.
        let path = dir.path().join(crate::hybrid::HYBRID_QUERIES_FILE);
        write_rows(&path, rows).unwrap();
        let back = crate::hybrid::read_rows(&path).unwrap();
        let summary = check_rows(rows, &back).unwrap();
        assert_eq!(summary.queries, 48);
        let mut tampered = back.clone();
        tampered[3].closest_ids.swap(0, 1);
        assert!(check_rows(rows, &tampered)
            .unwrap_err()
            .contains("ground truth"));
        let mut tampered = back.clone();
        tampered[3].matching_count += 1;
        assert!(check_rows(rows, &tampered)
            .unwrap_err()
            .contains("matching_count"));
        let mut tampered = back.clone();
        tampered[3].closest_scores[0] += 1e-3;
        assert!(check_rows(rows, &tampered)
            .unwrap_err()
            .contains("closest_scores"));
        let mut tampered = back;
        tampered[3].conditions = rows[4].conditions.clone();
        assert!(check_rows(rows, &tampered)
            .unwrap_err()
            .contains("predicate"));
        // The fulltext mix rule sees exactly the rows whose predicate filters
        // on `title` (the text predicates), and none for another text field.
        let text_rows = rows
            .iter()
            .filter(|r| r.conditions.to_string().contains("\"title\""))
            .count();
        assert!(text_rows > 0);
        assert_eq!(
            crate::hybrid::count_text_field_rows(dir.path(), &["title".to_string()]).unwrap(),
            text_rows
        );
        assert_eq!(
            crate::hybrid::count_text_field_rows(dir.path(), &["segment".to_string()]).unwrap(),
            0
        );
        // Bound to its corpus: the preflight accepts it here, and refuses it
        // next to a corpus of another size or another tests.jsonl.
        let schema = predicate_schema();
        assert_eq!(
            crate::hybrid::preflight(dir.path(), Some(&schema), &[10]).unwrap(),
            48
        );
        let npy = dir.path().join("vectors.npy");
        let tests_path = dir.path().join("tests.jsonl");
        let (npy_bytes, tests_text) = (
            std::fs::read(&npy).unwrap(),
            std::fs::read_to_string(&tests_path).unwrap(),
        );
        let (_, vecs) = crate::readers::read_npy_vectors(npy.to_str().unwrap(), false).unwrap();
        crate::readers::write_npy_vectors(npy.to_str().unwrap(), &vecs[..N - 1]).unwrap();
        let e = crate::hybrid::preflight(dir.path(), Some(&schema), &[10]).unwrap_err();
        assert!(e.contains("another corpus"), "{e}");
        std::fs::write(&npy, &npy_bytes).unwrap();
        let mut lines: Vec<Value> = tests_text
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        lines.swap(0, 1);
        std::fs::write(
            &tests_path,
            lines
                .iter()
                .map(|l| l.to_string() + "\n")
                .collect::<String>(),
        )
        .unwrap();
        let e = crate::hybrid::preflight(dir.path(), Some(&schema), &[10]).unwrap_err();
        assert!(e.contains("another tests.jsonl"), "{e}");
        std::fs::write(&tests_path, &tests_text).unwrap();
        assert!(crate::hybrid::preflight(dir.path(), Some(&schema), &[10]).is_ok());
        // The engines' reader accepts the file and aligns vectors by test_index.
        let set = crate::hybrid::read_query_set(dir.path(), false).unwrap();
        assert_eq!(set.queries.len(), 48);
        assert_eq!(set.queries[7], tests.vectors[7]);
        assert_eq!(set.by_bucket().len(), 4);
    }

    #[test]
    fn the_filtered_oracle_is_the_unfiltered_one_when_every_row_matches() {
        let dir = fixture();
        let tests = read_test_queries(&dir.path().join("tests.jsonl")).unwrap();
        let (_, mut vectors) = crate::readers::read_npy_vectors(
            dir.path().join("vectors.npy").to_str().unwrap(),
            false,
        )
        .unwrap();
        vectors
            .iter_mut()
            .for_each(|v| super::super::normalize_in_place(v));
        // The plain unfiltered path, block by block.
        let flat: Vec<f32> = tests.vectors.iter().flatten().copied().collect();
        let mut plain = super::super::TopK::new(flat, 8, GT_DEPTH).unwrap();
        for (i, chunk) in vectors.chunks(BLOCK_ROWS).enumerate() {
            let block: Vec<f32> = chunk.iter().flatten().copied().collect();
            plain.add_block(&block, (i * BLOCK_ROWS) as i64).unwrap();
        }
        let plain = plain.finish();
        let all = vec![vec![true; N]];
        let assign = vec![0; tests.vectors.len()];
        let gt = filtered_ground_truth(
            &dir.path().join("vectors.npy"),
            &tests.vectors,
            &assign,
            &all,
            GT_DEPTH,
        )
        .unwrap();
        assert_eq!(gt, plain);
        for (i, ranked) in gt.iter().enumerate() {
            let ids: Vec<i64> = ranked.iter().map(|(_, id)| *id).collect();
            assert_eq!(ids, tests.unfiltered[i]);
        }
        // A mask that drops rows drops exactly those rows from each ranking.
        let even: Vec<bool> = (0..N).map(|i| i % 2 == 0).collect();
        let gt = filtered_ground_truth(
            &dir.path().join("vectors.npy"),
            &tests.vectors,
            &assign,
            &[even],
            GT_DEPTH,
        )
        .unwrap();
        for (i, ranked) in gt.iter().enumerate() {
            assert!(ranked.iter().all(|(_, id)| id % 2 == 0));
            let head: Vec<i64> = plain[i]
                .iter()
                .map(|(_, id)| *id)
                .filter(|id| id % 2 == 0)
                .collect();
            let got: Vec<i64> = ranked.iter().take(head.len()).map(|(_, id)| *id).collect();
            assert_eq!(got, head);
        }
    }

    #[test]
    fn cross_check_catches_a_misaligned_mask_or_ranking() {
        let mask = vec![true, false, true, true];
        let filtered = vec![(0.9, 2), (0.8, 0), (0.1, 3)];
        assert_eq!(cross_check(&filtered, &[1, 2, 0], &mask).unwrap(), 2);
        assert!(cross_check(&filtered, &[0, 2], &mask).is_err());
        assert!(cross_check(&filtered[..1], &[2, 0], &mask).is_err());
    }

    #[test]
    fn a_predicate_outside_its_bucket_or_too_narrow_is_refused() {
        let dir = fixture();
        let corpus = HybridCorpus::from_payloads(&dir.path().join("payloads.jsonl")).unwrap();
        let t = |design, conditions| HybridTemplate {
            predicate_id: "x".into(),
            shape: "numeric_range",
            design,
            conditions,
        };
        let half = json!({"and":[rand_key_below(500_000)]});
        assert!(evaluate(&corpus, vec![t(Bucket::S50, half.clone())]).is_ok());
        assert!(evaluate(&corpus, vec![t(Bucket::S10, half)]).is_err());
        let narrow = json!({"and":[rand_key_below(100)]});
        assert!(evaluate(&corpus, vec![t(Bucket::S01, narrow)])
            .unwrap_err()
            .contains("fewer than"));
        // A text leaf on a keyword field is outside the schema.
        let wrong = json!({"and":[{"category":{"match":{"text":"quora"}}}]});
        assert!(evaluate(&corpus, vec![t(Bucket::S50, wrong)]).is_err());
        // A term the engines could tokenize differently cannot get truth.
        let amb = json!({"and":[{"title":{"match":{"text":"nature"}}}]});
        assert!(evaluate(&corpus, vec![t(Bucket::S10, amb)])
            .unwrap_err()
            .contains("tokenizer"));
    }

    #[test]
    fn unnormalized_queries_and_bad_npy_are_errors() {
        let dir = fixture();
        let tests = dir.path().join("tests.jsonl");
        let text = std::fs::read_to_string(&tests).unwrap();
        let mut row: Value = serde_json::from_str(text.lines().next().unwrap()).unwrap();
        row["query"][0] = json!(5.0);
        std::fs::write(&tests, format!("{row}\n")).unwrap();
        assert!(read_test_queries(&tests).unwrap_err().contains("norm"));
        let npy = dir.path().join("vectors.npy");
        let bytes = std::fs::read(&npy).unwrap();
        std::fs::write(&npy, &bytes[..bytes.len() - 4]).unwrap();
        let q = vec![vec![1.0f32, 0., 0., 0., 0., 0., 0., 0.]];
        assert!(filtered_ground_truth(&npy, &q, &[0], &[vec![true; N]], 10).is_err());
        let mut longer = bytes.clone();
        longer.extend_from_slice(&[0u8; 4]);
        std::fs::write(&npy, &longer).unwrap();
        assert!(filtered_ground_truth(&npy, &q, &[0], &[vec![true; N]], 10).is_err());
        std::fs::write(&npy, &bytes).unwrap();
        assert!(filtered_ground_truth(&npy, &q, &[0], &[vec![true; N - 1]], 10).is_err());
        assert!(filtered_ground_truth(&npy, &q, &[0], &[vec![true; N]], 10).is_ok());
        assert!(filtered_ground_truth(&npy, &[vec![1.0f32]], &[0], &[vec![true; N]], 10).is_err());
    }
}
