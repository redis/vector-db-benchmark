//! Derived exact-match fields for the MS MARCO corpus, and the filter-only query
//! set built over them (issue #340; the hybrid prefilters of #341 reuse both).
//!
//! # Why derive at all
//!
//! The real metadata has no field an exact-match benchmark can use well: `docid`
//! and `url` are unique keywords, `start_char`/`end_char` are offsets whose
//! distribution is a property of how upstream segmented documents. A tag/numeric
//! benchmark needs a tag with a spread of value frequencies, a low-cardinality
//! numeric (so a range covers many repeated values), a high-cardinality numeric
//! (so a range has a precise, tunable selectivity and a point lookup returns ~1
//! row) and a length-like numeric with a skewed distribution. Those are derived
//! here, per passage, as a **pure function of `docid`**: no RNG state, no
//! dependence on the row's position, the variant, the build host or the thread
//! count — so two builds of any variant agree byte for byte, and a passage
//! carries the same values in the 100K, 1M and 10M variants (which nest).
//!
//! # The rule (reproducible in five lines of Python)
//!
//! ```text
//! d = sha256(b"vdb-msmarco-derived/v1\0" + docid.encode()).digest()
//! w = [int.from_bytes(d[8*i:8*i+8], "big") for i in range(4)]
//! category = "cat%02d" % zipf_bucket(w[0] % sum(W)), W[i] = 1_000_000 // (i + 1), i < 32
//! e = 4 + w[1] % 12;  doc_len = 2**e + (w[1] >> 32) % 2**e      # 16..=65535, log-uniform
//! tier = w[2] % 10                                              # 0..=9, uniform
//! rand_key = w[3] >> 24                                         # 0..2**40, uniform
//! ```
//!
//! SHA-256 with a salt, NOT crc32: the variants are SELECTED by
//! `crc32(docid) % 1e6 < threshold`, so any crc32-derived value would be
//! confined to a sliver of its range inside a variant (crc32 is affine in its
//! input, so salting does not decorrelate it either). These fields are synthetic
//! labels attached to real passages; they carry no meaning about the text, and
//! a predicate over them has a selectivity set by construction, not by the data.
//!
//! # The predicate grammar this module evaluates
//!
//! The repo's existing `conditions` JSON (see `query_filter` and
//! `filter_only::domain`), restricted to the subset whose meaning every engine
//! adapter agrees on:
//!
//! * a **group**: an object with exactly ONE key, `and` or `or`, holding a
//!   nonempty array of groups or leaves;
//! * a **leaf**: an object with exactly ONE field, holding exactly ONE operation:
//!   `{"match": {"value": v}}`, `{"match": {"any": [..]}}` or
//!   `{"range": {gt|gte|lt|lte: n, ..}}`.
//!
//! The restriction is deliberate. A leaf with two operations, or an object
//! mixing `and` and `or`, has context-dependent meaning in the engine builders
//! (Redis pushes each operation of a leaf into the enclosing group, so a
//! two-op leaf inside `or` becomes a disjunction). Anything outside the subset
//! is an `Err`, never a guess — an expected count computed under a different
//! reading than the engine's is exactly the silently-wrong ground truth this
//! repo exists to avoid.

use rayon::prelude::*;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

/// Domain-separation prefix hashed in front of every `docid`. Bump the version
/// (and every expected value in the tests) to change the derivation; never
/// change the rule under the same salt.
pub const DERIVATION_SALT: &[u8] = b"vdb-msmarco-derived/v1\0";

/// The derived payload fields, with the `schema` type each is registered under
/// in `datasets/datasets.json`. `keyword` maps to a Redis TAG / OpenSearch
/// `keyword`; `int` to a Redis NUMERIC / OpenSearch `long`.
pub const DERIVED_FIELDS: &[(&str, &str)] = &[
    ("category", "keyword"),
    ("doc_len", "int"),
    ("tier", "int"),
    ("rand_key", "int"),
];

/// Number of `category` values.
pub const CATEGORY_COUNT: usize = 32;

/// Number of `tier` values (`0..TIER_COUNT`).
pub const TIER_COUNT: u64 = 10;

/// `doc_len` lies in `DOC_LEN_MIN..=DOC_LEN_MAX`.
pub const DOC_LEN_MIN: u32 = 1 << 4;
pub const DOC_LEN_MAX: u32 = (1 << 16) - 1;

/// `rand_key` lies in `0..RAND_KEY_SPAN` (40 bits — far inside the +/-2^53
/// range Redis NUMERIC represents exactly).
pub const RAND_KEY_SPAN: u64 = 1 << 40;

/// The `category` value names, by index. Zero-padded so lexical and numeric
/// order agree.
pub const CATEGORY_NAMES: [&str; CATEGORY_COUNT] = [
    "cat00", "cat01", "cat02", "cat03", "cat04", "cat05", "cat06", "cat07", "cat08", "cat09",
    "cat10", "cat11", "cat12", "cat13", "cat14", "cat15", "cat16", "cat17", "cat18", "cat19",
    "cat20", "cat21", "cat22", "cat23", "cat24", "cat25", "cat26", "cat27", "cat28", "cat29",
    "cat30", "cat31",
];

const fn category_weights() -> [u64; CATEGORY_COUNT] {
    let mut w = [0u64; CATEGORY_COUNT];
    let mut i = 0;
    while i < CATEGORY_COUNT {
        w[i] = 1_000_000 / (i as u64 + 1);
        i += 1;
    }
    w
}

/// Zipf(s=1) weights over the categories: `cat00` holds ~24.6% of passages,
/// `cat31` ~0.77%. Skewed on purpose, so tag equality alone spans a 32x range
/// of selectivities.
pub const CATEGORY_WEIGHTS: [u64; CATEGORY_COUNT] = category_weights();

const fn category_total() -> u64 {
    let mut t = 0;
    let mut i = 0;
    while i < CATEGORY_COUNT {
        t += CATEGORY_WEIGHTS[i];
        i += 1;
    }
    t
}

/// Sum of [`CATEGORY_WEIGHTS`].
pub const CATEGORY_WEIGHT_TOTAL: u64 = category_total();

/// One passage's derived values, compactly (16 bytes, so 10M rows fit in
/// 160 MB for counting).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DerivedFields {
    /// Index into [`CATEGORY_NAMES`].
    pub category: u8,
    pub doc_len: u32,
    pub tier: u8,
    pub rand_key: u64,
}

impl DerivedFields {
    /// The derivation. See the module docs for the rule and why it is shaped
    /// this way.
    pub fn from_docid(docid: &str) -> Self {
        let mut h = Sha256::new();
        h.update(DERIVATION_SALT);
        h.update(docid.as_bytes());
        let d = h.finalize();
        let word = |i: usize| {
            let mut b = [0u8; 8];
            b.copy_from_slice(&d[8 * i..8 * i + 8]);
            u64::from_be_bytes(b)
        };
        let (w0, w1, w2, w3) = (word(0), word(1), word(2), word(3));

        let mut r = w0 % CATEGORY_WEIGHT_TOTAL;
        let mut category = 0usize;
        while r >= CATEGORY_WEIGHTS[category] {
            r -= CATEGORY_WEIGHTS[category];
            category += 1;
        }

        let e = 4 + (w1 % 12) as u32;
        let doc_len = (1u64 << e) + (w1 >> 32) % (1u64 << e);

        Self {
            category: category as u8,
            doc_len: doc_len as u32,
            tier: (w2 % TIER_COUNT) as u8,
            rand_key: w3 >> 24,
        }
    }

    pub fn category_name(&self) -> &'static str {
        CATEGORY_NAMES[self.category as usize]
    }

    /// The derived fields as the JSON values written into `payloads.jsonl`.
    pub fn to_json_fields(&self) -> [(&'static str, Value); 4] {
        [
            ("category", json!(self.category_name())),
            ("doc_len", json!(self.doc_len)),
            ("tier", json!(self.tier)),
            ("rand_key", json!(self.rand_key)),
        ]
    }

    /// Add the derived fields to a payload object.
    ///
    /// A payload that already holds one of the names with a DIFFERENT value is
    /// an error: either upstream grew a field of that name (and overwriting it
    /// would silently replace real metadata) or the payload was prepared under
    /// another derivation. The same value is accepted, so re-deriving an
    /// already-derived payload is idempotent.
    pub fn insert_into(&self, payload: &mut Map<String, Value>) -> Result<(), String> {
        for (name, value) in self.to_json_fields() {
            if let Some(existing) = payload.get(name) {
                if *existing != value {
                    return Err(format!(
                        "payload already holds {name}={existing} but the derivation gives \
                         {value}; refusing to overwrite (different derivation, or an upstream \
                         field with the same name)"
                    ));
                }
            }
            payload.insert(name.to_string(), value);
        }
        Ok(())
    }
}

/// Derive from a payload's `docid` and check that every derived field the
/// payload stores equals the derivation. Missing fields are an error naming the
/// cure: the directory predates the derived fields.
pub fn check_payload(payload: &Value) -> Result<DerivedFields, String> {
    let obj = payload
        .as_object()
        .ok_or("payload line is not a JSON object")?;
    let docid = obj
        .get("docid")
        .and_then(Value::as_str)
        .ok_or("payload has no string 'docid'")?;
    let derived = DerivedFields::from_docid(docid);
    for (name, value) in derived.to_json_fields() {
        match obj.get(name) {
            None => {
                return Err(format!(
                    "payload for docid {docid:?} has no derived field {name:?}: this corpus was \
                     prepared before the derived fields existed. Upgrade it in place with \
                     `prepare-msmarco --add-derived-fields`."
                ))
            }
            Some(stored) if *stored != value => {
                return Err(format!(
                    "payload for docid {docid:?} stores {name}={stored} but the derivation gives \
                     {value}; the corpus was prepared under a different derivation"
                ))
            }
            Some(_) => {}
        }
    }
    Ok(derived)
}

// ---------------------------------------------------------------------------
// Predicate evaluation
// ---------------------------------------------------------------------------

/// A field's value, as a predicate sees it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FieldValue<'a> {
    Keyword(&'a str),
    Int(i64),
    /// Present, but not a type the exact-match grammar compares (float, bool,
    /// array, object). Evaluating a predicate against it is an error.
    Other,
    /// A name the row type cannot represent (the compact derived row has only
    /// the four derived fields). Not absence: evaluating against it is an
    /// error, so a predicate on `url` or a misspelt `catgory` cannot quietly
    /// count zero matches.
    Unknown,
}

/// Anything a predicate can be evaluated against: the compact derived row, or
/// a full payload object (so #341 can also filter on real metadata).
pub trait Fields {
    /// `None` when the field is absent; an absent field matches no leaf, which
    /// is what both engines do for a document without the field.
    fn field(&self, name: &str) -> Option<FieldValue<'_>>;
}

impl Fields for DerivedFields {
    fn field(&self, name: &str) -> Option<FieldValue<'_>> {
        Some(match name {
            "category" => FieldValue::Keyword(self.category_name()),
            "doc_len" => FieldValue::Int(self.doc_len as i64),
            "tier" => FieldValue::Int(self.tier as i64),
            "rand_key" => FieldValue::Int(self.rand_key as i64),
            _ => FieldValue::Unknown,
        })
    }
}

impl Fields for Map<String, Value> {
    fn field(&self, name: &str) -> Option<FieldValue<'_>> {
        Some(match self.get(name)? {
            Value::Null => return None,
            Value::String(s) => FieldValue::Keyword(s),
            v => v.as_i64().map(FieldValue::Int).unwrap_or(FieldValue::Other),
        })
    }
}

/// A `match` operand.
#[derive(Debug, Clone, PartialEq)]
pub enum Operand {
    Keyword(String),
    Int(i64),
}

/// One compiled leaf operation.
#[derive(Debug, Clone, PartialEq)]
pub enum Leaf {
    /// `match.value` (one operand) or `match.any` (several).
    Match(Vec<Operand>),
    Range {
        gt: Option<f64>,
        gte: Option<f64>,
        lt: Option<f64>,
        lte: Option<f64>,
    },
}

/// A `conditions` value compiled once and evaluated per row.
#[derive(Debug, Clone, PartialEq)]
pub enum Predicate {
    And(Vec<Predicate>),
    Or(Vec<Predicate>),
    Leaf { field: String, leaf: Leaf },
}

/// Largest integer every engine compares exactly (Redis NUMERIC is binary64).
const MAX_EXACT_INTEGER: i64 = 1 << 53;

fn operand(field: &str, v: &Value) -> Result<Operand, String> {
    if let Some(s) = v.as_str() {
        return Ok(Operand::Keyword(s.to_string()));
    }
    match v.as_i64() {
        Some(n) if n.unsigned_abs() <= MAX_EXACT_INTEGER as u64 => Ok(Operand::Int(n)),
        _ => Err(format!(
            "{field}: match operands must be strings or integers within +/-2^53, got {v}"
        )),
    }
}

impl Predicate {
    /// Compile a `conditions` value. The top level must be a group, as every
    /// engine adapter expects.
    pub fn compile(conditions: &Value) -> Result<Self, String> {
        let obj = conditions
            .as_object()
            .ok_or("conditions must be a JSON object")?;
        if !(obj.contains_key("and") || obj.contains_key("or")) {
            return Err("conditions must be an `and`/`or` group at the top level".into());
        }
        Self::node(conditions)
    }

    fn node(v: &Value) -> Result<Self, String> {
        let obj = v
            .as_object()
            .ok_or_else(|| format!("condition entry is not an object: {v}"))?;
        if obj.len() != 1 {
            return Err(format!(
                "a condition object must hold exactly one key (one group, or one field with one \
                 operation); got {} keys — multi-key objects mean different things in different \
                 engine builders",
                obj.len()
            ));
        }
        let (key, body) = obj.iter().next().expect("len checked");
        match key.as_str() {
            "and" | "or" => {
                let entries = body
                    .as_array()
                    .filter(|a| !a.is_empty())
                    .ok_or_else(|| format!("`{key}` must be a nonempty array"))?;
                let children = entries.iter().map(Self::node).collect::<Result<_, _>>()?;
                Ok(if key == "and" {
                    Predicate::And(children)
                } else {
                    Predicate::Or(children)
                })
            }
            field => Ok(Predicate::Leaf {
                field: field.to_string(),
                leaf: Self::leaf(field, body)?,
            }),
        }
    }

    fn leaf(field: &str, body: &Value) -> Result<Leaf, String> {
        let ops = body
            .as_object()
            .filter(|o| o.len() == 1)
            .ok_or_else(|| format!("{field}: a leaf must hold exactly one operation"))?;
        let (op, criteria) = ops.iter().next().expect("len checked");
        match op.as_str() {
            "match" => {
                let c = criteria
                    .as_object()
                    .filter(|c| c.len() == 1)
                    .ok_or_else(|| format!("{field}: match needs exactly one of value/any"))?;
                if let Some(v) = c.get("value") {
                    Ok(Leaf::Match(vec![operand(field, v)?]))
                } else if let Some(any) = c.get("any") {
                    let any = any
                        .as_array()
                        .filter(|a| !a.is_empty())
                        .ok_or_else(|| format!("{field}: match.any must be a nonempty array"))?;
                    Ok(Leaf::Match(
                        any.iter()
                            .map(|v| operand(field, v))
                            .collect::<Result<_, _>>()?,
                    ))
                } else {
                    Err(format!(
                        "{field}: match supports only value/any here (full-text `text` is not an \
                         exact-match predicate)"
                    ))
                }
            }
            "range" => {
                let b = criteria
                    .as_object()
                    .filter(|b| !b.is_empty())
                    .ok_or_else(|| format!("{field}: range needs at least one bound"))?;
                let mut out = (None, None, None, None);
                for (k, v) in b {
                    let n = v
                        .as_f64()
                        .filter(|n| n.is_finite() && n.abs() <= MAX_EXACT_INTEGER as f64)
                        .ok_or_else(|| format!("{field}: range bound {k} must be a number"))?;
                    let slot = match k.as_str() {
                        "gt" => &mut out.0,
                        "gte" => &mut out.1,
                        "lt" => &mut out.2,
                        "lte" => &mut out.3,
                        other => return Err(format!("{field}: unknown range bound {other:?}")),
                    };
                    *slot = Some(n);
                }
                if (out.0.is_some() && out.1.is_some()) || (out.2.is_some() && out.3.is_some()) {
                    return Err(format!(
                        "{field}: range permits at most one lower and one upper bound"
                    ));
                }
                Ok(Leaf::Range {
                    gt: out.0,
                    gte: out.1,
                    lt: out.2,
                    lte: out.3,
                })
            }
            other => Err(format!("{field}: unsupported operation {other:?}")),
        }
    }

    /// Evaluate against one row. Type mismatches (a keyword operand against an
    /// integer field, a range on a keyword) are errors, not `false`: a count
    /// computed by quietly treating them as non-matches is a count for a
    /// predicate no engine would run.
    pub fn matches<F: Fields + ?Sized>(&self, row: &F) -> Result<bool, String> {
        match self {
            Predicate::And(c) => {
                for p in c {
                    if !p.matches(row)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            Predicate::Or(c) => {
                for p in c {
                    if p.matches(row)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            Predicate::Leaf { field, leaf } => {
                let Some(value) = row.field(field) else {
                    return Ok(false);
                };
                match (leaf, value) {
                    (Leaf::Match(ops), FieldValue::Keyword(s)) => {
                        let mut hit = false;
                        for op in ops {
                            match op {
                                Operand::Keyword(k) => hit |= k == s,
                                Operand::Int(_) => {
                                    return Err(format!(
                                        "{field}: integer operand against a keyword field"
                                    ))
                                }
                            }
                        }
                        Ok(hit)
                    }
                    (Leaf::Match(ops), FieldValue::Int(n)) => {
                        let mut hit = false;
                        for op in ops {
                            match op {
                                Operand::Int(k) => hit |= *k == n,
                                Operand::Keyword(_) => {
                                    return Err(format!(
                                        "{field}: keyword operand against an integer field"
                                    ))
                                }
                            }
                        }
                        Ok(hit)
                    }
                    (Leaf::Range { gt, gte, lt, lte }, FieldValue::Int(n)) => {
                        if n.unsigned_abs() > MAX_EXACT_INTEGER as u64 {
                            return Err(format!("{field}: value {n} is outside +/-2^53"));
                        }
                        let x = n as f64;
                        Ok(gt.is_none_or(|b| x > b)
                            && gte.is_none_or(|b| x >= b)
                            && lt.is_none_or(|b| x < b)
                            && lte.is_none_or(|b| x <= b))
                    }
                    (Leaf::Range { .. }, FieldValue::Keyword(_)) => {
                        Err(format!("{field}: range predicate on a keyword field"))
                    }
                    (_, FieldValue::Other) => Err(format!(
                        "{field}: holds a value the exact-match grammar cannot compare"
                    )),
                    (_, FieldValue::Unknown) => Err(unknown_field(field)),
                }
            }
        }
    }
}

fn unknown_field(field: &str) -> String {
    format!(
        "{field}: not a field of this row type (the compact derived row holds only {})",
        DERIVED_FIELDS
            .iter()
            .map(|(n, _)| *n)
            .collect::<Vec<_>>()
            .join(", ")
    )
}

/// Every leaf field of `p` must be one the row type can represent. Checked up
/// front because `and`/`or` short-circuit: a disjunct naming an unknown field
/// after one every row satisfies would otherwise never be evaluated.
fn check_fields_known<F: Fields>(rows: &[F], p: &Predicate) -> Result<(), String> {
    let Some(row) = rows.first() else {
        return Ok(());
    };
    match p {
        Predicate::And(c) | Predicate::Or(c) => {
            c.iter().try_for_each(|q| check_fields_known(rows, q))
        }
        Predicate::Leaf { field, .. } => match row.field(field) {
            Some(FieldValue::Unknown) => Err(unknown_field(field)),
            _ => Ok(()),
        },
    }
}

/// How many rows satisfy `p`. Parallel; the count does not depend on the
/// thread count.
pub fn count_matches<F: Fields + Sync>(rows: &[F], p: &Predicate) -> Result<u64, String> {
    check_fields_known(rows, p)?;
    rows.par_iter()
        .map(|r| p.matches(r).map(u64::from))
        .try_reduce(|| 0, |a, b| Ok(a + b))
}

/// The ids (positions in `rows`, which are the corpus ids) satisfying `p`, in
/// ascending order. What a filtered brute-force ground truth (#341) iterates.
pub fn matching_ids<F: Fields + Sync>(rows: &[F], p: &Predicate) -> Result<Vec<u64>, String> {
    check_fields_known(rows, p)?;
    let hits: Vec<Option<u64>> = rows
        .par_iter()
        .enumerate()
        .map(|(i, r)| Ok(p.matches(r)?.then_some(i as u64)))
        .collect::<Result<_, String>>()?;
    Ok(hits.into_iter().flatten().collect())
}

// ---------------------------------------------------------------------------
// The filter-only query set
// ---------------------------------------------------------------------------

/// The query shapes `filter_queries.jsonl` covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Shape {
    TagEq,
    NumericPoint,
    NumericRange,
    RepeatedValueRange,
    TagAndRange,
    RangeOrRange,
}

impl Shape {
    pub const ALL: [Shape; 6] = [
        Shape::TagEq,
        Shape::NumericPoint,
        Shape::NumericRange,
        Shape::RepeatedValueRange,
        Shape::TagAndRange,
        Shape::RangeOrRange,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Shape::TagEq => "tag_eq",
            Shape::NumericPoint => "numeric_point",
            Shape::NumericRange => "numeric_range",
            Shape::RepeatedValueRange => "repeated_value_range",
            Shape::TagAndRange => "tag_and_range",
            Shape::RangeOrRange => "range_or_range",
        }
    }
}

/// How many `rand_key` point lookups the set carries.
pub const POINT_LOOKUP_KEYS: usize = 5;

/// Whether a passage's `rand_key` may serve as a point-lookup operand: it is in
/// the SMALLEST registered variant. The variants nest, so such a passage is in
/// every variant, and the first [`POINT_LOOKUP_KEYS`] of them in corpus order
/// are the same passages whichever variant is being prepared — making the
/// point-lookup rows identical across sizes and guaranteed to match >= 1 row.
pub fn is_point_lookup_source(docid: &str) -> bool {
    super::crc32_bucket(docid) < super::VARIANTS[0].threshold
}

/// One query of the set, before its expected count is known.
#[derive(Debug, Clone, PartialEq)]
pub struct FilterQueryTemplate {
    pub shape: Shape,
    pub conditions: Value,
}

fn leaf(field: &str, op: Value) -> Value {
    json!({ field: op })
}
fn eq(field: &str, v: Value) -> Value {
    leaf(field, json!({"match": {"value": v}}))
}
fn range(field: &str, bounds: Value) -> Value {
    leaf(field, json!({ "range": bounds }))
}
fn and(entries: Vec<Value>) -> Value {
    json!({ "and": entries })
}
fn or(entries: Vec<Value>) -> Value {
    json!({ "or": entries })
}
/// The `rand_key` value below which `per_million` / 1e6 of the key space lies.
/// Multiplies first (2^40 * 1e6 fits a u64 easily) so 500_000 gives exactly
/// 2^39, not a value truncated by dividing first.
fn rand_key_at(per_million: u64) -> u64 {
    RAND_KEY_SPAN * per_million / 1_000_000
}

/// `rand_key < fraction * RAND_KEY_SPAN`, i.e. a range of that selectivity.
fn rand_key_below(per_million: u64) -> Value {
    range("rand_key", json!({"lt": rand_key_at(per_million)}))
}

/// The query set. Every predicate except the `rand_key` point lookups is a
/// constant, so the set is the same for every variant and only the counts
/// differ. Approximate selectivities (by construction) are noted per row; the
/// realized ones are what `expected_count` records.
///
/// # Why the zero-count guard cannot newly fire on the 1M / 10M variants
///
/// [`DerivedCorpus::filter_query_rows`] refuses a query that matches nothing.
/// The rarest constant predicate, `doc_len == 3000`, expects ~4 rows at 100K
/// (P(0) ~ 1.6%), and it realized 6. Two facts make that result carry over to
/// the larger sizes. The variants nest (crc32 thresholds 886 < 8805 < 88075),
/// so every 100K passage is also in 1M and 10M. The derived values depend only
/// on `docid`, so each such passage carries the same values there. Every
/// predicate is therefore satisfied by a superset of its 100K matches, and its
/// count can only grow. The `rand_key` lookups use the same keys at every
/// size (see [`is_point_lookup_source`]), so they match >= 1 row everywhere.
/// The guard stays a hard error regardless: if an upstream re-export broke
/// the nesting, a regeneration would fail rather than publish a zero-count row.
pub fn filter_query_templates(lookup_keys: &[u64]) -> Result<Vec<FilterQueryTemplate>, String> {
    if lookup_keys.len() != POINT_LOOKUP_KEYS {
        return Err(format!(
            "need {POINT_LOOKUP_KEYS} point-lookup keys, got {} — the corpus holds fewer \
             passages from the smallest variant than the query set requires",
            lookup_keys.len()
        ));
    }
    let t = |shape, conditions| FilterQueryTemplate { shape, conditions };
    let mut out = Vec::new();

    // Tag equality: Zipf weights give ~24.6%, 12.3%, 6.2%, 3.1%, 1.5%, 0.77%.
    for c in [0usize, 1, 3, 7, 15, 31] {
        out.push(t(
            Shape::TagEq,
            and(vec![eq("category", json!(CATEGORY_NAMES[c]))]),
        ));
    }
    // Numeric point lookups. doc_len = v has probability 1/12 / 2^floor(log2 v):
    // 16 ~0.52%, 200 ~0.065%, 3000 ~0.0041%. rand_key: ~1 row each.
    for v in [16u32, 200, 3000] {
        out.push(t(Shape::NumericPoint, and(vec![eq("doc_len", json!(v))])));
    }
    for k in lookup_keys {
        out.push(t(Shape::NumericPoint, and(vec![eq("rand_key", json!(k))])));
    }
    // Numeric ranges on the high-cardinality field: 50%, 10%, 1%, 0.1%, 0.01%;
    // and a skewed-field range (doc_len in [1024, 4096): ~2/12 = 16.7%).
    for per_million in [500_000u64, 100_000, 10_000, 1_000, 100] {
        out.push(t(
            Shape::NumericRange,
            and(vec![rand_key_below(per_million)]),
        ));
    }
    out.push(t(
        Shape::NumericRange,
        and(vec![range("doc_len", json!({"gte": 1024, "lt": 4096}))]),
    ));
    // Ranges over a field where every value repeats ~N/10 times: 90%, 50%,
    // 20%, 10%.
    for (lo, hi) in [(0, 8), (0, 4), (2, 3), (5, 5)] {
        out.push(t(
            Shape::RepeatedValueRange,
            and(vec![range("tier", json!({"gte": lo, "lte": hi}))]),
        ));
    }
    // Tag AND range: ~2.5%, ~1.5%, ~1.0%, ~0.0077%.
    out.push(t(
        Shape::TagAndRange,
        and(vec![
            eq("category", json!("cat00")),
            rand_key_below(100_000),
        ]),
    ));
    out.push(t(
        Shape::TagAndRange,
        and(vec![
            eq("category", json!("cat07")),
            range("tier", json!({"gte": 0, "lte": 4})),
        ]),
    ));
    out.push(t(
        Shape::TagAndRange,
        and(vec![
            eq("category", json!("cat03")),
            range("doc_len", json!({"gte": 16, "lt": 64})),
        ]),
    ));
    out.push(t(
        Shape::TagAndRange,
        and(vec![eq("category", json!("cat31")), rand_key_below(10_000)]),
    ));
    // Range OR range: ~2% (two disjoint tails of one field), ~26.7% (two
    // fields), ~8.4% (two fields, one very selective).
    out.push(t(
        Shape::RangeOrRange,
        or(vec![
            rand_key_below(10_000),
            range("rand_key", json!({"gte": rand_key_at(990_000)})),
        ]),
    ));
    out.push(t(
        Shape::RangeOrRange,
        or(vec![
            range("tier", json!({"lte": 1})),
            range("doc_len", json!({"gte": 32768})),
        ]),
    ));
    out.push(t(
        Shape::RangeOrRange,
        or(vec![
            range("doc_len", json!({"lt": 32})),
            rand_key_below(1_000),
        ]),
    ));
    Ok(out)
}

/// The schema object the derived fields are registered under, for
/// `filter_only::domain::validate_predicate`.
pub fn derived_schema() -> Value {
    Value::Object(
        DERIVED_FIELDS
            .iter()
            .map(|(n, t)| (n.to_string(), json!(t)))
            .collect(),
    )
}

/// Every passage's derived values, in corpus-id order, plus the point-lookup
/// keys picked while streaming them.
#[derive(Debug, Default, Clone)]
pub struct DerivedCorpus {
    rows: Vec<DerivedFields>,
    lookup_keys: Vec<u64>,
}

impl DerivedCorpus {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append the passage with this `docid` (the next corpus id).
    pub fn push_docid(&mut self, docid: &str) -> DerivedFields {
        let d = DerivedFields::from_docid(docid);
        if self.lookup_keys.len() < POINT_LOOKUP_KEYS && is_point_lookup_source(docid) {
            self.lookup_keys.push(d.rand_key);
        }
        self.rows.push(d);
        d
    }

    /// Append a stored payload, checking its derived fields against the
    /// derivation ([`check_payload`]).
    pub fn push_payload(&mut self, payload: &Value) -> Result<DerivedFields, String> {
        check_payload(payload)?;
        let docid = payload["docid"]
            .as_str()
            .ok_or("payload has no string 'docid'")?;
        Ok(self.push_docid(docid))
    }

    pub fn rows(&self) -> &[DerivedFields] {
        &self.rows
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    pub fn lookup_keys(&self) -> &[u64] {
        &self.lookup_keys
    }

    /// The `filter_queries.jsonl` rows for this corpus, each with its exact
    /// expected count.
    ///
    /// Errors (rather than writing a weaker file) when a predicate falls outside
    /// the shared Redis/OpenSearch scalar domain, when any query matches zero
    /// passages — a zero expected count cannot tell a correct engine from one
    /// that returns nothing — or when the set fails to span at least 25% down
    /// to 0.1% selectivity.
    pub fn filter_query_rows(&self) -> Result<Vec<Value>, String> {
        let n = self.rows.len() as u64;
        if n == 0 {
            return Err("cannot build filter queries over an empty corpus".into());
        }
        let schema = derived_schema();
        let templates = filter_query_templates(&self.lookup_keys)?;
        let mut out = Vec::with_capacity(templates.len());
        let (mut widest, mut narrowest) = (0u64, u64::MAX);
        for (i, t) in templates.iter().enumerate() {
            let query_id = format!("msmarco-filter-{i:03}");
            crate::filter_only::domain::validate_predicate(&t.conditions, Some(&schema))
                .map_err(|e| format!("{query_id}: {e}"))?;
            let p = Predicate::compile(&t.conditions).map_err(|e| format!("{query_id}: {e}"))?;
            let count = count_matches(&self.rows, &p)?;
            if count == 0 {
                return Err(format!(
                    "{query_id} ({}) matches no passage of this {n}-passage corpus; a zero \
                     expected count cannot distinguish a correct engine from an empty result",
                    t.shape.as_str()
                ));
            }
            widest = widest.max(count);
            narrowest = narrowest.min(count);
            out.push(json!({
                "query_id": query_id,
                "shape": t.shape.as_str(),
                "conditions": t.conditions,
                "expected_count": count,
                "corpus_size": n,
            }));
        }
        check_spread(narrowest, widest, n)?;
        Ok(out)
    }
}

/// The query set must reach from at least 25% of the corpus down to at most
/// 0.1% of it; a set that does not measures one selectivity regime under six
/// shape names.
pub fn check_spread(narrowest: u64, widest: u64, n: u64) -> Result<(), String> {
    if widest.saturating_mul(4) < n || narrowest.saturating_mul(1000) > n {
        return Err(format!(
            "filter query selectivities span only {narrowest}..{widest} of {n} passages; the \
             set must reach >= 25% and <= 0.1%"
        ));
    }
    Ok(())
}

/// What [`check_filter_queries`] confirmed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FilterQuerySummary {
    pub queries: usize,
    pub min_count: u64,
    pub max_count: u64,
}

/// Compare a `filter_queries.jsonl` read from disk against the rows regenerated
/// from the corpus. Exact equality, row by row: a changed predicate and a
/// changed count are both errors, reported separately so the cause is clear.
pub fn check_filter_queries(
    expected: &[Value],
    found: &[Value],
) -> Result<FilterQuerySummary, String> {
    if expected.len() != found.len() {
        return Err(format!(
            "filter_queries.jsonl holds {} rows; the corpus regenerates {}",
            found.len(),
            expected.len()
        ));
    }
    for (i, (e, f)) in expected.iter().zip(found).enumerate() {
        for key in ["query_id", "shape", "conditions", "corpus_size"] {
            if e.get(key) != f.get(key) {
                return Err(format!(
                    "filter_queries.jsonl row {i}: {key} is {} but the derivation regenerates {} \
                     — the query set was built under different rules or for another corpus",
                    f.get(key).unwrap_or(&Value::Null),
                    e.get(key).unwrap_or(&Value::Null),
                ));
            }
        }
        if e.get("expected_count") != f.get("expected_count") {
            return Err(format!(
                "filter_queries.jsonl {}: expected_count is {} but {} passages of this corpus \
                 satisfy the predicate",
                e["query_id"],
                f.get("expected_count").unwrap_or(&Value::Null),
                e["expected_count"]
            ));
        }
        if e != f {
            return Err(format!(
                "filter_queries.jsonl row {i} carries keys the derivation does not produce"
            ));
        }
    }
    let counts = expected.iter().filter_map(|r| r["expected_count"].as_u64());
    Ok(FilterQuerySummary {
        queries: expected.len(),
        min_count: counts.clone().min().unwrap_or(0),
        max_count: counts.max().unwrap_or(0),
    })
}

/// File name of the filter-only query set inside a prepared directory.
pub const FILTER_QUERIES_FILE: &str = "filter_queries.jsonl";

/// Refuse a local MS MARCO directory prepared before the derived fields existed
/// when the registry already declares them.
///
/// The benchmark skips the download whenever the dataset directory exists. A
/// host that extracted an older tarball therefore keeps payloads without
/// `category`/`doc_len`/`tier`/`rand_key`, while the schema tells every engine
/// to build those columns. The result is columns no document fills (Milvus
/// stores an absent VarChar as `""`), and a filter run scores 0 hits as if they
/// were real. Cheap by design, so it can run on every dataset access:
///
/// * `filter_queries.jsonl` must exist. `--add-derived-fields` renames it LAST,
///   so its presence means the upgraded payloads were already in place.
/// * The FIRST payload line must carry every derived field with the derived
///   value ([`check_payload`]). That catches a hand-copied or mixed directory.
///   It does not scan the whole file; `prepare-msmarco --verify` does.
///
/// Does nothing for datasets that are not registered MS MARCO variants, or
/// whose schema declares none of the derived fields.
pub fn check_prepared_dir(
    dataset_name: &str,
    schema: Option<&Value>,
    dir: &std::path::Path,
) -> Result<(), String> {
    use std::io::BufRead as _;
    if super::variant(dataset_name).is_none() {
        return Ok(());
    }
    let declares = schema
        .and_then(Value::as_object)
        .is_some_and(|s| DERIVED_FIELDS.iter().any(|(f, _)| s.contains_key(*f)));
    if !declares {
        return Ok(());
    }
    let remedy = format!(
        "This directory predates the derived exact-match fields (#340). Either upgrade it in \
         place with `prepare-msmarco --dataset {dataset_name} --add-derived-fields --out-dir \
         <datasets dir>`, or delete {} so the current tarball is downloaded again.",
        dir.display()
    );
    if !dir.join(FILTER_QUERIES_FILE).exists() {
        return Err(format!(
            "{dataset_name}: {} has no {FILTER_QUERIES_FILE}, but the registry declares the \
             derived fields. {remedy}",
            dir.display()
        ));
    }
    let payloads = dir.join("payloads.jsonl");
    let f = std::fs::File::open(&payloads)
        .map_err(|e| format!("{dataset_name}: open {}: {e}", payloads.display()))?;
    let mut first = String::new();
    std::io::BufReader::new(f)
        .read_line(&mut first)
        .map_err(|e| format!("{dataset_name}: read {}: {e}", payloads.display()))?;
    let payload: Value = serde_json::from_str(first.trim_end()).map_err(|e| {
        format!(
            "{dataset_name}: first line of {} is not JSON: {e}",
            payloads.display()
        )
    })?;
    check_payload(&payload).map_err(|e| format!("{dataset_name}: {e} {remedy}"))?;
    Ok(())
}

/// The `derived_fields` block recorded in `PREPARED.json`.
pub fn manifest_entry(corpus: &DerivedCorpus, rows: &[Value]) -> Value {
    let mut per_category = vec![0u64; CATEGORY_COUNT];
    for r in corpus.rows() {
        per_category[r.category as usize] += 1;
    }
    json!({
        "derived_fields": {
            "rule": "d = sha256(b\"vdb-msmarco-derived/v1\\0\" + docid); w[i] = big-endian u64 of \
                     d[8i..8i+8]; category = zipf_bucket(w0 % sum(W)), W[i] = 1000000 // (i+1), \
                     i < 32, named cat00..cat31; e = 4 + w1 % 12, doc_len = 2^e + (w1 >> 32) % \
                     2^e; tier = w2 % 10; rand_key = w3 >> 24",
            "note": "synthetic labels, a pure function of docid; not properties of the text",
            "fields": DERIVED_FIELDS.iter().map(|(n, t)| json!({"field": n, "schema_type": t})).collect::<Vec<_>>(),
            "category_counts": CATEGORY_NAMES.iter().zip(&per_category)
                .map(|(n, c)| (n.to_string(), json!(c))).collect::<Map<String, Value>>(),
        },
        "filter_queries": {
            "file": "filter_queries.jsonl",
            "queries": rows.len(),
            "point_lookup_keys": corpus.lookup_keys(),
            "expected_counts": rows.iter().map(|r| json!({
                "query_id": r["query_id"], "shape": r["shape"], "expected_count": r["expected_count"],
            })).collect::<Vec<_>>(),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Values computed by the five-line Python rule in the module docs
    /// (`hashlib.sha256`), so the derivation is pinned across languages and a
    /// refactor cannot drift it without this going red.
    #[test]
    fn derivation_matches_the_python_reference() {
        for (docid, category, doc_len, tier, rand_key) in [
            (
                "msmarco_v2.1_doc_00_0#0_0",
                14u8,
                4476u32,
                1u8,
                923_388_215_550u64,
            ),
            (
                "msmarco_v2.1_doc_09_894803720#6_1517223658",
                1,
                17,
                3,
                432_755_501_277,
            ),
            ("", 20, 10218, 5, 314_801_767_012),
            ("hello", 5, 62, 9, 30_441_861_297),
        ] {
            assert_eq!(
                DerivedFields::from_docid(docid),
                DerivedFields {
                    category,
                    doc_len,
                    tier,
                    rand_key
                },
                "docid {docid:?}"
            );
        }
        assert_eq!(CATEGORY_WEIGHT_TOTAL, 4_058_487);
    }

    #[test]
    fn category_names_are_the_zero_padded_indices() {
        for (i, n) in CATEGORY_NAMES.iter().enumerate() {
            assert_eq!(*n, format!("cat{i:02}"));
        }
    }

    fn synthetic_docids(n: usize) -> Vec<String> {
        (0..n)
            .map(|i| format!("msmarco_v2.1_doc_{:02}_{}#{}_{}", i % 60, i * 7, i % 13, i))
            .collect()
    }

    #[test]
    fn derived_values_stay_in_their_declared_ranges_and_follow_their_distributions() {
        let docids = synthetic_docids(200_000);
        let rows: Vec<_> = docids
            .iter()
            .map(|d| DerivedFields::from_docid(d))
            .collect();
        let mut cats = [0u64; CATEGORY_COUNT];
        let mut tiers = [0u64; TIER_COUNT as usize];
        for r in &rows {
            assert!((DOC_LEN_MIN..=DOC_LEN_MAX).contains(&r.doc_len), "{r:?}");
            assert!(r.rand_key < RAND_KEY_SPAN);
            cats[r.category as usize] += 1;
            tiers[r.tier as usize] += 1;
        }
        let n = rows.len() as f64;
        // Zipf: cat00 ~24.6%, cat31 ~0.77%, and monotone-ish decreasing.
        let share = |c: usize| cats[c] as f64 / n;
        assert!((share(0) - 0.2464).abs() < 0.01, "{}", share(0));
        assert!((share(31) - 0.0077).abs() < 0.002, "{}", share(31));
        assert!(cats[0] > cats[1] && cats[1] > cats[3] && cats[3] > cats[15]);
        for t in tiers {
            assert!((t as f64 / n - 0.1).abs() < 0.01);
        }
        // High cardinality: essentially all distinct.
        let distinct: std::collections::HashSet<_> = rows.iter().map(|r| r.rand_key).collect();
        assert!(distinct.len() as f64 > 0.999 * n);
    }

    #[test]
    fn derivation_is_independent_of_the_crc32_sampling() {
        // Inside the 100K variant's selection (crc32 bucket < 886, 0.09% of the
        // hash space) the derived fields must still cover their full ranges —
        // the failure mode of deriving from crc32 itself.
        let docids: Vec<String> = synthetic_docids(3_000_000)
            .into_iter()
            .filter(|d| super::super::keeps(d, 886))
            .collect();
        assert!(docids.len() > 1_500, "{}", docids.len());
        let rows: Vec<_> = docids
            .iter()
            .map(|d| DerivedFields::from_docid(d))
            .collect();
        let max_key = rows.iter().map(|r| r.rand_key).max().unwrap();
        let min_key = rows.iter().map(|r| r.rand_key).min().unwrap();
        assert!(max_key > RAND_KEY_SPAN / 10 * 9 && min_key < RAND_KEY_SPAN / 10);
        let cats: std::collections::HashSet<_> = rows.iter().map(|r| r.category).collect();
        assert!(cats.len() >= 24, "{}", cats.len());
    }

    #[test]
    fn insert_is_idempotent_and_refuses_to_overwrite_a_different_value() {
        let d = DerivedFields::from_docid("x");
        let mut m = Map::new();
        d.insert_into(&mut m).unwrap();
        let once = m.clone();
        d.insert_into(&mut m).unwrap();
        assert_eq!(m, once);
        m.insert("tier".into(), json!(99));
        assert!(d.insert_into(&mut m).unwrap_err().contains("tier"));
    }

    #[test]
    fn check_payload_rejects_missing_and_drifted_fields() {
        let d = DerivedFields::from_docid("doc-a");
        let mut m = Map::new();
        m.insert("docid".into(), json!("doc-a"));
        let bare = Value::Object(m.clone());
        assert!(check_payload(&bare)
            .unwrap_err()
            .contains("--add-derived-fields"));
        d.insert_into(&mut m).unwrap();
        assert_eq!(check_payload(&Value::Object(m.clone())).unwrap(), d);
        for (name, _) in DERIVED_FIELDS {
            let mut bad = m.clone();
            bad.insert(name.to_string(), json!(-1));
            let e = check_payload(&Value::Object(bad)).unwrap_err();
            assert!(e.contains(name), "{e}");
        }
        assert!(check_payload(&json!({"url": "u"})).is_err());
    }

    fn row(category: u8, doc_len: u32, tier: u8, rand_key: u64) -> DerivedFields {
        DerivedFields {
            category,
            doc_len,
            tier,
            rand_key,
        }
    }

    #[test]
    fn predicate_semantics_for_every_operation() {
        let r = row(3, 100, 4, 1_000);
        let t = |c: Value| Predicate::compile(&c).unwrap().matches(&r).unwrap();
        assert!(t(json!({"and":[{"category":{"match":{"value":"cat03"}}}]})));
        assert!(!t(
            json!({"and":[{"category":{"match":{"value":"cat04"}}}]})
        ));
        assert!(t(
            json!({"and":[{"category":{"match":{"any":["cat01","cat03"]}}}]})
        ));
        assert!(t(json!({"and":[{"doc_len":{"match":{"value":100}}}]})));
        assert!(t(json!({"and":[{"tier":{"match":{"any":[1,4]}}}]})));
        // Bounds are honoured exactly, inclusive vs exclusive.
        assert!(t(json!({"and":[{"tier":{"range":{"gte":4,"lte":4}}}]})));
        assert!(!t(json!({"and":[{"tier":{"range":{"gt":4}}}]})));
        assert!(!t(json!({"and":[{"tier":{"range":{"lt":4}}}]})));
        assert!(t(json!({"and":[{"tier":{"range":{"gt":3.5,"lt":4.5}}}]})));
        // and / or / nesting.
        assert!(!t(
            json!({"and":[{"tier":{"match":{"value":4}}},{"doc_len":{"range":{"lt":100}}}]})
        ));
        assert!(t(
            json!({"or":[{"tier":{"match":{"value":5}}},{"doc_len":{"range":{"lte":100}}}]})
        ));
        assert!(t(
            json!({"and":[{"category":{"match":{"value":"cat03"}}},{"or":[{"tier":{"match":{"value":0}}},{"rand_key":{"range":{"gte":1000}}}]}]})
        ));
        // A field the compact row cannot represent is an error, not absence,
        // under `and` and under `or` alike, even when an earlier disjunct
        // already matches every row.
        for p in [
            json!({"and":[{"title":{"match":{"value":"x"}}}]}),
            json!({"or":[{"tier":{"match":{"value":4}}},{"catgory":{"match":{"value":"cat03"}}}]}),
        ] {
            let p = Predicate::compile(&p).unwrap();
            assert!(count_matches(std::slice::from_ref(&r), &p).is_err());
            assert!(matching_ids(std::slice::from_ref(&r), &p).is_err());
        }
        assert!(
            Predicate::compile(&json!({"and":[{"catgory":{"match":{"value":"cat03"}}}]}))
                .unwrap()
                .matches(&r)
                .is_err()
        );
        // On a payload object the same field is genuinely absent and matches
        // nothing, which is what both engines do.
        let payload: Map<String, Value> = json!({"tier": 4}).as_object().unwrap().clone();
        let p = Predicate::compile(&json!({"and":[{"title":{"match":{"value":"x"}}}]})).unwrap();
        assert!(!p.matches(&payload).unwrap());
    }

    #[test]
    fn predicates_outside_the_unambiguous_subset_are_errors() {
        for bad in [
            json!({"category":{"match":{"value":"cat00"}}}), // leaf at top level
            json!({"and":[]}),
            json!({"and":[{"tier":{"match":{"value":1}}}],"or":[{"tier":{"match":{"value":2}}}]}),
            json!({"and":[{"tier":{"match":{"value":1},"range":{"gte":0}}}]}),
            json!({"and":[{"tier":{"match":{"value":1}},"doc_len":{"match":{"value":2}}}]}),
            json!({"and":[{"title":{"match":{"text":"x"}}}]}),
            json!({"and":[{"tier":{"range":{}}}]}),
            json!({"and":[{"tier":{"range":{"gt":1,"gte":1}}}]}),
            json!({"and":[{"tier":{"range":{"between":1}}}]}),
            json!({"and":[{"tier":{"geo":{}}}]}),
            json!({"and":[{"tier":{"match":{"any":[]}}}]}),
            json!({"and":[{"tier":{"match":{"value":1.5}}}]}),
        ] {
            assert!(Predicate::compile(&bad).is_err(), "{bad}");
        }
        // Type mismatches are evaluation errors, not quiet non-matches.
        let r = row(0, 16, 0, 0);
        for bad in [
            json!({"and":[{"category":{"match":{"value":3}}}]}),
            json!({"and":[{"tier":{"match":{"value":"3"}}}]}),
            json!({"and":[{"category":{"range":{"gte":0}}}]}),
        ] {
            let p = Predicate::compile(&bad).unwrap();
            assert!(p.matches(&r).is_err(), "{bad}");
        }
    }

    #[test]
    fn payload_maps_evaluate_like_derived_rows() {
        let d = DerivedFields::from_docid("doc-b");
        let mut m = Map::new();
        m.insert("docid".into(), json!("doc-b"));
        d.insert_into(&mut m).unwrap();
        for t in filter_query_templates(&[d.rand_key; POINT_LOOKUP_KEYS]).unwrap() {
            let p = Predicate::compile(&t.conditions).unwrap();
            assert_eq!(
                p.matches(&d).unwrap(),
                p.matches(&m).unwrap(),
                "{}",
                t.conditions
            );
        }
        m.insert("tier".into(), json!(1.5));
        let p = Predicate::compile(&json!({"and":[{"tier":{"range":{"gte":0}}}]})).unwrap();
        assert!(p.matches(&m).is_err());
    }

    fn corpus(n: usize) -> DerivedCorpus {
        let mut c = DerivedCorpus::new();
        // Seed passages from the smallest variant first so point lookups exist.
        let mut seeded = 0;
        let mut i = 0usize;
        while seeded < POINT_LOOKUP_KEYS {
            let d = format!("seed-{i}");
            if is_point_lookup_source(&d) {
                c.push_docid(&d);
                seeded += 1;
            }
            i += 1;
        }
        for d in synthetic_docids(n) {
            c.push_docid(&d);
        }
        c
    }

    #[test]
    fn expected_counts_equal_a_naive_recount_and_cover_every_shape() {
        let c = corpus(100_000);
        let rows = c.filter_query_rows().unwrap();
        let shapes: std::collections::HashSet<_> =
            rows.iter().map(|r| r["shape"].as_str().unwrap()).collect();
        for s in Shape::ALL {
            assert!(shapes.contains(s.as_str()), "{}", s.as_str());
        }
        for r in &rows {
            let p = Predicate::compile(&r["conditions"]).unwrap();
            // A sequential recount through the payload-map path, independent
            // of the parallel compact-row path that produced the file.
            let naive = c
                .rows()
                .iter()
                .filter(|d| {
                    let mut m = Map::new();
                    d.insert_into(&mut m).unwrap();
                    p.matches(&m).unwrap()
                })
                .count() as u64;
            assert_eq!(r["expected_count"], json!(naive), "{}", r["query_id"]);
            assert_eq!(r["corpus_size"], json!(c.len()));
        }
        // rand_key point lookups hit their source passage.
        for r in rows.iter().filter(|r| r["shape"] == "numeric_point") {
            assert!(r["expected_count"].as_u64().unwrap() >= 1);
        }
        // Spread: from at least ~50% down to single rows.
        let counts: Vec<u64> = rows
            .iter()
            .map(|r| r["expected_count"].as_u64().unwrap())
            .collect();
        assert!(*counts.iter().max().unwrap() as f64 > 0.5 * c.len() as f64);
        assert!(*counts.iter().min().unwrap() <= 10);
    }

    #[test]
    fn two_builds_agree_byte_for_byte() {
        let a = corpus(20_000).filter_query_rows().unwrap();
        let b = corpus(20_000).filter_query_rows().unwrap();
        let ser = |v: &[Value]| {
            v.iter()
                .map(|r| serde_json::to_string(r).unwrap())
                .collect::<Vec<_>>()
                .join("\n")
        };
        assert_eq!(ser(&a), ser(&b));
    }

    #[test]
    fn point_lookup_keys_come_from_the_smallest_variant_only() {
        let mut c = DerivedCorpus::new();
        for i in 0..200_000 {
            c.push_docid(&format!("p-{i}"));
        }
        assert_eq!(c.lookup_keys().len(), POINT_LOOKUP_KEYS);
        // Computed WITHOUT `is_point_lookup_source`, so a wrong threshold in
        // it (e.g. a larger variant's, which would pick passages absent from
        // the 100K corpus) cannot pass by agreeing with itself.
        let smallest = super::super::VARIANTS
            .iter()
            .map(|v| v.threshold)
            .min()
            .unwrap();
        let sources: Vec<u64> = (0..200_000)
            .map(|i| format!("p-{i}"))
            .filter(|d| super::super::keeps(d, smallest))
            .take(POINT_LOOKUP_KEYS)
            .map(|d| DerivedFields::from_docid(&d).rand_key)
            .collect();
        assert_eq!(c.lookup_keys(), &sources[..]);
        assert!(filter_query_templates(&sources[..2]).is_err());
    }

    #[test]
    fn a_zero_count_query_is_refused() {
        // A 50-row corpus cannot give every query a nonzero count, and the
        // zero-count check fires before the spread check is reached.
        let e = corpus(50).filter_query_rows().unwrap_err();
        assert!(e.contains("matches no passage"), "{e}");
        assert!(DerivedCorpus::new().filter_query_rows().is_err());
    }

    #[test]
    fn a_narrow_selectivity_spread_is_refused() {
        // 100K corpus: 25_000 is exactly 25%, 100 exactly 0.1%.
        check_spread(100, 25_000, 100_000).unwrap();
        assert!(check_spread(100, 24_999, 100_000).is_err());
        assert!(check_spread(101, 25_000, 100_000).is_err());
    }

    #[test]
    fn check_filter_queries_catches_count_and_predicate_drift() {
        let c = corpus(20_000);
        let rows = c.filter_query_rows().unwrap();
        let s = check_filter_queries(&rows, &rows).unwrap();
        assert_eq!(s.queries, rows.len());

        let mut off_by_one = rows.clone();
        let n = off_by_one[3]["expected_count"].as_u64().unwrap();
        off_by_one[3]["expected_count"] = json!(n + 1);
        let e = check_filter_queries(&rows, &off_by_one).unwrap_err();
        assert!(e.contains("expected_count"), "{e}");

        let mut drifted = rows.clone();
        drifted[0]["conditions"] = json!({"and":[{"category":{"match":{"value":"cat02"}}}]});
        assert!(check_filter_queries(&rows, &drifted)
            .unwrap_err()
            .contains("conditions"));

        let mut extra = rows.clone();
        extra[1]["note"] = json!("x");
        assert!(check_filter_queries(&rows, &extra).is_err());

        assert!(check_filter_queries(&rows, &rows[1..]).is_err());
    }

    #[test]
    fn every_emitted_predicate_is_in_the_shared_scalar_domain() {
        assert_eq!(rand_key_at(500_000), 1 << 39);
        assert_eq!(rand_key_at(1_000_000), RAND_KEY_SPAN);
        let schema = derived_schema();
        for t in filter_query_templates(&[7; POINT_LOOKUP_KEYS]).unwrap() {
            crate::filter_only::domain::validate_predicate(&t.conditions, Some(&schema))
                .unwrap_or_else(|e| panic!("{}: {e}", t.conditions));
        }
    }

    #[test]
    fn matching_ids_agree_with_count_and_are_ascending() {
        let c = corpus(10_000);
        let p =
            Predicate::compile(&json!({"and":[{"category":{"match":{"value":"cat02"}}}]})).unwrap();
        let ids = matching_ids(c.rows(), &p).unwrap();
        assert_eq!(ids.len() as u64, count_matches(c.rows(), &p).unwrap());
        assert!(ids.windows(2).all(|w| w[0] < w[1]));
        assert!(ids.iter().all(|i| c.rows()[*i as usize].category == 2));
    }

    /// `i64::MIN.abs()` wraps back to `i64::MIN` in release builds, where CI
    /// runs, so an `abs()`-based bound check would accept it. Both the operand
    /// check and the field-value check must refuse it.
    #[test]
    fn i64_min_is_outside_the_exact_domain() {
        let e = Predicate::compile(&json!({"and":[{"tier":{"match":{"value":i64::MIN}}}]}))
            .unwrap_err();
        assert!(e.contains("2^53"), "{e}");
        let ok = MAX_EXACT_INTEGER;
        Predicate::compile(&json!({"and":[{"tier":{"match":{"value":-ok}}}]})).unwrap();
        Predicate::compile(&json!({"and":[{"tier":{"match":{"value":-ok - 1}}}]})).unwrap_err();

        let p = Predicate::compile(&json!({"and":[{"n":{"range":{"lte":0}}}]})).unwrap();
        let mut m = Map::new();
        m.insert("n".into(), json!(i64::MIN));
        assert!(p.matches(&m).is_err());
        m.insert("n".into(), json!(-ok));
        assert!(p.matches(&m).unwrap());
    }

    fn prepared_dir(fq: bool, payload: &Value) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("payloads.jsonl"),
            format!("{payload}\n{{\"docid\":\"later\"}}\n"),
        )
        .unwrap();
        if fq {
            std::fs::write(dir.path().join(FILTER_QUERIES_FILE), "{}\n").unwrap();
        }
        dir
    }

    #[test]
    fn a_prepared_dir_without_the_derived_fields_is_refused() {
        const NAME: &str = "msmarco-cohere-1024-100K-cosine";
        let schema = json!({"docid": "keyword", "category": "keyword", "rand_key": "int"});
        let old = json!({"docid": "doc-a"});
        let mut m = old.as_object().unwrap().clone();
        DerivedFields::from_docid("doc-a")
            .insert_into(&mut m)
            .unwrap();
        let new = Value::Object(m);

        // Upgraded directory: accepted.
        let d = prepared_dir(true, &new);
        check_prepared_dir(NAME, Some(&schema), d.path()).unwrap();

        // No filter_queries.jsonl: refused, naming both remedies.
        let d = prepared_dir(false, &new);
        let e = check_prepared_dir(NAME, Some(&schema), d.path()).unwrap_err();
        assert!(
            e.contains("--add-derived-fields") && e.contains("delete"),
            "{e}"
        );

        // filter_queries.jsonl present but old payloads (a mixed directory).
        let d = prepared_dir(true, &old);
        let e = check_prepared_dir(NAME, Some(&schema), d.path()).unwrap_err();
        assert!(
            e.contains("category") && e.contains("--add-derived-fields"),
            "{e}"
        );

        // A drifted value in the first payload is refused too.
        let mut drifted = new.clone();
        drifted["tier"] = json!(-1);
        let d = prepared_dir(true, &drifted);
        assert!(check_prepared_dir(NAME, Some(&schema), d.path()).is_err());

        // Not applicable: another dataset, or a schema without derived fields.
        let d = prepared_dir(false, &old);
        check_prepared_dir("glove-100-angular", Some(&schema), d.path()).unwrap();
        check_prepared_dir(NAME, Some(&json!({"docid": "keyword"})), d.path()).unwrap();
        check_prepared_dir(NAME, None, d.path()).unwrap();
    }
}
