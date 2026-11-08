//! `msmarco-disk-v1`: compound text/tag query families designed to stress
//! storage rather than CPU (cloud-benchmarks-private
//! `PLAN-disk-bound-text-tag-2026-11-08.md`).
//!
//! The `msmarco-v1` suite runs 32 queries per family on common terms and
//! low-cardinality tags, so its working set is a few posting lists that stay
//! in page cache on every engine. These families invert every one of those
//! properties, from corpus statistics only and identically for every engine:
//!
//! | family | query | matches |
//! |---|---|---|
//! | `rare-term` | one `segment` term, document frequency [`RARE_DF`] | the df |
//! | `two-rare-terms` | AND of two co-occurring `segment` terms, each df [`PAIR_DF`] | ≥1 |
//! | `rare-term-and-category` | a rare `segment` term AND the `category` of one of its passages | ≥1 |
//! | `url-eq` | `url` exact match | the passages of that URL |
//! | `docid-eq` | `docid` exact match | 1 |
//! | `deep-fetch` | one `segment` term, df [`DEEP_DF`], run with `top: 100` and `fetch` | the df |
//!
//! # Exact by construction
//!
//! Every term is a [`text_match`] term (one lowercase ASCII word, not a
//! stopword) with ZERO ambiguous passages in the whole corpus under the
//! [`text_match::classify`] rule, on the segment text as `analysis: parity`
//! uploads it (whitespace normalized, no stemming on either engine). A term
//! contained in any over-long ASCII run of the corpus is excluded too. So the
//! number of passages each engine matches is the document frequency counted
//! here, and every query carries that EXACT `expected_count`; the runner checks
//! it on every reply (the total under `total_hits: exact`, the page length
//! under `none`). Tag values are restricted to strings both engines index as
//! one exact tag (printable ASCII, no `;` — the Redis TAG separator).
//!
//! # Deterministic, uniform, never repeated
//!
//! Each family's items are ranked by a seeded SHA-256 key of the item and the
//! first `per_family + RESERVED` are kept, in key order, so any contiguous
//! slice of a family file is a uniform sample of that family's eligible pool.
//! The rare-term and rare-term-and-category term sets are disjoint;
//! deep-fetch is drawn from its own band independently (the overlap is
//! recorded in the manifest).
//! Line 0 of each family file is the connection-priming query and lines
//! 1..=[`PROBE_RESERVE`] are reserved for the Redis `WITHOUTCOUNT` probe:
//! neither ever appears in a timed window. A timed window is
//! `compound.offset` + `--queries` lines of the rest; the loader refuses a
//! window that runs past the end of the file instead of cycling.
//!
//! # Files
//!
//! `prepare-msmarco --add-disk-queries` writes `<dataset>/msmarco-disk-v1/`:
//! one `<family>.jsonl` per family (one JSON array per line, see
//! [`Family::conditions`]) and `manifest.json` (seed, df bands, pool sizes,
//! counts, expected-count statistics, per-file SHA-256, and the identity of the
//! corpus it was computed from). `PREPARED.json` is not modified, so the
//! preparation identity the existing results carry is unchanged.

use crate::compound_text::Query;
use crate::msmarco::text_match::{self, TextMatch};
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;

/// The `compound.suite` name.
pub const SUITE: &str = "msmarco-disk-v1";
/// Directory (inside the prepared dataset) holding the query files.
pub const DIR: &str = "msmarco-disk-v1";
pub const MANIFEST: &str = "manifest.json";
/// Version of the selection rules below. Bump it when any rule changes.
pub const GENERATOR: &str = "vdb-msmarco-disk/v1";
/// The only text field the families query.
pub const TEXT_FIELD: &str = "segment";
/// Document-frequency band of `rare-term` and `rare-term-and-category` terms.
pub const RARE_DF: (u64, u64) = (1, 100);
/// Document-frequency band of each `two-rare-terms` term.
pub const PAIR_DF: (u64, u64) = (100, 1000);
/// Document-frequency band of `deep-fetch` terms: the part of [`RARE_DF`]
/// above the default 10-hit page, so a `top: 100` fetch reads 11–100
/// documents per query. Uniform over the long tail, most `rare-term` terms
/// match one or two passages (on the 10M corpus 62% of eligible terms have
/// df 1), which would make a top-100 fetch no deeper than a top-10 one; and a
/// narrower band at the top (50–100) holds only ~15K eligible terms at 10M,
/// below [`MIN_PER_FAMILY`].
pub const DEEP_DF: (u64, u64) = (11, 100);
/// Line 0: the priming query. Lines 1..=PROBE_RESERVE: the probe reserve.
pub const PROBE_RESERVE: usize = 64;
/// Lines before the timed pool.
pub const RESERVED: usize = 1 + PROBE_RESERVE;
/// Every family must offer at least this many timed queries.
pub const MIN_PER_FAMILY: usize = 100_000;
/// Timed queries generated per family unless `--disk-queries-per-family` says
/// otherwise (capped by the family's eligible pool).
pub const DEFAULT_PER_FAMILY: usize = 1_000_000;
pub const DEFAULT_SEED: u64 = 1;
/// The fixed result size of `deep-fetch`.
pub const DEEP_FETCH_TOP: usize = 100;
/// Longest `url` used as a tag value.
pub const MAX_TAG_LEN: usize = 2048;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Family {
    RareTerm,
    TwoRareTerms,
    RareTermAndCategory,
    UrlEq,
    DocidEq,
    DeepFetch,
}

impl Family {
    pub const ALL: [Family; 6] = [
        Family::RareTerm,
        Family::TwoRareTerms,
        Family::RareTermAndCategory,
        Family::UrlEq,
        Family::DocidEq,
        Family::DeepFetch,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Family::RareTerm => "rare-term",
            Family::TwoRareTerms => "two-rare-terms",
            Family::RareTermAndCategory => "rare-term-and-category",
            Family::UrlEq => "url-eq",
            Family::DocidEq => "docid-eq",
            Family::DeepFetch => "deep-fetch",
        }
    }

    pub fn parse(name: &str) -> Result<Family, String> {
        Family::ALL
            .into_iter()
            .find(|f| f.name() == name)
            .ok_or_else(|| {
                format!(
                    "unknown {SUITE} family {name:?} (families: {})",
                    Family::ALL.map(Family::name).join(", ")
                )
            })
    }

    pub fn file_name(self) -> String {
        format!("{}.jsonl", self.name())
    }

    /// The document-frequency band of each text term, if the family has one.
    pub fn df_band(self) -> Option<(u64, u64)> {
        match self {
            Family::RareTerm | Family::RareTermAndCategory => Some(RARE_DF),
            Family::TwoRareTerms => Some(PAIR_DF),
            Family::DeepFetch => Some(DEEP_DF),
            Family::UrlEq | Family::DocidEq => None,
        }
    }

    /// Fields the family's conditions use (text, then tags).
    pub fn fields(self) -> &'static [&'static str] {
        match self {
            Family::RareTerm | Family::TwoRareTerms | Family::DeepFetch => &[TEXT_FIELD],
            Family::RareTermAndCategory => &[TEXT_FIELD, "category"],
            Family::UrlEq => &["url"],
            Family::DocidEq => &["docid"],
        }
    }

    /// Search-config requirements the family's name promises. `deep-fetch`
    /// is defined as the top-100 fetch; any other result size or the ids-only
    /// variant would be a different measurement under the same name.
    pub fn check_params(self, top: usize, fetch: bool) -> Result<(), String> {
        if self == Family::DeepFetch && (top != DEEP_FETCH_TOP || !fetch) {
            return Err(format!(
                "{SUITE} family deep-fetch is defined as `top: {DEEP_FETCH_TOP}` with \
                 `fulltext.fetch`; got top {top}{}",
                if fetch { "" } else { " without fetch" }
            ));
        }
        Ok(())
    }

    /// Logical conditions (the compound AST) and exact count of one line.
    pub fn conditions(self, line: &Value) -> Result<(Value, u64), String> {
        let a = line
            .as_array()
            .ok_or_else(|| format!("{SUITE} {}: line is not an array", self.name()))?;
        let s = |i: usize| -> Result<&str, String> {
            a.get(i)
                .and_then(Value::as_str)
                .ok_or_else(|| format!("{SUITE} {}: field {i} is not a string", self.name()))
        };
        let arity = match self {
            Family::RareTerm | Family::DeepFetch | Family::UrlEq | Family::DocidEq => 2,
            Family::TwoRareTerms | Family::RareTermAndCategory => 3,
        };
        if a.len() != arity {
            return Err(format!(
                "{SUITE} {}: expected {arity} fields, got {}",
                self.name(),
                a.len()
            ));
        }
        let count = a[arity - 1]
            .as_u64()
            .filter(|n| *n > 0)
            .ok_or_else(|| format!("{SUITE} {}: count must be a positive integer", self.name()))?;
        let text = |t: &str| json!({TEXT_FIELD: {"match": {"text": t}}});
        let tag = |f: &str, v: &str| json!({f: {"match": {"value": v}}});
        let conditions = match self {
            Family::RareTerm | Family::DeepFetch => text(s(0)?),
            Family::TwoRareTerms => json!({"and": [text(s(0)?), text(s(1)?)]}),
            Family::RareTermAndCategory => json!({"and": [text(s(0)?), tag("category", s(1)?)]}),
            Family::UrlEq => tag("url", s(0)?),
            Family::DocidEq => tag("docid", s(0)?),
        };
        Ok((conditions, count))
    }
}

/// A tag value both engines index and query as exactly one tag: printable
/// ASCII without space (so no edge whitespace, nothing a client or engine
/// normalizes) and without `;` (the Redis TAG separator).
pub fn is_safe_tag(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_TAG_LEN
        && value
            .bytes()
            .all(|b| (0x21..=0x7e).contains(&b) && b != b';')
}

fn sha256_hex(bytes: impl AsRef<[u8]>) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn sha256_file(path: &Path) -> Result<(String, u64), String> {
    let mut f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    let mut n = 0u64;
    loop {
        let k = f
            .read(&mut buf)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        if k == 0 {
            break;
        }
        h.update(&buf[..k]);
        n += k as u64;
    }
    Ok((h.finalize().iter().map(|b| format!("{b:02x}")).collect(), n))
}

/// The preparation identity every compound result already carries:
/// SHA-256 of `PREPARED.json`.
pub fn preparation_sha256(dir: &Path) -> Result<String, String> {
    let path = dir.join("PREPARED.json");
    std::fs::read(&path)
        .map(sha256_hex)
        .map_err(|e| format!("{SUITE}: {}: {e}", path.display()))
}

/// The seeded rank key of one item of one family.
pub fn rank_key(seed: u64, family: &str, item: &str) -> u64 {
    let mut h = Sha256::new();
    h.update(GENERATOR.as_bytes());
    h.update([0]);
    h.update(seed.to_be_bytes());
    h.update(family.as_bytes());
    h.update([0]);
    h.update(item.as_bytes());
    let d = h.finalize();
    u64::from_be_bytes(d[..8].try_into().unwrap())
}

/// The rank key of a term pair from its two terms' keys (`a` sorts first).
/// Plain integer arithmetic (SplitMix64 finalizer), so it is the same on
/// every platform.
pub fn pair_key(a: u64, b: u64) -> u64 {
    let mut z = a.rotate_left(32) ^ b.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The `cap` smallest distinct keys seen, each with how often it was seen.
///
/// The count of a key that is in the final set is exact: such a key was
/// inserted at its first occurrence (the set was not full, or the key was
/// below the maximum then, which only decreases) and was never evicted
/// (eviction is permanent: an evicted key is above every later maximum).
#[derive(Debug)]
pub struct Smallest<K: Ord + Clone> {
    cap: usize,
    map: BTreeMap<K, u64>,
}

impl<K: Ord + Clone> Smallest<K> {
    pub fn new(cap: usize) -> Self {
        Self {
            cap,
            map: BTreeMap::new(),
        }
    }
    pub fn add(&mut self, key: K) {
        if self.map.len() >= self.cap {
            match self.map.last_key_value() {
                Some((max, _)) if key > *max => return,
                _ => {}
            }
        }
        if let Some(n) = self.map.get_mut(&key) {
            *n += 1;
            return;
        }
        if self.cap == 0 {
            return;
        }
        if self.map.len() >= self.cap {
            self.map.pop_last();
        }
        self.map.insert(key, 1);
    }
    pub fn into_sorted(self) -> Vec<(K, u64)> {
        self.map.into_iter().collect()
    }
}

#[derive(Deserialize)]
struct Row<'a> {
    #[serde(borrow)]
    docid: Cow<'a, str>,
    #[serde(borrow)]
    url: Cow<'a, str>,
    #[serde(borrow)]
    category: Cow<'a, str>,
    #[serde(borrow)]
    segment: Cow<'a, str>,
}

/// What one passage contributes, computed in parallel.
struct Scanned {
    docid: String,
    url: String,
    category: String,
    /// (term, Yes?) for every term not `No` in this passage.
    terms: Vec<(String, bool)>,
    long_runs: Vec<String>,
}

fn scan_line(line: &str, row: usize) -> Result<Scanned, String> {
    let r: Row = serde_json::from_str(line)
        .map_err(|e| format!("{SUITE}: payloads.jsonl row {row}: {e}"))?;
    let text = crate::fulltext::normalize_whitespace(&r.segment);
    let mut verdicts = HashMap::new();
    let mut long_runs = Vec::new();
    text_match::doc_verdicts(&text, &mut verdicts, &mut long_runs);
    Ok(Scanned {
        docid: r.docid.into_owned(),
        url: r.url.into_owned(),
        category: r.category.into_owned(),
        terms: verdicts
            .into_iter()
            .map(|(t, v)| (t, v == TextMatch::Yes))
            .collect(),
        long_runs,
    })
}

/// Stream `payloads.jsonl` in parallel batches, in row order.
fn for_each_row(
    path: &Path,
    mut visit: impl FnMut(usize, Scanned) -> Result<(), String>,
) -> Result<(u64, String), String> {
    const BATCH: usize = 1 << 16;
    let file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut reader = BufReader::with_capacity(1 << 22, file);
    let mut hasher = Sha256::new();
    let mut bytes = 0u64;
    let mut row = 0usize;
    loop {
        let mut lines = Vec::with_capacity(BATCH);
        while lines.len() < BATCH {
            let mut line = String::new();
            let n = reader
                .read_line(&mut line)
                .map_err(|e| format!("{}: {e}", path.display()))?;
            if n == 0 {
                break;
            }
            hasher.update(line.as_bytes());
            bytes += n as u64;
            lines.push(line);
        }
        if lines.is_empty() {
            break;
        }
        let scanned: Vec<Result<Scanned, String>> = lines
            .par_iter()
            .enumerate()
            .map(|(i, l)| scan_line(l.trim_end_matches(['\n', '\r']), row + i))
            .collect();
        for s in scanned {
            visit(row, s?)?;
            row += 1;
        }
    }
    let _ = row;
    Ok((
        bytes,
        hasher
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect(),
    ))
}

/// Terms contained in an over-long ASCII run anywhere in the corpus: the
/// classifier calls them ambiguous in that passage, and a per-passage table
/// cannot see it (see [`text_match::doc_verdicts`]).
fn long_run_poison(runs: &[String], vocabulary: &HashMap<String, (u64, u64)>) -> HashSet<String> {
    let mut out = HashSet::new();
    for run in runs {
        for seg in run.split(|c: char| !c.is_ascii_lowercase()) {
            let b = seg.as_bytes();
            for i in 0..b.len() {
                for len in text_match::MIN_TERM_LEN..=text_match::MAX_TERM_LEN {
                    if i + len > b.len() {
                        break;
                    }
                    let t = &seg[i..i + len];
                    if vocabulary.contains_key(t) {
                        out.insert(t.to_string());
                    }
                }
            }
        }
    }
    out
}

fn stats(counts: &[u64]) -> Value {
    if counts.is_empty() {
        return Value::Null;
    }
    let mut sorted = counts.to_vec();
    sorted.sort_unstable();
    let q = |p: f64| sorted[((sorted.len() - 1) as f64 * p).round() as usize];
    let mut buckets: BTreeMap<String, u64> = BTreeMap::new();
    for &c in counts {
        let b = match c {
            1 => "1",
            2 => "2",
            3..=5 => "3-5",
            6..=10 => "6-10",
            11..=25 => "11-25",
            26..=50 => "26-50",
            51..=100 => "51-100",
            101..=1000 => "101-1000",
            _ => ">1000",
        };
        *buckets.entry(b.to_string()).or_default() += 1;
    }
    json!({
        "min": sorted[0],
        "p50": q(0.5),
        "p90": q(0.9),
        "p99": q(0.99),
        "max": sorted[sorted.len() - 1],
        "mean": counts.iter().sum::<u64>() as f64 / counts.len() as f64,
        "histogram": buckets,
    })
}

pub struct GenerateOptions {
    /// Timed queries per family (capped by each family's pool).
    pub per_family: usize,
    pub seed: u64,
    /// The floor every family must reach: [`MIN_PER_FAMILY`], except in unit
    /// fixtures (the CLI has no way to lower it).
    pub min_per_family: usize,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            per_family: DEFAULT_PER_FAMILY,
            seed: DEFAULT_SEED,
            min_per_family: MIN_PER_FAMILY,
        }
    }
}

/// Generate every family from the prepared corpus in `dir` and write the
/// family files and the manifest into `out` (an empty staging directory).
/// Two passes over `payloads.jsonl`. Returns the manifest.
pub fn generate(dir: &Path, opts: &GenerateOptions, out: &Path) -> Result<Value, String> {
    let min = opts.min_per_family;
    if opts.per_family < min || min == 0 {
        return Err(format!(
            "{SUITE}: --disk-queries-per-family {} is below the minimum {min}",
            opts.per_family
        ));
    }
    let prepared = crate::msmarco::preparation::read_manifest(dir)?;
    let passages = prepared["passages"]
        .as_u64()
        .ok_or("PREPARED.json lacks the corpus row count")?;
    let preparation_sha256 = preparation_sha256(dir)?;
    let payloads = dir.join("payloads.jsonl");
    let cap = opts.per_family + RESERVED;
    let seed = opts.seed;

    // Pass 1: document frequencies, URL and docid selection.
    let mut vocabulary: HashMap<String, (u64, u64)> = HashMap::new();
    let mut long_runs: Vec<String> = Vec::new();
    let mut docids = Smallest::<(u64, String)>::new(cap);
    let mut urls = Smallest::<(u64, String)>::new(cap);
    let mut url_keys = HashSet::<u64>::new();
    let mut urls_unsafe = 0u64;
    let mut docids_unsafe = 0u64;
    let mut rows = 0u64;
    let (payload_bytes, payload_sha256) = for_each_row(&payloads, |_, s| {
        rows += 1;
        for (t, yes) in s.terms {
            let e = vocabulary.entry(t).or_insert((0, 0));
            if yes {
                e.0 += 1
            } else {
                e.1 += 1
            }
        }
        long_runs.extend(s.long_runs);
        if is_safe_tag(&s.docid) {
            docids.add((rank_key(seed, "docid-eq", &s.docid), s.docid));
        } else {
            docids_unsafe += 1;
        }
        if is_safe_tag(&s.url) {
            let k = rank_key(seed, "url-eq", &s.url);
            url_keys.insert(k);
            urls.add((k, s.url));
        } else {
            urls_unsafe += 1;
        }
        Ok(())
    })?;
    if rows != passages {
        return Err(format!(
            "{SUITE}: payloads.jsonl has {rows} rows but PREPARED.json says {passages}"
        ));
    }
    long_runs.sort();
    long_runs.dedup();
    let poison = long_run_poison(&long_runs, &vocabulary);
    let ambiguous_terms = vocabulary.values().filter(|(_, a)| *a > 0).count();
    let eligible = |t: &str, (yes, amb): (u64, u64), band: (u64, u64)| {
        amb == 0
            && yes >= band.0
            && yes <= band.1
            && !poison.contains(t)
            && text_match::validate_term(t).is_ok()
    };
    let pool = |family: Family, exclude: &HashSet<String>| -> Vec<(u64, String, u64)> {
        let band = family.df_band().unwrap();
        let mut p: Vec<(u64, String, u64)> = vocabulary
            .iter()
            .filter(|(t, c)| eligible(t, **c, band) && !exclude.contains(t.as_str()))
            .map(|(t, c)| (rank_key(seed, family.name(), t), t.clone(), c.0))
            .collect();
        p.sort_unstable();
        p
    };
    // Eligible-term document-frequency profile (zero ambiguity, not in a long
    // run), recorded so the bands can be judged against the corpus.
    let mut df_profile: BTreeMap<String, u64> = BTreeMap::new();
    for (t, c) in &vocabulary {
        if eligible(t, *c, (1, u64::MAX)) {
            let b = match c.0 {
                1 => "0001",
                2 => "0002",
                3..=5 => "0003-0005",
                6..=10 => "0006-0010",
                11..=25 => "0011-0025",
                26..=49 => "0026-0049",
                50..=100 => "0050-0100",
                101..=1000 => "0101-1000",
                1001..=10000 => "1001-10000",
                _ => ">10000",
            };
            *df_profile.entry(b.to_string()).or_default() += 1;
        }
    }
    let mut pools: BTreeMap<Family, usize> = BTreeMap::new();
    let take = |family: Family, p: &[(u64, String, u64)]| -> Result<usize, String> {
        if p.len() < min + RESERVED {
            return Err(format!(
                "{SUITE}: family {} has {} eligible items; at least {} are needed \
                 ({min} timed + {RESERVED} reserved)",
                family.name(),
                p.len(),
                min + RESERVED
            ));
        }
        Ok(p.len().min(cap))
    };

    let mut lines: BTreeMap<Family, Vec<Value>> = BTreeMap::new();
    let mut chosen_terms: BTreeMap<Family, HashSet<String>> = BTreeMap::new();
    for family in [Family::RareTerm, Family::DeepFetch] {
        let p = pool(family, &HashSet::new());
        let n = take(family, &p)?;
        pools.insert(family, p.len());
        let chosen = &p[..n];
        chosen_terms.insert(family, chosen.iter().map(|(_, t, _)| t.clone()).collect());
        lines.insert(
            family,
            chosen.iter().map(|(_, t, df)| json!([t, df])).collect(),
        );
    }
    let rac_pool = pool(
        Family::RareTermAndCategory,
        &chosen_terms[&Family::RareTerm],
    );
    let rac_n = take(Family::RareTermAndCategory, &rac_pool)?;
    pools.insert(Family::RareTermAndCategory, rac_pool.len());
    chosen_terms.insert(
        Family::RareTermAndCategory,
        rac_pool[..rac_n]
            .iter()
            .map(|(_, t, _)| t.clone())
            .collect(),
    );
    let overlap = |a: Family, b: Family| chosen_terms[&a].intersection(&chosen_terms[&b]).count();
    let term_overlap = json!({
        "rare-term & rare-term-and-category": overlap(Family::RareTerm, Family::RareTermAndCategory),
        "deep-fetch & rare-term": overlap(Family::DeepFetch, Family::RareTerm),
        "deep-fetch & rare-term-and-category": overlap(Family::DeepFetch, Family::RareTermAndCategory),
    });
    let rac_terms: Vec<(String, u64)> = rac_pool[..rac_n]
        .iter()
        .map(|(_, t, df)| (t.clone(), *df))
        .collect();
    let rac_index: HashMap<&str, usize> = rac_terms
        .iter()
        .enumerate()
        .map(|(i, (t, _))| (t.as_str(), i))
        .collect();
    let pair_pool = pool(Family::TwoRareTerms, &HashSet::new());
    // Pair-pool term ids in lexicographic order, so an id-sorted pair is a
    // term-sorted pair.
    let mut pair_terms: Vec<(&str, u64)> =
        pair_pool.iter().map(|(k, t, _)| (t.as_str(), *k)).collect();
    pair_terms.sort_unstable();
    let pair_index: HashMap<&str, u32> = pair_terms
        .iter()
        .enumerate()
        .map(|(i, (t, _))| (*t, i as u32))
        .collect();

    // Pass 2: postings of the rare-term-and-category terms, and pairs.
    let mut rac_cats: Vec<Vec<String>> = vec![Vec::new(); rac_terms.len()];
    let mut rac_best: Vec<Option<(u64, String)>> = vec![None; rac_terms.len()];
    let mut pairs = Smallest::<(u64, u32, u32)>::new(cap);
    let mut ids: Vec<u32> = Vec::new();
    let (bytes2, sha2) = for_each_row(&payloads, |_, s| {
        ids.clear();
        for (t, yes) in &s.terms {
            if !yes {
                continue;
            }
            if let Some(&i) = rac_index.get(t.as_str()) {
                rac_cats[i].push(s.category.clone());
                let k = rank_key(seed, "rare-term-and-category/passage", &s.docid);
                if rac_best[i].as_ref().is_none_or(|(b, _)| k < *b) {
                    rac_best[i] = Some((k, s.category.clone()));
                }
            }
            if let Some(&i) = pair_index.get(t.as_str()) {
                ids.push(i);
            }
        }
        ids.sort_unstable();
        for (x, &a) in ids.iter().enumerate() {
            for &b in &ids[x + 1..] {
                let k = pair_key(pair_terms[a as usize].1, pair_terms[b as usize].1);
                pairs.add((k, a, b));
            }
        }
        Ok(())
    })?;
    if (bytes2, &sha2) != (payload_bytes, &payload_sha256) {
        return Err(format!(
            "{SUITE}: payloads.jsonl changed between the two passes"
        ));
    }
    let mut rac_lines = Vec::with_capacity(rac_terms.len());
    for (i, (t, df)) in rac_terms.iter().enumerate() {
        if rac_cats[i].len() as u64 != *df {
            return Err(format!(
                "{SUITE}: term {t:?} matched {} passages in pass 2 but {df} in pass 1",
                rac_cats[i].len()
            ));
        }
        let (_, cat) = rac_best[i].as_ref().ok_or("rare term without a passage")?;
        if !is_safe_tag(cat) {
            return Err(format!("{SUITE}: category {cat:?} is not a safe tag value"));
        }
        let n = rac_cats[i].iter().filter(|c| *c == cat).count();
        rac_lines.push(json!([t, cat, n]));
    }
    lines.insert(Family::RareTermAndCategory, rac_lines);
    let pairs = pairs.into_sorted();
    if pairs.len() < min + RESERVED {
        return Err(format!(
            "{SUITE}: only {} co-occurring term pairs in the {PAIR_DF:?} band",
            pairs.len()
        ));
    }
    pools.insert(Family::TwoRareTerms, pairs.len());
    lines.insert(
        Family::TwoRareTerms,
        pairs
            .iter()
            .map(|((_, a, b), n)| json!([pair_terms[*a as usize].0, pair_terms[*b as usize].0, n]))
            .collect(),
    );
    for (family, sel, distinct) in [
        (Family::UrlEq, urls, url_keys.len()),
        (Family::DocidEq, docids, (rows - docids_unsafe) as usize),
    ] {
        let sel = sel.into_sorted();
        if sel.len() < min + RESERVED {
            return Err(format!(
                "{SUITE}: family {} has {} eligible values; at least {} are needed",
                family.name(),
                sel.len(),
                min + RESERVED
            ));
        }
        pools.insert(family, distinct);
        lines.insert(
            family,
            sel.into_iter().map(|((_, v), n)| json!([v, n])).collect(),
        );
    }

    // Write the files.
    std::fs::create_dir_all(out).map_err(|e| format!("{}: {e}", out.display()))?;
    let mut families = serde_json::Map::new();
    for family in Family::ALL {
        let l = &lines[&family];
        let path = out.join(family.file_name());
        let mut w = std::io::BufWriter::new(
            std::fs::File::create(&path).map_err(|e| format!("{}: {e}", path.display()))?,
        );
        let mut counts = Vec::with_capacity(l.len());
        for line in l {
            let (_, n) = family.conditions(line)?;
            counts.push(n);
            writeln!(w, "{line}").map_err(|e| format!("{}: {e}", path.display()))?;
        }
        w.flush().map_err(|e| format!("{}: {e}", path.display()))?;
        drop(w);
        let (sha, _) = sha256_file(&path)?;
        families.insert(
            family.name().into(),
            json!({
                "file": family.file_name(),
                "sha256": sha,
                "lines": l.len(),
                "timed_queries": l.len() - RESERVED,
                "df_band": family.df_band().map(|(a, b)| json!([a, b])),
                "fields": family.fields(),
                "eligible_pool": pools[&family],
                "expected_count": stats(&counts),
                "expected_count_timed": stats(&counts[RESERVED..]),
            }),
        );
    }
    let manifest = json!({
        "suite": SUITE,
        "generator": GENERATOR,
        "seed": seed,
        "per_family_requested": opts.per_family,
        "per_family_minimum": min,
        "preparation_sha256": preparation_sha256,
        "dataset": prepared["dataset"],
        "payloads": {"rows": rows, "bytes": payload_bytes, "sha256": payload_sha256},
        "text_field": TEXT_FIELD,
        "text_match_rules": text_match::RULE_VERSION,
        "analysis": "parity: segment text with every whitespace other than U+0020 replaced by a space (as uploaded), no stemming",
        "df_bands": {"rare": RARE_DF, "pair": PAIR_DF, "deep": DEEP_DF},
        "layout": {"prime_line": 0, "probe_reserve_lines": [1, PROBE_RESERVE], "timed_from_line": RESERVED},
        "vocabulary": {
            "eligible_terms_by_df": df_profile,
            "terms": vocabulary.len(),
            "terms_with_ambiguous_passages": ambiguous_terms,
            "long_runs": long_runs.len(),
            "terms_in_long_runs": poison.len(),
        },
        "tags": {"urls_not_safe": urls_unsafe, "docids_not_safe": docids_unsafe,
                 "safe": "printable ASCII 0x21-0x7E, no ';', at most 2048 bytes"},
        "term_overlap": term_overlap,
        "selection": "per family: eligible items ranked by SHA-256(generator, seed, family, item); the smallest kept in key order. rare-term-and-category terms are disjoint from rare-term terms; deep-fetch is drawn from its own band independently (overlaps recorded in term_overlap). Terms: zero ambiguous passages, not in any over-long run. rare-term-and-category: the category of the term's passage with the smallest seeded docid key. two-rare-terms: co-occurring pairs (one passage has both) ranked by a SplitMix64 mix of the two term keys.",
        "families": families,
    });
    let text = serde_json::to_string_pretty(&manifest).map_err(|e| e.to_string())?;
    std::fs::write(out.join(MANIFEST), text + "\n")
        .map_err(|e| format!("{}: {e}", out.display()))?;
    Ok(manifest)
}

/// One timed window of one family, loaded and verified.
#[derive(Debug, Clone)]
pub struct Window {
    pub family: Family,
    /// `[prime, probe reserve…, timed…]`, each with `query_id` = file line.
    pub queries: Vec<(usize, Query)>,
    /// Index into `queries` of the first timed query.
    pub timed_from: usize,
    pub offset: u64,
    pub manifest: Value,
    pub manifest_sha256: String,
}

impl Window {
    /// Record of the window for the result metadata.
    pub fn record(&self) -> Value {
        let timed = &self.queries[self.timed_from..];
        json!({
            "suite": SUITE,
            "family": self.family.name(),
            "generator": self.manifest["generator"],
            "seed": self.manifest["seed"],
            "manifest_sha256": self.manifest_sha256,
            "family_file_sha256": self.manifest["families"][self.family.name()]["sha256"],
            "df_band": self.family.df_band().map(|(a, b)| json!([a, b])),
            "eligible_pool": self.manifest["families"][self.family.name()]["eligible_pool"],
            "timed_pool": self.manifest["families"][self.family.name()]["timed_queries"],
            "offset": self.offset,
            "timed_queries": timed.len(),
            "first_line": timed.first().map(|q| q.0),
            "last_line": timed.last().map(|q| q.0),
            "prime_line": 0,
            "probe_reserve_lines": [1, PROBE_RESERVE],
            "payloads": self.manifest["payloads"],
        })
    }
}

fn read_manifest(dir: &Path) -> Result<(Value, String), String> {
    let path = dir.join(DIR).join(MANIFEST);
    let bytes = std::fs::read(&path).map_err(|e| {
        format!(
            "{SUITE}: {} is missing ({e}); write it with `prepare-msmarco --dataset <name> \
             --add-disk-queries`",
            path.display()
        )
    })?;
    let manifest: Value =
        serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok((manifest, sha256_hex(&bytes)))
}

/// Check the manifest belongs to this corpus and this generator.
pub fn check_manifest(dir: &Path, manifest: &Value) -> Result<(), String> {
    if manifest["suite"] != SUITE || manifest["generator"] != GENERATOR {
        return Err(format!(
            "{SUITE}: {} was written by generator {} (suite {}); this build needs {GENERATOR} \
             — regenerate with --add-disk-queries",
            dir.join(DIR).join(MANIFEST).display(),
            manifest["generator"],
            manifest["suite"]
        ));
    }
    let prep = preparation_sha256(dir)?;
    if manifest["preparation_sha256"] != prep.as_str() {
        return Err(format!(
            "{SUITE}: the query files were computed from preparation {} but PREPARED.json is \
             {prep}; regenerate with --add-disk-queries",
            manifest["preparation_sha256"]
        ));
    }
    let payloads = dir.join("payloads.jsonl");
    let size = std::fs::metadata(&payloads)
        .map_err(|e| format!("{}: {e}", payloads.display()))?
        .len();
    if manifest["payloads"]["bytes"].as_u64() != Some(size) {
        return Err(format!(
            "{SUITE}: payloads.jsonl is {size} bytes but the query files were computed from {} \
             bytes; regenerate with --add-disk-queries",
            manifest["payloads"]["bytes"]
        ));
    }
    Ok(())
}

/// Load the window `[offset, offset + count)` of `family`'s timed pool, plus
/// the priming query and the probe reserve. `count` `None` = the rest of the
/// pool. A window past the end of the pool is an error: queries never cycle.
pub fn load_window(
    dir: &Path,
    family: Family,
    offset: u64,
    count: Option<usize>,
) -> Result<Window, String> {
    let (manifest, manifest_sha256) = read_manifest(dir)?;
    check_manifest(dir, &manifest)?;
    let entry = &manifest["families"][family.name()];
    let path = dir.join(DIR).join(family.file_name());
    let (sha, _) = sha256_file(&path)?;
    if entry["sha256"] != sha.as_str() {
        return Err(format!(
            "{SUITE}: {} does not match its manifest SHA-256; regenerate with \
             --add-disk-queries",
            path.display()
        ));
    }
    let lines = entry["lines"]
        .as_u64()
        .ok_or("disk manifest lacks a line count")? as usize;
    let pool = lines
        .checked_sub(RESERVED)
        .ok_or("disk family file is shorter than its reserved lines")?;
    let offset_us = usize::try_from(offset).map_err(|_| "offset out of range")?;
    let count = match count {
        Some(n) => n,
        None => pool.saturating_sub(offset_us),
    };
    if count == 0 || offset_us.saturating_add(count) > pool {
        return Err(format!(
            "{SUITE} family {}: the window needs {count} queries from offset {offset}, but the \
             timed pool holds {pool}; queries are never repeated within a run — use a smaller \
             --queries or another offset, or regenerate with a larger \
             --disk-queries-per-family",
            family.name()
        ));
    }
    let first = RESERVED + offset_us;
    let last = first + count;
    let file = std::fs::File::open(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut queries = Vec::with_capacity(RESERVED + count);
    let mut seen = 0usize;
    for (line_no, line) in BufReader::new(file).lines().enumerate() {
        seen += 1;
        if line_no >= last {
            continue;
        }
        if line_no >= RESERVED && line_no < first {
            continue;
        }
        let line = line.map_err(|e| format!("{}: {e}", path.display()))?;
        let v: Value =
            serde_json::from_str(&line).map_err(|e| format!("{}: {e}", path.display()))?;
        let (conditions, n) = family.conditions(&v)?;
        queries.push((
            line_no,
            Query {
                name: format!("{}-{line_no:07}", family.name()),
                family: family.name().into(),
                conditions,
                expected_count: Some(n),
                minimum_count: 1,
            },
        ));
    }
    if seen != lines {
        return Err(format!(
            "{SUITE}: {} has {seen} lines but its manifest says {lines}",
            path.display()
        ));
    }
    Ok(Window {
        family,
        queries,
        timed_from: RESERVED,
        offset,
        manifest,
        manifest_sha256,
    })
}

/// A small MS MARCO-shaped corpus for tests (unit and integration): `n`
/// passages with `docid`, `url`, `title`, `segment`, `category` and
/// `doc_len`, unit vectors of `dim`, a `tests.jsonl` and a `PREPARED.json`.
/// The vocabulary is built so every family has eligible items at small
/// minimums, and so every exclusion rule has something to exclude: unique
/// terms (df 1), `mid*` terms (df n/150, the deep band at n = 3000), `pair*`
/// terms (df ≈ n/10), a term ambiguous in some passages (`edge`), a term
/// that only reaches df through newline normalization (`newl`), a term
/// inside an over-long run, a folding collision (`cafe`), and unsafe URLs.
pub fn write_fixture_corpus(dir: &Path, n: usize, dim: usize) -> Result<(), String> {
    use rand::{Rng, SeedableRng};
    fn letters(mut i: usize) -> String {
        let mut s = String::new();
        loop {
            s.push((b'a' + (i % 26) as u8) as char);
            i /= 26;
            if i == 0 {
                return s;
            }
        }
    }
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let mut rng = rand::rngs::StdRng::seed_from_u64(0xD15C);
    let mut payloads = String::new();
    let mut vectors = Vec::with_capacity(n);
    for i in 0..n {
        let a = i % 20;
        let b = (a + 1 + (i / 20) % 19) % 20;
        let mut words = vec![
            format!("uniq{}", letters(i)),
            format!("mid{}", letters(i % 150)),
            format!("pair{}", letters(a)),
            format!("pair{}", letters(b)),
        ];
        match i % 50 {
            0 => words.push("edge.Case".into()),
            1 => words.push("edge".into()),
            _ => {}
        }
        if i % 30 == 5 {
            words.push("\nnewl".into());
        }
        if i == 7 {
            words.push(format!("{}uniq{}", "a".repeat(260), letters(8)));
        }
        match i {
            9 => words.push("caf\u{e9}".into()),
            10 => words.push("cafe".into()),
            _ => {}
        }
        let url = match i % 100 {
            3 => format!("http://bad site/{i}"),
            4 => format!("http://x/caf\u{e9}/{i}"),
            _ => format!("https://example.org/{}", i % 400),
        };
        let docid = format!("fix_doc_{i}#0_{i}");
        payloads.push_str(
            &json!({
                "docid": docid,
                "url": url,
                "title": format!("title {}", letters(i % 10)),
                "segment": words.join(" ").replace(" \n", "\n"),
                "category": format!("cat{:02}", i % 7),
                "doc_len": i,
            })
            .to_string(),
        );
        payloads.push('\n');
        let mut v: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
        super::normalize_in_place(&mut v);
        vectors.push(v);
    }
    std::fs::write(dir.join("payloads.jsonl"), payloads).map_err(|e| e.to_string())?;
    crate::readers::write_npy_vectors(dir.join("vectors.npy").to_str().ok_or("path")?, &vectors)?;
    let tests: String = (0..4)
        .map(|q| {
            format!(
                "{}\n",
                json!({"query": vectors[q], "closest_ids": [q], "query_text": "fixture"})
            )
        })
        .collect();
    std::fs::write(dir.join("tests.jsonl"), tests).map_err(|e| e.to_string())?;
    std::fs::write(
        dir.join("PREPARED.json"),
        json!({"dataset": "disk-fixture", "passages": n}).to_string(),
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(n: usize) -> (tempfile::TempDir, std::path::PathBuf) {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("corpus");
        write_fixture_corpus(&dir, n, 4).unwrap();
        (tmp, dir)
    }

    fn opts(per_family: usize, seed: u64) -> GenerateOptions {
        GenerateOptions {
            per_family,
            seed,
            min_per_family: 5,
        }
    }

    fn rows(dir: &Path) -> Vec<Value> {
        std::fs::read_to_string(dir.join("payloads.jsonl"))
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect()
    }

    fn verdict(row: &Value, term: &str) -> TextMatch {
        text_match::classify(
            &crate::fulltext::normalize_whitespace(row["segment"].as_str().unwrap()),
            term,
        )
    }

    /// Brute-force oracle, independent of the generator's single-pass table:
    /// `classify` per (passage, term) on the normalized text.
    fn df(rows: &[Value], term: &str) -> (u64, u64) {
        rows.iter()
            .fold((0, 0), |(y, a), r| match verdict(r, term) {
                TextMatch::Yes => (y + 1, a),
                TextMatch::Ambiguous => (y, a + 1),
                TextMatch::No => (y, a),
            })
    }

    fn lines(dir: &Path, f: Family) -> Vec<Value> {
        std::fs::read_to_string(dir.join(f.file_name()))
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect()
    }

    #[test]
    fn generated_counts_match_a_brute_force_oracle() {
        let (_tmp, dir) = fixture(3000);
        let out = dir.join(DIR);
        let m = generate(&dir, &opts(200, 7), &out).unwrap();
        let corpus = rows(&dir);
        // As `analysis: parity` uploads it.
        let uploaded: Vec<Value> = corpus
            .iter()
            .map(|r| {
                let mut r = r.clone();
                r["segment"] = json!(crate::fulltext::normalize_whitespace(
                    r["segment"].as_str().unwrap()
                ));
                r
            })
            .collect();
        for family in Family::ALL {
            let l = lines(&out, family);
            assert_eq!(m["families"][family.name()]["lines"], l.len(), "{family:?}");
            let mut distinct = HashSet::new();
            for line in &l {
                assert!(distinct.insert(line.to_string()), "duplicate {line}");
                let (conditions, count) = family.conditions(line).unwrap();
                let expr = crate::compound_text::Expr::compile(
                    &conditions,
                    &json!({"segment":"text","category":"keyword","url":"keyword","docid":"keyword"}),
                )
                .unwrap();
                let mut n = 0;
                for r in &uploaded {
                    // The oracle refuses an ambiguous passage outright.
                    if expr.matches(r.as_object().unwrap()).unwrap() {
                        n += 1;
                    }
                }
                assert_eq!(n, count, "{family:?} {line}");
                if let Some((lo, hi)) = family.df_band() {
                    for t in line.as_array().unwrap().iter().filter_map(Value::as_str) {
                        if t.starts_with("cat") {
                            continue;
                        }
                        let (yes, amb) = df(&corpus, t);
                        assert_eq!(amb, 0, "{t}");
                        assert!((lo..=hi).contains(&yes), "{family:?} {t} df {yes}");
                    }
                }
            }
        }
        // Pool sizes equal the brute-force eligible sets: a rule that excluded
        // too much (newline normalization lost) or too little (an ambiguous,
        // folded or long-run term kept) changes them.
        let vocab: HashSet<String> = corpus
            .iter()
            .flat_map(|r| {
                r["segment"]
                    .as_str()
                    .unwrap()
                    .split(|c: char| !c.is_ascii_alphabetic())
                    .map(str::to_ascii_lowercase)
                    .collect::<Vec<_>>()
            })
            .filter(|t| text_match::validate_term(t).is_ok())
            .collect();
        let eligible = |lo: u64, hi: u64| {
            vocab
                .iter()
                .filter(|t| {
                    let (yes, amb) = df(&corpus, t);
                    amb == 0 && (lo..=hi).contains(&yes)
                })
                .count()
        };
        assert_eq!(
            m["families"]["rare-term"]["eligible_pool"],
            eligible(1, 100)
        );
        assert_eq!(
            m["families"]["deep-fetch"]["eligible_pool"],
            eligible(11, 100)
        );
        for (term, why) in [
            ("edge", "ambiguous passage"),
            ("cafe", "folding"),
            ("uniqi", "inside an over-long run"),
        ] {
            assert!(
                !Family::ALL
                    .iter()
                    .any(|f| lines(&out, *f).iter().any(|l| l[0] == term)),
                "{term} kept despite {why}"
            );
        }
        assert_eq!(df(&corpus, "newl").1, 0, "newline normalization is applied");
        // Single-term families are in seeded key order (uniform slices).
        for family in [
            Family::RareTerm,
            Family::DeepFetch,
            Family::DocidEq,
            Family::UrlEq,
        ] {
            let keys: Vec<u64> = lines(&out, family)
                .iter()
                .map(|l| rank_key(7, family.name(), l[0].as_str().unwrap()))
                .collect();
            assert!(keys.windows(2).all(|w| w[0] < w[1]), "{family:?}");
        }
        // Unsafe URLs never become queries.
        for l in lines(&out, Family::UrlEq) {
            assert!(is_safe_tag(l[0].as_str().unwrap()), "{l}");
        }
        assert_eq!(m["tags"]["urls_not_safe"], 60);
        // rare-term-and-category: the category of the term's passage with the
        // smallest seeded docid key (the documented rule, not just any one).
        for l in lines(&out, Family::RareTermAndCategory) {
            let t = l[0].as_str().unwrap();
            let best = corpus
                .iter()
                .filter(|r| verdict(r, t) == TextMatch::Yes)
                .min_by_key(|r| {
                    rank_key(
                        7,
                        "rare-term-and-category/passage",
                        r["docid"].as_str().unwrap(),
                    )
                })
                .unwrap();
            assert_eq!(l[1], best["category"], "{l}");
        }
        // rare-term-and-category terms are disjoint from rare-term terms.
        let rare: HashSet<Value> = lines(&out, Family::RareTerm)
            .iter()
            .map(|l| l[0].clone())
            .collect();
        assert!(lines(&out, Family::RareTermAndCategory)
            .iter()
            .all(|l| !rare.contains(&l[0])));
        // Deterministic for a seed, and the seed matters.
        let again = dir.join("again");
        let m2 = generate(&dir, &opts(200, 7), &again).unwrap();
        assert_eq!(m, m2);
        for family in Family::ALL {
            assert_eq!(
                std::fs::read(out.join(family.file_name())).unwrap(),
                std::fs::read(again.join(family.file_name())).unwrap()
            );
        }
        let other = dir.join("other");
        generate(&dir, &opts(200, 8), &other).unwrap();
        assert_ne!(
            std::fs::read(out.join("rare-term.jsonl")).unwrap(),
            std::fs::read(other.join("rare-term.jsonl")).unwrap()
        );
    }

    #[test]
    fn a_family_below_its_floor_is_an_error_not_a_smaller_set() {
        let (_tmp, dir) = fixture(3000);
        let mut o = opts(200, 1);
        o.min_per_family = 100_000;
        o.per_family = 100_000;
        let e = generate(&dir, &o, &dir.join("x")).unwrap_err();
        assert!(e.contains("eligible"), "{e}");
        assert!(generate(&dir, &opts(4, 1), &dir.join("y")).is_err());
    }

    #[test]
    fn windows_never_cycle_and_check_corpus_identity() {
        let (_tmp, dir) = fixture(3000);
        generate(&dir, &opts(100, 1), &dir.join(DIR)).unwrap();
        let pool = lines(&dir.join(DIR), Family::RareTerm).len() - RESERVED;
        let w = load_window(&dir, Family::RareTerm, 10, Some(20)).unwrap();
        let ids: Vec<usize> = w.queries.iter().map(|q| q.0).collect();
        let want: Vec<usize> = (0..RESERVED).chain(RESERVED + 10..RESERVED + 30).collect();
        assert_eq!(ids, want);
        assert_eq!(w.timed_from, RESERVED);
        assert_eq!(w.record()["first_line"], RESERVED + 10);
        assert_eq!(w.record()["last_line"], RESERVED + 29);
        let rest = load_window(&dir, Family::RareTerm, 5, None).unwrap();
        assert_eq!(rest.queries.len() - RESERVED, pool - 5);
        for (offset, count) in [(0, pool + 1), (pool as u64 - 3, 4), (pool as u64, 1)] {
            let e = load_window(&dir, Family::RareTerm, offset, Some(count)).unwrap_err();
            assert!(e.contains("never repeated"), "{e}");
        }
        assert!(load_window(&dir, Family::RareTerm, pool as u64, None).is_err());
        // Corpus identity: PREPARED.json, payload size, the family file.
        let prepared = std::fs::read(dir.join("PREPARED.json")).unwrap();
        std::fs::write(
            dir.join("PREPARED.json"),
            b"{\"dataset\":\"x\",\"passages\":3000}",
        )
        .unwrap();
        assert!(load_window(&dir, Family::RareTerm, 0, Some(1))
            .unwrap_err()
            .contains("preparation"));
        std::fs::write(dir.join("PREPARED.json"), &prepared).unwrap();
        let payloads = std::fs::read(dir.join("payloads.jsonl")).unwrap();
        std::fs::write(dir.join("payloads.jsonl"), [&payloads[..], b"\n"].concat()).unwrap();
        assert!(load_window(&dir, Family::RareTerm, 0, Some(1))
            .unwrap_err()
            .contains("bytes"));
        std::fs::write(dir.join("payloads.jsonl"), &payloads).unwrap();
        let f = dir.join(DIR).join(Family::UrlEq.file_name());
        let body = std::fs::read_to_string(&f).unwrap();
        std::fs::write(&f, body.replacen("example", "exbmple", 1)).unwrap();
        assert!(load_window(&dir, Family::UrlEq, 0, Some(1))
            .unwrap_err()
            .contains("SHA-256"));
        std::fs::write(&f, body).unwrap();
        assert!(load_window(&dir, Family::UrlEq, 0, Some(1)).is_ok());
        std::fs::remove_dir_all(dir.join(DIR)).unwrap();
        assert!(load_window(&dir, Family::UrlEq, 0, Some(1))
            .unwrap_err()
            .contains("--add-disk-queries"));
    }

    #[test]
    fn smallest_keeps_exact_counts_of_survivors() {
        // Brute force over a fixed stream: the kept set is the `cap`
        // smallest distinct keys, and each kept count is its full count.
        let stream = [9u64, 3, 7, 3, 1, 9, 8, 1, 2, 3, 7, 0, 9, 2, 2];
        for cap in 0..8 {
            let mut s = Smallest::new(cap);
            for k in stream {
                s.add(k);
            }
            let mut distinct: Vec<u64> = stream.to_vec();
            distinct.sort();
            distinct.dedup();
            let want: Vec<(u64, u64)> = distinct
                .into_iter()
                .take(cap)
                .map(|k| (k, stream.iter().filter(|x| **x == k).count() as u64))
                .collect();
            assert_eq!(s.into_sorted(), want, "cap {cap}");
        }
    }

    #[test]
    fn tag_safety() {
        assert!(is_safe_tag("http://0-60.reviews/0-60-times/"));
        assert!(is_safe_tag("msmarco_v2.1_doc_00_0#2_3101"));
        for bad in [
            "",
            "a;b",
            "a b",
            " a",
            "caf\u{e9}",
            "a\tb",
            &"x".repeat(2049),
        ] {
            assert!(!is_safe_tag(bad), "{bad:?}");
        }
    }

    #[test]
    fn family_lines_round_trip_and_reject_malformed() {
        let (c, n) = Family::TwoRareTerms
            .conditions(&json!(["alpha", "beta", 3]))
            .unwrap();
        assert_eq!(n, 3);
        assert_eq!(
            c,
            json!({"and":[{"segment":{"match":{"text":"alpha"}}},{"segment":{"match":{"text":"beta"}}}]})
        );
        assert_eq!(
            Family::UrlEq.conditions(&json!(["u", 1])).unwrap().0,
            json!({"url":{"match":{"value":"u"}}})
        );
        for bad in [
            json!(["alpha", 0]),
            json!(["alpha"]),
            json!(["alpha", "beta", 1]),
            json!([1, 1]),
            json!({"t": "alpha"}),
        ] {
            assert!(Family::RareTerm.conditions(&bad).is_err(), "{bad}");
        }
        for f in Family::ALL {
            assert_eq!(Family::parse(f.name()).unwrap(), f);
        }
        assert!(Family::parse("segment").is_err());
        assert!(Family::DeepFetch.check_params(100, true).is_ok());
        assert!(Family::DeepFetch.check_params(10, true).is_err());
        assert!(Family::DeepFetch.check_params(100, false).is_err());
        assert!(Family::RareTerm.check_params(10, false).is_ok());
    }

    const RARE_FRIZZES: u64 = 0x6859_78ae_b850_f4a3;

    #[test]
    fn pair_key_is_order_sensitive_and_stable() {
        assert_ne!(pair_key(1, 2), pair_key(2, 1));
        // Pinned: a platform or refactor that changes the mix changes every
        // generated two-rare-terms file.
        assert_eq!(pair_key(1, 2), 0x37ad_5fdd_5756_bd3d);
        // Pinned against an independent implementation (Python hashlib):
        // sha256(b"vdb-msmarco-disk/v1\0" + (1).to_bytes(8, "big") +
        // b"rare-term\0frizzes")[:8], big-endian.
        assert_eq!(rank_key(1, "rare-term", "frizzes"), RARE_FRIZZES);
    }
}
