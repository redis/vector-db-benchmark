//! Full-text `{"match": {"text": term}}` semantics that the brute-force oracle,
//! Redis and OpenSearch provably agree on (#341).
//!
//! # The problem
//!
//! The two engines do not tokenize alike. Redis (RediSearch `TEXT`) splits on
//! ASCII whitespace and a fixed punctuation list, lowercases, treats `\` as an
//! escape, and STEMS (English Snowball) at query time; OpenSearch's default
//! `text` mapping uses the `standard` analyzer (Unicode UAX#29 word breaks,
//! lowercase, no stemming, tokens capped at 255 chars). On real titles they
//! disagree: `Nature's Way` is `nature` + `s` to Redis and `nature's` to
//! OpenSearch; `nature.com` is two tokens to Redis and one to OpenSearch;
//! `x\-run` is one Redis token and two OpenSearch tokens; a Redis query for
//! `run` also matches `Running` unless the query is sent `VERBATIM`. No single
//! tokenizer written here can reproduce both engines on every string.
//!
//! # What this module does instead
//!
//! It restricts the query side and classifies the document side:
//!
//! * A query term is ONE lowercase ASCII word, `[a-z]{3,64}`, that is not a
//!   RediSearch default stopword. It analyzes to itself in both engines, and
//!   Redis must run it `VERBATIM` (no stemming expansion) — the engine arm
//!   does that; this module states the requirement.
//! * For a document's text and a term, [`classify`] answers `Yes` (both engines
//!   certainly match), `No` (both certainly do not) or `Ambiguous` (the
//!   engines might disagree, or this model cannot tell). It only answers
//!   `Yes`/`No` from contexts where both tokenizers are known to agree:
//!   - an occurrence is the term as an entire maximal run of ASCII
//!     alphanumerics (a longer run — `running`, `abc123`, `foo_bar` — is one
//!     token in both engines, so it is `No` for the term);
//!   - each side of that run is the string boundary, an ASCII space, or a
//!     character from [`SAFE_SEPARATORS`] (separators in RediSearch's
//!     documented list that UAX#29 also always breaks on), not escaped by a
//!     preceding `\`; or one of `.`, `'`, `:` (RediSearch separators that
//!     UAX#29 joins only when a letter or digit is on the far side) whose far
//!     side is again a boundary, space or safe separator.
//!
//!   Anything else next to an occurrence — other ASCII punctuation, tabs,
//!   any non-ASCII character — makes that occurrence `Ambiguous`. A text that
//!   contains a character which case-folds or diacritic-folds to ASCII letters
//!   (`İ`, `ß`, `ﬁ`, `é`, the Kelvin sign...) is `Ambiguous` for every term the
//!   folded text could spell, and an ASCII run longer than OpenSearch's 255-char
//!   token cap is `Ambiguous` for every term it contains.
//! * A document matches a term if ANY occurrence is `Yes`; otherwise it is
//!   `Ambiguous` if any occurrence is; otherwise `No`.
//!
//! The predicate evaluator (`derived::Predicate::matches`) turns `Ambiguous`
//! into an error, so ground truth is never computed for a document whose
//! membership the engines could dispute. Hybrid query generation therefore only
//! uses terms with ZERO ambiguous documents in the whole corpus
//! ([`choose_term`]); on the 100K MS MARCO corpus ~49K title terms qualify.
//!
//! This is exact agreement by construction for every document of a corpus the
//! term was chosen on — the guarantee is about the chosen (term, corpus) pair,
//! not a claim that the classifier models either engine in general.

use std::collections::HashMap;

/// RediSearch's default stopword list. A stopword is not indexed by Redis
/// (OpenSearch's default analyzer keeps it), so a stopword term is refused.
pub const REDIS_STOPWORDS: &[&str] = &[
    "a", "is", "the", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into",
    "it", "no", "not", "of", "on", "or", "such", "that", "their", "then", "there", "these", "they",
    "this", "to", "was", "will", "with",
];

/// ASCII characters both engines always split on, whatever is on either side.
/// RediSearch's documented separators minus `.`, `'` and `:` (see
/// [`MID_SEPARATORS`]).
pub const SAFE_SEPARATORS: &str = ",<>{}[]\";!@#$%^&*()-+=~";

/// RediSearch separators that UAX#29 joins when a letter is on both sides.
pub const MID_SEPARATORS: &str = ".':";

/// Shortest and longest accepted query term.
pub const MIN_TERM_LEN: usize = 3;
pub const MAX_TERM_LEN: usize = 64;

/// OpenSearch's `standard` tokenizer splits longer runs into 255-char pieces.
pub const LUCENE_MAX_TOKEN_LEN: usize = 255;

/// The version of these rules, recorded with every generated query set so a
/// change of rule is visible in the artifacts.
pub const RULE_VERSION: &str = "vdb-text-match/v1";

/// One text's membership for one term.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextMatch {
    Yes,
    No,
    Ambiguous,
}

impl TextMatch {
    /// Combine two occurrences of the same term in one document.
    fn or(self, other: TextMatch) -> TextMatch {
        match (self, other) {
            (TextMatch::Yes, _) | (_, TextMatch::Yes) => TextMatch::Yes,
            (TextMatch::Ambiguous, _) | (_, TextMatch::Ambiguous) => TextMatch::Ambiguous,
            _ => TextMatch::No,
        }
    }
}

/// Refuse a term the engines would not both analyze to exactly itself.
pub fn validate_term(term: &str) -> Result<(), String> {
    if !(MIN_TERM_LEN..=MAX_TERM_LEN).contains(&term.len())
        || !term.bytes().all(|b| b.is_ascii_lowercase())
    {
        return Err(format!(
            "text match term {term:?} must be one lowercase ASCII word of {MIN_TERM_LEN}..= \
             {MAX_TERM_LEN} letters: anything else analyzes differently in Redis and OpenSearch"
        ));
    }
    if REDIS_STOPWORDS.contains(&term) {
        return Err(format!(
            "text match term {term:?} is a RediSearch stopword: Redis does not index it, \
             OpenSearch does, so the engines would disagree on every document"
        ));
    }
    Ok(())
}

fn is_safe(c: char) -> bool {
    c == ' ' || SAFE_SEPARATORS.contains(c)
}

/// What one side of a maximal ASCII-alphanumeric run means for a token match.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Side {
    /// A token boundary in both engines.
    Boundary,
    /// Joined to more token characters in both engines: not a token match.
    Joined,
    Unknown,
}

/// Classify the side of the run at `chars[i]` (the neighbour), stepping
/// outward by `step` (-1 left, +1 right).
fn side(chars: &[char], i: isize, step: isize) -> Side {
    let at = |j: isize| -> Option<char> {
        if j < 0 {
            None
        } else {
            chars.get(j as usize).copied()
        }
    };
    let Some(c) = at(i) else {
        return Side::Boundary;
    };
    if c == '_' {
        return Side::Joined;
    }
    if is_safe(c) {
        // `\-` is an escaped, NON-separating `-` to RediSearch.
        if step < 0 && c != ' ' && at(i - 1) == Some('\\') {
            return Side::Unknown;
        }
        return Side::Boundary;
    }
    if MID_SEPARATORS.contains(c) {
        return match at(i + step) {
            None => Side::Boundary,
            Some(d) if is_safe(d) => Side::Boundary,
            Some(_) => Side::Unknown,
        };
    }
    Side::Unknown
}

/// Every maximal run of ASCII alphanumerics, as (start, end) char indices.
fn ascii_runs(chars: &[char]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        if chars[i].is_ascii_alphanumeric() {
            let start = i;
            while i < chars.len() && chars[i].is_ascii_alphanumeric() {
                i += 1;
            }
            out.push((start, i));
        } else {
            i += 1;
        }
    }
    out
}

/// The ASCII letters a non-ASCII character may lowercase, case-fold or
/// diacritic-fold to in some engine, or `None` when it cannot become ASCII.
/// Conservative on purpose: a superset of what either engine actually does.
fn ascii_fold(c: char) -> Option<&'static str> {
    // Latin-1 Supplement and Latin Extended-A letters, indexed from U+00C0.
    const LATIN: [&str; 192] = [
        "a", "a", "a", "a", "a", "a", "ae", "c", "e", "e", "e", "e", "i", "i", "i", "i", // C0
        "d", "n", "o", "o", "o", "o", "o", "", "o", "u", "u", "u", "u", "y", "th", "ss", // D0
        "a", "a", "a", "a", "a", "a", "ae", "c", "e", "e", "e", "e", "i", "i", "i", "i", // E0
        "d", "n", "o", "o", "o", "o", "o", "", "o", "u", "u", "u", "u", "y", "th", "y", // F0
        "a", "a", "a", "a", "a", "a", "c", "c", "c", "c", "c", "c", "c", "c", "d", "d", // 100
        "d", "d", "e", "e", "e", "e", "e", "e", "e", "e", "e", "e", "g", "g", "g", "g", // 110
        "g", "g", "g", "g", "h", "h", "h", "h", "i", "i", "i", "i", "i", "i", "i", "i", // 120
        "i", "i", "ij", "ij", "j", "j", "k", "k", "k", "l", "l", "l", "l", "l", "l",
        "l", // 130
        "l", "l", "l", "n", "n", "n", "n", "n", "n", "n", "n", "n", "o", "o", "o", "o", // 140
        "o", "o", "oe", "oe", "r", "r", "r", "r", "r", "r", "s", "s", "s", "s", "s",
        "s", // 150
        "s", "s", "t", "t", "t", "t", "t", "t", "u", "u", "u", "u", "u", "u", "u", "u", // 160
        "u", "u", "u", "u", "w", "w", "y", "y", "y", "z", "z", "z", "z", "z", "z", "s", // 170
    ];
    let cp = c as u32;
    if (0xC0..0x180).contains(&cp) {
        let s = LATIN[(cp - 0xC0) as usize];
        return (!s.is_empty()).then_some(s);
    }
    match c {
        '\u{1E9E}' => Some("ss"),
        '\u{212A}' => Some("k"),
        '\u{FB00}' => Some("ff"),
        '\u{FB01}' => Some("fi"),
        '\u{FB02}' => Some("fl"),
        '\u{FB03}' => Some("ffi"),
        '\u{FB04}' => Some("ffl"),
        '\u{FB05}' | '\u{FB06}' => Some("st"),
        _ => None,
    }
}

/// Raw classification: occurrences of `term` as whole ASCII runs, plus the
/// over-long-run rule.
fn classify_runs(chars: &[char], term: &str) -> TextMatch {
    let mut verdict = TextMatch::No;
    for (start, end) in ascii_runs(chars) {
        let len = end - start;
        if len > LUCENE_MAX_TOKEN_LEN {
            let run: String = chars[start..end]
                .iter()
                .map(|c| c.to_ascii_lowercase())
                .collect();
            if run.contains(term) {
                verdict = verdict.or(TextMatch::Ambiguous);
            }
            continue;
        }
        if len != term.len()
            || !chars[start..end]
                .iter()
                .zip(term.chars())
                .all(|(a, b)| a.to_ascii_lowercase() == b)
        {
            continue;
        }
        let l = side(chars, start as isize - 1, -1);
        let r = side(chars, end as isize, 1);
        let here = match (l, r) {
            (Side::Joined, _) | (_, Side::Joined) => TextMatch::No,
            (Side::Boundary, Side::Boundary) => TextMatch::Yes,
            _ => TextMatch::Ambiguous,
        };
        verdict = verdict.or(here);
    }
    verdict
}

/// Folding rule: if the text holds a character that may fold to ASCII
/// letters, `term` as a whole alphanumeric run of the folded text (other
/// non-ASCII characters acting as breaks) is a possible match the raw view
/// cannot see.
fn folded_could_spell(chars: &[char], term: &str) -> bool {
    if !chars.iter().any(|&c| ascii_fold(c).is_some()) {
        return false;
    }
    let mut folded = String::with_capacity(chars.len() + 8);
    for &c in chars {
        if c.is_ascii() {
            folded.push(c.to_ascii_lowercase());
        } else if let Some(s) = ascii_fold(c) {
            folded.push_str(s);
        } else {
            folded.push('\u{0}');
        }
    }
    folded
        .split(|c: char| !c.is_ascii_alphanumeric())
        .any(|run| run == term)
}

/// Whether `text` matches `term` in both engines, neither, or possibly only
/// one. `term` must satisfy [`validate_term`]; an invalid term is `Ambiguous`.
pub fn classify(text: &str, term: &str) -> TextMatch {
    if validate_term(term).is_err() {
        return TextMatch::Ambiguous;
    }
    let chars: Vec<char> = text.chars().collect();
    match classify_runs(&chars, term) {
        TextMatch::Yes => TextMatch::Yes,
        raw => {
            if folded_could_spell(&chars, term) {
                TextMatch::Ambiguous
            } else {
                raw
            }
        }
    }
}

/// Per-term document frequency over the RAW view (every candidate term that
/// is a whole ASCII-letter run somewhere), used to rank candidates before the
/// exact [`classify`] scan confirms one.
#[derive(Debug, Default, Clone)]
pub struct TermCounter {
    /// term -> (documents where it is `Yes`, documents where it is `Ambiguous`).
    counts: HashMap<String, (u64, u64)>,
    docs: u64,
}

impl TermCounter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn docs(&self) -> u64 {
        self.docs
    }

    pub fn add_doc(&mut self, text: &str) {
        self.docs += 1;
        let chars: Vec<char> = text.chars().collect();
        let mut seen: HashMap<String, TextMatch> = HashMap::new();
        for (start, end) in ascii_runs(&chars) {
            let len = end - start;
            if !(MIN_TERM_LEN..=MAX_TERM_LEN).contains(&len)
                || !chars[start..end].iter().all(|c| c.is_ascii_alphabetic())
            {
                continue;
            }
            let term: String = chars[start..end]
                .iter()
                .map(|c| c.to_ascii_lowercase())
                .collect();
            if seen.contains_key(&term) {
                continue;
            }
            let m = classify_runs(&chars, &term);
            seen.insert(term, m);
        }
        for (term, m) in seen {
            let e = self.counts.entry(term).or_insert((0, 0));
            match m {
                TextMatch::Yes => e.0 += 1,
                TextMatch::Ambiguous => e.1 += 1,
                TextMatch::No => {}
            }
        }
    }

    /// Valid terms with no raw-view ambiguity, as (yes-documents, term).
    pub fn candidates(&self) -> Vec<(u64, &str)> {
        self.counts
            .iter()
            .filter(|(t, (yes, amb))| *yes > 0 && *amb == 0 && validate_term(t).is_ok())
            .map(|(t, (yes, _))| (*yes, t.as_str()))
            .collect()
    }
}

/// The text-match term whose document frequency is nearest `target` (ties to
/// the lexicographically smallest term), among terms with ZERO ambiguous
/// documents under the full [`classify`] rule. `excluded` terms are skipped.
///
/// `scan(term)` must count, over the whole corpus, documents classified
/// `Yes` and `Ambiguous` for `term`; it is a callback so the caller can scan
/// however its corpus is stored (in memory, or streamed from disk, where it
/// can fail with an I/O error). Candidates are confirmed nearest-first, so
/// the scan count is small in practice.
pub fn choose_term(
    counter: &TermCounter,
    target: u64,
    excluded: &[&str],
    mut scan: impl FnMut(&str) -> Result<(u64, u64), String>,
) -> Result<(String, u64), String> {
    let mut cands = counter.candidates();
    cands.retain(|(_, t)| !excluded.contains(t));
    cands.sort_by(|a, b| {
        a.0.abs_diff(target)
            .cmp(&b.0.abs_diff(target))
            .then(a.1.cmp(b.1))
    });
    for (raw_yes, term) in cands {
        let (yes, amb) = scan(term)?;
        if yes != raw_yes {
            return Err(format!(
                "text term {term:?}: the full classifier counts {yes} matching documents but \
                 the raw count was {raw_yes}; the two passes saw different corpora"
            ));
        }
        if amb == 0 {
            return Ok((term.to_string(), yes));
        }
    }
    Err(format!(
        "no text term with zero ambiguous documents near {target} documents exists in this \
         corpus ({} raw candidates)",
        counter.docs()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use TextMatch::*;

    #[test]
    fn terms_must_be_one_lowercase_ascii_non_stopword() {
        for ok in ["run", "quora", "fandom"] {
            assert!(validate_term(ok).is_ok(), "{ok}");
        }
        for bad in [
            "", "ab", "Run", "run fast", "naïve", "r2d2", "the", "with", "a_b",
        ] {
            assert!(validate_term(bad).is_err(), "{bad:?}");
        }
        assert!(validate_term(&"a".repeat(65)).is_err());
        assert_eq!(classify("the run", "the"), Ambiguous);
    }

    /// Each row was checked against Redis 8.8 (`FT.SEARCH ... VERBATIM`) and
    /// OpenSearch 3.7 (`match` on a default `text` field) where both agree;
    /// where they disagree the classifier must say Ambiguous.
    #[test]
    fn classification_matches_what_both_engines_agree_on() {
        for (text, term, want) in [
            // Both engines: token "run".
            ("run fast", "run", Yes),
            ("the run", "run", Yes),
            ("u.s.a run", "run", Yes),
            ("e.g run", "run", Yes),
            ("(run)", "run", Yes),
            ("run-walk", "run", Yes),
            ("Health.", "health", Yes),
            ("RUN", "run", Yes),
            ("Students' union", "students", Yes),
            ("naïve run", "run", Yes),
            // Both engines: no token "run".
            ("Running shoes", "run", No),
            ("rerun", "run", No),
            ("run_fast", "run", No),
            ("run2", "run", No),
            ("na\u{ef}ve running", "run", No),
            ("", "run", No),
            // The engines disagree (Redis splits, Lucene joins, or the escape).
            ("Nature's Way", "nature", Ambiguous),
            ("nature.com", "nature", Ambiguous),
            ("a:run", "run", Ambiguous),
            ("run.Net", "run", Ambiguous),
            ("x\\-run", "run", Ambiguous),
            // Conservative: separators only verified on one engine version.
            ("abc|run", "run", Ambiguous),
            ("run/walk", "run", Ambiguous),
            ("run\twalk", "run", Ambiguous),
            // Non-ASCII neighbours and folding.
            ("Nature\u{2019}s", "nature", Ambiguous),
            ("\u{0130}stanbul", "istanbul", Ambiguous),
            ("\u{FB01}le", "file", Ambiguous),
            ("caf\u{e9}", "cafe", Ambiguous),
            ("\u{212A}ey", "key", Ambiguous),
            // One clean occurrence decides, whatever else is in the text.
            ("Nature's nature", "nature", Yes),
            ("caf\u{e9} cafe", "cafe", Yes),
        ] {
            assert_eq!(classify(text, term), want, "{text:?} / {term:?}");
        }
    }

    #[test]
    fn a_run_longer_than_the_lucene_token_cap_is_ambiguous_for_its_substrings() {
        let long = format!("{}quora{}", "x".repeat(200), "y".repeat(100));
        assert_eq!(classify(&long, "quora"), Ambiguous);
        let short = format!("{}quora", "x".repeat(200));
        assert_eq!(classify(&short, "quora"), No);
    }

    #[test]
    fn mid_separators_count_only_when_the_far_side_breaks_too() {
        assert_eq!(classify("run. walk", "run"), Yes);
        assert_eq!(classify("run.", "run"), Yes);
        assert_eq!(classify("run.(", "run"), Yes);
        assert_eq!(classify("run.x", "run"), Ambiguous);
        assert_eq!(classify("x.run", "run"), Ambiguous);
        assert_eq!(classify("'run'", "run"), Yes);
    }

    #[test]
    fn counter_and_choice_agree_with_classify() {
        let texts = [
            "Quora answers",
            "quora",
            "Nature's way",
            "nature walk",
            "fandom wiki",
            "fandom",
            "fandom page",
            "caf\u{e9} society",
            "society news",
        ];
        let mut c = TermCounter::new();
        for t in texts {
            c.add_doc(t);
        }
        let scan = |term: &str| {
            Ok(texts
                .iter()
                .fold((0, 0), |(y, a), t| match classify(t, term) {
                    Yes => (y + 1, a),
                    Ambiguous => (y, a + 1),
                    No => (y, a),
                }))
        };
        // "nature" has a raw-ambiguous document, so it is never a candidate.
        assert!(c.candidates().iter().all(|(_, t)| *t != "nature"));
        assert_eq!(choose_term(&c, 3, &[], scan).unwrap(), ("fandom".into(), 3));
        assert_eq!(choose_term(&c, 2, &[], scan).unwrap(), ("quora".into(), 2));
        // Excluding the nearest moves to the next one.
        assert_eq!(
            choose_term(&c, 3, &["fandom"], scan).unwrap().0,
            "quora".to_string()
        );
        // "cafe" is only raw-visible in no document; "society" is Yes twice.
        assert_eq!(choose_term(&c, 2, &["quora"], scan).unwrap().0, "society");
        // A scan that disagrees with the counter is an error, not a choice.
        assert!(choose_term(&c, 3, &[], |_| Ok((99, 0))).is_err());
        // A scan that fails (a streamed corpus hitting an I/O error) is
        // propagated as itself.
        assert_eq!(
            choose_term(&c, 3, &[], |_| Err("disk gone".into())).unwrap_err(),
            "disk gone"
        );
        // A candidate the full rule finds ambiguous is skipped.
        let picky = |term: &str| {
            if term == "fandom" {
                Ok((3, 1))
            } else {
                scan(term)
            }
        };
        assert_eq!(choose_term(&c, 3, &[], picky).unwrap().0, "quora");
    }
}
