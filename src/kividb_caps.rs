//! Server capabilities of a KiviDB instance that change what the `kividb`
//! engine may safely do, decided from the version the server reports.
//!
//! Lives in the library (not `src/bin/`) so the version gate is unit-tested:
//! a gate that silently opens on a server the tool does not support could
//! declare a TAG field that matches the wrong documents and publish a recall
//! for a filter that was never applied (the failure `validate_dataset_support`
//! exists to prevent). A gate that silently stays shut on a NEWER server is
//! wrong the other way round: it refuses datasets the server handles.

use std::collections::BTreeMap;

/// The `major.minor.patch` core of a KiviDB version.
pub type VersionCore = (u32, u32, u32);

/// A parsed `kividb_version`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KividbVersion {
    pub core: VersionCore,
    /// Whether the reported value carried a SemVer pre-release suffix
    /// (`1.1.2-rc1`). Recorded, but NOT used to order versions: see
    /// [`TAG_MIN_VERSION`].
    pub prerelease: bool,
}

/// The oldest KiviDB release this tool declares TAG fields on: 1.1.2, the
/// current stable release, which is what CI runs (`make
/// integration-test-kividb`).
///
/// On it TAG fields follow RediSearch semantics: `FT.CREATE` accepts
/// `SEPARATOR` and splits stored values on it, and a bare `TAG` (no
/// `SEPARATOR`) splits on RediSearch's default `,`. The engine therefore
/// declares every TAG `SEPARATOR ;`, exactly as redis.rs does. The query side
/// honours RediSearch backslash escapes, so the filter emitter escapes every
/// TAG value (`kividb_filter::escape_tag_value` in the engine).
///
/// An older server, or one whose version cannot be read, is not supported for
/// TAG fields: every dataset that declares one is refused (vector-only and
/// numeric/text-only datasets still run). This is a SUPPORT boundary, not the
/// point where the behaviour changed. Measured during #375: 1.0.2, 1.0.3,
/// 1.0.4-rc3 and 1.0.4 reject `SEPARATOR`, store a bare TAG as one atomic token
/// and match only raw (unescaped) query values; 1.0.5-rc1, 1.0.5 and 1.0.6-rc4
/// already split on `SEPARATOR` and honour escapes as 1.1.2 does. 1.0.7 to
/// 1.1.1 were not measured (1.1.1 was withdrawn). CI runs 1.1.2 only, so
/// nothing older is declared on, rather than relying on an encoding no test
/// exercises.
///
/// A pre-release is gated on its core version (`1.1.2-rc1` counts as 1.1.2,
/// unlike strict SemVer precedence): a release candidate carries the behaviour
/// of the release it leads to.
pub const TAG_MIN_VERSION: VersionCore = (1, 1, 2);

/// `TAG_MIN_VERSION` as text, for messages.
pub const TAG_MIN_VERSION_STR: &str = "1.1.2";

/// What a KiviDB server reported about itself in `INFO server`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KividbServer {
    /// The raw `kividb_version` value, if the field was present.
    pub reported: Option<String>,
    /// `reported`, parsed. `None` when absent or not a recognised version.
    pub version: Option<KividbVersion>,
}

impl KividbServer {
    /// Read the server's identity out of an `INFO` (or `INFO server`) reply.
    pub fn from_info(info: &str) -> Self {
        let reported = raw_kividb_version(info).map(str::to_string);
        let version = reported.as_deref().and_then(parse_version);
        Self { reported, version }
    }

    /// Whether the engine may declare TAG fields on this server (`TAG
    /// SEPARATOR ;`): its version is known and at least [`TAG_MIN_VERSION`].
    pub fn tags_supported(&self) -> bool {
        self.version.is_some_and(|v| v.core >= TAG_MIN_VERSION)
    }

    /// One clause naming what the server reported, for error messages, so
    /// "why is my server rejected?" is answerable from the message alone.
    pub fn describe(&self) -> String {
        match (&self.reported, &self.version) {
            (None, _) => "`INFO server` reported no `kividb_version` field".to_string(),
            (Some(raw), None) => format!(
                "`INFO server` reported `kividb_version:{raw}`, which is not a \
                 `major.minor.patch[-pre][+build]` version"
            ),
            (Some(raw), Some(v)) if v.core < TAG_MIN_VERSION => format!(
                "`INFO server` reported `kividb_version:{raw}`, older than \
                 {TAG_MIN_VERSION_STR}, the oldest KiviDB this tool supports TAG fields on"
            ),
            (Some(raw), Some(_)) => format!("`INFO server` reported `kividb_version:{raw}`"),
        }
    }
}

/// The `SEPARATOR` each TAG attribute of an index reports in `FT.INFO`, by
/// attribute name (`None`: a TAG attribute that reports no `SEPARATOR`).
///
/// KiviDB lists `attributes` as flat `identifier, …, attribute, …, type, TAG,
/// SEPARATOR, …` arrays and reports a bare TAG as `SEPARATOR ,` (measured on
/// 1.1.2). An `FT.INFO` reply with no `attributes` section at all is
/// `Ok(None)`.
pub fn ft_info_tag_separators(
    info: &redis::Value,
) -> Result<Option<BTreeMap<String, Option<String>>>, String> {
    use crate::indexing_wait::{ft_info_pairs, value_text};
    let Some((_, attributes)) = ft_info_pairs(info)?
        .into_iter()
        .find(|(k, _)| k == "attributes")
    else {
        return Ok(None);
    };
    let redis::Value::Array(attributes) = attributes else {
        return Err(format!(
            "FT.INFO `attributes` is not an array: {attributes:?}"
        ));
    };
    let mut out = BTreeMap::new();
    for attr in attributes {
        let mut name = None;
        let mut is_tag = false;
        let mut separator = None;
        for (k, v) in ft_info_pairs(attr)? {
            match k.as_str() {
                "attribute" => name = value_text(v),
                "identifier" if name.is_none() => name = value_text(v),
                "type" => is_tag = value_text(v).as_deref() == Some("TAG"),
                "SEPARATOR" => separator = value_text(v),
                _ => {}
            }
        }
        if is_tag {
            let name =
                name.ok_or_else(|| format!("FT.INFO TAG attribute without a name: {attr:?}"))?;
            out.insert(name, separator);
        }
    }
    Ok(Some(out))
}

/// Check that an EXISTING index declares every TAG field in `tag_fields`
/// `SEPARATOR ;`, as `configure()` would have built it.
///
/// This is what makes `--skip-upload` safe to reuse an index built elsewhere:
/// an index built bare (by a binary that declared scalar TAGs bare, or by
/// hand) reports `SEPARATOR ,`, splits every stored `Smith, John`, and would
/// publish a recall for filters that matched the wrong documents.
/// `tags_supported` is [`KividbServer::tags_supported`]: on a server where it
/// is false no TAG declaration is known to be right, so a TAG-bearing index
/// cannot be verified at all.
///
/// `reported` is [`KividbServer::describe`], so every refusal names the version
/// the server reported.
pub fn check_index_tag_separators(
    index: &str,
    separators: Option<&BTreeMap<String, Option<String>>>,
    tag_fields: &[String],
    tags_supported: bool,
    reported: &str,
) -> Result<(), String> {
    if tag_fields.is_empty() {
        return Ok(());
    }
    if !tags_supported {
        return Err(format!(
            "KiviDB: cannot verify the TAG declarations of index `{index}`: {reported}, and \
             this tool declares TAG fields only on KiviDB >= {TAG_MIN_VERSION_STR}"
        ));
    }
    let rebuild = format!(
        "({reported}.) Re-run without `--skip-upload` so `configure()` rebuilds the index \
         with `TAG SEPARATOR ;`."
    );
    let Some(separators) = separators else {
        return Err(format!(
            "KiviDB: `FT.INFO {index}` reports no `attributes`, so the TAG declarations of \
             the index cannot be verified. {rebuild}"
        ));
    };
    for field in tag_fields {
        match separators.get(field) {
            Some(Some(sep)) if sep == ";" => {}
            Some(sep) => {
                let consequence = match sep.as_deref() {
                    Some(",") => "That splits a stored value such as `Smith, John` into \
                                  `Smith` + `John`, so a filter on the whole value \
                                  under-matches and a filter on a fragment over-matches."
                        .to_string(),
                    Some(other) => format!(
                        "That splits every stored value on {other:?}, so a filter on a whole \
                         value containing it under-matches and a filter on a fragment \
                         over-matches."
                    ),
                    None => "The server reports no SEPARATOR for it, so how it tokenises \
                             stored values is unknown."
                        .to_string(),
                };
                let sep = sep
                    .as_deref()
                    .map_or("none".to_string(), |s| format!("{s:?}"));
                return Err(format!(
                    "KiviDB: index `{index}` declares TAG field `{field}` with SEPARATOR \
                     {sep}, not \";\". {consequence} {rebuild}"
                ));
            }
            None => {
                return Err(format!(
                    "KiviDB: index `{index}` has no TAG field `{field}` (FT.INFO attributes: \
                     {:?}). {rebuild}",
                    separators.keys().collect::<Vec<_>>()
                ));
            }
        }
    }
    Ok(())
}

// ── TAG query values ─────────────────────────────────────────────────────
//
// How the `kividb` engine's filter builder writes a string between a TAG
// clause's braces. Here rather than in the engine so the live suite
// (`tests/integration_kividb.rs`) can query the engine-built index with
// exactly the clause the engine sends and assert exact document counts.

/// What goes between a TAG clause's braces for `value`: the value escaped
/// with [`escape_tag_value`], or an error if it cannot be matched exactly.
///
/// `tags_supported` is [`KividbServer::tags_supported`] (KiviDB >= 1.1.2).
/// The escaped form is only known to be right there; the release before
/// 1.0.5 matched escaped values against nothing. So on any other server a
/// TAG value is an error. The engine already refuses a dataset whose schema
/// declares a TAG field on such a server; this also covers a string
/// condition on a field the schema does not declare as a TAG.
pub fn tag_literal(field: &str, value: &str, tags_supported: bool) -> Result<String, String> {
    if !tags_supported {
        return Err(format!(
            "KiviDB filter: cannot encode TAG value {value:?} for field `{field}`: the \
             server's `kividb_version` is unknown or older than {TAG_MIN_VERSION_STR}, the \
             oldest KiviDB on which this tool's (escaped) TAG encoding is tested."
        ));
    }
    if let Some(problem) = tag_value_problem(value) {
        return Err(format!(
            "KiviDB filter: TAG value {value:?} for field `{field}` cannot be expressed \
             exactly on KiviDB: {problem}. Refusing to emit a filter that would report a \
             wrong recall."
        ));
    }
    Ok(escape_tag_value(value))
}

/// RediSearch-style TAG escaping, as KiviDB 1.1.2 accepts it: a backslash
/// before every ASCII punctuation character and before space.
///
/// Measured on `kividb_version:1.1.2` (a `TAG SEPARATOR ;` index): storing
/// `zz<c>zz`, `zz<c>` and `<c>zz` for each of the 32 ASCII punctuation
/// characters plus space and querying the escaped value bare, inside a
/// parenthesised OR AND'd with a second clause, and AND'd after another
/// clause returned exactly the expected documents in all three positions,
/// except the shapes [`tag_value_problem`] refuses. Raw, 22 of 146 probe
/// values were wrong: a raw `\` is read as an escape (`a\b` matched `ab`,
/// `J\"org Hennig` matched `J"org Hennig`), a raw `|` is the TAG-OR
/// separator (`x|y` matched `x` and `y`), and a raw `}` or leading `"` is a
/// syntax error. KiviDB 1.0.5 measured the same way, which is how the
/// shipped arxiv `submitter` values were found: escaped they return their
/// exact corpus counts (`J\"org Hennig` → 31, 0 raw; `Ulf-G. Mei{\ss}ner` →
/// 108, a syntax error raw).
///
/// Non-ASCII characters are passed through: non-ASCII punctuation and
/// fullwidth look-alikes (`’ – （ ） ｜ ＠ ＊ ＄`) and accented letters
/// matched exactly unescaped on 1.1.2. Escaping everything rather than a
/// minimal set is deliberate: it is the RediSearch rule, so every character
/// this emitter could meet in a value has one encoding with a measurement
/// behind it.
pub fn escape_tag_value(value: &str) -> String {
    let mut out = String::with_capacity(value.len() * 2);
    for c in value.chars() {
        // MUTATION (throwaway): escaping reverted, values sent raw.
        let _ = c;
        out.push(c);
    }
    out
}

/// Why `value` cannot be expressed exactly as a KiviDB TAG query value, or
/// `None` if it can. Measured on 1.1.2 (and on 1.0.5 before it), with the
/// value escaped by [`escape_tag_value`]:
///
/// * **Leading or trailing whitespace.** The query side trims it, and so
///   does the index: `@t:{\ sp}` matched the stored `sp`, ` sp` and `sp `.
///   Rust's `char::is_whitespace` is used, which covers the tab, U+3000
///   and U+2028 that were measured to be trimmed as well (on 1.0.5).
/// * **`;`**, the SEPARATOR every TAG field is declared with, so the stored
///   value was split into tokens and no query matches it whole
///   (`@t:{zz\;zz}` → 0).
/// * **A trailing `*`**, read as a prefix wildcard even when escaped:
///   `@t:{zz\*}` matched 73 documents (`zz`, `zzzz`, `zza`, `zz!zz`, …) and
///   a lone `\*` is a syntax error. A `*` elsewhere (`zz\*zz`, `\*zz`) is
///   exact.
/// * **A leading `$`**, read as a query parameter even when escaped:
///   `@t:{\$zz}` fails with `No such parameter`. That error is per query,
///   and `search()` drops a failed query with only a log line, so it is
///   refused here instead. A `$` elsewhere (`zz\$`, `zz\$zz`) is exact.
///
/// No shipped query value hits any of these. Only the QUERY side is
/// checked: a stored value with one of these shapes does not perturb any
/// other query (on 1.1.2 the stored `zz*`, `$zz` and `zz*zz` leave
/// `@t:{zz}` at exactly its own documents). Two corpus-side effects that do
/// reach other queries are in the README, because they are shared with
/// redis.rs's `vector` layout: TAG matching is case-insensitive, and stored
/// whitespace is trimmed (so `@t:{sp}` also matches a stored ` sp `).
pub fn tag_value_problem(value: &str) -> Option<String> {
    if value.starts_with(char::is_whitespace) || value.ends_with(char::is_whitespace) {
        Some(
            "it has leading or trailing whitespace, which KiviDB trims from a TAG query \
             value (`@t:{\\ sp}` is looked up as `sp`)"
                .to_string(),
        )
    } else if value.contains(';') {
        Some(
            "it contains `;`, the SEPARATOR every TAG field is declared with, so the \
             stored value was split into tokens and no query can match it whole"
                .to_string(),
        )
    } else if value.ends_with('*') {
        Some(
            "it ends in `*`, which KiviDB reads as a prefix wildcard even when escaped \
             (`@t:{zz\\*}` also matches `zz`, `zzab`, ...)"
                .to_string(),
        )
    } else if value.starts_with('$') {
        Some(
            "it starts with `$`, which KiviDB reads as a query parameter even when \
             escaped (`@t:{\\$zz}` fails with `No such parameter`)"
                .to_string(),
        )
    } else {
        None
    }
}

/// The raw `kividb_version` value in an `INFO` reply, trimmed.
fn raw_kividb_version(info: &str) -> Option<&str> {
    info.lines()
        .find_map(|line| line.trim().strip_prefix("kividb_version:"))
        .map(str::trim)
}

/// Parse `major.minor.patch[-prerelease][+build]` (SemVer shape). Anything else
/// (`1.1`, `v1.1.2`, `1.1.2.1`, `+1.1.2`) is `None`: an unrecognised version is
/// reported as such, never guessed at.
fn parse_version(value: &str) -> Option<KividbVersion> {
    // A SemVer pre-release or build identifier: non-empty, `[0-9A-Za-z.-]`.
    let valid_ident = |s: &str| {
        !s.is_empty()
            && s.chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '-')
    };
    // Build metadata never affects precedence, but must still be well-formed.
    let value = match value.split_once('+') {
        Some((v, build)) if valid_ident(build) => v,
        Some(_) => return None,
        None => value,
    };
    let (core, prerelease) = match value.split_once('-') {
        Some((core, pre)) if valid_ident(pre) => (core, true),
        Some(_) => return None,
        None => (value, false),
    };
    // `u32::from_str` would accept a leading `+`, but none survives here: the
    // build-metadata split above already cut the value at the first `+`.
    let num = |s: &str| s.parse::<u32>().ok();
    let mut parts = core.split('.');
    let major = num(parts.next()?)?;
    let minor = num(parts.next()?)?;
    let patch = num(parts.next()?)?;
    if parts.next().is_some() {
        return None;
    }
    Some(KividbVersion {
        core: (major, minor, patch),
        prerelease,
    })
}

/// Extract and parse `kividb_version` from an `INFO` (or `INFO server`) reply.
pub fn parse_kividb_version(info: &str) -> Option<KividbVersion> {
    KividbServer::from_info(info).version
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The head of a real KiviDB 1.1.2 `INFO server` reply.
    const INFO_112: &str =
        "# Server\r\nredis_version:8.10.0\r\nkividb_version:1.1.2\r\ngcc_version:0.0.0\r\n";

    fn info(v: &str) -> String {
        format!("# Server\r\nkividb_version:{v}\r\n")
    }

    fn supported(v: &str) -> bool {
        KividbServer::from_info(&info(v)).tags_supported()
    }

    #[test]
    fn parses_the_field_out_of_a_real_info_reply() {
        assert_eq!(
            parse_kividb_version(INFO_112),
            Some(KividbVersion {
                core: (1, 1, 2),
                prerelease: false
            })
        );
        assert!(KividbServer::from_info(INFO_112).tags_supported());
    }

    #[test]
    fn does_not_mistake_redis_version_for_kividb_version() {
        // A plain Redis (or a KiviDB build without the field) reports only
        // redis_version; that must read as "unknown", not as 8.10.0.
        let s = KividbServer::from_info("redis_version:8.10.0\r\n");
        assert_eq!(s.version, None);
        assert!(!s.tags_supported());
    }

    #[test]
    fn rejects_malformed_versions() {
        for v in [
            "1.1",
            "1.1.2.1",
            "x.y.z",
            "",
            "v1.1.2",
            "+1.1.2",
            "1.+1.2",
            "1.1.2-",
            "1.1.2-r c",
            "1.1.2+",
            "1.1.2+b d",
        ] {
            assert_eq!(parse_kividb_version(&info(v)), None, "{v:?} must not parse");
            assert!(!supported(v), "{v:?} must read as unknown, so unsupported");
        }
    }

    #[test]
    fn parses_prerelease_and_build_metadata() {
        let v = parse_kividb_version(&info("1.2.0-rc2")).unwrap();
        assert_eq!(v.core, (1, 2, 0));
        assert!(v.prerelease);
        let v = parse_kividb_version(&info("1.1.2+build.7")).unwrap();
        assert_eq!(v.core, (1, 1, 2));
        assert!(!v.prerelease);
    }

    /// Both sides of the gate at `TAG_MIN_VERSION`, plus the numeric-ordering
    /// traps.
    #[test]
    fn tags_are_supported_from_the_minimum_version() {
        for (v, want) in [
            // Older than the current stable release: unsupported.
            ("0.10.5", false),
            ("0.9.9", false),
            ("1.1.0-rc1", false),
            ("1.1.0", false),
            ("1.1.1-rc3", false),
            ("1.1.1", false),
            // The current stable release, its pre-releases (gated on their
            // core version) and everything newer: supported.
            ("1.1.2-rc1", true),
            ("1.1.2", true),
            ("1.1.3", true),
            ("1.2.0-rc2", true),
            ("1.2.0", true),
            ("2.0.0", true),
            // "1.1.10" < "1.1.2" as strings; as a version it is newer.
            ("1.1.10", true),
        ] {
            assert_eq!(supported(v), want, "kividb_version:{v}");
        }
    }

    fn bulk(s: &str) -> redis::Value {
        redis::Value::BulkString(s.as_bytes().to_vec())
    }

    /// The RESP2 `FT.INFO` shape KiviDB 1.1.2 returns (measured), with one
    /// TAG attribute per `(name, SEPARATOR)` and one NUMERIC.
    fn ft_info_112(tags: &[(&str, &str)]) -> redis::Value {
        let mut attrs: Vec<redis::Value> = tags
            .iter()
            .map(|(n, sep)| {
                redis::Value::Array(
                    [
                        "identifier",
                        n,
                        "attribute",
                        n,
                        "type",
                        "TAG",
                        "SEPARATOR",
                        sep,
                    ]
                    .iter()
                    .map(|s| bulk(s))
                    .collect(),
                )
            })
            .collect();
        attrs.push(redis::Value::Array(
            ["identifier", "n", "attribute", "n", "type", "NUMERIC"]
                .iter()
                .map(|s| bulk(s))
                .collect(),
        ));
        redis::Value::Array(vec![
            bulk("index_name"),
            bulk("idx:x"),
            bulk("attributes"),
            redis::Value::Array(attrs),
            bulk("num_docs"),
            bulk("0"),
        ])
    }

    /// An `FT.INFO` reply without an `attributes` section.
    fn ft_info_without_attributes() -> redis::Value {
        redis::Value::Array(vec![
            bulk("index_name"),
            bulk("idx:x"),
            bulk("num_fields"),
            bulk("1"),
        ])
    }

    fn check(info: &redis::Value, fields: &[&str], tags_supported: bool) -> Result<(), String> {
        let seps = ft_info_tag_separators(info).unwrap();
        let fields: Vec<String> = fields.iter().map(|s| s.to_string()).collect();
        check_index_tag_separators("idx:x", seps.as_ref(), &fields, tags_supported, REPORTED)
    }

    const REPORTED: &str = "`INFO server` reported `kividb_version:1.1.3`";

    #[test]
    fn reads_tag_separators_from_ft_info() {
        let seps = ft_info_tag_separators(&ft_info_112(&[("s", ","), ("t", ";")]))
            .unwrap()
            .unwrap();
        assert_eq!(seps.get("s"), Some(&Some(",".to_string())));
        assert_eq!(seps.get("t"), Some(&Some(";".to_string())));
        assert!(!seps.contains_key("n"), "NUMERIC is not a TAG");
        assert_eq!(
            ft_info_tag_separators(&ft_info_without_attributes()).unwrap(),
            None
        );
    }

    /// #375 review, optional item 2: `--skip-upload` must refuse an index whose
    /// TAG fields the server splits on `,` (built bare, e.g. by a binary that
    /// declared scalar TAGs bare).
    #[test]
    fn existing_index_must_declare_semicolon() {
        let good = ft_info_112(&[("s", ";"), ("labels", ";")]);
        check(&good, &["s", "labels"], true).unwrap();
        let stale = ft_info_112(&[("s", ","), ("labels", ";")]);
        let err = check(&stale, &["s", "labels"], true).unwrap_err();
        assert!(
            err.contains("`s`") && err.contains("SEPARATOR \",\""),
            "{err}"
        );
        let err = check(&good, &["s", "missing"], true).unwrap_err();
        assert!(err.contains("no TAG field `missing`"), "{err}");
        let bare = ft_info_without_attributes();
        assert!(check(&bare, &["s"], true).is_err(), "unverifiable");
        // On an unsupported server nothing can be verified, not even a
        // `SEPARATOR ;` index.
        let err = check(&good, &["s"], false).unwrap_err();
        assert!(err.contains(TAG_MIN_VERSION_STR), "{err}");
        // No TAG field: nothing to check, whatever the server.
        check(&bare, &[], false).unwrap();
        // Every refusal names what the server reported.
        let stale_err = check(&stale, &["s", "labels"], true).unwrap_err();
        for e in [
            stale_err.clone(),
            check(&bare, &["s"], true).unwrap_err(),
            check(&good, &["s"], false).unwrap_err(),
        ] {
            assert!(e.contains("kividb_version:1.1.3"), "{e}");
        }
        // The `Smith, John` story is told only for a `,` separator.
        assert!(stale_err.contains("Smith, John"), "{stale_err}");
        let pipe = check(&ft_info_112(&[("s", "|")]), &["s"], true).unwrap_err();
        assert!(pipe.contains("\"|\"") && !pipe.contains("Smith"), "{pipe}");
    }

    /// A TAG attribute that reports no `SEPARATOR` at all (`Some(None)`) is
    /// refused, and the two malformed `FT.INFO` shapes are errors rather than
    /// an empty (vacuously passing) map.
    #[test]
    fn tag_attribute_without_separator_and_malformed_ft_info_are_refused() {
        let attr = |pairs: &[&str]| redis::Value::Array(pairs.iter().map(|s| bulk(s)).collect());
        let info = |attrs: redis::Value| {
            redis::Value::Array(vec![
                bulk("index_name"),
                bulk("idx:x"),
                bulk("attributes"),
                attrs,
            ])
        };
        let no_sep = info(redis::Value::Array(vec![attr(&[
            "identifier",
            "s",
            "attribute",
            "s",
            "type",
            "TAG",
        ])]));
        assert_eq!(
            ft_info_tag_separators(&no_sep).unwrap().unwrap().get("s"),
            Some(&None)
        );
        let e = check(&no_sep, &["s"], true).unwrap_err();
        assert!(e.contains("SEPARATOR none"), "{e}");

        let not_array = info(bulk("oops"));
        let e = ft_info_tag_separators(&not_array).unwrap_err();
        assert!(e.contains("not an array"), "{e}");

        let nameless = info(redis::Value::Array(vec![attr(&["type", "TAG"])]));
        let e = ft_info_tag_separators(&nameless).unwrap_err();
        assert!(e.contains("without a name"), "{e}");
    }

    #[test]
    fn describe_names_what_the_server_reported() {
        let absent = KividbServer::from_info("redis_version:8.10.0\r\n").describe();
        assert!(absent.contains("no `kividb_version`"), "{absent}");
        let garbage = KividbServer::from_info(&info("v1.1.2")).describe();
        assert!(
            garbage.contains("`kividb_version:v1.1.2`") && garbage.contains("not a"),
            "{garbage}"
        );
        let old = KividbServer::from_info(&info("1.1.1")).describe();
        assert!(
            old.contains("`kividb_version:1.1.1`") && old.contains("older than 1.1.2"),
            "{old}"
        );
        let current = KividbServer::from_info(&info("1.1.2")).describe();
        assert!(current.contains("`kividb_version:1.1.2`"), "{current}");
        assert!(
            !current.contains("not a") && !current.contains("older"),
            "{current}"
        );
    }
    #[test]
    fn the_inexpressible_set_is_semicolon_trailing_star_leading_dollar() {
        for (v, needle) in [
            ("a;b", "`;`"),
            (";", "`;`"),
            ("zz*", "prefix wildcard"),
            ("a*b*", "prefix wildcard"),
            ("*", "prefix wildcard"),
            ("$p", "query parameter"),
            ("$", "query parameter"),
        ] {
            let p = tag_value_problem(v).expect(v);
            assert!(p.contains(needle), "{v}: {p}");
            let e = tag_literal("c", v, true).unwrap_err();
            assert!(e.contains(needle), "{v}: {e}");
        }
        // Measured exact once escaped on 1.1.2: `*` / `$` anywhere else,
        // and the four characters the early release could not express.
        for v in [
            "a*b",
            "*zz",
            "a$b",
            "zz$",
            "a|b",
            "a@b",
            "Bag (Small)",
            "x{y}z",
            "a\\b",
        ] {
            assert_eq!(tag_value_problem(v), None, "{v}");
        }
        for c in "-_.:/\\+*?~^$#%&[]{}<>=!,'\"`|@() ".chars() {
            assert_eq!(
                tag_value_problem(&format!("zz{c}zz")),
                None,
                "{c:?} must be expressible (escaped) on KiviDB 1.1.2"
            );
        }
    }

    #[test]
    fn leading_or_trailing_whitespace_is_inexpressible() {
        for v in [
            " sp",
            "sp ",
            " sp ",
            "\tlead",
            "trail\t",
            "zz\u{3000}",
            "\u{2028}zz",
        ] {
            let p = tag_value_problem(v).unwrap_or_else(|| panic!("{v:?}"));
            assert!(p.contains("whitespace"), "{v:?}: {p}");
        }
        for v in ["two  spaces", "tab\there", "sp"] {
            assert_eq!(tag_value_problem(v), None, "{v:?}");
        }
        let e = tag_literal("c", " sp", true).unwrap_err();
        assert!(e.contains("whitespace"), "{e}");
    }

    #[test]
    fn every_ascii_punctuation_character_and_space_is_escaped() {
        // The measured set: all 32 ASCII punctuation characters plus space.
        let specials: String = (0u8..128)
            .map(char::from)
            .filter(|c| c.is_ascii_punctuation() || *c == ' ')
            .collect();
        assert_eq!(specials.len(), 33);
        for c in specials.chars() {
            assert_eq!(
                escape_tag_value(&format!("a{c}b")),
                format!("a\\{c}b"),
                "{c:?}"
            );
        }
        // Alphanumerics, other ASCII whitespace inside a value, and every
        // non-ASCII character (measured exact unescaped) pass through.
        for v in [
            "abcXYZ019",
            "tab\there",
            "Jörg Müller",
            "王小明",
            "（）｜＠＊＄",
            "nbsp\u{a0}x",
        ] {
            let want: String = v
                .chars()
                .flat_map(|c| (c == ' ').then_some('\\').into_iter().chain([c]))
                .collect();
            assert_eq!(escape_tag_value(v), want, "{v}");
        }
    }

    /// The escaped form is only known to be right on a server TAG fields are
    /// supported on; anywhere else every TAG value is refused, even one that
    /// needs no escaping.
    #[test]
    fn tag_literal_escapes_on_a_supported_server_and_refuses_elsewhere() {
        assert_eq!(tag_literal("c", "red", true).unwrap(), "red");
        assert_eq!(
            tag_literal("s", r"Rasmus Bj{\o}rk", true).unwrap(),
            r"Rasmus\ Bj\{\\o\}rk"
        );
        for v in ["red", r"a\b"] {
            let e = tag_literal("c", v, false).unwrap_err();
            assert!(e.contains("unknown or older than 1.1.2"), "{e}");
        }
    }
}
