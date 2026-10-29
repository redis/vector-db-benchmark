//! Server capabilities of a KiviDB instance that change what the `kividb`
//! engine may safely do, decided from the version the server reports.
//!
//! Lives in the library (not `src/bin/`) so the version gate is unit-tested:
//! a gate that silently opens on an older server would index a multi-valued
//! field into a TAG that matches nothing and publish recall 0.0 for a filter
//! that was never applied (the failure `validate_dataset_support` exists to
//! prevent). A gate that silently stays shut on a NEWER server is just as
//! wrong, the other way round: from 1.0.5 a bare `TAG` splits on `,` (see
//! [`TagSemantics`]), so a scalar keyword value containing a comma would stop
//! matching itself.

use std::collections::BTreeMap;

/// The `major.minor.patch` core of a KiviDB version.
pub type VersionCore = (u32, u32, u32);

/// A parsed `kividb_version`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KividbVersion {
    pub core: VersionCore,
    /// Whether the reported value carried a SemVer pre-release suffix
    /// (`1.0.5-rc1`). Recorded, but NOT used to order versions: see
    /// [`TAG_SEPARATOR_MIN_VERSION`].
    pub prerelease: bool,
}

/// First KiviDB version whose TAG fields follow RediSearch semantics: `FT.CREATE`
/// accepts `SEPARATOR` and splits stored values on it, and a bare `TAG` (no
/// `SEPARATOR`) splits on RediSearch's default `,`.
///
/// Measured on every published image around the boundary, by declaring a bare
/// TAG and a `TAG SEPARATOR ;`, storing `Smith, John` / `a;b;c`, and querying
/// `@s:{Smith}`, `@s:{Smith, John}` and `@l:{b}`:
///
/// | `kividb_version`                    | `SEPARATOR`              | bare TAG        |
/// |-------------------------------------|--------------------------|-----------------|
/// | 1.0.2, 1.0.3, 1.0.4-rc3, 1.0.4      | `ERR unknown field type` | atomic          |
/// | 1.0.5-rc1, 1.0.5, 1.0.6-rc4         | accepted, splits         | splits on `,`   |
///
/// The pre-releases behave like their own release, so a pre-release is gated
/// on its core version (`1.0.5-rc1` is >= 1.0.5 here, unlike strict SemVer
/// precedence). Treating it as older would put a `1.0.5-rc1` server on the bare
/// TAG declaration, which that server splits on `,`.
pub const TAG_SEPARATOR_MIN_VERSION: VersionCore = (1, 0, 5);

/// How a server treats TAG fields, which decides how the engine must declare
/// them for an exact whole-value match.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TagSemantics {
    /// Before 1.0.5: `SEPARATOR` is rejected and a bare TAG value is ONE atomic
    /// token. Scalar fields are declared bare; a multi-valued field cannot be
    /// indexed at all (`@f:{b}` never matches a stored `a;b;c`).
    Atomic,
    /// From 1.0.5: RediSearch semantics. Every TAG field must be declared
    /// `SEPARATOR ;` (as redis.rs does): bare, a stored `Smith, John` is split
    /// into `Smith` / `John` and `@f:{Smith, John}` matches nothing.
    Separator,
}

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

    /// The server's TAG semantics, or `None` when its version is unknown. An
    /// unknown server gets NO TAG declaration: neither choice is safe without
    /// knowing which semantics it has (bare splits on `,` on a new server;
    /// `SEPARATOR` is rejected on an old one, and a multi-valued field cannot be
    /// indexed there at all), so the engine refuses TAG-bearing schemas instead.
    pub fn tag_semantics(&self) -> Option<TagSemantics> {
        self.version.map(|v| {
            if v.core >= TAG_SEPARATOR_MIN_VERSION {
                TagSemantics::Separator
            } else {
                TagSemantics::Atomic
            }
        })
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
            (Some(raw), Some(_)) => format!("`INFO server` reported `kividb_version:{raw}`"),
        }
    }
}

/// The `SEPARATOR` each TAG attribute of an index reports in `FT.INFO`, by
/// attribute name (`None`: a TAG attribute that reports no `SEPARATOR`).
///
/// KiviDB >= 1.0.5 lists `attributes` as flat `identifier, …, attribute, …,
/// type, TAG, SEPARATOR, …` arrays and reports a bare TAG as `SEPARATOR ,`
/// (measured). KiviDB 1.0.2 has no `attributes` section at all: `Ok(None)`.
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

/// Check that an EXISTING index declares every TAG field in `tag_fields` the
/// way a server with `tags` semantics requires.
///
/// This is what makes `--skip-upload` safe to reuse an index built elsewhere:
/// on KiviDB >= 1.0.5 an index built bare (by a binary predating this check,
/// whose scalar TAGs were bare, or by hand) reports `SEPARATOR ,`, splits every
/// stored `Smith, John`, and would publish a recall for filters that matched
/// the wrong documents. Before 1.0.5 there is nothing to check (no `SEPARATOR`
/// can be declared, and `FT.INFO` reports no attributes), so `Atomic` passes.
///
/// `reported` is [`KividbServer::describe`], so every refusal names the version
/// the server reported.
pub fn check_index_tag_separators(
    index: &str,
    separators: Option<&BTreeMap<String, Option<String>>>,
    tag_fields: &[String],
    tags: Option<TagSemantics>,
    reported: &str,
) -> Result<(), String> {
    if tag_fields.is_empty() {
        return Ok(());
    }
    let rebuild = format!(
        "({reported}.) Re-run without `--skip-upload` so `configure()` rebuilds the index \
         with `TAG SEPARATOR ;`."
    );
    match tags {
        Some(TagSemantics::Atomic) => Ok(()),
        None => Err(format!(
            "KiviDB: cannot verify the TAG declarations of index `{index}`: {reported}, so \
             the server's TAG semantics are unknown"
        )),
        Some(TagSemantics::Separator) => {
            let Some(separators) = separators else {
                return Err(format!(
                    "KiviDB: `FT.INFO {index}` reports no `attributes`, so the TAG \
                     declarations of a >= 1.0.5 server's index cannot be verified. {rebuild}"
                ));
            };
            for field in tag_fields {
                match separators.get(field) {
                    Some(Some(sep)) if sep == ";" => {}
                    Some(sep) => {
                        let consequence = match sep.as_deref() {
                            Some(",") => "On KiviDB >= 1.0.5 that splits a stored value such \
                                          as `Smith, John` into `Smith` + `John`, so a filter \
                                          on the whole value under-matches and a filter on a \
                                          fragment over-matches."
                                .to_string(),
                            Some(other) => format!(
                                "On KiviDB >= 1.0.5 that splits every stored value on \
                                 {other:?}, so a filter on a whole value containing it \
                                 under-matches and a filter on a fragment over-matches."
                            ),
                            None => "The server reports no SEPARATOR for it, so how it \
                                     tokenises stored values is unknown."
                                .to_string(),
                        };
                        let sep = sep
                            .as_deref()
                            .map_or("none".to_string(), |s| format!("{s:?}"));
                        return Err(format!(
                            "KiviDB: index `{index}` declares TAG field `{field}` with \
                             SEPARATOR {sep}, not \";\". {consequence} {rebuild}"
                        ));
                    }
                    None => {
                        return Err(format!(
                            "KiviDB: index `{index}` has no TAG field `{field}` (FT.INFO \
                             attributes: {:?}). {rebuild}",
                            separators.keys().collect::<Vec<_>>()
                        ));
                    }
                }
            }
            Ok(())
        }
    }
}

/// The raw `kividb_version` value in an `INFO` reply, trimmed.
fn raw_kividb_version(info: &str) -> Option<&str> {
    info.lines()
        .find_map(|line| line.trim().strip_prefix("kividb_version:"))
        .map(str::trim)
}

/// Parse `major.minor.patch[-prerelease][+build]` (SemVer shape). Anything else
/// (`1.0`, `v1.0.5`, `1.0.5.1`, `+1.0.5`) is `None`: an unrecognised version is
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

    const INFO_105: &str =
        "# Server\r\nredis_version:8.10.0\r\nkividb_version:1.0.5\r\ngcc_version:0.0.0\r\n";

    fn info(v: &str) -> String {
        format!("# Server\r\nkividb_version:{v}\r\n")
    }

    fn semantics(v: &str) -> Option<TagSemantics> {
        KividbServer::from_info(&info(v)).tag_semantics()
    }

    #[test]
    fn parses_the_field_out_of_a_real_info_reply() {
        assert_eq!(
            parse_kividb_version(INFO_105),
            Some(KividbVersion {
                core: (1, 0, 5),
                prerelease: false
            })
        );
    }

    #[test]
    fn does_not_mistake_redis_version_for_kividb_version() {
        // A plain Redis (or a KiviDB build without the field) reports only
        // redis_version; that must read as "unknown", not as 8.10.0.
        let s = KividbServer::from_info("redis_version:8.10.0\r\n");
        assert_eq!(s.version, None);
        assert_eq!(s.tag_semantics(), None);
    }

    #[test]
    fn rejects_malformed_versions() {
        for v in [
            "1.0",
            "1.0.5.1",
            "x.y.z",
            "",
            "v1.0.5",
            "+1.0.5",
            "1.+0.5",
            "1.0.5-",
            "1.0.5-r c",
            "1.0.5+",
            "1.0.5+b d",
        ] {
            assert_eq!(parse_kividb_version(&info(v)), None, "{v:?} must not parse");
            assert_eq!(semantics(v), None, "{v:?} must read as unknown");
        }
    }

    #[test]
    fn parses_prerelease_and_build_metadata() {
        let v = parse_kividb_version(&info("1.0.6-rc4")).unwrap();
        assert_eq!(v.core, (1, 0, 6));
        assert!(v.prerelease);
        let v = parse_kividb_version(&info("1.0.6+build.7")).unwrap();
        assert_eq!(v.core, (1, 0, 6));
        assert!(!v.prerelease);
    }

    /// The measured boundary (`TAG_SEPARATOR_MIN_VERSION`'s table), one row per
    /// image probed, plus the numeric-ordering traps.
    #[test]
    fn tag_semantics_follow_the_measured_boundary() {
        use TagSemantics::*;
        for (v, want) in [
            ("1.0.2", Atomic),
            ("1.0.3", Atomic),
            ("1.0.4-rc3", Atomic),
            ("1.0.4", Atomic),
            // Measured: 1.0.5-rc1 already splits a bare TAG on `,`.
            ("1.0.5-rc1", Separator),
            ("1.0.5", Separator),
            ("1.0.6-rc4", Separator),
            ("1.1.0", Separator),
            ("2.0.0", Separator),
            // "1.0.10" < "1.0.5" as strings; as a version it is newer.
            ("1.0.10", Separator),
            ("0.10.5", Atomic),
        ] {
            assert_eq!(semantics(v), Some(want), "kividb_version:{v}");
        }
    }

    fn bulk(s: &str) -> redis::Value {
        redis::Value::BulkString(s.as_bytes().to_vec())
    }

    /// The RESP2 `FT.INFO` shape KiviDB 1.0.5 returns (measured), with one
    /// TAG attribute per `(name, SEPARATOR)` and one NUMERIC.
    fn ft_info_105(tags: &[(&str, &str)]) -> redis::Value {
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

    fn check(
        info: &redis::Value,
        fields: &[&str],
        tags: Option<TagSemantics>,
    ) -> Result<(), String> {
        let seps = ft_info_tag_separators(info).unwrap();
        let fields: Vec<String> = fields.iter().map(|s| s.to_string()).collect();
        check_index_tag_separators("idx:x", seps.as_ref(), &fields, tags, REPORTED)
    }

    const REPORTED: &str = "`INFO server` reported `kividb_version:1.0.6`";

    #[test]
    fn reads_tag_separators_from_ft_info() {
        let seps = ft_info_tag_separators(&ft_info_105(&[("s", ","), ("t", ";")]))
            .unwrap()
            .unwrap();
        assert_eq!(seps.get("s"), Some(&Some(",".to_string())));
        assert_eq!(seps.get("t"), Some(&Some(";".to_string())));
        assert!(!seps.contains_key("n"), "NUMERIC is not a TAG");
        // KiviDB 1.0.2 reports no attributes at all.
        let v102 = redis::Value::Array(vec![
            bulk("index_name"),
            bulk("idx:x"),
            bulk("num_fields"),
            bulk("1"),
        ]);
        assert_eq!(ft_info_tag_separators(&v102).unwrap(), None);
    }

    /// #375 review, optional item 2: `--skip-upload` must refuse an index whose
    /// TAG fields a >= 1.0.5 server splits on `,` (built bare, e.g. by a binary
    /// predating this check).
    #[test]
    fn existing_index_must_declare_semicolon_on_separator_servers() {
        use TagSemantics::*;
        let good = ft_info_105(&[("s", ";"), ("labels", ";")]);
        check(&good, &["s", "labels"], Some(Separator)).unwrap();
        let stale = ft_info_105(&[("s", ","), ("labels", ";")]);
        let err = check(&stale, &["s", "labels"], Some(Separator)).unwrap_err();
        assert!(
            err.contains("`s`") && err.contains("SEPARATOR \",\""),
            "{err}"
        );
        let err = check(&good, &["s", "missing"], Some(Separator)).unwrap_err();
        assert!(err.contains("no TAG field `missing`"), "{err}");
        let v102 = redis::Value::Array(vec![bulk("index_name"), bulk("idx:x")]);
        assert!(
            check(&v102, &["s"], Some(Separator)).is_err(),
            "unverifiable on >= 1.0.5"
        );
        // Before 1.0.5 nothing can be declared, so nothing is checked.
        check(&v102, &["s"], Some(Atomic)).unwrap();
        check(&stale, &["s"], Some(Atomic)).unwrap();
        assert!(check(&good, &["s"], None).is_err());
        // No TAG field: nothing to check, whatever the server.
        check(&v102, &[], None).unwrap();
        // Every refusal names what the server reported.
        let stale_err = check(&stale, &["s", "labels"], Some(Separator)).unwrap_err();
        for e in [
            stale_err.clone(),
            check(&v102, &["s"], Some(Separator)).unwrap_err(),
            check(&good, &["s"], None).unwrap_err(),
        ] {
            assert!(e.contains("kividb_version:1.0.6"), "{e}");
        }
        // The `Smith, John` story is told only for a `,` separator.
        assert!(stale_err.contains("Smith, John"), "{stale_err}");
        let pipe = check(&ft_info_105(&[("s", "|")]), &["s"], Some(Separator)).unwrap_err();
        assert!(pipe.contains("\"|\"") && !pipe.contains("Smith"), "{pipe}");
    }

    /// A TAG attribute that reports no `SEPARATOR` at all (`Some(None)`) is
    /// refused on a >= 1.0.5 server, and the two malformed `FT.INFO` shapes
    /// are errors rather than an empty (vacuously passing) map.
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
        let e = check(&no_sep, &["s"], Some(TagSemantics::Separator)).unwrap_err();
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
        let garbage = KividbServer::from_info(&info("v1.0.5")).describe();
        assert!(
            garbage.contains("`kividb_version:v1.0.5`") && garbage.contains("not a"),
            "{garbage}"
        );
        let old = KividbServer::from_info(&info("1.0.2")).describe();
        assert!(old.contains("`kividb_version:1.0.2`"), "{old}");
        assert!(!old.contains("not a"), "{old}");
    }
}
