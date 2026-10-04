//! Server capabilities of a KiviDB instance that change what the `kividb`
//! engine may safely do, decided from the version the server reports.
//!
//! Lives in the library (not `src/bin/`) so the version gate is unit-tested:
//! a gate that silently opens on an older server would index a multi-valued
//! field into a TAG that matches nothing and publish recall 0.0 for a filter
//! that was never applied (the failure `validate_dataset_support` exists to
//! prevent).

/// A `major.minor.patch` KiviDB version.
pub type KividbVersion = (u32, u32, u32);

/// First KiviDB release whose `FT.CREATE` accepts a TAG `SEPARATOR` and splits
/// stored values on it, so `@labels:{b}` matches a stored `a;b;c`. Measured on
/// 1.0.5 against Redis Stack with the same fixture; 1.0.2 rejects the modifier
/// (`ERR unknown field type`) and stores TAG values atomically.
pub const MULTIVALUED_TAG_MIN_VERSION: KividbVersion = (1, 0, 5);

/// Extract `kividb_version` from an `INFO` (or `INFO server`) reply.
///
/// Returns `None` when the field is absent or not a `x.y.z` triple, so callers
/// treat an unknown server as the OLDEST capability set rather than guessing.
/// Pre-release suffixes (`1.0.5-rc1`) are deliberately NOT accepted: an rc of
/// 1.0.5 predates the behaviour 1.0.5 ships, and rejecting it errs towards the
/// hard error rather than a silently wrong filter.
pub fn parse_kividb_version(info: &str) -> Option<KividbVersion> {
    let value = info
        .lines()
        .find_map(|line| line.trim().strip_prefix("kividb_version:"))?
        .trim();
    let mut parts = value.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next()?.parse().ok()?;
    let patch = parts.next()?.parse().ok()?;
    if parts.next().is_some() {
        return None;
    }
    Some((major, minor, patch))
}

/// Whether a server of `version` indexes a multi-valued keyword field faithfully
/// as `TAG SEPARATOR ;`. `None` (version unknown) is `false`.
pub fn supports_multivalued_tags(version: Option<KividbVersion>) -> bool {
    matches!(version, Some(v) if v >= MULTIVALUED_TAG_MIN_VERSION)
}

#[cfg(test)]
mod tests {
    use super::*;

    const INFO_105: &str =
        "# Server\r\nredis_version:8.10.0\r\nkividb_version:1.0.5\r\ngcc_version:0.0.0\r\n";

    #[test]
    fn parses_the_field_out_of_a_real_info_reply() {
        assert_eq!(parse_kividb_version(INFO_105), Some((1, 0, 5)));
    }

    #[test]
    fn does_not_mistake_redis_version_for_kividb_version() {
        // A plain Redis (or a KiviDB build without the field) reports only
        // redis_version; that must read as "unknown", not as 8.10.0.
        assert_eq!(parse_kividb_version("redis_version:8.10.0\r\n"), None);
    }

    #[test]
    fn rejects_malformed_and_prerelease_versions() {
        for v in ["1.0", "1.0.5-rc1", "1.0.5.1", "x.y.z", ""] {
            let info = format!("kividb_version:{v}\r\n");
            assert_eq!(parse_kividb_version(&info), None, "{v:?} must not parse");
        }
    }

    #[test]
    fn gate_opens_exactly_at_1_0_5() {
        assert!(!supports_multivalued_tags(None));
        assert!(!supports_multivalued_tags(Some((1, 0, 2))));
        assert!(!supports_multivalued_tags(Some((1, 0, 4))));
        assert!(supports_multivalued_tags(Some((1, 0, 5))));
        assert!(supports_multivalued_tags(Some((1, 1, 0))));
        assert!(supports_multivalued_tags(Some((2, 0, 0))));
    }

    #[test]
    fn compares_numerically_not_lexically() {
        // "1.0.10" < "1.0.5" as strings; as a version it is newer.
        assert!(supports_multivalued_tags(parse_kividb_version(
            "kividb_version:1.0.10\r\n"
        )));
        assert!(!supports_multivalued_tags(parse_kividb_version(
            "kividb_version:0.10.5\r\n"
        )));
    }
}
