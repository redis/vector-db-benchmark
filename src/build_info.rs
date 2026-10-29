//! Which source these binaries were built from, and how (#366).
//!
//! `build.rs` resolves the commit with [`resolve`] and passes the answer in
//! through `VDB_*` compile-time variables; this module exposes it, adds the
//! versions of the C libraries actually loaded at run time, and renders the
//! `params.build` block of result files. The resolution rules, and why
//! "unknown" is preferred to a guess, are on [`resolve`].

pub mod resolve;

/// The crate version (`Cargo.toml`).
pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Full commit SHA, or the literal `unknown`.
pub const GIT_SHA: &str = env!("VDB_GIT_SHA");

/// `clean`, `dirty` or `unknown` — see [`resolve::tree_state`].
pub const GIT_TREE: &str = env!("VDB_GIT_TREE");

/// `git`, `env` or `unknown` — see [`resolve::Origin`].
pub const GIT_SHA_SOURCE: &str = env!("VDB_GIT_SHA_SOURCE");

/// What every binary prints after its name for `--version`, e.g.
/// `0.1.0 (git 0123…cdef)`. Built by [`resolve::version_line`].
pub const LONG_VERSION: &str = env!("VDB_LONG_VERSION");

/// Target triple the binaries were compiled for, e.g.
/// `x86_64-unknown-linux-gnu`.
pub const BUILD_TARGET: &str = env!("VDB_BUILD_TARGET");

/// Cargo profile the crate was compiled with: `release` or `debug`.
pub const BUILD_PROFILE: &str = env!("VDB_BUILD_PROFILE");

/// `rustc -V` of the compiler that built the crate, or `unknown`.
pub const RUSTC_VERSION: &str = env!("VDB_RUSTC_VERSION");

/// Comma-separated cargo features, derived by `build.rs` from cargo's own
/// `CARGO_FEATURE_*` variables (see [`resolve::features_from_env`]).
const FEATURES: &str = env!("VDB_FEATURES");

/// The non-default cargo features compiled in. `release-static` (what the
/// release archives are built with) swaps the client stack: vendored OpenSSL
/// and bundled HDF5 instead of the system libraries, and TLS cost sits in the
/// timed path of the managed engines.
pub fn features() -> Vec<&'static str> {
    FEATURES.split(',').filter(|f| !f.is_empty()).collect()
}

/// The OpenSSL the process actually runs with (`OpenSSL_version()`), e.g.
/// `OpenSSL 3.0.2 15 Mar 2022` from the system library, or the vendored one
/// in a release archive. `None` on macOS, where TLS goes through
/// Security.framework and OpenSSL is not linked.
pub fn openssl_runtime_version() -> Option<String> {
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        Some(openssl::version::version().to_string())
    }
    #[cfg(not(all(unix, not(target_os = "macos"))))]
    {
        None
    }
}

/// The HDF5 library linked in (`H5get_libversion`), e.g. `1.10.7`.
pub fn hdf5_runtime_version() -> String {
    let (a, b, c) = hdf5::library_version();
    format!("{a}.{b}.{c}")
}

/// Everything `params.build` records, gathered so the rendering is a pure
/// function that tests can feed any value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuildFacts {
    pub crate_version: String,
    /// Full SHA, or `unknown`.
    pub git_sha: String,
    pub git_tree: String,
    pub git_sha_source: String,
    pub target: String,
    pub profile: String,
    pub features: Vec<String>,
    pub rustc: String,
    pub openssl: Option<String>,
    pub hdf5: String,
}

impl BuildFacts {
    /// This binary's facts: the compiled-in constants plus the libraries
    /// loaded now.
    pub fn current() -> Self {
        BuildFacts {
            crate_version: CRATE_VERSION.into(),
            git_sha: GIT_SHA.into(),
            git_tree: GIT_TREE.into(),
            git_sha_source: GIT_SHA_SOURCE.into(),
            target: BUILD_TARGET.into(),
            profile: BUILD_PROFILE.into(),
            features: features().into_iter().map(str::to_string).collect(),
            rustc: RUSTC_VERSION.into(),
            openssl: openssl_runtime_version(),
            hdf5: hdf5_runtime_version(),
        }
    }
}

/// Render `params.build`. `git_sha` is JSON `null` when unknown, never the
/// string "unknown"; `openssl` is `null` where OpenSSL is not linked.
pub fn provenance_of(f: &BuildFacts) -> serde_json::Value {
    let git_sha = if f.git_sha == "unknown" {
        serde_json::Value::Null
    } else {
        f.git_sha.clone().into()
    };
    serde_json::json!({
        "crate_version": f.crate_version,
        "git_sha": git_sha,
        "git_tree": f.git_tree,
        "git_sha_source": f.git_sha_source,
        "target": f.target,
        "profile": f.profile,
        "features": f.features,
        "rustc": f.rustc,
        "openssl": f.openssl,
        "hdf5": f.hdf5,
    })
}

/// The `params.build` block of every result file: which build produced the
/// numbers, so a result can be traced to a release archive and told apart
/// from a source build of the same commit (`features`, `target`, and the
/// OpenSSL / HDF5 actually loaded). Also what `--describe build` prints.
pub fn provenance() -> serde_json::Value {
    provenance_of(&BuildFacts::current())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The four compile-time values must agree with each other: a build that
    /// knows its SHA prints it in `--version`, and one that does not says so.
    #[test]
    fn compiled_in_values_are_consistent() {
        assert!(LONG_VERSION.starts_with(&format!("{CRATE_VERSION} (git ")));
        assert!(
            matches!(GIT_TREE, "clean" | "dirty" | "unknown"),
            "{GIT_TREE}"
        );
        match GIT_SHA_SOURCE {
            "git" | "env" => {
                assert!(resolve::is_full_sha(GIT_SHA), "{GIT_SHA}");
                assert!(LONG_VERSION.contains(GIT_SHA), "{LONG_VERSION}");
                assert_eq!(LONG_VERSION.contains("dirty tree"), GIT_TREE == "dirty");
            }
            "unknown" => {
                assert_eq!(GIT_SHA, "unknown");
                assert!(
                    LONG_VERSION.ends_with("(git commit unknown)"),
                    "{LONG_VERSION}"
                );
            }
            other => panic!("unexpected VDB_GIT_SHA_SOURCE {other:?}"),
        }
    }

    fn facts(sha: &str) -> BuildFacts {
        BuildFacts {
            crate_version: "1.2.3".into(),
            git_sha: sha.into(),
            git_tree: "clean".into(),
            git_sha_source: "git".into(),
            target: "x86_64-unknown-linux-gnu".into(),
            profile: "release".into(),
            features: vec!["release-static".into()],
            rustc: "rustc 1.99.1 (abc 2026-01-01)".into(),
            openssl: Some("OpenSSL 3.6.3".into()),
            hdf5: "1.10.7".into(),
        }
    }

    #[test]
    fn provenance_renders_every_fact_and_an_unknown_sha_as_null() {
        let sha = "0123456789abcdef0123456789abcdef01234567";
        assert_eq!(
            provenance_of(&facts(sha)),
            serde_json::json!({
                "crate_version": "1.2.3",
                "git_sha": sha,
                "git_tree": "clean",
                "git_sha_source": "git",
                "target": "x86_64-unknown-linux-gnu",
                "profile": "release",
                "features": ["release-static"],
                "rustc": "rustc 1.99.1 (abc 2026-01-01)",
                "openssl": "OpenSSL 3.6.3",
                "hdf5": "1.10.7",
            })
        );

        let p = provenance_of(&BuildFacts {
            openssl: None,
            features: vec![],
            ..facts("unknown")
        });
        assert!(p["git_sha"].is_null(), "unknown must be null, not a string");
        assert!(p["openssl"].is_null());
        assert_eq!(p["features"], serde_json::json!([]));
    }

    #[test]
    fn provenance_block_mirrors_this_build() {
        let p = provenance();
        assert_eq!(p, provenance_of(&BuildFacts::current()));
        assert_eq!(p["crate_version"], CRATE_VERSION);
        assert_eq!(p["git_tree"], GIT_TREE);
        assert_eq!(p["git_sha_source"], GIT_SHA_SOURCE);
        assert_eq!(p["target"], BUILD_TARGET);
        // The release archives are built with `release-static` (and their
        // unit tests run with it in release-binaries.yml); `make agent-check`
        // builds without it. So both directions are checked, against the
        // compiler's own view of the feature rather than build.rs's.
        let expect: Vec<&str> = if cfg!(feature = "release-static") {
            vec!["release-static"]
        } else {
            vec![]
        };
        assert_eq!(p["features"], serde_json::json!(expect));
        let want_profile = if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        };
        assert_eq!(p["profile"], want_profile);
        assert!(
            RUSTC_VERSION.starts_with("rustc "),
            "VDB_RUSTC_VERSION {RUSTC_VERSION:?}"
        );
    }

    /// The triple is the one this test binary runs as: its arch and OS agree
    /// with the compiled `std::env::consts`, checked independently of how
    /// `build.rs` obtained it.
    #[test]
    fn build_target_names_the_platform_the_code_was_compiled_for() {
        let arch = std::env::consts::ARCH;
        assert!(
            BUILD_TARGET.starts_with(&format!("{arch}-")),
            "{BUILD_TARGET} vs arch {arch}"
        );
        let os = match std::env::consts::OS {
            "macos" => "darwin",
            o => o,
        };
        assert!(BUILD_TARGET.contains(os), "{BUILD_TARGET} vs os {os}");
    }

    /// The library versions are the ones loaded, in the form a reader expects.
    #[test]
    fn runtime_library_versions_are_reported() {
        let h = hdf5_runtime_version();
        assert_eq!(h.split('.').count(), 3, "{h}");
        assert!(h.starts_with("1."), "{h}");
        // Linked (and reported) everywhere but macOS.
        assert_eq!(
            openssl_runtime_version().is_some(),
            !cfg!(target_os = "macos")
        );
        if let Some(v) = openssl_runtime_version() {
            assert!(v.starts_with("OpenSSL "), "{v}");
        }
        // A release-static build reports the bundled HDF5's version and an
        // OpenSSL 3.x. This is a sanity check of what is recorded, NOT what
        // catches a system link: Ubuntu 22.04's own libhdf5 is also 1.10.7,
        // and its OpenSSL is 3.0. That is caught by check_portable_linkage
        // (no NEEDED libhdf5/libssl/libcrypto) in package.sh and by verify.sh's
        // `ldd` check on a host where those libraries are absent.
        if cfg!(feature = "release-static") {
            assert_eq!(h, BUNDLED_HDF5, "release-static must run the bundled HDF5");
            if let Some(v) = openssl_runtime_version() {
                assert!(v.starts_with("OpenSSL 3."), "{v}");
            }
        }
    }

    /// The HDF5 version `hdf5-src` (pulled in by `release-static`) bundles;
    /// the same as Ubuntu 22.04's system library, so it identifies nothing.
    const BUNDLED_HDF5: &str = "1.10.7";
}
