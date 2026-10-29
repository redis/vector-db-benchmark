//! Decides which git commit a build claims to come from (#366).
//!
//! Compiled TWICE: by `build.rs` (via `#[path]`), which runs it at build time
//! and bakes the answer into the binaries, and by the library as
//! `build_info::resolve`, which is where its unit tests live. It therefore
//! uses `std` only — a build script cannot see the crate's dependencies.
//!
//! The failure this exists to prevent is a binary that names a commit it was
//! not built from. A release archive is traced back to its tag through this
//! SHA, so "unknown" is an acceptable answer and a wrong SHA is not:
//!
//! * A source tree with no `.git` of its own (the Docker build context, an
//!   unpacked source tarball) can sit INSIDE some other git checkout. Asking
//!   git there answers with the enclosing repository's HEAD, so the probe is
//!   only believed when git's top level is the crate's own directory.
//! * An explicit `VDB_BUILD_GIT_SHA` (how a build with no `.git` can still
//!   name its commit) must be a full 40-character SHA. A short or malformed
//!   value — `$(git rev-parse HEAD)` that was never expanded, a branch name —
//!   fails the build rather than being published as provenance.
//! * Uncommitted changes are reported (`dirty`), because the commit alone does
//!   not describe what was compiled.

use std::path::Path;

/// Where the SHA came from. Recorded next to it in result files, so a reader
/// can tell a probed commit from an asserted one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Origin {
    /// `git` was asked in the crate's own checkout.
    Git,
    /// `VDB_BUILD_GIT_SHA` was set by whoever ran the build.
    Env,
    /// Neither was available; the build does not know its commit.
    Unknown,
}

impl Origin {
    pub fn as_str(self) -> &'static str {
        match self {
            Origin::Git => "git",
            Origin::Env => "env",
            Origin::Unknown => "unknown",
        }
    }
}

/// What `git` said about the crate directory. Every field is `None` when the
/// corresponding command failed or git is absent.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GitProbe {
    /// `git rev-parse --show-toplevel`, canonicalized by the caller.
    pub toplevel: Option<std::path::PathBuf>,
    /// `git rev-parse HEAD`.
    pub head: Option<String>,
    /// Whether `git status --porcelain --untracked-files=all -- <compiled
    /// inputs>` printed anything: a tracked change or an untracked file under
    /// `build.rs`, `src`, `Cargo.toml` or `Cargo.lock` (see `build.rs`).
    pub dirty: Option<bool>,
}

/// The resolved build identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuildSource {
    /// Full lowercase 40-hex commit, or `None` when unknown.
    pub sha: Option<String>,
    /// `Some(true)` when tracked files differed from `sha`; `None` when that
    /// could not be determined (always the case for [`Origin::Env`]).
    pub dirty: Option<bool>,
    pub origin: Origin,
}

/// `true` for exactly 40 lowercase hex digits — the form `git rev-parse HEAD`
/// prints for a SHA-1 repository.
pub fn is_full_sha(s: &str) -> bool {
    s.len() == 40
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// Pick the build's commit. `manifest_dir` must be canonicalized the same way
/// as `git.toplevel`.
///
/// An explicit override wins over git, and an invalid override is an error
/// rather than a fallback: whoever set it meant to assert a commit.
pub fn resolve(
    env_override: Option<&str>,
    manifest_dir: &Path,
    git: &GitProbe,
) -> Result<BuildSource, String> {
    if let Some(raw) = env_override {
        let v = raw.trim();
        if !v.is_empty() {
            let lower = v.to_ascii_lowercase();
            if !is_full_sha(&lower) {
                return Err(format!(
                    "VDB_BUILD_GIT_SHA={raw:?} is not a full 40-character git commit SHA. \
                     Set it to the output of `git rev-parse HEAD`, or leave it unset."
                ));
            }
            return Ok(BuildSource {
                sha: Some(lower),
                dirty: None,
                origin: Origin::Env,
            });
        }
    }

    let own_checkout = git.toplevel.as_deref() == Some(manifest_dir);
    match git.head.as_deref().map(str::trim) {
        Some(head) if own_checkout && is_full_sha(head) => Ok(BuildSource {
            sha: Some(head.to_string()),
            dirty: git.dirty,
            origin: Origin::Git,
        }),
        _ => Ok(BuildSource {
            sha: None,
            dirty: None,
            origin: Origin::Unknown,
        }),
    }
}

/// The text after the program name in `--version`, e.g.
/// `0.1.0 (git 0123…cdef)` or `0.1.0 (git 0123…cdef, dirty tree)`.
///
/// The SHA is printed in full: the release workflow compares it against the
/// tagged commit, and a prefix would let two commits share an identity.
pub fn version_line(crate_version: &str, src: &BuildSource) -> String {
    match (&src.sha, src.dirty) {
        (Some(sha), Some(true)) => format!("{crate_version} (git {sha}, dirty tree)"),
        (Some(sha), _) => format!("{crate_version} (git {sha})"),
        (None, _) => format!("{crate_version} (git commit unknown)"),
    }
}

/// The crate's enabled cargo features, from the `CARGO_FEATURE_<NAME>`
/// variables cargo gives a build script: lowercased, `_` back to `-`, sorted,
/// without `default` (which names no behaviour of its own).
///
/// Cargo upper-cases names and maps `-` to `_`, so `release-static` arrives as
/// `CARGO_FEATURE_RELEASE_STATIC`. A feature with a real `_` in its name would
/// come back as `-`; this crate has none.
pub fn features_from_env<I>(vars: I) -> Vec<String>
where
    I: IntoIterator<Item = (String, String)>,
{
    let mut f: Vec<String> = vars
        .into_iter()
        .filter_map(|(k, _)| {
            k.strip_prefix("CARGO_FEATURE_")
                .map(|n| n.to_ascii_lowercase().replace('_', "-"))
        })
        .filter(|n| !n.is_empty() && n != "default")
        .collect();
    f.sort();
    f.dedup();
    f
}

/// Tree state as one word, for result files: `clean`, `dirty` or `unknown`.
pub fn tree_state(src: &BuildSource) -> &'static str {
    match src.dirty {
        Some(true) => "dirty",
        Some(false) => "clean",
        None => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    const SHA: &str = "0123456789abcdef0123456789abcdef01234567";

    fn probe(toplevel: &str, head: Option<&str>, dirty: Option<bool>) -> GitProbe {
        GitProbe {
            toplevel: Some(PathBuf::from(toplevel)),
            head: head.map(str::to_string),
            dirty,
        }
    }

    #[test]
    fn git_head_of_the_crates_own_checkout_is_used() {
        let src = resolve(
            None,
            Path::new("/src/vdb"),
            &probe("/src/vdb", Some(SHA), Some(false)),
        )
        .unwrap();
        assert_eq!(src.sha.as_deref(), Some(SHA));
        assert_eq!(src.origin, Origin::Git);
        assert_eq!(src.dirty, Some(false));
    }

    /// The Docker build context has no `.git`; if it sat inside another
    /// checkout, git would answer with THAT repository's HEAD.
    #[test]
    fn an_enclosing_repositorys_head_is_not_claimed() {
        let src = resolve(
            None,
            Path::new("/outer/vendor/vdb"),
            &probe("/outer", Some(SHA), Some(false)),
        )
        .unwrap();
        assert_eq!(src.sha, None);
        assert_eq!(src.origin, Origin::Unknown);
        assert_eq!(
            src.dirty, None,
            "the outer tree's state says nothing about ours"
        );
    }

    #[test]
    fn no_git_at_all_is_unknown_not_an_error() {
        let src = resolve(None, Path::new("/build"), &GitProbe::default()).unwrap();
        assert_eq!(src.sha, None);
        assert_eq!(src.origin, Origin::Unknown);
        assert_eq!(version_line("0.1.0", &src), "0.1.0 (git commit unknown)");
        assert_eq!(tree_state(&src), "unknown");
    }

    #[test]
    fn a_head_that_is_not_a_full_sha_is_unknown() {
        for head in ["", "HEAD", &SHA[..12], "fatal: not a git repository"] {
            let src = resolve(
                None,
                Path::new("/src/vdb"),
                &probe("/src/vdb", Some(head), Some(false)),
            )
            .unwrap();
            assert_eq!(src.sha, None, "head {head:?}");
        }
        // Trailing newline from the command output is not a reason to drop it.
        let src = resolve(
            None,
            Path::new("/src/vdb"),
            &probe("/src/vdb", Some(&format!("{SHA}\n")), Some(false)),
        )
        .unwrap();
        assert_eq!(src.sha.as_deref(), Some(SHA));
    }

    #[test]
    fn override_wins_over_git_and_is_normalised() {
        let other = "fedcba9876543210fedcba9876543210fedcba98";
        let src = resolve(
            Some(&format!(" {} ", SHA.to_ascii_uppercase())),
            Path::new("/src/vdb"),
            &probe("/src/vdb", Some(other), Some(true)),
        )
        .unwrap();
        assert_eq!(src.sha.as_deref(), Some(SHA));
        assert_eq!(src.origin, Origin::Env);
        assert_eq!(
            src.dirty, None,
            "an asserted SHA says nothing about the tree"
        );
    }

    #[test]
    fn an_empty_override_means_unset() {
        // `ARG VDB_BUILD_GIT_SHA=""` in the Dockerfile reaches cargo as "".
        let src = resolve(
            Some(""),
            Path::new("/src/vdb"),
            &probe("/src/vdb", Some(SHA), Some(false)),
        )
        .unwrap();
        assert_eq!(src.origin, Origin::Git);
    }

    #[test]
    fn a_malformed_override_fails_the_build() {
        for bad in [
            "$(git rev-parse HEAD)",
            "master",
            &SHA[..7],
            &format!("{SHA}0"),
            "g123456789abcdef0123456789abcdef01234567",
        ] {
            let err = resolve(Some(bad), Path::new("/b"), &GitProbe::default()).unwrap_err();
            assert!(err.contains("VDB_BUILD_GIT_SHA"), "{err}");
        }
    }

    #[test]
    fn version_line_names_the_full_sha_and_a_dirty_tree() {
        let clean = BuildSource {
            sha: Some(SHA.into()),
            dirty: Some(false),
            origin: Origin::Git,
        };
        assert_eq!(version_line("1.2.3", &clean), format!("1.2.3 (git {SHA})"));
        assert_eq!(tree_state(&clean), "clean");

        let dirty = BuildSource {
            dirty: Some(true),
            ..clean.clone()
        };
        assert_eq!(
            version_line("1.2.3", &dirty),
            format!("1.2.3 (git {SHA}, dirty tree)")
        );
        assert_eq!(tree_state(&dirty), "dirty");

        let asserted = BuildSource {
            dirty: None,
            origin: Origin::Env,
            ..clean
        };
        assert_eq!(
            version_line("1.2.3", &asserted),
            format!("1.2.3 (git {SHA})")
        );
        assert_eq!(tree_state(&asserted), "unknown");
    }

    /// The dirty probe must reach the result: a tree git reports as modified
    /// is `dirty` in result files and says so in `--version`.
    #[test]
    fn a_dirty_probe_is_reported_as_a_dirty_tree() {
        let src = resolve(
            None,
            Path::new("/src/vdb"),
            &probe("/src/vdb", Some(SHA), Some(true)),
        )
        .unwrap();
        assert_eq!(src.dirty, Some(true));
        assert_eq!(tree_state(&src), "dirty");
        assert!(
            version_line("0.1.0", &src).ends_with(", dirty tree)"),
            "{}",
            version_line("0.1.0", &src)
        );
        // And a probe that could not tell is unknown, not clean.
        let src = resolve(
            None,
            Path::new("/src/vdb"),
            &probe("/src/vdb", Some(SHA), None),
        )
        .unwrap();
        assert_eq!(tree_state(&src), "unknown");
    }

    #[test]
    fn features_come_from_cargos_variables() {
        let vars = |ks: &[&str]| {
            ks.iter()
                .map(|k| (k.to_string(), "1".to_string()))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            features_from_env(vars(&[
                "CARGO_FEATURE_RELEASE_STATIC",
                "CARGO_FEATURE_DEFAULT",
                "CARGO_PKG_VERSION",
                "PATH",
                "CARGO_FEATURE_A",
            ])),
            ["a", "release-static"]
        );
        assert!(features_from_env(vars(&["CARGO_FEATURE_DEFAULT", "HOME"])).is_empty());
        assert!(features_from_env(Vec::new()).is_empty());
    }

    #[test]
    fn is_full_sha_is_exact() {
        assert!(is_full_sha(SHA));
        assert!(!is_full_sha(&SHA.to_ascii_uppercase()));
        assert!(!is_full_sha(&SHA[1..]));
        assert!(!is_full_sha(&format!("{SHA}a")));
        assert!(!is_full_sha("z123456789abcdef0123456789abcdef01234567"));
    }
}
