#!/usr/bin/env bash
# Build the vector-db-benchmark RPM natively on an el8/el9/el10/Amazon Linux 2023 host.
#
#   scripts/release/build-rpm.sh --out <dir> [--install-deps] [--release --tag <tag>]
#
# Run inside the target container in CI (rockylinux:8/9/10, amazonlinux:2023),
# or locally through `make release-rpm`. Tars HEAD with `git archive`, builds
# rpm/vector-db-benchmark.spec with the crate version and the commit passed as
# defines, and leaves <out>/vector-db-benchmark-<version>-1.<dist>.<arch>.rpm.
#
# --install-deps  dnf/yum the spec's BuildRequires plus rpm-build and a current
#                 stable Rust toolchain via rustup (the distros' are too old).
# --release       the tag must name the crate version (see package.sh).
set -euo pipefail

out="" install_deps=0 release=0 tag=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) out="$2"; shift 2 ;;
    --install-deps) install_deps=1; shift ;;
    --release) release=1; shift ;;
    --tag) tag="$2"; shift 2 ;;
    *) echo "build-rpm.sh: unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$out" ]] || { echo "usage: build-rpm.sh --out <dir> [--install-deps] [--release --tag <tag>]" >&2; exit 2; }

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/release/lib.sh
source "$here/lib.sh"
repo="$(cd "$here/../.." && pwd)"
cd "$repo"

pkgmgr=dnf; command -v dnf >/dev/null || pkgmgr=yum
if [[ "$install_deps" == 1 ]]; then
  "$pkgmgr" install -y rpm-build rpmdevtools cmake gcc gcc-c++ make perl pkgconfig git tar gzip findutils curl ca-certificates
  # OpenSSL's Configure needs these; el9+/AL2023 ship them as separate packages.
  "$pkgmgr" install -y perl-IPC-Cmd perl-FindBin 2>/dev/null || true
fi
# RELEASE_RUST_TOOLCHAIN (the pin release-binaries.yml builds the archives
# with) selects the toolchain; unset, rustup's current stable is used.
toolchain="${RELEASE_RUST_TOOLCHAIN:-stable}"
if [[ "$install_deps" == 1 ]]; then
  if ! command -v rustup >/dev/null && [[ ! -x "${CARGO_HOME:-$HOME/.cargo}/bin/rustup" ]]; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain "$toolchain"
  fi
  export PATH="${CARGO_HOME:-$HOME/.cargo}/bin:$PATH"
  rustup toolchain install "$toolchain" --profile minimal
  rustup default "$toolchain"
fi
export PATH="${CARGO_HOME:-$HOME/.cargo}/bin:$PATH"
command -v cargo >/dev/null || { echo "build-rpm.sh: cargo not on PATH (pass --install-deps or install rustup)" >&2; exit 1; }
if [[ -n "${RELEASE_RUST_TOOLCHAIN:-}" ]]; then
  # Not `rustc -vV | grep -q`: grep exits at the match, rustc takes SIGPIPE, pipefail fails.
  test "$(rustc -vV | sed -n 's/^release: //p')" = "$RELEASE_RUST_TOOLCHAIN" ||
    { echo "build-rpm.sh: rustc is $(rustc -vV | sed -n 's/^release: //p'), not the pinned $RELEASE_RUST_TOOLCHAIN" >&2; exit 1; }
fi
# hdf5-sys builds its bundled HDF5 only when HDF5_DIR is UNSET; the Makefile
# exports it for normal builds, and with it set release-static would silently
# link the system libhdf5 (then dh_shlibdeps / rpm would add a libhdf5 dependency).
unset HDF5_DIR HDF5_VERSION
command -v rpmbuild >/dev/null || { echo "build-rpm.sh: rpmbuild missing (pass --install-deps)" >&2; exit 1; }

git config --global --add safe.directory "$repo" 2>/dev/null || true
git config --global --add safe.directory '*' 2>/dev/null || true

version="$(crate_version)"
[[ -n "$version" ]] || { echo "build-rpm.sh: no [package].version in Cargo.toml" >&2; exit 1; }
[[ "$version" =~ ^[0-9A-Za-z.]+$ ]] || {
  # An RPM Version may not contain '-': a pre-release crate version (0.2.0-rc.1)
  # would have to be mapped to 0.2.0~rc.1; refuse until that is decided.
  echo "build-rpm.sh: crate version '$version' is not a valid RPM Version (letters, digits and dots only)" >&2; exit 1; }
if [[ "$release" == 1 ]]; then
  [[ -n "$tag" ]] || { echo "build-rpm.sh: --release needs --tag" >&2; exit 1; }
  check_tag "$tag"
  tag_matches_version "$tag" "$version" || {
    echo "build-rpm.sh: release tag '$tag' does not name the crate version '$version' (Cargo.toml)." >&2; exit 1; }
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "build-rpm.sh: refusing to package a tree with uncommitted changes:" >&2
  git status --porcelain --untracked-files=no >&2
  exit 1
fi
sha="$(git rev-parse HEAD)"
[[ -z "${VDB_BUILD_GIT_SHA:-}" || "$VDB_BUILD_GIT_SHA" == "$sha" ]] || {
  echo "build-rpm.sh: VDB_BUILD_GIT_SHA=$VDB_BUILD_GIT_SHA is not HEAD ($sha)" >&2; exit 1; }

topdir="$(mktemp -d)"
trap 'rm -rf "$topdir"' EXIT
mkdir -p "$topdir"/{BUILD,RPMS,SOURCES,SPECS,SRPMS}
# The tarball is the committed tree (what the commit in --version names), not
# the working directory with its target/ cache.
git archive --format=tar.gz --prefix="vector-db-benchmark-$version/" -o "$topdir/SOURCES/vector-db-benchmark-$version.tar.gz" HEAD
cp rpm/vector-db-benchmark.spec "$topdir/SPECS/"

# shellcheck disable=SC1091
echo "build-rpm.sh: $version ($sha) on $(. /etc/os-release 2>/dev/null; echo "${PRETTY_NAME:-unknown host}") $(uname -m)"
rpmbuild -bb \
  --define "_topdir $topdir" \
  --define "vdb_version $version" \
  --define "vdb_git_sha $sha" \
  "$topdir/SPECS/vector-db-benchmark.spec"

mkdir -p "$out"
found=0
while IFS= read -r -d '' rpmfile; do
  cp "$rpmfile" "$out/"
  found=$((found + 1))
  rpm -qp --queryformat 'Name: %{NAME}\nVersion: %{VERSION}-%{RELEASE}\nArch: %{ARCH}\n' "$rpmfile"
  echo "Requires:"; rpm -qpR "$rpmfile" | sed 's/^/  /'
  echo "$out/$(basename "$rpmfile")"
done < <(find "$topdir/RPMS" -name '*.rpm' -type f -print0)
[[ "$found" -eq 1 ]] || { echo "build-rpm.sh: expected exactly one RPM, found $found" >&2; exit 1; }
