#!/usr/bin/env bash
# Build the vector-db-benchmark .deb for ONE codename, natively, on that codename.
#
#   scripts/release/build-deb.sh --dist <codename> --out <dir> [--install-deps] [--release --tag <tag>]
#
# Run inside (or on) the target distribution: an `ubuntu:<codename>` /
# `debian:<codename>` container in CI, the same thing locally through
# `make release-deb DIST=jammy`. It writes debian/changelog as
# `<crate version>~<codename>` (the memtier-benchmark convention on
# packages.redis.io), runs dpkg-buildpackage and leaves
# <out>/vector-db-benchmark_<version>~<codename>_<arch>.deb.
#
# --install-deps  apt-get the build dependencies (debian/control Build-Depends
#                 plus curl) and a current stable Rust toolchain via rustup.
#                 The distros' cargo/rustc are too old for this crate.
# --release       a real release: --tag must name the crate version, as with
#                 package.sh, so the .deb never carries a version its binaries
#                 do not report.
set -euo pipefail

dist="" out="" install_deps=0 release=0 tag=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dist) dist="$2"; shift 2 ;;
    --out) out="$2"; shift 2 ;;
    --install-deps) install_deps=1; shift ;;
    --release) release=1; shift ;;
    --tag) tag="$2"; shift 2 ;;
    *) echo "build-deb.sh: unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$dist" && -n "$out" ]] || {
  echo "usage: build-deb.sh --dist <codename> --out <dir> [--install-deps] [--release --tag <tag>]" >&2; exit 2; }
[[ "$dist" =~ ^[a-z]+$ ]] || { echo "build-deb.sh: codename '$dist' is not a plain lowercase word" >&2; exit 2; }

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/release/lib.sh
source "$here/lib.sh"
repo="$(cd "$here/../.." && pwd)"
cd "$repo"

if [[ -r /etc/os-release ]]; then
  # shellcheck disable=SC1091
  host_codename="$(. /etc/os-release; echo "${VERSION_CODENAME:-}")"
  if [[ -n "$host_codename" && "$host_codename" != "$dist" ]]; then
    echo "build-deb.sh: building for '$dist' on a '$host_codename' host; the .deb's glibc floor will be the host's" >&2
  fi
fi

if [[ "$install_deps" == 1 ]]; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y --no-install-recommends \
    build-essential debhelper cmake perl pkg-config ca-certificates git curl
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
command -v cargo >/dev/null || { echo "build-deb.sh: cargo not on PATH (pass --install-deps or install rustup)" >&2; exit 1; }
if [[ -n "${RELEASE_RUST_TOOLCHAIN:-}" ]]; then
  # Not `rustc -vV | grep -q`: grep exits at the match, rustc takes SIGPIPE, pipefail fails.
  test "$(rustc -vV | sed -n 's/^release: //p')" = "$RELEASE_RUST_TOOLCHAIN" ||
    { echo "build-deb.sh: rustc is $(rustc -vV | sed -n 's/^release: //p'), not the pinned $RELEASE_RUST_TOOLCHAIN" >&2; exit 1; }
fi
# hdf5-sys builds its bundled HDF5 only when HDF5_DIR is UNSET; the Makefile
# exports it for normal builds, and with it set release-static would silently
# link the system libhdf5 (then dh_shlibdeps / rpm would add a libhdf5 dependency).
unset HDF5_DIR HDF5_VERSION
command -v dpkg-buildpackage >/dev/null || { echo "build-deb.sh: dpkg-buildpackage missing (pass --install-deps)" >&2; exit 1; }

# A checkout owned by another uid (CI containers run as root over the runner's
# files) makes git refuse to look at it; the build only reads from it.
git config --global --add safe.directory "$repo" 2>/dev/null || true
git config --global --add safe.directory '*' 2>/dev/null || true

version="$(crate_version)"
[[ -n "$version" ]] || { echo "build-deb.sh: no [package].version in Cargo.toml" >&2; exit 1; }
if [[ "$release" == 1 ]]; then
  [[ -n "$tag" ]] || { echo "build-deb.sh: --release needs --tag" >&2; exit 1; }
  check_tag "$tag"
  tag_matches_version "$tag" "$version" || {
    echo "build-deb.sh: release tag '$tag' does not name the crate version '$version' (Cargo.toml)." >&2; exit 1; }
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "build-deb.sh: refusing to package a tree with uncommitted changes:" >&2
  git status --porcelain --untracked-files=no >&2
  exit 1
fi
sha="$(git rev-parse HEAD)"
export VDB_BUILD_GIT_SHA="${VDB_BUILD_GIT_SHA:-$sha}"
[[ "$VDB_BUILD_GIT_SHA" == "$sha" ]] || {
  echo "build-deb.sh: VDB_BUILD_GIT_SHA=$VDB_BUILD_GIT_SHA is not HEAD ($sha)" >&2; exit 1; }

deb_version="$version~$dist"
release_url=""
[[ -n "$tag" ]] && release_url=": https://github.com/redis/vector-db-benchmark/releases/tag/$tag"
cat > debian/changelog <<CHANGELOG
vector-db-benchmark ($deb_version) $dist; urgency=medium

  * Release $version$release_url (git $sha)

 -- Redis Team <oss@redis.com>  $(date -R)
CHANGELOG

arch="$(dpkg --print-architecture)"
# shellcheck disable=SC1091
echo "build-deb.sh: $deb_version for $arch on $(. /etc/os-release 2>/dev/null; echo "${PRETTY_NAME:-unknown host}")"
dpkg-buildpackage -b -us -uc

mkdir -p "$out"
deb="$(cd "$repo/.." && pwd)/vector-db-benchmark_${deb_version}_${arch}.deb"
[[ -f "$deb" ]] || { echo "build-deb.sh: expected $deb after dpkg-buildpackage" >&2; ls -l "$repo/.." >&2; exit 1; }
mv "$deb" "$out/"
rm -f "$repo"/../vector-db-benchmark_"${deb_version}"_"${arch}".{buildinfo,changes}
# What the archive layout promises: the package needs glibc (and libgcc) only.
dpkg-deb -f "$out/$(basename "$deb")" Package Version Architecture Depends
echo "$out/$(basename "$deb")"
