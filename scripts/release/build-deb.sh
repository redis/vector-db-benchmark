#!/usr/bin/env bash
# Build the vector-db-benchmark .deb for ONE codename, natively, on that codename.
#
#   scripts/release/build-deb.sh --dist <codename> --out <dir> [--install-deps] [--release --tag <tag>] [--allow-host-mismatch]
#
# Run ON the target distribution (it refuses another codename's host, since
# the .deb's glibc floor is the host's): an `ubuntu:<codename>` /
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

dist="" out="" install_deps=0 release=0 tag="" allow_host_mismatch=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dist) dist="$2"; shift 2 ;;
    --out) out="$2"; shift 2 ;;
    --install-deps) install_deps=1; shift ;;
    --release) release=1; shift ;;
    --tag) tag="$2"; shift 2 ;;
    --allow-host-mismatch) allow_host_mismatch=1; shift ;;
    *) echo "build-deb.sh: unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$dist" && -n "$out" ]] || {
  echo "usage: build-deb.sh --dist <codename> --out <dir> [--install-deps] [--release --tag <tag>] [--allow-host-mismatch]" >&2; exit 2; }
[[ "$dist" =~ ^[a-z]+$ ]] || { echo "build-deb.sh: codename '$dist' is not a plain lowercase word" >&2; exit 2; }

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/release/lib.sh
source "$here/lib.sh"
repo="$(cd "$here/../.." && pwd)"
cd "$repo"

# shellcheck disable=SC1091
host_codename="$(. /etc/os-release 2>/dev/null; echo "${VERSION_CODENAME:-}")"
if [[ -n "$host_codename" && "$host_codename" != "$dist" && "$allow_host_mismatch" != 1 ]]; then
  # The .deb's glibc floor is the HOST's: `--dist jammy` built on noble would
  # publish a jammy package that jammy cannot install.
  echo "build-deb.sh: building for '$dist' on a '$host_codename' host; run it on $dist (a container will do), or pass --allow-host-mismatch" >&2
  exit 1
fi

if [[ "$install_deps" == 1 ]]; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y --no-install-recommends \
    build-essential debhelper cmake perl pkg-config ca-certificates git curl
fi
ensure_release_toolchain "$install_deps"
command -v dpkg-buildpackage >/dev/null || { echo "build-deb.sh: dpkg-buildpackage missing (pass --install-deps)" >&2; exit 1; }

# A checkout owned by another uid (CI containers run as root over the runner's
# files) makes git refuse to look at it; the build only reads from it.
git config --global --add safe.directory "$repo" 2>/dev/null || true
git config --global --add safe.directory '*' 2>/dev/null || true

version="$(crate_version)"
[[ -n "$version" ]] || { echo "build-deb.sh: no [package].version in Cargo.toml" >&2; exit 1; }
[[ "$version" =~ ^[0-9A-Za-z.+]+$ ]] || {
  # `3.0 (native)` forbids a Debian revision, and dpkg would read `0.2.0-rc.1`
  # as upstream 0.2.0 with revision rc.1; the RPM side refuses the same
  # versions. A pre-release scheme (0.2.0~rc.1) is a decision for later.
  echo "build-deb.sh: crate version '$version' is not packageable as a native Debian version (no '-'; letters, digits, dots only)" >&2; exit 1; }
if [[ "$release" == 1 ]]; then
  [[ -n "$tag" ]] || { echo "build-deb.sh: --release needs --tag" >&2; exit 1; }
  check_tag "$tag"
  tag_matches_version "$tag" "$version" || {
    echo "build-deb.sh: release tag '$tag' does not name the crate version '$version' (Cargo.toml)." >&2; exit 1; }
fi
sha="$(require_clean_head)"
export VDB_BUILD_GIT_SHA="$sha"

deb_version="$version~$dist"
release_url=""
[[ -n "$tag" ]] && release_url=": https://github.com/redis/vector-db-benchmark/releases/tag/$tag"
cat > debian/changelog <<CHANGELOG
vector-db-benchmark ($deb_version) $dist; urgency=medium

  * Release $version$release_url (git $sha)

 -- Redis Ltd. <oss@redislabs.com>  $(LC_ALL=C date -R)
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
# What release-static promises, asserted: the package needs glibc (and
# libgcc) only. A libhdf5/libssl dependency here means the static feature did
# not apply, and the bare-container install test would NOT catch it (apt would
# simply install the library).
dpkg-deb -f "$out/$(basename "$deb")" Package Version Architecture Depends
depends="$(dpkg-deb -f "$out/$(basename "$deb")" Depends)"
glibc_only='^libc6 \([^)]*\)(, libgcc-s1 \([^)]*\))?$'
[[ "$depends" =~ $glibc_only ]] || {
  echo "build-deb.sh: Depends is '$depends', expected glibc (and libgcc) only; release-static not applied?" >&2; exit 1; }
echo "$out/$(basename "$deb")"
