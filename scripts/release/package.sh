#!/usr/bin/env bash
# Build and package the prebuilt release archive for one target (#366).
#
#   scripts/release/package.sh --target <triple> --tag <tag> --out <dir> [--release]
#
# Writes <out>/vector-db-benchmark-<tag>-<platform>.tar.gz, where <platform> is
# linux-x86_64, linux-aarch64 or macos-aarch64. Called through
# `make release-archive`; the release workflow and its PR check both use it,
# so what a PR exercises is what a release ships.
#
# The archive is FLAT, so `curl -L <asset> | tar xz && ./vector-db-benchmark`
# works in the directory it was extracted into. Next to the binaries it carries
# what the CLI resolves relative to the working directory (`datasets/datasets.json`,
# `experiments/configurations/`, the bundled `random-100` smoke dataset), the
# LICENSE, the README and BUILD-INFO.txt.
#
# --release: this is a real release. The tag must then name the crate version
# (`0.2.0` or `v0.2.0`; a pre-release may add a suffix, `0.2.0-rc.1`), so a
# binary never reports a different version than the archive it came in.
set -euo pipefail

target="" tag="" out="" release=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) target="$2"; shift 2 ;;
    --tag) tag="$2"; shift 2 ;;
    --out) out="$2"; shift 2 ;;
    --release) release=1; shift ;;
    *) echo "package.sh: unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$target" && -n "$tag" && -n "$out" ]] || {
  echo "usage: package.sh --target <triple> --tag <tag> --out <dir> [--release]" >&2; exit 2; }

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/release/lib.sh
source "$here/lib.sh"
repo="$(cd "$here/../.." && pwd)"
cd "$repo"

platform="$(platform_for_target "$target")"
check_tag "$tag"

version="$(crate_version)"
if [[ "$release" == 1 ]]; then
  tag_matches_version "$tag" "$version" || {
    echo "package.sh: release tag '$tag' does not name the crate version '$version' (Cargo.toml)." >&2
    echo "  Bump [package].version and re-tag, so the binaries report the version they are released as." >&2
    exit 1; }
fi

# The archive must be built from a committed tree: its binaries report the
# commit, and a dirty tree makes that a lie (`--version` would say so too).
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "package.sh: refusing to package a tree with uncommitted changes:" >&2
  git status --porcelain --untracked-files=no >&2
  exit 1
fi
sha="$(git rev-parse HEAD)"

export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$repo/target}"
# `--locked`: the archive is built from the tag's Cargo.lock, never a re-resolve.
cargo build --release --locked --features release-static --target "$target" --bins

bin_dir="$CARGO_TARGET_DIR/$target/release"
bins=()
while IFS= read -r b; do bins+=("$b"); done < <(package_bins)
[[ ${#bins[@]} -gt 0 ]] || { echo "package.sh: cargo metadata listed no binaries" >&2; exit 1; }
for b in "${REQUIRED_BINS[@]}"; do
  printf '%s\n' "${bins[@]}" | grep -qx "$b" || {
    echo "package.sh: required binary '$b' is not a target of this crate" >&2; exit 1; }
done

for b in "${bins[@]}"; do
  check_portable_linkage "$bin_dir/$b" "$target"
done

name="vector-db-benchmark-$tag-$platform"
stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT
for b in "${bins[@]}"; do
  install -m 0755 "$bin_dir/$b" "$stage/$b"
done
mkdir -p "$stage/datasets" "$stage/experiments"
cp datasets/datasets.json "$stage/datasets/"
cp -R datasets/random-100 "$stage/datasets/"
cp -R experiments/configurations "$stage/experiments/"
cp LICENSE README.md "$stage/"
{
  echo "archive: $name.tar.gz"
  echo "tag: $tag"
  echo "crate_version: $version"
  echo "git_sha: $sha"
  echo "target: $target"
  echo "features: release-static"
  echo "binaries: ${bins[*]}"
  echo "built_at_commit_time: $(git log -1 --format=%cI HEAD)"
  echo "--- rustc -vV"
  rustc -vV
  echo "--- cargo -V"
  cargo -V
} > "$stage/BUILD-INFO.txt"

mkdir -p "$out"
archive="$(cd "$out" && pwd)/$name.tar.gz"
# Stable member order, owner and timestamps (the commit time), so repacking the
# same binaries gives the same archive.
epoch="$(git log -1 --format=%ct HEAD)"
tarbin=tar
command -v gtar >/dev/null 2>&1 && tarbin=gtar
(cd "$stage" && find . -mindepth 1 -print0 | LC_ALL=C sort -z |
  "$tarbin" --null --no-recursion -T - --owner=0 --group=0 --numeric-owner \
    --mtime="@$epoch" -cf - | gzip -n -9 > "$archive")
echo "$archive"
