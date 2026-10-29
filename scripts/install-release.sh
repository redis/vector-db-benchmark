#!/usr/bin/env bash
# Download a prebuilt vector-db-benchmark release for this host and verify it (#366).
#
#   curl -fsSL https://raw.githubusercontent.com/redis/vector-db-benchmark/master/scripts/install-release.sh | bash
#   curl -fsSL …/install-release.sh | bash -s -- 0.2.0 /opt/vdb     # tag, destination
#   make install-release [TAG=0.2.0] [DEST=…]
#
# Picks linux-x86_64, linux-aarch64 or macos-aarch64 from `uname`, downloads that
# archive and SHA256SUMS from the GitHub release, refuses on a checksum mismatch,
# and extracts into DEST (default ./vector-db-benchmark-<tag>). Without a tag it
# installs the latest release. Needs curl, tar, gzip and sha256sum/shasum.
#
# VDB_RELEASE_BASE_URL overrides https://github.com/redis/vector-db-benchmark/releases
# (CI points it at a file:// directory of freshly built archives).
set -euo pipefail

tag="${1:-}"
base="${VDB_RELEASE_BASE_URL:-https://github.com/redis/vector-db-benchmark/releases}"

die() { echo "install-release: $*" >&2; exit 1; }

case "$(uname -s)" in
  Linux) os=linux ;;
  Darwin) os=macos ;;
  *) die "no prebuilt binaries for $(uname -s); build from source (see README)" ;;
esac
case "$(uname -m)" in
  x86_64|amd64) arch=x86_64 ;;
  aarch64|arm64) arch=aarch64 ;;
  *) die "no prebuilt binaries for $(uname -m); build from source (see README)" ;;
esac
[[ "$os-$arch" != macos-x86_64 ]] || die "no prebuilt binaries for Intel macOS; build from source (see README)"

if [[ -z "$tag" ]]; then
  # /releases/latest redirects to /releases/tag/<tag>; no API token or rate limit.
  latest="$(curl -fsSL -o /dev/null -w '%{url_effective}' "$base/latest")" ||
    die "could not resolve the latest release from $base/latest"
  tag="${latest##*/}"
  # With no published release, /releases/latest lands on the /releases page.
  [[ "$tag" != releases && "$latest" == */tag/* ]] ||
    die "$base has no published release to install; pass a tag, or build from source (see README)"
  latest_resolved=1
fi
[[ "$tag" =~ ^[A-Za-z0-9][A-Za-z0-9._+-]*$ ]] || die "invalid tag '$tag'"

name="vector-db-benchmark-$tag-$os-$arch.tar.gz"
dest="${2:-${DEST:-./vector-db-benchmark-$tag}}"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo "Downloading $name ($tag) ..."
if ! curl -fsSL -o "$tmp/$name" "$base/download/$tag/$name"; then
  if [[ -n "${latest_resolved:-}" ]]; then
    die "the latest release, $tag, has no prebuilt archive $name (releases before prebuilt binaries ship none); pass a tag that has one, or build from source (see README)"
  fi
  die "no prebuilt archive $name at $base/download/$tag/ (no such release, or it predates prebuilt binaries); build from source (see README)"
fi
curl -fsSL -o "$tmp/SHA256SUMS" "$base/download/$tag/SHA256SUMS" || die "download failed: SHA256SUMS"

want="$(awk -v n="$name" '$2 == n {print $1}' "$tmp/SHA256SUMS")"
[[ -n "$want" ]] || die "SHA256SUMS of $tag does not list $name"
if command -v sha256sum >/dev/null 2>&1; then
  got="$(sha256sum "$tmp/$name" | cut -d' ' -f1)"
else
  got="$(shasum -a 256 "$tmp/$name" | cut -d' ' -f1)"
fi
[[ "$got" == "$want" ]] || die "checksum mismatch for $name: got $got, SHA256SUMS says $want"

mkdir -p "$dest"
tar -xzf "$tmp/$name" -C "$dest"
echo "Installed $tag ($os-$arch) into $dest:"
"$dest/vector-db-benchmark" --version
echo "Run it from that directory (it reads datasets/ and experiments/ relative to the working directory):"
echo "  cd $dest && ./vector-db-benchmark --help"
