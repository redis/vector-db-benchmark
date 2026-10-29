#!/usr/bin/env bash
# Download a prebuilt vector-db-benchmark release for this host and verify it (#366).
#
#   curl -fsSL https://raw.githubusercontent.com/redis/vector-db-benchmark/master/scripts/install-release.sh | bash
#   curl -fsSL …/install-release.sh | bash -s -- 0.2.0 /opt/vdb     # tag, destination
#   make install-release [TAG=0.2.0] [DEST=…]
#
# Picks linux-x86_64, linux-aarch64 or macos-aarch64 from `uname`, downloads that
# archive and SHA256SUMS from the GitHub release, refuses on a checksum mismatch,
# and extracts into DEST (default ./vector-db-benchmark-<tag>), which must not
# exist or be empty. Without a tag it installs the latest release. Needs curl,
# tar, gzip and sha256sum/shasum. Archive members are checked before anything
# is extracted: plain files and directories inside DEST only.
#
# VDB_RELEASE_BASE_URL overrides https://github.com/redis/vector-db-benchmark/releases
# (CI's scripts/release/install-test.sh points it at a local HTTP stand-in
# for GitHub serving freshly built archives).
set -euo pipefail

tag="${1:-}"
base="${VDB_RELEASE_BASE_URL:-https://github.com/redis/vector-db-benchmark/releases}"

die() { echo "install-release: $*" >&2; exit 1; }

if command -v sha256sum >/dev/null 2>&1; then sha256() { sha256sum "$1" | cut -d' ' -f1; }
elif command -v shasum >/dev/null 2>&1; then sha256() { shasum -a 256 "$1" | cut -d' ' -f1; }
else die "needs sha256sum or shasum to check the download, and found neither"
fi

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

# The Linux archives need glibc 2.34 or newer (MAX_GLIBC in
# scripts/release/lib.sh). Say so before downloading, rather than letting the
# binary fail later with "GLIBC_2.34 not found". Skipped where getconf cannot
# tell (musl, or no getconf): the archive is then simply tried.
MIN_GLIBC=2.34
if [[ "$os" == linux ]] && glibc="$(getconf GNU_LIBC_VERSION 2>/dev/null)" && [[ "$glibc" == glibc\ * ]]; then
  have="${glibc#glibc }"
  if [[ "$(printf '%s\n%s\n' "$have" "$MIN_GLIBC" | sort -V | head -n1)" != "$MIN_GLIBC" ]]; then
    die "this host has glibc $have; the prebuilt Linux binaries need $MIN_GLIBC or newer (Amazon Linux 2023, RHEL 9, Ubuntu 22.04, Debian 12). Use the Docker image or build from source (see README)"
  fi
fi

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
if [[ -e "$dest" ]]; then
  [[ -d "$dest" ]] || die "$dest exists and is not a directory"
  [[ -z "$(ls -A "$dest")" ]] || die "$dest exists and is not empty; remove it or pass another destination"
fi
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
got="$(sha256 "$tmp/$name")"
[[ "$got" == "$want" ]] || die "checksum mismatch for $name: got $got, SHA256SUMS says $want"

# Before extracting: no absolute path or `..` (a write outside DEST), and no
# link or special member (which a later member could be written through).
members="$(tar -tzf "$tmp/$name")" || die "cannot list $name"
while IFS= read -r p; do
  case "/$p/" in //*|*/../*) die "$name has a member outside the destination: $p" ;; esac
done <<<"$members"
listing="$(tar -tvzf "$tmp/$name")" || die "cannot list $name"
odd="$(grep -v '^[-d]' <<<"$listing" || true)"
[[ -z "$odd" ]] || die "$name has a link or special member: $(head -n1 <<<"$odd")"

mkdir -p "$dest"
tar -xzf "$tmp/$name" -C "$dest"
echo "Installed $tag ($os-$arch) into $dest:"
"$dest/vector-db-benchmark" --version
echo "Run it from that directory (it reads datasets/ and experiments/ relative to the working directory):"
echo "  cd $dest && ./vector-db-benchmark --help"
