#!/usr/bin/env bash
# SHA256SUMS for a set of release archives (#366).
#
#   scripts/release/checksums.sh write <dir>   # <dir>/SHA256SUMS over <dir>/*.tar.gz
#   scripts/release/checksums.sh check <dir>   # verify it
#
# `check` fails unless SHA256SUMS names EXACTLY the archives in <dir> — none
# missing, none extra — and every digest matches. `sha256sum -c` alone accepts
# a checksum file that silently omits an archive, which is the mismatched-set
# state a re-run or a partial upload can leave behind.
#
# The format is `sha256sum`'s (`<hex>  <name>`), so users can run
# `sha256sum -c SHA256SUMS` (or `shasum -a 256 -c`) themselves.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/release/lib.sh
source "$here/lib.sh"

mode="${1:-}" dir="${2:-}"
[[ ( "$mode" == write || "$mode" == check ) && -d "$dir" ]] || {
  echo "usage: checksums.sh write|check <dir>" >&2; exit 2; }
cd "$dir"

archives="$(find . -maxdepth 1 -type f -name '*.tar.gz' | sed 's|^\./||' | LC_ALL=C sort)"
[[ -n "$archives" ]] || { echo "checksums.sh: no *.tar.gz in $dir" >&2; exit 1; }

if [[ "$mode" == write ]]; then
  while read -r a; do
    printf '%s  %s\n' "$(sha256_file "$a")" "$a"
  done <<<"$archives" > SHA256SUMS
  cat SHA256SUMS
  exit 0
fi

[[ -f SHA256SUMS ]] || { echo "checksums.sh: no SHA256SUMS in $dir" >&2; exit 1; }
listed="$(awk '{print $2}' SHA256SUMS | LC_ALL=C sort)"
if [[ "$listed" != "$archives" ]]; then
  echo "checksums.sh: SHA256SUMS lists a different set of archives than $dir holds" >&2
  echo "  listed:  ${listed//$'\n'/ }" >&2
  echo "  present: ${archives//$'\n'/ }" >&2
  exit 1
fi
while read -r want name; do
  got="$(sha256_file "$name")"
  [[ "$got" == "$want" ]] || { echo "checksums.sh: $name: sha256 $got != SHA256SUMS $want" >&2; exit 1; }
  echo "$name: OK"
done < SHA256SUMS
