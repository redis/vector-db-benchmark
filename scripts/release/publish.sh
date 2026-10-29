#!/usr/bin/env bash
# Upload a checked set of release archives + SHA256SUMS to a GitHub release (#366).
#
#   REPO=owner/name [CLOBBER=true] scripts/release/publish.sh <tag> <dir>
#
# <dir> holds exactly the archives and the SHA256SUMS that `checksums.sh check`
# accepts. The release ends up serving that set and nothing else of this tag:
#
#  * Nothing of ours there yet: upload the archives, then SHA256SUMS last, so a
#    reader never sees checksums for archives that are not there yet.
#  * Some or all of ours already there with IDENTICAL digests (a re-run after a
#    partial upload): upload only what is missing. Re-running is idempotent.
#  * Anything of ours there with a DIFFERENT (or unknown) digest, or a stale
#    `vector-db-benchmark-<tag>-*.tar.gz` that this set does not contain:
#    refuse, unless CLOBBER=true. A release must not mix archives from two
#    builds, and SHA256SUMS must describe what is actually served.
#  * CLOBBER=true: delete SHA256SUMS FIRST (so no stale checksum file is live
#    while archives change), delete stale archives, replace the archives, then
#    upload the new SHA256SUMS.
#
# `gh` is invoked as $GH (default `gh`) so the self-test can substitute a fake.
# Every `gh` failure aborts: the asset listing is a plain command substitution,
# so an API error cannot read as "the release is empty" and let an upload
# overwrite what is there.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/release/lib.sh
source "$here/lib.sh"

GH="${GH:-gh}"
CLOBBER="${CLOBBER:-false}"
tag="${1:-}" dir="${2:-}"
[[ -n "$tag" && -d "$dir" && -n "${REPO:-}" ]] || {
  echo "usage: REPO=owner/name [CLOBBER=true] publish.sh <tag> <dir>" >&2; exit 2; }
check_tag "$tag"
bash "$here/checksums.sh" check "$dir" >/dev/null

ours=()
while IFS= read -r f; do ours+=("$f"); done < <(cd "$dir" && find . -maxdepth 1 -type f | sed 's|^\./||' | LC_ALL=C sort)
is_ours() { printf '%s\n' "${ours[@]}" | grep -qxF "$1"; }

# name<TAB>digest for every asset currently on the release. Fails the script
# (set -e) if gh fails.
listing="$("$GH" api "repos/$REPO/releases/tags/$tag" --jq '.assets[] | [.name, (.digest // "")] | @tsv')"

present=() conflicting=() stale=()
while IFS=$'\t' read -r name digest; do
  [[ -n "$name" ]] || continue
  if is_ours "$name"; then
    if [[ "$digest" == "sha256:$(sha256_file "$dir/$name")" ]]; then
      present+=("$name")
    else
      conflicting+=("$name")
    fi
  elif [[ "$name" == "vector-db-benchmark-$tag-"*.tar.gz ]]; then
    stale+=("$name")
  fi
done <<<"$listing"

upload() { # upload <extra gh flags...> -- <names...>
  local flags=() files=()
  while [[ "$1" != -- ]]; do flags+=("$1"); shift; done
  shift
  for n in "$@"; do files+=("$dir/$n"); done
  [[ ${#files[@]} -gt 0 ]] || return 0
  echo "uploading: $*"
  "$GH" release upload "$tag" "${files[@]}" -R "$REPO" ${flags[@]+"${flags[@]}"}
}

archives=()
for f in "${ours[@]}"; do [[ "$f" == SHA256SUMS ]] || archives+=("$f"); done

if [[ ${#conflicting[@]} -eq 0 && ${#stale[@]} -eq 0 ]]; then
  missing=()
  for f in "${archives[@]}"; do
    printf '%s\n' ${present[@]+"${present[@]}"} | grep -qxF "$f" || missing+=("$f")
  done
  [[ ${#present[@]} -eq 0 ]] || echo "already on the release with identical digests: ${present[*]}"
  # No --clobber: gh itself refuses to overwrite an existing name.
  upload -- ${missing[@]+"${missing[@]}"}
  if ! printf '%s\n' ${present[@]+"${present[@]}"} | grep -qxF SHA256SUMS; then
    upload -- SHA256SUMS
  fi
  echo "publish.sh: $tag serves exactly this set"
  exit 0
fi

if [[ "$CLOBBER" != true ]]; then
  echo "publish.sh: release $tag already has assets from a different build:" >&2
  [[ ${#conflicting[@]} -eq 0 ]] || echo "  different digest: ${conflicting[*]}" >&2
  [[ ${#stale[@]} -eq 0 ]] || echo "  not in this set:  ${stale[*]}" >&2
  echo "Refusing to mix builds. Re-run via workflow_dispatch with clobber: true to replace the whole set." >&2
  exit 1
fi

echo "CLOBBER: replacing the whole set on $tag"
if printf '%s\n' ${present[@]+"${present[@]}"} ${conflicting[@]+"${conflicting[@]}"} | grep -qxF SHA256SUMS; then
  "$GH" release delete-asset "$tag" SHA256SUMS -R "$REPO" --yes
fi
for s in ${stale[@]+"${stale[@]}"}; do
  "$GH" release delete-asset "$tag" "$s" -R "$REPO" --yes
done
upload --clobber -- "${archives[@]}"
upload -- SHA256SUMS
echo "publish.sh: $tag serves exactly this set"
