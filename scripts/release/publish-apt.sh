#!/usr/bin/env bash
# Publish built .debs to an S3-backed APT repository with deb-s3, one codename at a time.
#
#   scripts/release/publish-apt.sh --bucket <bucket> --region <region> --in <dir> [--prefix deb] [--origin packages.redis.io] [--overwrite] [--dry-run]
#
# <dir> is searched recursively for *.deb; each package's codename is read from
# its own Version (`<crate version>~<codename>`), so the directory layout the
# workflow artifacts happen to have does not matter. The repository layout is
# the one packages.redis.io serves: <prefix>/dists/<codename>/... and
# <prefix>/pool/<codename>/..., objects public-read, `Origin: packages.redis.io`.
#
# Signing: if the GPG keyring holds a secret key (the workflow imports
# APT_SIGNING_KEY), Release is signed with it (InRelease + Release.gpg).
# Without one the repo is published UNSIGNED and this script says so loudly;
# that is acceptable for a staging rehearsal only.
#
# --overwrite  allow re-publishing a version that is already in the index
#              (deb-s3's default `--fail-if-exists` is what keeps a re-run from
#              silently replacing a shipped package). Staging rehearsals only.
# AWS credentials come from the environment (AWS_ACCESS_KEY_ID/... or an
# assumed role), as deb-s3 reads them through the AWS SDK.
set -euo pipefail

bucket="" region="" indir="" prefix=deb origin=packages.redis.io overwrite=0 dry_run=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --bucket) bucket="$2"; shift 2 ;;
    --region) region="$2"; shift 2 ;;
    --in) indir="$2"; shift 2 ;;
    --prefix) prefix="$2"; shift 2 ;;
    --origin) origin="$2"; shift 2 ;;
    --overwrite) overwrite=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    *) echo "publish-apt.sh: unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$bucket" && -n "$region" && -d "$indir" ]] || {
  echo "usage: publish-apt.sh --bucket <bucket> --region <region> --in <dir> [--prefix deb] [--origin <origin>] [--overwrite] [--dry-run]" >&2; exit 2; }
command -v deb-s3 >/dev/null || { echo "publish-apt.sh: deb-s3 not on PATH" >&2; exit 1; }
command -v dpkg-deb >/dev/null || { echo "publish-apt.sh: dpkg-deb not on PATH" >&2; exit 1; }

sign=()
# Captured first, not `gpg | grep -q`: under pipefail grep's early exit would
# read as "no key" (see lib.sh on producer | grep -q).
secret_keys="$(gpg --list-secret-keys --with-colons 2>/dev/null || true)"
if grep -q '^sec:' <<<"$secret_keys"; then
  sign=(--sign)
  echo "publish-apt.sh: signing with $(awk -F: '/^fpr:/{print $10; exit}' <<<"$secret_keys")"
else
  echo "::warning::publish-apt.sh: no GPG secret key in the keyring, publishing an UNSIGNED repository (staging rehearsal only)"
fi

# codename -> list of .deb files
declare -A by_dist=()
while IFS= read -r -d '' deb; do
  version="$(dpkg-deb -f "$deb" Version)"
  [[ "$version" == *~* ]] || { echo "publish-apt.sh: $deb: Version '$version' has no ~<codename> suffix" >&2; exit 1; }
  dist="${version##*~}"
  [[ "$dist" =~ ^[a-z]+$ ]] || { echo "publish-apt.sh: $deb: '$dist' is not a codename" >&2; exit 1; }
  by_dist[$dist]+="$deb"$'\n'
done < <(find "$indir" -type f -name '*.deb' -print0 | LC_ALL=C sort -z)
[[ ${#by_dist[@]} -gt 0 ]] || { echo "publish-apt.sh: no .deb under $indir" >&2; exit 1; }

exists=(--fail-if-exists)
[[ "$overwrite" == 1 ]] && exists=()

for dist in $(printf '%s\n' "${!by_dist[@]}" | LC_ALL=C sort); do
  debs=()
  while IFS= read -r f; do [[ -n "$f" ]] && debs+=("$f"); done <<<"${by_dist[$dist]}"
  echo "--- $dist: ${#debs[@]} package(s)"
  for f in "${debs[@]}"; do dpkg-deb -f "$f" Package Version Architecture | paste -sd' '; done
  cmd=(deb-s3 upload
    --bucket "$bucket" --s3-region "$region" --prefix "$prefix"
    --codename "$dist" --component main --origin "$origin"
    --visibility public --preserve-versions --lock "${exists[@]}" "${sign[@]}"
    "${debs[@]}")
  if [[ "$dry_run" == 1 ]]; then
    printf 'dry-run:'; printf ' %q' "${cmd[@]}"; echo
  else
    "${cmd[@]}"
  fi
done
echo "publish-apt.sh: done (s3://$bucket/$prefix, codenames: $(printf '%s ' "${!by_dist[@]}"))"
