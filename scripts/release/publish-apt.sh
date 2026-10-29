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
# SIGNING IS MANDATORY. The dists/<codename>/ of both buckets is shared with
# the other Redis products and already signed (InRelease + Release.gpg): an
# unsigned deb-s3 run deletes Release.gpg and leaves the old InRelease behind,
# so every user of that codename gets "Hash Sum mismatch" from apt update.
# The keyring must hold exactly one secret key (the workflow imports
# APT_SIGNING_KEY); its fingerprint is passed as `--sign=<fpr>`. Never a bare
# `--sign`: it is a string option, and the next argument (the first .deb) would
# be taken as the key id and silently not uploaded.
#
# THE KEY MUST BE THE ONE THE REPOSITORY IS ALREADY SIGNED WITH. deb-s3
# re-signs the whole codename (every product's index), so a wrong key (the
# staging throwaway key, a rotated one) would leave every redis / memtier APT
# user with NO_PUBKEY on `apt update`. Before anything is written, each target
# codename's existing dists/<codename>/InRelease is read (aws s3, the object
# deb-s3 replaces) and must verify as signed by our key; a codename with no
# InRelease yet (a fresh staging tree or prefix) is new and accepted. The key
# must also be able to sign at all (no passphrase, not expired): deb-s3
# uploads the new Release BEFORE signing it, so a signing failure mid-run
# would leave Release and InRelease disagreeing ("Hash Sum mismatch").
#
# deb-s3 refuses (`--fail-if-exists`) a package whose name+version is already
# in the codename's index under a different file, and a pool object whose
# content differs from what is there; an identical re-upload is a no-op, so a
# re-run after a partial publish completes it. --overwrite drops that refusal
# (staging rehearsals only; on production the CDN would keep serving the old
# bytes under the new hash for a while).
#
# `--lock` takes deb-s3's per-codename lock (deb/dists/<codename>/main/binary-/lockfile),
# the same one redis-debian's publish takes, so two products do not rewrite one
# Release concurrently. A job killed mid-publish leaves it behind; the next run
# waits ten minutes and fails with "Repository is locked": remove the lockfile
# object by hand after checking nothing else is publishing.
#
# AWS credentials come from the environment (static keys or an assumed role;
# deb-s3 reads them through the AWS SDK, session tokens included). The
# identity needs s3:PutObjectAcl as well as Put/Get/Delete (public-read objects).
# --dry-run prints the deb-s3 commands instead of running them and needs
# neither deb-s3, credentials nor a key.
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
command -v dpkg-deb >/dev/null || { echo "publish-apt.sh: dpkg-deb not on PATH" >&2; exit 1; }
[[ "$dry_run" == 1 ]] || command -v deb-s3 >/dev/null || { echo "publish-apt.sh: deb-s3 not on PATH" >&2; exit 1; }
[[ "$dry_run" == 1 ]] || command -v aws >/dev/null || { echo "publish-apt.sh: aws not on PATH" >&2; exit 1; }

# Captured first, not `gpg | grep`: under pipefail grep's early exit would read
# as "no key" (see lib.sh on producer | grep -q).
secret_keys="$(gpg --list-secret-keys --with-colons 2>/dev/null || true)"
n_keys="$(grep -c '^sec:' <<<"$secret_keys" || true)"
fpr="$(awk -F: '/^fpr:/{print $10; exit}' <<<"$secret_keys")"
if [[ "$n_keys" -ne 1 || -z "$fpr" ]]; then
  if [[ "$dry_run" == 1 ]]; then
    fpr="DRYRUN0000000000000000000000000000000000"
    echo "publish-apt.sh: dry run with no signing key; would sign with the keyring's one secret key"
  else
    echo "publish-apt.sh: the GPG keyring must hold exactly ONE secret key (found $n_keys); refusing to publish unsigned into a shared, signed repository" >&2
    exit 1
  fi
fi
echo "publish-apt.sh: signing with $fpr"

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

if [[ "$dry_run" != 1 ]]; then
  work="$(mktemp -d)"
  trap 'rm -rf "$work"' EXIT
  gpg --batch --no-tty --local-user "$fpr" --clearsign <<<"publish-apt.sh signing test" >/dev/null 2>"$work/sign.err" || {
    echo "publish-apt.sh: the key $fpr cannot sign non-interactively (passphrase? expired?):" >&2; cat "$work/sign.err" >&2; exit 1; }
  for dist in $(printf '%s\n' "${!by_dist[@]}" | LC_ALL=C sort); do
    obj="s3://$bucket/$prefix/dists/$dist/InRelease"
    rc=0; listing="$(aws s3 ls "$obj" 2>"$work/ls.err")" || rc=$?
    if [[ "$rc" -eq 1 && -z "$listing" && ! -s "$work/ls.err" ]]; then
      echo "$dist: no InRelease yet, a new codename in this repository"
      continue
    fi
    [[ "$rc" -eq 0 ]] || { echo "publish-apt.sh: aws s3 ls $obj failed (exit $rc):" >&2; cat "$work/ls.err" >&2; exit 1; }
    aws s3 cp "$obj" "$work/InRelease-$dist" --only-show-errors
    status="$(gpg --batch --no-tty --status-fd 1 --verify "$work/InRelease-$dist" 2>/dev/null || true)"
    if ! grep -q "^\[GNUPG:\] VALIDSIG $fpr " <<<"$status"; then
      signer="$(sed -n 's/^\[GNUPG:\] \(VALIDSIG\|ERRSIG\|NO_PUBKEY\|BADSIG\) \([0-9A-F]*\).*/\2/p' <<<"$status" | head -n1)"
      echo "publish-apt.sh: $obj is signed by ${signer:-an unreadable signature}, not by $fpr." >&2
      echo "  Publishing would re-sign every product's $dist index with a key its users do not trust; refusing." >&2
      echo "  Check the environment's APT_SIGNING_KEY (and APT_SIGNING_FPR)." >&2
      exit 1
    fi
    echo "$dist: existing InRelease is signed by $fpr"
  done
fi

exists=(--fail-if-exists)
[[ "$overwrite" == 1 ]] && exists=()

for dist in $(printf '%s\n' "${!by_dist[@]}" | LC_ALL=C sort); do
  debs=()
  while IFS= read -r f; do [[ -n "$f" ]] && debs+=("$f"); done <<<"${by_dist[$dist]}"
  echo "--- $dist: ${#debs[@]} package(s)"
  for f in "${debs[@]}"; do dpkg-deb -f "$f" Package Version Architecture | paste -sd' '; done
  # Every option in `--opt=value` form, and the packages last: a value-taking
  # option written as two tokens could swallow the first package path.
  cmd=(deb-s3 upload
    "--bucket=$bucket" "--s3-region=$region" "--prefix=$prefix"
    "--codename=$dist" "--suite=$dist" --component=main "--origin=$origin"
    --visibility=public --preserve-versions --lock "${exists[@]}"
    "--sign=$fpr" "--gpg-options=--batch --no-tty"
    "${debs[@]}")
  if [[ "$dry_run" == 1 ]]; then
    printf 'dry-run:'; printf ' %q' "${cmd[@]}"; echo
  else
    "${cmd[@]}"
  fi
done
echo "publish-apt.sh: done (s3://$bucket/$prefix, codenames: $(printf '%s ' "${!by_dist[@]}"))"
