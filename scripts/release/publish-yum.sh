#!/usr/bin/env bash
# Publish built RPMs into an S3-backed YUM repository laid out like packages.redis.io.
#
#   scripts/release/publish-yum.sh --bucket <bucket> --in <dir> [--prefix rpm] [--overwrite] [--dry-run]
#
# <dir> is searched recursively for *.rpm. Each RPM goes to
# <prefix>/<distro>/<arch>/ where <distro> comes from its own dist tag
# (Release `1.el9` -> el9, `1.amzn2023` -> amzn2023; lib.sh rpm_release_to_distro)
# and <arch> from its header, i.e. the README's baseurl pattern
# rpm/el$releasever/$basearch and rpm/amzn2023/$basearch. The bucket's rpm/
# prefix is SHARED with other Redis products (prod: redis, redis-stack; staging:
# redis-rpm's rockylinux*/ dirs), so this script:
#  * touches only the <distro>/<arch>/ directories it has an RPM for,
#  * downloads each such directory in full before regenerating its metadata,
#    so createrepo_c --update indexes everything already there (and keeps any
#    extra metadata other products placed: --keep-all-metadata),
#  * never deletes an RPM; only the directory's repodata/ is replaced, RPMs
#    are uploaded before it, and the previous hashed metadata files are kept
#    one generation (--retain-old-md 1) for clients mid-refresh.
# Objects are uploaded public-read (the CDN serves per-object ACLs).
#
# Re-runs are idempotent: an RPM already in the repo with the same NEVRA and
# build time is skipped (a partial publish — one directory uploaded, the next
# failed on a transient S3 error — completes on re-run); a DIFFERENT build
# under the same file name is refused. --overwrite replaces it anyway
# (staging rehearsals only).
#
# SIGNING IS MANDATORY (except --dry-run): the keyring must hold exactly one
# secret key (the workflow imports APT_SIGNING_KEY); every RPM is signed with
# it and the signature verified against a private rpm database before upload.
# --dry-run maps, copies and generates metadata locally, prints the aws
# commands instead of running them, and needs no network, credentials or key.
set -euo pipefail

bucket="" indir="" prefix=rpm overwrite=0 dry_run=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --bucket) bucket="$2"; shift 2 ;;
    --in) indir="$2"; shift 2 ;;
    --prefix) prefix="$2"; shift 2 ;;
    --overwrite) overwrite=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    *) echo "publish-yum.sh: unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$bucket" && -d "$indir" ]] || {
  echo "usage: publish-yum.sh --bucket <bucket> --in <dir> [--prefix rpm] [--overwrite] [--dry-run]" >&2; exit 2; }
for t in createrepo_c rpm; do
  command -v "$t" >/dev/null || { echo "publish-yum.sh: $t not on PATH" >&2; exit 1; }
done
[[ "$dry_run" == 1 ]] || command -v aws >/dev/null || { echo "publish-yum.sh: aws not on PATH" >&2; exit 1; }

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/release/lib.sh
source "$here/lib.sh"

run() { if [[ "$dry_run" == 1 ]]; then printf 'dry-run:'; printf ' %q' "$@"; echo; else "$@"; fi; }

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

rpms=()
while IFS= read -r -d '' f; do rpms+=("$f"); done < <(find "$indir" -type f -name '*.rpm' -print0 | LC_ALL=C sort -z)
[[ ${#rpms[@]} -gt 0 ]] || { echo "publish-yum.sh: no .rpm under $indir" >&2; exit 1; }

secret_keys="$(gpg --list-secret-keys --with-colons 2>/dev/null || true)"
n_keys="$(grep -c '^sec:' <<<"$secret_keys" || true)"
fpr="$(awk -F: '/^fpr:/{print $10; exit}' <<<"$secret_keys")"
if [[ "$n_keys" -eq 1 && -n "$fpr" && "$dry_run" != 1 ]]; then
  echo "publish-yum.sh: signing RPMs with $fpr"
  # rpm's %__gpg defaults to a gpg2 binary name some distros (Ubuntu) do not
  # ship; point it at the gpg on PATH, and name the key by fingerprint.
  grep -q '^%__gpg ' ~/.rpmmacros 2>/dev/null || echo "%__gpg $(command -v gpg)" >> ~/.rpmmacros
  grep -q '^%_gpg_name' ~/.rpmmacros 2>/dev/null || echo "%_gpg_name $fpr" >> ~/.rpmmacros
  for f in "${rpms[@]}"; do rpm --addsign "$f"; done
  # Verify against a private rpm database: no root, no host keyring.
  mkdir -p "$work/rpmdb"
  rpm --dbpath "$work/rpmdb" --initdb
  gpg --export --armor "$fpr" > "$work/signing-pub.asc"
  rpm --dbpath "$work/rpmdb" --import "$work/signing-pub.asc"
  for f in "${rpms[@]}"; do
    out="$(rpm --dbpath "$work/rpmdb" -Kv "$f" 2>&1 || true)"
    if grep -qwE 'BAD|NOTTRUSTED|NOKEY' <<<"$out"; then echo "publish-yum.sh: $f: signature does not verify:" >&2; echo "$out" >&2; exit 1; fi
    if ! grep -qE 'Header V[34] (RSA|DSA|EdDSA)/.*: OK' <<<"$out"; then echo "publish-yum.sh: $f: not signed:" >&2; echo "$out" >&2; exit 1; fi
  done
elif [[ "$dry_run" == 1 ]]; then
  echo "publish-yum.sh: dry run, RPMs left unsigned"
else
  echo "publish-yum.sh: the GPG keyring must hold exactly ONE secret key (found $n_keys); refusing to publish unsigned RPMs" >&2
  exit 1
fi

declare -A dirs=()
for f in "${rpms[@]}"; do
  distro="$(rpm_release_to_distro "$(rpm -qp --queryformat '%{RELEASE}' "$f")")"
  arch="$(rpm -qp --queryformat '%{ARCH}' "$f")"
  [[ "$arch" =~ ^(x86_64|aarch64)$ ]] || { echo "publish-yum.sh: $f: refusing unexpected arch '$arch'" >&2; exit 1; }
  mkdir -p "$work/$distro/$arch"
  cp "$f" "$work/$distro/$arch/"
  dirs["$distro/$arch"]=1
done

nevra_of() { rpm -qp --queryformat '%{NAME}-%{EPOCHNUM}:%{VERSION}-%{RELEASE}.%{ARCH} %{BUILDTIME}' "$1" 2>/dev/null || echo unreadable; }

for d in $(printf '%s\n' "${!dirs[@]}" | LC_ALL=C sort); do
  remote="s3://$bucket/$prefix/$d"
  local_dir="$work/.repo/$d"
  mkdir -p "$local_dir"
  echo "--- $d"
  # Everything already there, so the regenerated metadata covers it. A sync
  # error is a real error: without the existing files, --update would publish
  # metadata that forgets the other products' RPMs in this directory.
  run aws s3 sync "$remote/" "$local_dir/" --only-show-errors
  upload=()
  for f in "$work/$d"/*.rpm; do
    b="$(basename "$f")"
    if [[ -f "$local_dir/$b" ]]; then
      have="$(nevra_of "$local_dir/$b")"; want="$(nevra_of "$f")"
      if [[ "$have" == "$want" && "$overwrite" != 1 ]]; then
        echo "already published, skipping upload: $prefix/$d/$b ($have)"
        continue
      fi
      if [[ "$overwrite" != 1 ]]; then
        echo "publish-yum.sh: $prefix/$d/$b already exists in s3://$bucket as a DIFFERENT build ($have vs $want); refusing to replace a published RPM (--overwrite for staging rehearsals)" >&2
        exit 1
      fi
    fi
    cp "$f" "$local_dir/$b"
    upload+=("$b")
  done
  createrepo_c --update --keep-all-metadata --retain-old-md 1 "$local_dir" >/dev/null
  # RPMs first, metadata last, so no index ever points at a missing file.
  for b in ${upload[@]+"${upload[@]}"}; do
    run aws s3 cp "$local_dir/$b" "$remote/$b" --acl public-read --only-show-errors
  done
  run aws s3 sync "$local_dir/repodata/" "$remote/repodata/" --delete --acl public-read --only-show-errors
  echo "published: $(ls "$work/$d") -> $remote/ (uploaded: ${upload[*]:-none, all already there})"
done
echo "publish-yum.sh: done (s3://$bucket/$prefix, dirs: $(printf '%s ' "${!dirs[@]}"))"
