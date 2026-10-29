#!/usr/bin/env bash
# Publish built RPMs into an S3-backed YUM repository laid out like packages.redis.io.
#
#   scripts/release/publish-yum.sh --bucket <bucket> --in <dir> [--prefix rpm] [--overwrite] [--dry-run]
#
# <dir> is searched recursively for *.rpm. Each RPM goes to
# <prefix>/<distro>/<arch>/ where <distro> comes from its own dist tag
# (Release `1.el9` -> el9, `1.amzn2023` -> amzn2023) and <arch> from its
# header, i.e. the README's baseurl pattern rpm/el$releasever/$basearch and
# rpm/amzn2023/$basearch. The bucket's rpm/ prefix is SHARED with other Redis
# products (prod: redis, redis-stack; staging: redis-rpm's rockylinux*/ dirs),
# so this script:
#  * touches only the <distro>/<arch>/ directories it has an RPM for,
#  * downloads each such directory in full before regenerating its metadata,
#    so createrepo_c --update indexes everything already there, and
#  * never deletes an RPM; only the directory's repodata/ is replaced.
# Objects are uploaded public-read (the CDN serves per-object ACLs).
#
# Signing: with a GPG secret key in the keyring (the workflow imports
# APT_SIGNING_KEY) every RPM is signed before upload. Without one the RPMs
# are published UNSIGNED and this script says so loudly (staging only).
# Re-runs are idempotent: an RPM already in the repo with the same NEVRA is
# skipped (a partial publish — one directory uploaded, the next one failed on a
# transient S3 error — completes on re-run), and only a DIFFERENT build under
# the same file name is refused.
# --overwrite  replace such a different build anyway (staging rehearsals).
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
for t in aws createrepo_c rpm; do
  command -v "$t" >/dev/null || { echo "publish-yum.sh: $t not on PATH" >&2; exit 1; }
done

rpms=()
while IFS= read -r -d '' f; do rpms+=("$f"); done < <(find "$indir" -type f -name '*.rpm' -print0 | LC_ALL=C sort -z)
[[ ${#rpms[@]} -gt 0 ]] || { echo "publish-yum.sh: no .rpm under $indir" >&2; exit 1; }

secret_keys="$(gpg --list-secret-keys --with-colons 2>/dev/null || true)"
if grep -q '^sec:' <<<"$secret_keys"; then   # captured first: no producer | grep -q under pipefail
  fpr="$(awk -F: '/^fpr:/{print $10; exit}' <<<"$secret_keys")"
  echo "publish-yum.sh: signing RPMs with $fpr"
  grep -q '^%_gpg_name' ~/.rpmmacros 2>/dev/null || echo "%_gpg_name $fpr" >> ~/.rpmmacros
  for f in "${rpms[@]}"; do rpm --addsign "$f"; done
  pubkey="$(mktemp)"
  gpg --export --armor "$fpr" > "$pubkey"
  rpm --import "$pubkey" 2>/dev/null || sudo rpm --import "$pubkey"
  rm -f "$pubkey"
  for f in "${rpms[@]}"; do
    out="$(rpm -Kv "$f" 2>&1 || true)"
    if grep -qwE 'BAD|NOTTRUSTED' <<<"$out"; then echo "publish-yum.sh: $f: bad signature:" >&2; echo "$out" >&2; exit 1; fi
    if ! grep -qE 'Header V[34] (RSA|DSA|EdDSA)/.*: (OK|NOKEY)' <<<"$out"; then echo "publish-yum.sh: $f: not signed:" >&2; echo "$out" >&2; exit 1; fi
  done
else
  echo "::warning::publish-yum.sh: no GPG secret key in the keyring, publishing UNSIGNED RPMs (staging rehearsal only)"
fi

run() { if [[ "$dry_run" == 1 ]]; then printf 'dry-run:'; printf ' %q' "$@"; echo; else "$@"; fi; }

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
declare -A dirs=()
for f in "${rpms[@]}"; do
  rel="$(rpm -qp --queryformat '%{RELEASE}' "$f")"
  arch="$(rpm -qp --queryformat '%{ARCH}' "$f")"
  case "$rel" in
    *.el[0-9]*) distro="el${rel##*.el}" ;;
    *.amzn2023*) distro=amzn2023 ;;
    *) echo "publish-yum.sh: $f: cannot map Release '$rel' to a repo directory (expected .elN or .amzn2023)" >&2; exit 1 ;;
  esac
  distro="${distro%%_*}"
  [[ "$distro" =~ ^(el[0-9]+|amzn2023)$ && "$arch" =~ ^(x86_64|aarch64)$ ]] ||
    { echo "publish-yum.sh: $f: refusing unexpected distro/arch '$distro/$arch'" >&2; exit 1; }
  mkdir -p "$work/$distro/$arch"
  cp "$f" "$work/$distro/$arch/"
  dirs["$distro/$arch"]=1
done

for d in $(printf '%s\n' "${!dirs[@]}" | LC_ALL=C sort); do
  remote="s3://$bucket/$prefix/$d"
  local_dir="$work/.repo/$d"
  mkdir -p "$local_dir"
  echo "--- $d"
  # Everything already there, so the regenerated metadata covers it. A sync
  # error is a real error: without the existing files, --update would publish
  # metadata that forgets the other products' RPMs in this directory.
  aws s3 sync "$remote/" "$local_dir/" --only-show-errors
  upload=()
  for f in "$work/$d"/*.rpm; do
    b="$(basename "$f")"
    if [[ -f "$local_dir/$b" ]]; then
      # Same NEVRA = the same build, re-signed or not: nothing to upload. A
      # different build under the same name is what must not be replaced.
      have="$(rpm -qp --queryformat '%{NAME}-%{EPOCHNUM}:%{VERSION}-%{RELEASE}.%{ARCH} %{BUILDTIME}' "$local_dir/$b" 2>/dev/null || echo unreadable)"
      want="$(rpm -qp --queryformat '%{NAME}-%{EPOCHNUM}:%{VERSION}-%{RELEASE}.%{ARCH} %{BUILDTIME}' "$f")"
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
  # --keep-all-metadata: other products' extra metadata in this directory
  # (updateinfo, comps, modules, a detached repomd.xml.asc) survives the
  # regeneration. --retain-old-md keeps the previous hashed files so a client
  # mid-refresh does not 404 on them the moment repodata/ is synced.
  createrepo_c --update --keep-all-metadata --retain-old-md 1 "$local_dir" >/dev/null
  # RPMs first, metadata last, so no index ever points at a missing file.
  for b in ${upload[@]+"${upload[@]}"}; do
    run aws s3 cp "$local_dir/$b" "$remote/$b" --acl public-read --only-show-errors
  done
  run aws s3 sync "$local_dir/repodata/" "$remote/repodata/" --delete --acl public-read --only-show-errors
  echo "published: $(ls "$work/$d") -> $remote/ (uploaded: ${upload[*]:-none, all already there})"
done
echo "publish-yum.sh: done (s3://$bucket/$prefix, dirs: $(printf '%s ' "${!dirs[@]}"))"
