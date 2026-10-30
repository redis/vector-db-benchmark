#!/usr/bin/env bash
# One-time setup of the production package signing key and the `build`
# environment's secrets and APT_SIGNING_FPR variable (release-packages.yml).
#
#   scripts/release/setup-signing-key.sh [OUT_DIR]      # default ~/vdb-packages-signing
#
# Run after scripts/release/setup-production-repo.sh --apply (which creates the
# bucket, the publisher role and the protected `build` environment).
# Needs: gpg (GnuPG 2.2+), gh logged in with admin on redis/vector-db-benchmark.
#  * Generates an RSA 4096 signing key (no passphrase: the publish job signs
#    non-interactively; 5-year expiry) in a throwaway GNUPGHOME, so your own
#    keyring is not touched.
#  * Writes the secret key, public key, revocation certificate and fingerprint
#    to OUT_DIR (created with mode 700; refused inside a git checkout, so key
#    material can never be committed). Move them to the team vault afterwards
#    and delete OUT_DIR.
#  * Sets the `build` secrets APT_S3_BUCKET, APT_S3_REGION, APT_S3_IAM_ARN,
#    APT_SIGNING_KEY and the variable APT_SIGNING_FPR.
#  * Does NOT set PACKAGES_PRODUCTION: that is switched on only once the code
#    of PR #386 is on master (setup-production-repo.sh prints that step then).
# Refuses to run if `build` already pins APT_SIGNING_FPR: that would rotate the
# key, which is a manual procedure (AGENTS.md "Rotating the signing key").
set -euo pipefail
unset GH_TOKEN

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/release/lib.sh
source "$here/lib.sh"

REPO=redis/vector-db-benchmark
ENV=build
ACCOUNT=726902207197
ROLE_ARN="arn:aws:iam::$ACCOUNT:role/vdb-packages-publisher"
UID_NAME='vector-db-benchmark package signing key'
OUT="${1:-$HOME/vdb-packages-signing}"

die() { echo "setup-signing-key.sh: $*" >&2; exit 1; }

command -v gpg >/dev/null || die "gpg not found (install GnuPG, or run this on a machine that has it)"
command -v gh  >/dev/null || die "gh not found"
gh auth status >/dev/null 2>&1 || die "gh is not logged in (gh auth login)"
gh api "repos/$REPO/environments/$ENV" >/dev/null 2>&1 || die "environment '$ENV' not found on $REPO (run setup-production-repo.sh --apply first)"

existing="$(gh api "repos/$REPO/environments/$ENV/variables/APT_SIGNING_FPR" --jq .value 2>/dev/null)" || existing=""
[[ -z "$existing" ]] || die "'$ENV' already pins APT_SIGNING_FPR=$existing; refusing to replace the signing key"
[[ ! -e "$OUT" ]] || die "$OUT already exists; move it away or pass another directory"
parent="$(cd "$(dirname "$OUT")" && pwd)" || die "the parent of $OUT does not exist"
if git -C "$parent" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  die "$OUT is inside a git checkout ($(git -C "$parent" rev-parse --show-toplevel)); key material must not land where it can be committed"
fi

mkdir -m 700 "$OUT"
GNUPGHOME="$(mktemp -d)"; export GNUPGHOME; chmod 700 "$GNUPGHOME"
trap 'gpgconf --kill gpg-agent >/dev/null 2>&1 || true; rm -rf "$GNUPGHOME"' EXIT

echo "== generating RSA 4096 signing key"
gpg --batch --pinentry-mode loopback --passphrase '' --quick-gen-key "$UID_NAME" rsa4096 sign 5y
FPR="$(gpg --list-secret-keys --with-colons | awk -F: '/^fpr:/{print $10; exit}')"
[[ "$FPR" =~ ^[0-9A-F]{40}$ ]] || die "could not read the new key's fingerprint"

umask 077
gpg --armor --export-secret-keys "$FPR" > "$OUT/vdb-packages-signing.secret.asc"
gpg --armor --export "$FPR"             > "$OUT/vdb-packages-signing.public.asc"
cp "$GNUPGHOME/openpgp-revocs.d/$FPR.rev" "$OUT/vdb-packages-signing.rev"
echo "$FPR" > "$OUT/FINGERPRINT"
[[ -s "$OUT/vdb-packages-signing.secret.asc" ]] || die "secret key export is empty"

echo "== setting '$ENV' secrets and variables on $REPO"
gh secret set APT_S3_BUCKET   --env "$ENV" -R "$REPO" --body "$PRODUCTION_BUCKET"
gh secret set APT_S3_REGION   --env "$ENV" -R "$REPO" --body "$PRODUCTION_REGION"
gh secret set APT_S3_IAM_ARN  --env "$ENV" -R "$REPO" --body "$ROLE_ARN"
gh secret set APT_SIGNING_KEY --env "$ENV" -R "$REPO" < "$OUT/vdb-packages-signing.secret.asc"
gh variable set APT_SIGNING_FPR --env "$ENV" -R "$REPO" --body "$FPR"

echo
echo "Done. Signing key fingerprint: $FPR"
echo "Files in $OUT:"
ls -l "$OUT"
cat <<EOF

Next:
  1. Store $OUT/vdb-packages-signing.secret.asc and .rev in the team vault,
     then: rm -rf "$OUT"
  2. Once #386 is merged to master (not before: master's older workflow would
     publish with the wrong Origin and no <base>/gpg), re-run
     scripts/release/setup-production-repo.sh, which then prints the
     PACKAGES_PRODUCTION=true and release_type=public tag=0.2.0 commands.
EOF
