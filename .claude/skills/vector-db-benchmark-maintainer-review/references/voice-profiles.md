# Voice profiles — real vector-db-benchmark contributors

Mined from actual GitHub history on `redis/vector-db-benchmark` (`gh pr list`, `gh pr view
--json body,reviews,comments`, `gh issue list/view`) as of September 2026: **197 merged PRs, 100 issues
sampled, at least 8 distinct human contributors across two clearly different eras.** Read this alongside
`nitpick-taxonomy.md`.

## The two eras, and why they read completely differently

**Python era (2023 – Feb/March 2026).** This is the original shared codebase (a fork of
`qdrant/vector-db-benchmark`), and it has a real, if thin, multi-author cast: `filipecosta90` (4 PRs, including
the fork's own PR #2), `mpozniak95` (17 PRs — mostly dataset registrations and small CLI additions, e.g. "Add
1M,10M,20M,40M,100M,200M laion datasets", "Add REDIS_KEEP_DOCUMENTS and --only-configure flag"), `mihaic` (6,
mostly SVS/calibration work), `paulorsousa` (3), `slice4e` (3), `mpdimitr` (2, mixed-workload metrics). Review
in this era is real but bare: `filipecosta90 APPROVED` with an empty body (PR #21, #37) is the norm. Nobody in
this era left a substantive inline comment thread comparable to what you'll find in smaller sibling repos'
histories — if you need that kind of example, it isn't here either; don't invent one.

**Rust-rewrite era (roughly July 2026 onward).** `fcostaoliveira` authored 161 of 197 merged PRs overall, and
156 of the 197 merged in just July–August 2026 — essentially a from-scratch Rust rewrite of the harness, done
by one person at high velocity. This is the era almost every "which PR would this look like" question will
land in. Outside contributions still happen — `murtazayusuf` added the 15th engine (KiviDB, PR #203),
`paulorsousa` rewrote Weaviate's search path onto gRPC (PR #79, #90) — but they are the exception, not the
pattern.

## fcostaoliveira — the dominant author and, in this era, the dominant reviewer of his own work

### As author: PR descriptions are the real, evidenced standard

Recurring, real structure across dozens of PRs (e.g. #308, #233, #215, #286): a `## The bug` section that
names the exact mechanism and file (*"`configured_index_name()` returns one fixed name... and `create_index()`
runs `FLUSHALL` on every upload"*), a `## The fix` section, and a `## Tests` section that is frequently a table
mapping each new test to **what specifically it breaks when reverted** (*"verified RED by mutation... reverting
derivation to `env_or`, replacing the hash with plain truncation"*). Quantitative claims are live-verified
against a real, version-pinned server, not asserted (*"verified live against `mongodb-atlas-local:8.0.17` — 200
bytes accepted, 250 rejected"*). PR bodies frequently carry a `🤖 Generated with Claude Code` footer — this is
real evidence of the standard this repo's PRs are held to, not necessarily evidence of unaided human prose.

### As reviewer of his own work: a genuine, distinctive "adversarial review" methodology

This is the single most important, best-evidenced pattern in this repo's real history, and it is *not* a
formality. Real, verbatim structure across PR #233, #215, #286:

- Multiple labeled rounds against the same PR as it evolves (*"Updated after review round 2 — `273039b`"*,
  *"`dba6f23` — review round 3"*, *"Adversarial review addressed"*).
- A numbered defect scheme (B1/B2/B3/D1/D2/D3 in PR #233) with each defect's real-world effect **quantified**,
  not just described: *"interleaved A/B (n=1600/arm) puts the empty filter at +6.61% mean / +4.60% p50
  latency"*; *"recall 0.520 → 1.000 (closes #220)"*.
- Willingness to correct his own earlier claim in the same thread, explicitly: *"Correction to my own earlier
  comment: the base test count was wrong... The real base is 872"*; *"You were right and I was wrong. `dba6f23`
  claimed the fixture decorrelation in the commit message and the PR body; it was not in the tree."* This
  self-correction, stated plainly rather than glossed over, is real and worth imitating when your own earlier
  read of something in this review turns out to be wrong.
- A closing verification block naming exact numbers, not "tests pass": *"586 unit tests (was 573), 22 live
  Qdrant tests against `qdrant/qdrant:v1.18.2` (was 20), `cargo clippy --all-targets -- -D warnings` clean."*
- Explicit follow-up triage: a real bug found but out of scope for *this* PR gets named and either filed as a
  separate issue or explicitly deferred, never silently dropped (*"Worth filing before this merges, because the
  new error makes it reachable"*).

**What this means for the bot's voice**: when reviewing what would be a `fcostaoliveira`-authored PR (the
large majority of real traffic here), hold it to exactly this bar — cite exact file:line, quantify any effect
you claim, and separate "this changes correctness/fairness" from "this is a style/parity nit" — rather than a
generic checklist pass.

### As reviewer of someone else's PR: the one real, evidenced cross-author deep review

PR #203 (KiviDB, by `murtazayusuf`, a genuine outside contributor adding the 15th engine) is the template for
reviewing an outside contribution here. Real, verbatim structure:

- States what was actually checked, specifically, up front: *"multi-angle: RESP2/RESP3 parsing, vector
  encoding, filtering, concurrency, connection lifecycle, namespacing, EF gating, distance mapping, error
  handling, config, trait completeness, tests, and an injection/secrets sweep"* — then states the clean result
  plainly (*"found no correctness, concurrency, measurement-fidelity, security, or secrets defects in the
  production paths"*) before listing what needs fixing.
- Splits findings into **"Worth fixing before merge"** (two items, each with exact file:line and a proposed
  fix, e.g. *"The read timeout (300s) is shorter than `wait_for_indexing`'s `max_wait` (600s)... kividb.rs:190
  ... kividb.rs:365"*) and **"Optional / follow-up (at parity with the `dragonfly` template, so not blockers)"**
  — never mixes the two severities in one list.
- One of the two "before merge" items is a **test that doesn't test what it claims to** (a regression guard
  that reimplements the production logic inline instead of calling it) — a real, evidenced category worth
  checking for on any new engine's test suite.
- Closes with one plain sentence naming the net assessment and explicitly hands the merge decision to a human:
  *"Net: clean, well-tested, faithful clone — nice work, especially catching the `#151-4` per-config namespacing
  nuance. The two 'before merge' items are small and localized. The accept decision on a new third-party engine
  is the maintainer's call."*
- The contributor's real reply (*"Thanks for the thorough writeup — I ran the verification you asked for. Checked
  out on top of current `master` and it's fully green locally"*) shows this pattern gets genuine engagement, not
  silence — if you're modeling how a contributor might respond, this is the real precedent for a good-faith one.

### Issue-triage voice: the one real external-user reply

Issue #52 (Python era, a genuine external OOM report from a non-contributor) got a substantive, technically
precise reply once it became moot: names the exact old mechanism (*"process-per-client model — at 100 parallel
clients it spawned 100 worker processes, each of which held its own copy of the dataset"*), the exact new one
(*"N threads that share a single in-memory copy... adds ~N threads and N connections (tens to low-hundreds of
MB), not N× the data"*), and a concrete memory figure for the reporter's own dataset. This is the real template
for responding to a genuine external report: name the actual mechanism, give a concrete number, don't hand-wave.

## Other contributors — real but sparse voice evidence

`mpozniak95`, `mihaic`, `paulorsousa`, `slice4e`, `mpdimitr`, `filipecosta90` appear almost exclusively as PR
*authors* in the mined sample, with terse, functional PR titles and no recorded substantive review text of
their own (their PRs get bare `APPROVED`s, not the reverse). There is no real quote to draw a richer "how X
reviews" profile from for any of them — if asked to imitate one of these contributors specifically, say plainly
that the record doesn't support more than "terse, feature/fix-focused PR titles, no recorded review voice."

## CI bots — real signal, not maintainer voice

- **`github-actions[bot]` "Docker Build Validation"** (`.github/workflows/docker-build-pr.yml`) posts a
  standardized comment on most PRs listing exactly what it built and ran (amd64 build+test, arm64 build
  validation, `--help`, `--describe datasets`, a Redis integration smoke test). Real and deterministic — cite
  it as "already covered," never rewrite it as if a human said it.
- **`ci.yml`** blocks on `cargo fmt --check` and `cargo clippy --all-targets -- -D warnings`, plus `cargo test
  --lib --bins --release`, a CLI smoke suite, and named measurement/harness invariant suites
  (`overhead_invariants`, `INV-P1..P6`). Treat a PR that doesn't touch these as passing them; don't re-derive
  formatting/lint nits from the diff by eye.
- No GitHub Copilot review activity was found in this repo's sampled history (unlike some sibling repos) —
  don't invent Copilot commentary here.
