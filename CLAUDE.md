# CLAUDE.md

All agent instructions for this repository live in **[AGENTS.md](./AGENTS.md)** —
read it before making changes. It is the same file Codex and other agents use, so
there is one set of instructions rather than two that can drift apart.

The short version, if you read nothing else:

- Verify with **`make agent-check`**, not `make check`. `make check` runs no tests.
- Run it unpiped and check `$?` — `make agent-check | tail && git commit` reads
  `tail`'s exit status, not `make`'s, and will commit over a failing gate.
- CI runs the suites in **`--release`**, so a `debug_assert!` does not exist there.
- Put logic in `src/` (unit-tested), not in `src/bin/`.
- When you add a guard, **break the thing it guards** and confirm it goes red.
