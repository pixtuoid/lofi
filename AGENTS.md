# AGENTS.md

Instructions for any AI coding agent in this repo. A rule that governs one item lives on that item; read its doc comment and header before changing it.

## The goal

Lofi that sounds better without leaving the genre. The genre is pinned by measurement: the targets [`LOFI-BIBLE.md`](.claude/skills/procedural-lofi/reference/LOFI-BIBLE.md) marks MEASURED and the fingerprint pins; its literature numbers are starting points. The first listener is [pixtuoid](https://github.com/IvanWng97/pixtuoid)'s office, which loops these lanes for as long as it runs, so a take must not fatigue on repeat.

## Changing the sound

Measurement steers a change toward a target; only the owner's LISTEN says it sounds good. A music change ends in an audition batch the owner blind-listens (`cargo run --release --example lofi_audition -- --seeds N`), and its commit records the verdict. Where each taste axis lives and how to add an instrument: the `procedural-lofi` skill's "GENERATOR loop". The taste anchor is in [`docs/phase0-ratification.md`](docs/phase0-ratification.md).

## Map

- `compose`: seed → `Score`. The rng draw order is the sound.
- `synth`: voices, the master chain, `render_lane`.
- `dsp`: the public primitives, dependency-free on purpose.
- `score`: test-only frozen takes that the fingerprint pins guard.

## Invariants no single item owns

- **The score is bit-identical everywhere; rendered audio is not.** `+ - * /` are exact ([RFC 3514](https://rust-lang.github.io/rfcs/3514-float-semantics.html)) but `sin`/`exp` [are not](https://doc.rust-lang.org/std/primitive.f32.html#method.sin), so tests pin fingerprints, never sample values.
- **A musical lane is a whole number of loops long**, counted in samples, so stems loop in phase (`generated_beds_are_finite_phase_locked_and_in_the_sound_world`).
- **The public API is what `src/lib.rs` exports**: every `pub` is a semver promise.
- **No audio file is ever committed** (`.gitignore`, CI): the references are copyrighted.

## Workflow

- [`ci.yml`](.github/workflows/ci.yml) is the gate, and no hook runs it: run its steps before pushing.
- A session pushes its own PR branches and merges its own PR once the ruleset's checks pass, the [review](.github/workflows/claude-review.yml)'s among them (it fails unless the review at the head reports `Blocking: 0`), and every review thread is resolved. A blocking finding it refutes gets its disposition in its thread, and every such thread is resolved before `gh run rerun <id> --failed` re-runs the review, which judges no new change. It never force-pushes or pushes to `main`, never skips hooks (`--no-verify`), and never files issues.
- A commit's title is its changelog line. In 0.x, `feat`/`fix` bump the patch and `!` bumps the minor ([release-plz](https://release-plz.dev/docs/config)). A new `Mood`/`Lane` or a changed `dsp` output takes `!`: pixtuoid's foley hears the latter, and no semver check does.
- Releasing is the owner's: a hand-dispatched release PR, then the owner-approved `release` environment ([`release-plz.yml`](.github/workflows/release-plz.yml)).

## Conventions

- TDD; every mechanism names the failure it prevents; reuse before you build, except a dependency.
- Comments, doc prose and PR bodies included, are WHY only and sit on the narrowest thing they constrain. Name the authority instead of restating its value (`` [`SAMPLE_RATE`] ``, not `44100`). Measurements go in commit messages.
- No magic numbers: one named `const`, apart from self-evident `0`/`1`/`2`, indices and test fixtures. No `unwrap()` outside tests.
- An external claim, "best practice" included, cites a doc fetched this session or is labeled opinion.
- Check a review finding's premise before accepting it. A refuted finding produces a mechanism (a test, a CI gate) or nothing.
- Docs change in the same commit as what they describe. No new doc lands unless the owner asks.
