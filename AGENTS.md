# AGENTS.md

Instructions for any AI coding agent in this repo. A rule that governs one item lives on that item; read its doc comment and header before changing it.

## The goal

Lofi that sounds better without leaving the genre. The A/B judge holds the genre; [`LOFI-BIBLE.md`](.claude/skills/procedural-lofi/reference/LOFI-BIBLE.md)'s MEASURED tempo and the reference corpus's spectral spread are guard rails `lofi_measure` prints on every sound change, and a change that leaves them says so beside the judge's verdict. The first listener is [pixtuoid](https://github.com/pixtuoid/pixtuoid)'s office, which plays lofi for as long as it runs, so the music must not fatigue over hours.

## Changing the sound

Measurement steers a change toward a target; CI's [`judge.yml`](.github/workflows/judge.yml) posts `ab_judge.py`'s score on every PR that changes the render, and the owner's blind listen, when given, overrides it. A music change ends in an A/B batch the judge scores (the `procedural-lofi` skill's "GENERATOR loop" step 5), and its commit records the verdict. Where each taste axis lives and how to add an instrument: the `procedural-lofi` skill's "GENERATOR loop". The taste anchor is in [`docs/phase0-ratification.md`](docs/phase0-ratification.md).

## Map

- `perform`: the public engine, `Performance`, its `Controls` and lofi's output level.
- `stream`: the engine's chunk-invariant parts: the song schedule, the lane chains and their filters, and the master: each lane's gain for the intensity, and the ceiling.
- `compose`: seed → a `Score` and a `Form`, each `Stage` drawn from a stream of its own. Each stream's draw order is the sound.
- `synth`: voices, the lane recipe both chains read, and the whole-lane chain behind `render_lane`.
- `dsp`: the public primitives, dependency-free on purpose.
- `score`: test-only frozen takes that the fingerprint pins guard.
- `audition`: lofi's own tools' access to the parts; no contract.

## Invariants no single item owns

- **The score is bit-identical everywhere; rendered audio is not.** `+ - * /` are exact ([RFC 3514](https://rust-lang.github.io/rfcs/3514-float-semantics.html)) but `sin`/`exp` [are not](https://doc.rust-lang.org/std/primitive.f32.html#method.sin), so tests pin fingerprints, never sample values.
- **Output is a pure function of the seed and the `(position, Controls)` timeline**: no `render` chunking changes it, and song k's noise comes from `(seed, k)`, never from render history (the chunk-invariance and no-history tests).
- **The public API is what `src/lib.rs` exports, `#[doc(hidden)]` items aside** ([they promise nothing](https://predr.ag/blog/checking-semver-for-doc-hidden-items/)): every other `pub` is a semver promise.
- **No audio file is ever committed** (`.gitignore`, CI): the references are copyrighted.

## Workflow

- [`ci.yml`](.github/workflows/ci.yml) is the gate, and runs the [`justfile`](justfile)'s recipes. `just setup` turns on the hooks that run them first: pre-commit checks formatting and pre-push runs `just preflight`.
- [`lofi_bench`](examples/lofi_bench.rs) gates what CI can't afford: `Performance::render`'s cost, and a run of each room playing finite at a steady level. A change on render's path runs it natively and on wasm, and the PR names the machine and the numbers.
- A session pushes its own PR branches and merges its own PR once the ruleset's checks pass, the org's [shared review](https://github.com/pixtuoid/.github/blob/main/.github/workflows/claude-review.yml) among them, and every review thread is resolved. A blocking finding it refutes is dispositioned and resolved in its thread before the review re-runs (`gh run rerun <id> --failed`). It never force-pushes or pushes to `main`, never skips hooks (`--no-verify`), and never files issues.
- A commit's title is its changelog line. In 0.x, `feat`/`fix` bump the patch and `!` bumps the minor ([release-plz](https://release-plz.dev/docs/config)). A changed `dsp` output takes `!`: pixtuoid's foley hears it, and no semver check does. A new `Controls` field doesn't ([`#[non_exhaustive]`](https://doc.rust-lang.org/cargo/reference/semver.html#struct-add-public-field-when-no-private)), as long as its default keeps the old sound.
- Releasing is the owner's: a hand-dispatched release PR, then the owner-approved `release` environment ([`release-plz.yml`](.github/workflows/release-plz.yml)).

## Conventions

- TDD; every mechanism names the failure it prevents; reuse before you build, except a dependency.
- Comments, doc prose and PR bodies included, are WHY only and sit on the narrowest thing they constrain. Name the authority instead of restating its value (`` [`SAMPLE_RATE`] ``, not `44100`). Measurements go in commit messages.
- No magic numbers: one named `const`, apart from self-evident `0`/`1`/`2`, indices and test fixtures. No `unwrap()` outside tests.
- An external claim, "best practice" included, cites a doc fetched this session or is labeled opinion, and a design change first looks up the field's standard design for its seam.
- Check a review finding's premise before accepting it. A refuted finding produces a mechanism (a test, a CI gate) or nothing.
- Docs change in the same commit as what they describe. No new doc lands unless the owner asks.
