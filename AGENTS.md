# AGENTS.md

Instructions for any AI coding agent in this repo. A rule that governs one item lives on that item; read its doc comment and header before changing it.

<!-- org-rules:begin -->
## Org rules

Shared by every pixtuoid repo. The source is pixtuoid/.github's `.github/actions/org-rules/org-rules.md`, and that action fails CI when a repo's copy differs: change the rules there, then sync each repo's copy.

The owner's bar, in their words:

> 反正一定要确保所有的改动是best pracitce + best arch design + best future proof + 符合业界标准做法 + 符合md 规则 如果existing codebase不符合规范就去refactor 不做低ROI 不over engineer 不向legacy/下兼容

- **Evidence.** An external claim, "best practice" included, cites a doc URL or `path:line` fetched this session, or is labelled opinion.
- **Refactor.** Existing code a change touches that breaks a written rule is refactored in that change. Any other code may be, in a commit of its own, when that removes a named defect, never on taste alone.
- **Every mechanism earns its upkeep.** Code, config, a test, a CI job or a doc stays only if it names a failure it prevents that nothing else already does. A maintained library, the platform or the upstream tool's own mechanism is reused before one is built.
- **TDD.** A failing test, then the minimal implementation, then the commit.
- **Comments are WHY only**, doc prose and PR bodies included, and sit on the narrowest thing they constrain. Name the authority instead of restating its value. Measurements go in commit messages.
- **No magic numbers.** Reuse the existing authority, else one named constant at the narrowest scope; when a copy of a value must cross a boundary, a test pins the pair. No `unwrap()` outside tests.
- **Review findings.** A finding's premise is checked against the item it names before it is accepted. A refuted finding produces a mechanism (a test, a compile-time constraint, a CI gate) or nothing.
- **Docs** change in the same commit as what they describe, and no new doc lands unless the owner asks.
- **Git.** Never force-push and never skip hooks. How a change reaches `main` is each repo's own rule.
- **The [Roadmap](https://github.com/orgs/pixtuoid/projects/1) is the plan of record**, the owner's rule ("每做一步都需要sync board"): every step (an item started or finished, a PR opened, merged or closed, a reorder) updates its item, the draft or the issue it tracks, in the same turn. Who may file an issue is each repo's own rule.
<!-- org-rules:end -->

## The goal

Our own lofi, in several styles, each learned from a Lofi Girl playlist of its own, without leaving the genre. The A/B judge holds the genre; [`LOFI-BIBLE.md`](.claude/skills/procedural-lofi/reference/LOFI-BIBLE.md)'s MEASURED tempo and the hiphop corpus's spectral spread are guard rails `lofi_measure` prints on every sound change, and a change that leaves them says so beside the judge's verdict. The first listener is [pixtuoid](https://github.com/pixtuoid/pixtuoid)'s office, which plays lofi for as long as it runs, so the music must not fatigue over hours.

## Changing the sound

[`study/score.sh`](.claude/skills/procedural-lofi/study/score.sh) prints a scorecard a style: every measured dimension against its playlist's quartiles, which [`study/targets.sh`](.claude/skills/procedural-lofi/study/targets.sh) measures, by layer of the signal chain. Work goes down it from the top (opinion: a row means little to tune until the layers above it hold), and a PR names the rows it moves; going out of order takes a stated dependency. The org's [Roadmap](https://github.com/orgs/pixtuoid/projects/1) holds the order and the work no row holds, one issue an item: read it before choosing a change, and close an item from the PR that finishes it (`Fixes #n`). The judge's panel and FAD read a mono fold, so the side rows rest on the card and the owner's ear; so do the harmony rows, on #65's factor runs (opinion).

Measurement steers a change toward a target; CI's [`judge.yml`](.github/workflows/judge.yml) posts `ab_judge.py`'s score on every PR that changes the render, and the owner's blind listen, when given, overrides it. A music change ends in an A/B batch the judge scores (the `procedural-lofi` skill's "GENERATOR loop" step 5), and its commit records the verdict. Where each taste axis lives and how to add an instrument: the `procedural-lofi` skill's "GENERATOR loop". The taste anchor is in [`docs/phase0-ratification.md`](docs/phase0-ratification.md).

## Map

- `perform`: the public engine, `Performance`, its `Controls` and lofi's output level.
- `stream`: the engine's chunk-invariant parts: the song schedule, the lane chains and their filters, and the master: each lane's gain for the intensity, and the ceiling.
- `compose`: seed → a `Score` and a `Form`, each `Stage` drawn from a stream of its own. Each stream's draw order is the sound.
- `style`: one record per style, holding every knob its songs are written, sounded and staged with.
- `synth`: voices, the lane recipe both chains read, and the whole-lane chain behind `render_lane`.
- `dsp`: the public primitives, dependency-free on purpose.
- `score`: test-only frozen takes that the fingerprint pins guard.
- `audition`: lofi's own tools' access to the parts; no contract.

## Invariants no single item owns

- **The score is bit-identical everywhere; rendered audio is not.** `+ - * /` are exact ([RFC 3514](https://rust-lang.github.io/rfcs/3514-float-semantics.html)) but `sin`/`exp` [are not](https://doc.rust-lang.org/std/primitive.f32.html#method.sin), so tests pin fingerprints, never sample values.
- **Output is a pure function of the seed and the `(position, Controls)` timeline**: no `render` chunking changes it, and song k's noise comes from `(seed, k)`, never from render history (the chunk-invariance and no-history tests).
- **The public API is what `src/lib.rs` exports, `#[doc(hidden)]` items aside** ([they promise nothing](https://predr.ag/blog/checking-semver-for-doc-hidden-items/)): every other `pub` is a semver promise. [`api/pixtuoid-lofi.txt`](api/pixtuoid-lofi.txt) lists it, and `just api` rewrites it.
- **No audio file is ever committed** (`.gitignore`, CI): the references are copyrighted.

## Workflow

- [`ci.yml`](.github/workflows/ci.yml) is the gate, and runs the [`justfile`](justfile)'s recipes. `mise exec -- just setup`, once per clone, installs the tools and turns on the pre-push hook, which runs `just preflight` first.
- [`lofi_bench`](examples/lofi_bench.rs) gates what CI can't afford: `Performance::render`'s cost, and a run of each room playing finite at a steady level. It is [REVIEW.md](REVIEW.md#must-run)'s must-run item.
- A session pushes its own PR branches and merges its own PR once the ruleset's checks pass, the org's [shared review](https://github.com/pixtuoid/.github/blob/main/.github/workflows/claude-review.yml) among them, and every review thread is resolved. A blocking finding it refutes is dispositioned and resolved in its thread before the review re-runs (`gh run rerun <id> --failed`). It never pushes to `main`, and files an issue for work no issue holds, in the repo whose code it changes and on the Roadmap, never a second one for work an issue holds.
- A commit's title is its changelog line. In 0.x, `feat`/`fix` bump the patch and `!` bumps the minor ([release-plz](https://release-plz.dev/docs/config)). A changed `dsp` output takes `!`: pixtuoid's foley hears it, and no semver check does. A new `Controls` field doesn't ([`#[non_exhaustive]`](https://doc.rust-lang.org/cargo/reference/semver.html#struct-add-public-field-when-no-private)), as long as its default keeps the old sound.
- Releasing is the owner's: a hand-dispatched release PR, then the owner-approved `release` environment ([`release-plz.yml`](.github/workflows/release-plz.yml)).

## Conventions

- Reuse never adds a dependency: the crate has none, on purpose.
- Self-evident `0`/`1`/`2`, indices and test fixtures aren't magic numbers.
- A design change first looks up the field's standard design for its seam.
