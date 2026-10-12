# AGENTS.md

Instructions for any AI coding agent in this repo.

## The goal

Lofi that sounds better without leaving the genre. The character is measured, not described: [`LOFI-BIBLE.md`](.claude/skills/procedural-lofi/reference/LOFI-BIBLE.md)'s groove, sound and mix targets, fingerprinted from the reference recordings, so a change that sounds better but leaves them is not one. The first listener is [pixtuoid](https://github.com/IvanWng97/pixtuoid)'s office, which loops these lanes for as long as its agents work, so a take must not fatigue on repeat (the `procedural-lofi` skill, "Never repeats").

## Who judges "good"

- **Measurement is the machine's ears; taste is the owner's.** Band shares, centroids and onset rates drive a change toward a target; only the owner's LISTEN says it sounds good. Hand over a finished audition, never each tweak (the `procedural-lofi` skill, "The core idea").
- **A music change ends in an audition batch**: `cargo run --release --example lofi_audition -- --seeds N` renders random seeds through the real `render_lane` chain, and the owner blind-listens the composer, not one take (the example's header). One dud means tightening a constraint and re-batching (the skill's "GENERATOR loop").
- **Where each taste axis lives**: that section's table; a new instrument follows its checklist.
- **The ratified anchors** are in [`docs/phase0-ratification.md`](docs/phase0-ratification.md). A synth constant changes there first, and where a measurement and the literature disagree, the measurement wins (`LOFI-BIBLE.md`'s header).

## What this is

`compose(mood, seed)` yields a `Score`, and `render_lane` synthesizes one `Lane` of it to mono `f32` at `dsp::SAMPLE_RATE`. Modules:

- `dsp` — public primitives: FFT-backed filters, shaped noise, `NoiseStream`, analysis helpers.
- `compose` — seed → `Score` (progression, melody, groove); the rng draw order lives here.
- `score` — the frozen, owner-ratified takes (`#[cfg(test)]` fingerprint anchors).
- `synth` — voices, the lofi master chain, `render_lane`.

## Invariants

- **The rng draw order is the sound.** The voice draw stays the stream's tail so a new voice never recomposes a blessed seed (`src/compose.rs`, the comment at the end of `compose`).
- **Dependency-free on purpose**: a wasm consumer ships `dsp`, and an FFT crate is a download its visitors pay (`src/dsp.rs` header).
- **Playback never synthesizes**: rendering is a once-per-track pre-render (`src/synth.rs` header).
- **Frozen takes keep their darker chain**, and `lane_recipe` (`src/synth.rs`) is the one authority the fingerprint pins guard; brightening a frozen chain invalidates them.
- **A regenerated score is a new take** and needs the owner's LISTEN gate (`src/score.rs` header).
- **The score is bit-identical everywhere; rendered audio is not**: the composer uses no transcendental function (`+ - * /` are exact, [RFC 3514](https://rust-lang.github.io/rfcs/3514-float-semantics.html)), but `sin`/`exp` precision is [non-deterministic across platforms and Rust versions](https://doc.rust-lang.org/std/primitive.f32.html#method.sin), so tests pin fingerprints, never sample values.
- **A musical lane's length is an exact integer multiple of the loop's sample count**, so stems loop in phase (`musical_stems_share_one_loop_length`, `src/synth.rs`). A lane may span several loops, which is how variation and arrangement are baked in, but always a whole number of them.
- **The public API is what `src/lib.rs` exports**; everything else stays private or `pub(crate)`, because every `pub` item is a semver promise.

## Never commit audio

`.gitignore` blocks audio extensions and CI guards it; the reference recordings are copyrighted and rendered takes are local auditions.

## Release

- The release PR is dispatched by hand (Actions → release-plz → Run workflow), and merging it publishes through the owner-approved `release` environment by crates.io trusted publishing; nothing else releases ([`release-plz.yml`](.github/workflows/release-plz.yml) header, [`release-plz.toml`](release-plz.toml)).
- Versions follow release-plz's 0.x defaults ([config](https://release-plz.dev/docs/config), `features_always_increment_minor`): `feat`/`fix` bump the patch, a breaking commit (`!`) or an API break its semver check finds bumps the minor. A music-only change is a patch, so a consumer's `0.y` requirement takes it; a new `Mood`/`Lane` or an API break is a minor, which a consumer bumps by hand.
- **A `dsp` output change is breaking**: pixtuoid's office foley builds on these primitives, and no semver check sees a changed buffer, so its commit carries `!`.
- A commit's title is its changelog line: release-plz writes `CHANGELOG.md` from them.

## Conventions

- **TDD first**: failing test → minimal implementation → commit. A sound's test asserts a fingerprint (the invariant above).
- **Every mechanism earns its upkeep.** Code, config, a test, a CI job or a doc stays only if it names a failure it prevents that nothing else already does, or a need someone has.
- **Reuse before you build**, except a dependency: the crate stays dependency-free (above), so an algorithm it needs is written here.
- **Comments: WHY only**, and every comment the repo ships is in scope: doc prose, workflow and manifest comments, PR bodies. The first sentence is the whole answer, and every later one must add information. A comment sits on the narrowest thing it constrains. Name the authority, never restate its value (`` [`SAMPLE_RATE`] ``, not `44100`). Measurements belong in commit messages.
- **No magic numbers**: one named `const` at the narrowest covering scope, or a type for a related set. Two copies of one value are a drift bug; pin a pair that must cross a boundary with a test.
- **No `unwrap()` outside tests.**
- **External claims are fetched, not remembered**, and so is "best practice": cite the doc URL or `path:line` fetched this session, else label it opinion.
- **A refuted review finding produces a mechanism** (a test, a compile-time constraint, a CI gate) **or nothing**, never prose.
- **Docs stay current in the same commit** as any structure, API or workflow change.
- A session pushes its own PR branches; it never merges, force-pushes or pushes to `main` without the owner, never skips hooks (`--no-verify`), and never files issues.
