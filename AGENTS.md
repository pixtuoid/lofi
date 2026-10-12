# AGENTS.md

Instructions for any AI coding agent in this repo.

## What this is

Procedural lofi: `compose(mood, seed)` yields a `Score`, `render_lane` synthesizes one `Lane` of it to mono `f32` at `dsp::SAMPLE_RATE`. Modules:

- `dsp` — public primitives: FFT-backed filters, shaped noise, `NoiseStream`, analysis helpers.
- `compose` — seed → `Score` (progression, melody, groove); the rng draw order lives here.
- `score` — the frozen, owner-ratified takes (`#[cfg(test)]` fingerprint anchors).
- `synth` — voices, the lofi master chain, `render_lane`.

## Invariants

- **The rng draw order is the sound.** The voice draw stays the stream's tail so a new voice never recomposes a blessed seed (`src/compose.rs`, the comment at the end of `compose`).
- **Dependency-free on purpose**: a wasm consumer ships `dsp`, and an FFT crate is a download its visitors pay (`src/dsp.rs` header).
- **Playback never synthesizes**: rendering is a once-per-track pre-render (`src/synth.rs` header).
- **Frozen takes keep their darker chain**, and `lane_recipe` (`src/synth.rs`) is the one authority the fingerprint pins guard; brightening a frozen chain invalidates them.
- **A regenerated score is a new take** and needs the owner's LISTEN gate (`src/score.rs` header; ratified targets in `docs/phase0-ratification.md`).
- **Same synthesis, not the same bits, across platforms**: `sin`/`exp` differ, so tests pin fingerprints, never sample values.
- **A musical lane's length is an exact integer multiple of the loop's sample count**, so stems loop in phase (`musical_stems_share_one_loop_length`, `src/synth.rs`).
- **A `dsp` output change ships as a minor bump with a changelog line**: pixtuoid's foley builds on these primitives.

## Never commit audio

`.gitignore` blocks audio extensions and CI guards it; the reference recordings are copyrighted and rendered takes are local auditions.

## Dev loop

Follow the `procedural-lofi` skill (`.claude/skills/procedural-lofi/SKILL.md`). `cargo test --all-targets` gates; `cargo run --release --example lofi_audition` is for ears.

## Release

release-plz on `main`: a release PR, then publish on merge via crates.io trusted publishing. A music-only change is a patch; a new `Mood`/`Lane` or any API break is a minor (0.x), and pixtuoid follows by hand.

## Rules

- Never `--no-verify`; never force-push or push to `main` without the owner.
- Comments: WHY only.
