# Ambient sound Phase 0 — audition prototype (pixtuoid/pixtuoid#633)

The Phase 0 record: the taste anchor and the one-texture-stem noise floor the lanes keep; the foley it ratifies (typing, rain, chime, printer) stayed in pixtuoid. The music takes' fingerprints are pinned in `src/synth/frozen.rs`'s tests.

Owner-ratified 2026-07-16. Tracks issue pixtuoid/pixtuoid#633 (locked decisions in the issue body;
Phase 0 results in the comments). Everything here is deterministic numpy — run:
`python3 .claude/skills/procedural-lofi/scripts/synth_audition.py` → `audio-demos/*.wav` beside the script.

## Ratified sounds (taste anchor: demo_2_moderate)

- **demo_2_moderate** = THE anchor ("挺好听的"): lofi band (pad 0.7 / keys 0.6 /
  drums 0.35 / texture 0.30) + typing 0.5. Empty subtracts, busy adds, rain overlays.
- **noise floor**: ONE texture stem carries the medium noise (per-stem hiss
  stacks ≈ +6dB); texture sits 25-35dB below the music (fixed from -17dB).
