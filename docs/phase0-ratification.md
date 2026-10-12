# Ambient sound Phase 0 — audition prototype (IvanWng97/pixtuoid#633)

The ratified versions and measured targets the synth constants in `src/synth.rs` cite; the foley sounds it records (typing, rain, chime, printer) stayed in pixtuoid.

Owner-ratified 2026-07-16. Tracks issue IvanWng97/pixtuoid#633 (locked decisions in the issue body;
Phase 0 results in the comments). Everything here is deterministic numpy — run:
`python3 .claude/skills/procedural-lofi/scripts/synth_audition.py` → `audio-demos/*.wav`.

## Ratified sounds (taste anchor: demo_2_moderate)

- **demo_2_moderate** = THE anchor ("挺好听的"): lofi band (pad 0.7 / keys 0.6 /
  drums 0.35 / texture 0.30) + typing 0.5. Empty subtracts, busy adds, rain overlays.
- **typing v4** ("这个可以了"): BRIGHT office clack matched to yt 2BUNHd7ENZk —
  click 1250-3700Hz decay~330 + body 1-2k + >4k spice; strokes ~12.9/s, inter-key
  66-96ms jitter; converged to ±8pp of the reference bands. (v3 thock — centroid
  474Hz per ASMR-community lore — measured OPPOSITE to the owner's reference:
  **the reference outranks the literature**.)
- **rain v9 — RATIFIED ("这个可以了 雨声定稿")**: the reference changed mid-flight —
  RainyMood's 70%-below-250Hz wash was the WRONG CHARACTER ("比较是gentle rain");
  the owner's chosen ref is yt 42M3esYyHdw (Gentle Night Rain, LIVE stream —
  capture via `yt-dlp -g` + `ffmpeg -t 180`). Gentle envelope: energy 500Hz-2k
  (~50%), rumble <12%, real 2-8k air; converged ±0.7pp. DROP LAYER is the soul
  ("有雨声落下的声音"): audible foreground drops 12-14dB ABOVE the bed (temporal
  fingerprint via analyze_drops.py — spectral averages can't see drops), ~0.5/s,
  three surface populations (dull plop 20% / water plip 55% / bright ping 25%),
  35% fast pairs; stereo standalone (decorrelated L/R beds + per-drop pan), 60s
  render kills loop deja-vu. PRODUCT: engine loops only the BED, scatters drops
  at runtime → never repeats.
- **noise floor**: ONE texture stem carries the medium noise (per-stem hiss
  stacks ≈ +6dB); texture sits 25-35dB below the music (fixed from -17dB).

### Revisit round (all units auditioned/rebuilt)

- printer_whir v2: structured laser-printer (motor 80→130Hz spin-up + 11/s
  roller ticks + paper swoosh + spin-down) — v1 was an unstructured noise blob.
- door_chime v2: DESCENDING ding-dong E5→C5, warm/slow (centroid 881→556Hz) —
  v1 nearly collided with the elevator ding (872Hz); one-shots must be
  spectrally DISTINCT per cue role. Loudness matched across one-shots (±2.2dB).
- cooler_glug: owner-confirmed KEEP (the pantry water cooler's bubble animation).
- demo_1/elevator/glug: provisionally OK — final verdict lands in Phase 2's
  in-engine dogfood.

References used (re-downloadable, NOT stored): media.rainymood.com/0.m4a
(superseded); youtube 2BUNHd7ENZk (typing); youtube 42M3esYyHdw (gentle rain,
live stream). Live streams: yt-dlp direct download never ends — `yt-dlp -g`
for the HLS URL then `ffmpeg -t <s>`. Physics notes: Minnaert bubble resonance
(glug), inharmonic chime-bar modes 1:2.76:5.40 (ding), tape wow/flutter+head-bump
(lofi chain) — citations in IvanWng97/pixtuoid#633.
