---
name: procedural-lofi
version: 1.0.0
description: "Generate a lofi soundtrack ENTIRELY in code — no sampled audio ships. Fingerprint a beloved reference recording, shape synthesis to the measured spectral + temporal curve, freeze one human-blessed take into constant tables, then grow a seeded generator judged by A/B batches (~0 KB, no sample to license). Use when adding generative lofi to an app, game, terminal UI, or site, or on 'add another lofi voice'. Bundles the parameter tables (LOFI-BIBLE.md) + the numpy synth/freeze pipeline."
---

# Procedural Lofi — build a soundtrack in code, not in a sample pack

This skill is the end-to-end recipe for a lofi soundtrack that is **synthesized at
runtime from constants** — no audio file ships, and no two minutes sound the same.

**`reference/LOFI-BIBLE.md`** holds the parameter tables: harmony (chord grammar,
voice-leading, register clamps), groove (swing %, drag ms, velocity curves), per-voice
sound design (pad / bass / EP keys / sparkle / drums / texture / tape chain), mix targets
(band shares, HPF strategy, loudness), and generative lessons from prior art. Every number
is cited and, where it conflicts with a measurement, the measurement wins. Read it when you
reach the step that needs it. This file is the map.

---

## The core idea (why code, not samples)

A sampled lofi loop is: (a) a licensing liability, (b) heavy to bundle, and (c) audibly
repetitive — the brain catches a loop seam on its 3rd or 4th pass (LOFI-BIBLE's
anti-repetition row cites the measurements). Synthesizing it solves all three:

- **Legal**: you measure a reference to get target numbers (this band has 55% of the power,
  the centroid sits at 150 Hz) and never keep or ship its bytes; what ships is your own
  oscillator code hitting those numbers. US copyright doesn't extend to "any idea,
  procedure, process, system, method of operation, concept, principle, or discovery" (17
  U.S.C. § 102(b)); that this covers measured numbers is opinion, not legal advice.
- **Size**: a few kilobytes of constant tables + a synth function vs. megabytes of PCM.
- **Never repeats (in practice)**, from stacked tricks: (1) every seed composes a new song
  (`compose`), and within a song the musical lanes loop in lockstep on a loop made **long
  enough** not to fatigue (the day loop doubled from 4 to 8 bars because a short looped
  melody *was* audibly repetitive — "loop the bed" does not mean "make it short"); (2)
  each song has a form (LOFI-BIBLE R11) that mutes and restores lanes over
  its loop, and a `Performance` layers each lane's gain by `Controls::intensity` on top, which
  a consumer sets, e.g. by how busy an office is.

The one law of the whole method:

> **Measurement is the machine's ears; taste is the human's.** You (or your tooling)
> cannot reliably *hear* whether it sounds good — you can only measure proxies (band
> energies, onset rate, dB deltas) and drive them to a target. A human listens **once per
> major revision** and gives a yes/no. Pick your measurable proxy carefully, iterate on it
> alone, and hand over a finished audition — don't ask the human to babysit each tweak.

---

## The pipeline (five phases)

### Phase 0 — Audition in a scripting language first (numpy), NOT in your ship language

Prototype the whole sound in Python/numpy where the write→hear loop is seconds. Only port
to Rust/C++/wasm once a human has ratified the *sound*. `scripts/synth_audition.py` is a
worked example: deterministic numpy that writes `audio-demos/*.wav` beside the script. Iterating synth
recipes in a compiled language first is the classic time sink.

### Phase 1 — Reference-fingerprint each sound

For the bed and each voice in it:

1. **Get a beloved reference.** An owner-supplied reference beats a "community ideal"
   every time — build to what *they* love, not to what a forum says lofi should be.
   (`yt-dlp` / `curl` for analysis only; for an endless live stream, `yt-dlp -g` gets the
   HLS URL, then `ffmpeg -t 180` grabs a finite slice.)
2. **Confirm the CHARACTER before deep-matching.** One sentence — "a sleepy night or a
   bright study session?" — saves wasted versions built to the wrong reference.
3. **Fingerprint it.** Measure **9 octave-band energies + spectral centroid + rolloff**.
   For anything *rhythmic or event-bearing* ALSO measure the **temporal** fingerprint —
   onset rate, inter-onset-interval spread, per-stroke decay, and the dB level of
   foreground events vs. the bed. Spectral averages are blind to events: a snare ghost
   doesn't show up in an averaged spectrum at all, only in the temporal pass.

### Phase 2 — Shape synthesis to the measured curve

Build your oscillators/noise-shapers and drive their parameters until a re-measurement of
*your* output lands within a few percentage points of the reference fingerprint. The
`LOFI-BIBLE.md` gives you the starting parameter values per voice; the fingerprint tells
you which way to push them. Search the literature for the *physics* (inharmonic cantilever
modes for a kalimba, tape wow/flutter + head-bump for the lofi chain) but *tune to the
reference*, not to the physics ideal.

### Phase 3 — Human LISTEN gate → freeze the realization

When the numbers converge, a human auditions once (`afplay file.wav` on macOS, or hand
them the wav). On yes:

> **The RNG was the composer. Freeze the one take they blessed.**

The generative script drew notes/velocities/timings from a seeded RNG. That *one seed's
output* is what got ratified — so capture its exact event stream into **constant tables**
(`export_score.py` → a `.rs` / `.h` table), plus a **full-table checksum**. Do NOT re-run
the RNG in production and hope; a later library bump silently redraws and you ship a
different, un-ratified take. **The freeze is the contract.** (Subtlety: your exporter must
reproduce the *exact draw order* of the original — argument-evaluation order, nested draws
inside a `choice()` — or the frozen table desyncs from what was auditioned.)

### Phase 4 — Port + ship (synthesize at launch)

Port the numpy synth to your runtime language reading the frozen tables. Key engineering:

- **The port must keep the audition's fingerprints.** Keep the ratified fingerprint and
  checksum tests; passing them after the port is your oracle that nothing shifted.
- **No wall-clock reads inside the audio math.** Pass `dt` / `now` in as parameters. On
  wasm especially, `SystemTime::now()` isn't available — and a backgrounded tab that jumps
  the clock will otherwise ramp-snap your crossfades and burst-replay every queued event
  (the "stall-clock" bug). Clamp big `dt` gaps.
- **Loop the frozen composition, scatter only the stochastic layers.** All the musical
  stems (pad/drums/keys/sparkle) tile in lockstep on one loop length — make that loop long
  enough to not fatigue, and let the song's form and the busy-ness mute and fade whole
  stems (see "Never repeats" above). Tiny RAM: one copy of each stem loop.
- **Keep the sub band clear.** The sub-bass register must belong to the bass alone or the
  low end turns to mud. Two ways to get there: the textbook one is to high-pass every
  non-bass stem ~140 Hz (LOFI-BIBLE §4); the cheaper one this project used is to *voice the
  other stems out of that register in the first place* (mid-register EP plucks; the sub
  register belongs to the bass lane alone) so there's nothing to filter. Either way, keep ONE
  texture stem carrying all the "medium" noise — per-stem hiss *stacks* (every musical stem
  adding a little sums to real mud); texture sits 25–35 dB below the music.

---

## The failure catalog (bugs this method hit, so you don't)

- **Reference outranks literature.** Search for physics; tune to the reference: the
  BIBLE's measured rows override its literature rows wherever they disagree.
- **Phantom-octave measurement bug.** Fingerprints extracted from the *listening artifacts*
  (stereo wav, soft-clipped) read every frequency halved (interleaved stereo read as mono =
  2× time-stretch). **Pin spectral references on the raw float synthesis chain, never on the
  exported/played wav.**
- **Spectral averages can't see events.** A snare ghost, a hat accent —
  measure them *temporally* (level vs. bed, onset rate) or you'll "match" a reference and
  still be missing its soul.
- **Head-only crossfade ≠ a seamless loop.** Fading the *start* of a loop back over its end
  leaves the wrap audible. A real seamless loop synthesizes a genuine *continuation* past the
  loop point and blends that in.
- **Two frozen copies of one truth desync silently.** If a kick-times table duplicates the
  drum table's kicks, add an equality test or a later edit to one drifts from the other with
  no error.
- **Control plane off the data plane.** A mute/volume press must not ride the same queue as
  audio frames — a synthesis burst can saturate the queue and eat the keypress, so the beds
  fade in *unmuted* while the UI says muted. Put mute/volume on an atomic flag the audio
  thread reads, separate from the frame channel.
- **Empty-room-uncanny.** A truly silent "idle" state feels broken, but a full bed under
  nothing feels wrong too. Keep the register (don't octave-drop), voice harmonically, sparse
  EP notes, and let the *texture* bed carry the quiet — don't drop to pure silence.

---

## Quickstart for a fresh project

1. Copy `reference/LOFI-BIBLE.md` and `scripts/` into your repo (or just read them).
2. Install: `python3 -m venv .venv && .venv/bin/pip install numpy scipy` (+ `yt-dlp`,
   `ffmpeg` for grabbing references).
3. Pick and *character-confirm* a reference, and fingerprint it (Phase 1).
4. Adapt `synth_audition.py`'s voices toward the fingerprint; re-measure to convergence.
5. Human listens once. On yes, `export_score.py` freezes the take + checksum.
6. Port to your runtime; keep the fingerprint tests as the oracle; synthesize at launch,
   loop the frozen composition, gate stems by busy-ness.

(The `scripts/` are the real working prototypes from pixtuoid's Phase 0, kept as a concrete
starting point — not a turnkey CLI; `synth_audition.py` runs standalone, and `export_score.py`
imports `phase2_audition.py` — both are bundled. Adapt the voices to your own reference.)

---

## The GENERATOR loop (this crate's composer — iterate in minutes, not design cycles)

Once a project graduates from frozen takes to a theory-constrained generator
(`compose`'s scores, which a `Performance` plays through `synth`'s voices and `stream`'s
chains), the iteration loop changes shape: the gate becomes **statistical** (`ab_judge.py`
judges a batch of seeds, step 5; PASS = the GENERATOR ships — REGRESSION = tighten a
constraint and re-batch), and every taste axis is a data table or named const with ONE home:

| Want to change…            | Edit (all in `compose.rs` unless noted)                  |
|----------------------------|----------------------------------------------------------|
| harmony vocabulary         | `DAY_PROGRESSIONS` / `NIGHT_PROGRESSIONS` (pre-voiced templates + `roots_pc` + `scale_pcs`) |
| tempo feel                 | `DAY_BPM` / `NIGHT_BPM` windows                          |
| melody character           | `lead_events` rules (density draws, leap bound, peak bar, grid) + `*_LEAD_LO/HI` registers |
| groove feel                | `DAY_GROOVES` templates / `night_drums` pattern + the swing/drag consts |
| comping density            | `keys_events` density draws                              |
| room-tone crackle          | `synth::CRACKLE_POPS_PER_SEC` (one knob, both beds)      |
| lane gains, mix level      | `stream::master`: `tier_gains`; `Performance::MASTER_GAIN` |
| staging, bed crossfade     | `stream::lanes`: `staging`, `BED_CROSSFADE_S`            |
| song form (intro, sections, outro, edges) | `compose::form`'s consts; edges' depth and corner: `stream::schedule::FADE_FLOOR_DB`, `stream::lanes::FILTERED_HZ` |
| **a new instrument**       | see the checklist below                                  |

**Add-an-instrument checklist** (executed once for the `Pluck` lead — repeat verbatim):
1. Write the voice as a pure note fn in `synth.rs`, a `NoteVoice` rendered at the rate it
   is given, physics-first then tuned by ear/fingerprint (Phase 1-2 of this skill still apply).
2. Add a `compose::LeadVoice` variant + its arm in `LeadVoice::voice` (the ONE dispatch).
3. Give it a draw weight at the END of `compose()` — the voice draw is deliberately LAST in
   the seed stream so new voices never redraw an already-blessed seed's notes (one banded
   unit-draw per mood — re-weight the bands rather than adding draws).
4. Extend the distribution property test; run the compose suite (fast — no synthesis).
5. Render the batch the judge scores (`cargo run --release --example lofi_audition -- --batch
   judge`; `--engine whole --solo sparkle` isolates a lane) → `ab_judge.py` gates it → tighten
   or ship. A change to a shipped sound renders it at the merge-base (`git merge-base origin/main HEAD`) into `DIR/base` and at
   the branch into `DIR/cand`, DIR durable and outside the repo (`~/Desktop/lofi-listen/<change>`),
   each from a clean tree with its `git rev-parse HEAD` in `DIR/{base,cand}/COMMIT`, and
   `ab_judge.py` gates them (its header holds the verdict rule and each measure's blind spots)
   against REFS, the private reference corpus (`audio/` and its `manifest.json` of sha256s, never
   committed), from a venv outside the repo, set up once; CI's `judge.yml` renders the same batch and
   judges it on Modal (`judge/modal_judge.py`) for every PR that changes the render:
   ```bash
   J=~/.cache/lofi-judge; S=.claude/skills/procedural-lofi; uv venv --python 3.12 $J
   uv pip install --python $J/bin/python --require-hashes -r $S/judge/judge-requirements.txt
   git clone https://github.com/dgtql/MuQ-Eval $J/MuQ-Eval
   git -C $J/MuQ-Eval checkout --detach "$(sed -nE 's/^MUQ_EVAL_COMMIT = "(.*)"$/\1/p' $S/scripts/ab_judge.py)"
   $J/bin/python $S/scripts/ab_judge.py DIR --refs REFS
   ```
   The owner's blind listen is optional and, when given, overrides the judge:
   `python3 .claude/skills/procedural-lofi/scripts/ab_listen.py DIR/base DIR/cand --out DIR/verdict.json`.

Mix LANES stay instrument-blind (`stream::master::lane_gain` sees lanes, never voices; a
consumer sets `Controls` and receives `Frame`s) — a lane is a busy-ness ROLE, an instrument a
timbre WITHIN it, which keeps "blend in another instrument" a minutes-scale edit, not a design cycle.
