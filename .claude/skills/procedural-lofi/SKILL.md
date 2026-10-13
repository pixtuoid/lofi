---
name: procedural-lofi
version: 1.0.0
description: "Generate a royalty-free lofi (or rain / typing / chime / any ambient) soundtrack ENTIRELY in code — no sampled audio ships. Fingerprint a beloved reference recording, shape synthesis to the measured spectral + temporal curve, freeze one human-blessed take into constant tables, and re-synthesize at launch (loop the bed, scatter the foreground → never repeats, ~0 KB, no licensing risk). Use when adding ambient/generative audio to an app, game, terminal UI, or site, or on 'add another lofi/rain/ambient sound'. Bundles the parameter tables (LOFI-BIBLE.md) + the numpy fingerprint/synth/freeze pipeline."
---

# Procedural Lofi — build a soundtrack in code, not in a sample pack

This skill is the end-to-end recipe for a lofi bed (and its sibling ambient sounds:
rain, keystrokes, door chimes, printer, water cooler…) that is **synthesized at
runtime from constants** — zero audio files, zero royalties, and no two minutes ever
sound the same.

Two documents ship alongside this one:

- **`reference/LOFI-BIBLE.md`** — the parameter tables. Harmony (chord grammar,
  voice-leading, register clamps), groove (swing %, drag ms, velocity curves), per-voice
  sound design (pad / bass / EP keys / sparkle / drums / texture / tape chain), mix
  targets (band shares, HPF strategy, loudness), and generative lessons from prior art.
  Every number is cited and, where it conflicts with a measurement, the measurement wins.
- **`scripts/`** — the Python (numpy) pipeline: `analyze_*.py` (fingerprint a reference),
  `synth_audition.py` (a runnable numpy synth that produces `.wav` auditions), and
  `export_score.py` (freeze a good take into a constants table for the port).

Read those two when you reach the step that needs them. This file is the map.

---

## The core idea (why code, not samples)

A sampled lofi loop is: (a) a licensing liability, (b) heavy to bundle, and (c) audibly
repetitive — the human brain catches a loop seam on the 3rd or 4th pass. Synthesizing it
solves all three:

- **Legal**: *acoustic parameters are not copyrightable.* You measure a reference to get
  target numbers (this band has 55% of the power, the centroid sits at 150 Hz, drops land
  13 dB above the bed). You never keep or ship the reference bytes. What ships is your own
  oscillator code hitting those numbers.
- **Size**: a few kilobytes of constant tables + a synth function vs. megabytes of PCM.
- **Never repeats (in practice)**, from stacked tricks: (1) every seed composes a new song
  (`compose`), and within a song the musical lanes loop in lockstep on a loop made **long
  enough** not to fatigue (the day loop doubled from 4 to 8 bars because a short looped
  melody *was* audibly repetitive — "loop the bed" does not mean "make it short"); (2) a
  consumer scatters genuinely-stochastic layers fresh at runtime (pixtuoid's rain drops and
  keystrokes); (3) a consumer's mixer fades whole lanes in and out (pixtuoid's by how busy
  its office is), so the arrangement keeps changing even while the notes don't.

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

For every distinct sound (the lofi bed, rain, typing, each one-shot):

1. **Get a beloved reference.** An owner-supplied reference beats a "community ideal"
   every time — build to what *they* love, not to what a forum says lofi should be.
   (`yt-dlp` / `curl` for analysis only; for an endless live stream, `yt-dlp -g` gets the
   HLS URL, then `ffmpeg -t 180` grabs a finite slice.)
2. **Confirm the CHARACTER before deep-matching.** One sentence — "gentle rain or heavy
   downpour?" — saves three wasted versions. On pixtuoid, three rain versions were
   built to the wrong reference (a heavy wash) before the owner clarified they wanted
   *gentle* rain.
3. **Fingerprint it.** Measure **9 octave-band energies + spectral centroid + rolloff**
   (`analyze_rain.py` / `analyze_typing.py`). For anything *rhythmic or event-bearing*
   ALSO measure the **temporal** fingerprint — onset rate, inter-onset-interval spread,
   per-stroke decay, and the dB level of foreground events vs. the bed
   (`analyze_drops.py`). Spectral averages are blind to events: rain's audible *drops*
   don't show up in an averaged spectrum at all, only in the temporal pass.

### Phase 2 — Shape synthesis to the measured curve

Build your oscillators/noise-shapers and drive their parameters until a re-measurement of
*your* output lands within a few percentage points of the reference fingerprint. The
`LOFI-BIBLE.md` gives you the starting parameter values per voice; the fingerprint tells
you which way to push them. Search the literature for the *physics* (Minnaert resonance
for a water glug, inharmonic bar modes 1 : 2.76 : 5.40 for a chime, tape wow/flutter +
head-bump for the lofi chain) but *tune to the reference*, not to the physics ideal.

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

- **The port must be byte-identical to the audition.** Keep the ratified fingerprint/
  checksum tests; passing them verbatim after the port is your oracle that nothing shifted.
- **No wall-clock reads inside the audio math.** Pass `dt` / `now` in as parameters. On
  wasm especially, `SystemTime::now()` isn't available — and a backgrounded tab that jumps
  the clock will otherwise ramp-snap your crossfades and burst-replay every queued event
  (the "stall-clock" bug). Clamp big `dt` gaps.
- **Loop the frozen composition, scatter only the stochastic layers.** All the musical
  stems (pad/drums/keys/sparkle) tile in lockstep on one loop length — make that loop long
  enough to not fatigue; then scatter the truly-random layers (rain drops, keystrokes) fresh
  each frame and fade whole stems by busy-ness (see "Never repeats" above). Tiny RAM: one
  copy of each stem loop.
- **Keep the sub band clear.** The sub-bass register must belong to the bass alone or the
  low end turns to mud. Two ways to get there: the textbook one is to high-pass every
  non-bass stem ~140 Hz (LOFI-BIBLE §4); the cheaper one this project used is to *voice the
  other stems out of that register in the first place* (mid-register EP plucks; the sub
  register belongs to the bass lane alone) so there's nothing to filter. Either way, keep ONE
  texture stem carrying all the "medium" noise — per-stem hiss *stacks* (every musical stem
  adding a little sums to real mud); texture sits 25–35 dB below the music.
- **Perceptual volume.** Map a user volume slider as `amplitude = user²` (loudness is
  logarithmic). Linear volume feels "still too loud at 5%" — the classic trap.

---

## The failure catalog (bugs this method hit, so you don't)

- **Reference outranks literature.** A "thock" keyboard sound built to the ASMR-community
  ideal measured the *opposite* of the owner's clacky reference. Search for physics; tune
  to the reference.
- **Phantom-octave measurement bug.** Fingerprints extracted from the *listening artifacts*
  (stereo wav, soft-clipped) read every frequency halved (interleaved stereo read as mono =
  2× time-stretch). **Pin spectral references on the raw float synthesis chain, never on the
  exported/played wav.**
- **Spectral averages can't see events.** Rain drops, a snare ghost, a keystroke accent —
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
3. Pick and *character-confirm* a reference. Fingerprint it with the matching `analyze_*.py`.
4. Adapt `synth_audition.py`'s voices toward the fingerprint; re-measure to convergence.
5. Human listens once. On yes, `export_score.py` freezes the take + checksum.
6. Port to your runtime; keep the fingerprint tests as the byte-identity oracle; synthesize
   at launch, loop the frozen composition, scatter the stochastic layers, gate stems by
   busy-ness.

(The `scripts/` are the real working prototypes from pixtuoid's Phase 0, kept as a concrete
starting point — not a turnkey CLI; `synth_audition.py` runs standalone, and `export_score.py`
imports `phase2_audition.py` — both are bundled. Adapt the voices to your own reference.)

---

## The GENERATOR loop (this crate's composer — iterate in minutes, not design cycles)

Once a project graduates from frozen takes to a theory-constrained generator
(`src/compose.rs` + `synth::render_lane` in `src/synth.rs`), the iteration loop changes
shape: the gate becomes **statistical** (`ab_judge.py` judges a batch of seeds, step 5;
PASS = the GENERATOR ships — REGRESSION = tighten a constraint and re-batch), and
every taste axis is a data table or named const with ONE home:

| Want to change…            | Edit (all in `compose.rs` unless noted)                  |
|----------------------------|----------------------------------------------------------|
| harmony vocabulary         | `DAY_PROGRESSIONS` / `NIGHT_PROGRESSIONS` (pre-voiced templates + `roots_pc` + `scale_pcs`) |
| tempo feel                 | `DAY_BPM` / `NIGHT_BPM` windows                          |
| melody character           | `lead_events` rules (density draws, leap bound, peak bar, grid) + `*_LEAD_LO/HI` registers |
| groove feel                | `DAY_GROOVES` templates / `night_drums` pattern + the swing/drag consts |
| comping density            | `keys_events` density draws                              |
| room-tone crackle          | `synth::CRACKLE_POPS_PER_SEC` (one knob, both beds)      |
| **a new instrument**       | see the checklist below                                  |

**Add-an-instrument checklist** (executed once for the `Pluck` lead — repeat verbatim):
1. Write the voice as a pure note fn in `synth.rs`, a `NoteVoice` rendered at the rate it
   is given, physics-first then tuned by ear/fingerprint (Phase 1-2 of this skill still apply).
2. Add a `compose::LeadVoice` variant + its arm in `synth::lead_voice_fn` (the ONE dispatch).
3. Give it a draw weight at the END of `compose()` — the voice draw is deliberately LAST in
   the seed stream so new voices never redraw an already-blessed seed's notes (one banded
   unit-draw per mood — re-weight the bands rather than adding draws).
4. Extend the distribution property test; run the compose suite (fast — no synthesis).
5. Render a batch (`cargo run --release --example lofi_audition -- --seeds N`;
   `--engine v1 --solo sparkle` isolates the lane, rendered whole) → `ab_judge.py` gates it →
   tighten or ship. A change to a shipped sound first diffs the tier gains moved here
   against pixtuoid's (no output: in step),
   ```bash
   diff <(gh api repos/pixtuoid/pixtuoid/contents/crates/pixtuoid-scene/src/audio/mod.rs --jq .content | base64 -d |
            sed -nE 's/^const (PAD|SPARKLE|KEYS|DRUMS|TEXTURE|BASS)_GAIN: TierGain = TierGain::new\((.*)\);$/\1 \2/p' | tr '[:upper:]' '[:lower:]') \
        <(sed -nE 's/^ *Lane::([A-Za-z]+) => (\[.*\]),$/\1 \2/p' src/stream/master.rs | tr '[:upper:]' '[:lower:]')
   ```
   then renders the batch on `main` into `DIR/base` and on the branch into `DIR/cand`, DIR
   durable and outside the repo (`~/Desktop/lofi-listen/<change>`), each from a clean tree
   with its `git rev-parse HEAD` in `DIR/{base,cand}/COMMIT`, and `ab_judge.py` gates them
   (its header holds the verdict rule and each measure's blind spots) against REFS, the
   private reference corpus (`audio/` and its `manifest.json` of sha256s, never committed),
   from a venv outside the repo, set up once:
   ```bash
   # Python 3.12: fadtk 1.1.0 pins numpy<2
   # (https://github.com/microsoft/fadtk/blob/6f815282007288a7165d037a95dacd6d783b6f32/pyproject.toml#L9),
   # and numpy's last 1.x, 1.26.4, has wheels up to cp312 (https://pypi.org/pypi/numpy/1.26.4/json)
   J=~/.cache/lofi-judge; uv venv --python 3.12 $J
   # transformers 4: 5's conformer reads `config._attn_implementation`
   # (https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/masking_utils.py#L822),
   # which MuQ's EasyDict config lacks
   # (https://github.com/tencent-ailab/MuQ/blob/28847ea50cd31ac4b8b6a7dacc051ad7d1c7606a/src/muq/muq/models/muq_model.py#L134)
   uv pip install --python $J/bin/python audiobox_aesthetics==0.0.4 torch==2.14.1 \
     torchaudio==2.11.0 numpy==1.26.4 scipy==1.17.1 soundfile==0.14.0 requests==2.34.2 \
     huggingface_hub==0.36.2 muq==0.1.0 peft==0.21.2 omegaconf==2.4.0 transformers==4.57.6 \
     fadtk==1.1.0 pyloudnorm==0.2.0
   git clone https://github.com/dgtql/MuQ-Eval $J/MuQ-Eval
   git -C $J/MuQ-Eval checkout 60a88f8ac0909ca1fd1a78af3660f1fc376977a1
   $J/bin/python .claude/skills/procedural-lofi/scripts/ab_judge.py DIR --refs REFS
   ```
   The owner's blind listen is optional and, when given, overrides the judge:
   `python3 .claude/skills/procedural-lofi/scripts/ab_listen.py DIR/base DIR/cand --out DIR/verdict.json`.
   A change of engine renders both on one commit, `--engine v1 --out DIR/base` and
   `--engine v2 --out DIR/cand` with the same `--mood`, `--tier` and seeds: each take keeps
   its name and its score under either engine.

Mix LANES stay instrument-blind (a consumer's mixer sees lanes, never voices) — a
lane is a busy-ness ROLE, an instrument is a timbre WITHIN it. That split is what keeps
"blend in another instrument" a minutes-scale edit instead of a design cycle.
