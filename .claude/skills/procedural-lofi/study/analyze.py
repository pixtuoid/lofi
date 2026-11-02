"""What a Lofi Girl track does, from its stems: the drums' grid and sound, the
bass, the harmony, the top line. A take of our own goes through the same code,
so its known score is this measure's control.

  modal run analyze.py::run --out DIR --prefix take-NAME_
  modal run analyze.py::run --out DIR --hints JSON      the references

Each track's statistics, its drum grid and chord counts among them, come home
in DIR's study*.json, which stays private: only their quartiles are committed,
in targets.json. Its notes are discarded.
"""

import json
import sys
from collections import Counter
from pathlib import Path

import modal

STUDY = "/study"
CONCURRENT = 24
TIMEOUT_S = 1200

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install("torch==2.5.1", "torchaudio==2.5.1", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("basic-pitch==0.4.0", "soundfile==0.12.1", "numpy<2", "beat-this==1.1.0", "lv-chordia==1.1.0")
)
app = modal.App("lofi-study-analyze", image=image)
study = modal.Volume.from_name("lofi-study", create_if_missing=True)

SR = 22050
HOP = 256
STEPS = 16
KICK, SNARE, HAT = 0, 1, 2
NMF_FFT, NMF_ITER = 1024, 120
# where each voice may sit: a kick's click and a snare's rattle overlap the
# hat's floor, but a hat holds nothing under it
NMF_BANDS = ((20, 200), (150, 6000), (3000, 11025))
HIT_DELTA = 0.12
WAIT_FRAMES = 6
STACK_S = 0.03
NEAR_S = 0.05
OFF_EIGHTH = (0.40, 0.70)
HITS_AVERAGED = 32
ENV_N = 220
DECAY_DB = 20.0
PITCH = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5, "F#": 6, "Gb": 6,
    "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11, "Cb": 11,
}
TEMPO_AGREES = 0.08
# Krumhansl & Kessler's key profiles
MAJOR = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88)
MINOR = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17)
OVERLAP_S = 0.03
LEAD_MIN_S = 0.06
PENTATONIC = {"major": {0, 2, 4, 7, 9}, "minor": {0, 3, 5, 7, 10}}
CPU = 4.0
MEMORY_MIB = 8192
ERROR_CHARS = 300
PITCH_CLASSES = 12
BEATS_PER_BAR = 4
STEPS_PER_BEAT = 4
NMF_INIT_LO, NMF_INIT_HI = 0.5, 1.0
NMF_FLOOR = 1e-9
LEVEL_PERCENTILE = 99.5
PEAK_MAX_FRAMES = 3
PEAK_AVG_FRAMES = 10
HIT_LEVEL_FRAMES = 4
LEVEL_FLOOR = 1e-6
LOG_FLOOR = 1e-12
MIN_CLEAR_HITS = 4
BEAT_TIGHTNESS = 400.0
ON_BEAT_EARLY, ON_BEAT_LATE = 0.12, 0.88
MIN_OFF_EIGHTHS = 8
MIN_ON_BEATS = 8
MIN_LINE_NOTES = 8
NO_KEY_R = -2.0
CYCLE_BARS = (1, 2, 3, 4, 6, 8, 12, 16)
CYCLE_MARGIN = 0.03
REPEAT_LAGS = (1, 2, 4, 8)
REGISTER_PERCENTILES = [10, 50, 90]
BASS_REGISTER_PERCENTILES = [5, 25, 50, 75, 95]
INTERVAL_CAP = 13
INTERVAL_BINS = 14
STEP_SEMITONES = 2
LEGATO_GAP_S = 2.0
BASS_LOW_HZ, BASS_HIGH_HZ = 28.0, 400.0
OTHER_LOW_HZ, OTHER_HIGH_HZ = 80.0, 2400.0
KICK_LENGTH_S = 0.3
SNARE_LENGTH_S = 0.3
HAT_LENGTH_S = 0.12


def stem(root, name):
    import librosa

    y, _ = librosa.load(root / f"{name}.flac", sr=SR, mono=True)
    return y


def drum_hits(drums):
    """The drum stem's hits by NMF with band-bound templates: each voice's
    template starts as noise over its band and stays zero outside it, as
    multiplicative updates keep a zero, so overlapping hits still split.
    Returns (time, voice, level dB) rows."""
    import librosa
    import numpy as np
    from sklearn.decomposition import NMF

    spec = np.abs(librosa.stft(drums, n_fft=NMF_FFT, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=NMF_FFT)
    rng = np.random.default_rng(0)
    templates = np.stack([((freqs >= lo) & (freqs < hi)) * rng.uniform(NMF_INIT_LO, NMF_INIT_HI, freqs.size) for lo, hi in NMF_BANDS])
    gains = rng.uniform(NMF_INIT_LO, NMF_INIT_HI, (spec.shape[1], len(NMF_BANDS))) * spec.mean()
    model = NMF(n_components=len(NMF_BANDS), init="custom", solver="mu", beta_loss="kullback-leibler", max_iter=NMF_ITER)
    active = model.fit_transform((spec.T + NMF_FLOOR).astype(np.float64), W=gains, H=templates)
    rows = []
    for voice, track in zip((KICK, SNARE, HAT), active.T):
        top = float(np.percentile(track, LEVEL_PERCENTILE)) or 1.0
        flux = np.maximum(0.0, np.diff(track, prepend=track[0]))
        frames = librosa.util.peak_pick(flux, pre_max=PEAK_MAX_FRAMES, post_max=PEAK_MAX_FRAMES, pre_avg=PEAK_AVG_FRAMES, post_avg=PEAK_AVG_FRAMES, delta=HIT_DELTA * top, wait=WAIT_FRAMES)
        rows += [(f * HOP / SR, voice, 20 * np.log10(track[f : f + HIT_LEVEL_FRAMES].max() / top + LEVEL_FLOOR)) for f in frames]
    rows = np.array(sorted(rows)) if rows else np.zeros((0, 3))
    # a kick's click sounds in the snare's band too, and a kick under a snare
    # is rare: a hit with a kick on it is that kick
    kicks = rows[rows[:, 1] == KICK, 0]
    stacked = np.array([v == SNARE and kicks.size and np.abs(kicks - t).min() < STACK_S for t, v, _ in rows], dtype=bool)
    rows = rows[~stacked]
    return rows[:, 0], rows[:, 1].astype(int), rows[:, 2]


def beat_grid(mix, drums, times, voices, hint):
    """Beats, tempo, the first downbeat's index, the share of 4-beat bars and
    who said so. Beat This! (Foscarin et al., ISMIR 2024) unless it disagrees
    with `hint`, a tempo from a measure whose controls passed: on swung beats
    Beat This! reads 2/3 or 3/2 of it. Then librosa's tracker follows `hint`,
    and the downbeat puts the kicks on 1 and the snares on 2 and 4."""
    import librosa
    import numpy as np
    from beat_this.inference import File2Beats

    beats, downbeats = File2Beats(checkpoint_path="final0", device="cpu", dbn=False)(str(mix))
    beats, downbeats = np.asarray(beats), np.asarray(downbeats)
    ibi = float(np.median(np.diff(beats)))
    if hint is None or abs(60.0 / ibi / hint - 1.0) <= TEMPO_AGREES:
        first = int(np.argmin(np.abs(beats - downbeats[0])))
        lengths = np.round(np.diff(downbeats) / ibi)
        return beats, 60.0 / ibi, first, round(float(np.mean(lengths == BEATS_PER_BAR)), 3) if lengths.size else 0.0, "beat-this"
    onset = librosa.onset.onset_strength(y=drums, sr=SR, hop_length=HOP)
    _, frames = librosa.beat.beat_track(onset_envelope=onset, sr=SR, hop_length=HOP, bpm=hint, tightness=BEAT_TIGHTNESS, trim=False)
    beats = librosa.frames_to_time(frames, sr=SR, hop_length=HOP)
    kicks, snares = times[voices == KICK], times[voices == SNARE]

    def landed(hits_at, where):
        return sum(float(np.min(np.abs(where - t))) < NEAR_S for t in hits_at) if where.size else 0

    score = [landed(kicks, beats[p::BEATS_PER_BAR]) + landed(snares, beats[(p + 1) % BEATS_PER_BAR :: 2]) for p in range(BEATS_PER_BAR)]
    return beats, 60.0 / float(np.median(np.diff(beats))), int(np.argmax(score)), None, "form"


def placed(t, beats):
    """Where `t` falls on the grid `beats` lays from a downbeat: its 16th in
    its bar, its phase in its beat and that beat's length; None off the grid."""
    import numpy as np

    at = int(np.searchsorted(beats, t, side="right")) - 1
    if at < 0 or at + 1 >= beats.size:
        return None
    span = beats[at + 1] - beats[at]
    phase = (t - beats[at]) / span
    return int(round((at % BEATS_PER_BAR) * STEPS_PER_BEAT + phase * STEPS_PER_BEAT)) % STEPS, phase, span


def drum_grid(times, voices, levels, beats, downbeat):
    """Each voice's share of bars with a hit on each 16th, its hits a bar and
    the spread of its levels; where an off-eighth sits in its beat; how late
    the snare sits behind the kick."""
    import numpy as np

    bars = beats[downbeat:]
    n_bars = max(1, (bars.size - 1) // BEATS_PER_BAR)
    out, offs, on_beat = {}, [], {KICK: [], SNARE: []}
    for code, name in ((KICK, "kick"), (SNARE, "snare"), (HAT, "hat")):
        grid = np.zeros(STEPS)
        mine = levels[voices == code]
        for t in times[voices == code]:
            where = placed(t, bars)
            if where is None:
                continue
            step, phase, span = where
            grid[step] += 1
            if OFF_EIGHTH[0] <= phase <= OFF_EIGHTH[1]:
                offs.append(float(phase))
            if code in on_beat and (phase < ON_BEAT_EARLY or phase > ON_BEAT_LATE):
                on_beat[code].append(float((phase if phase < 0.5 else phase - 1) * span * 1000))
        out[name] = {
            "steps": (grid / n_bars).round(3).tolist(),
            "per_bar": round(float(mine.size) / n_bars, 2),
            "level_sd_db": round(float(np.std(mine)), 2) if mine.size else None,
            "level_db": round(float(np.median(mine)), 1) if mine.size else None,
        }
    out["off_eighth"] = round(float(np.median(offs)), 3) if len(offs) >= MIN_OFF_EIGHTHS else None
    if len(on_beat[KICK]) >= MIN_ON_BEATS and len(on_beat[SNARE]) >= MIN_ON_BEATS:
        out["snare_behind_kick_ms"] = round(float(np.median(on_beat[SNARE]) - np.median(on_beat[KICK])), 1)
    return out


def drum_sound(drums, times, voices, levels, code, length_s):
    """A voice's loudest clear hits, averaged: where its power sits, and how
    long its level takes to fall `DECAY_DB`."""
    import librosa
    import numpy as np

    length = int(length_s * SR)
    following = np.append(np.diff(times), np.inf)
    others = times[voices != code]
    alone = np.array([others.size == 0 or np.abs(others - t).min() > STACK_S for t in times], dtype=bool)
    picks = np.flatnonzero((voices == code) & (following > length_s) & alone)
    picks = picks[np.argsort(-levels[picks])][:HITS_AVERAGED]
    if picks.size < MIN_CLEAR_HITS:
        return None
    takes = np.array([drums[int(times[i] * SR) :][:length] for i in picks if int(times[i] * SR) + length < drums.size])
    if not takes.size:
        return None
    spectrum = (np.abs(np.fft.rfft(takes * np.hanning(length), axis=1)) ** 2).mean(axis=0)
    freqs = np.fft.rfftfreq(length, 1 / SR)
    env = librosa.power_to_db(np.convolve((takes**2).mean(axis=0), np.ones(ENV_N) / ENV_N, "same") + LOG_FLOOR)
    top = int(env.argmax())
    below = np.flatnonzero(env[top:] < env[top] - DECAY_DB)
    return {
        "centroid_hz": round(float((freqs * spectrum).sum() / spectrum.sum()), 1),
        "peak_hz": round(float(freqs[spectrum.argmax()]), 1),
        "decay_ms": round(float(below[0]) / SR * 1000, 1) if below.size else None,
    }


def transcribe(path, low, high):
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import predict

    _, _, notes = predict(str(path), ICASSP_2022_MODEL_PATH, minimum_frequency=low, maximum_frequency=high)
    return [(float(s), float(e), int(p), float(a)) for s, e, p, a, _ in notes]


def chroma_of(notes, edges):
    import numpy as np

    chroma = np.zeros((len(edges) - 1, PITCH_CLASSES))
    for start, end, pitch, amp in notes:
        first = max(np.searchsorted(edges, start, side="right") - 1, 0)
        last = min(np.searchsorted(edges, end, side="left"), len(edges) - 1)
        for i in range(first, last):
            overlap = min(end, edges[i + 1]) - max(start, edges[i])
            if overlap > 0:
                chroma[i, pitch % PITCH_CLASSES] += overlap * amp
    return chroma


def key_of(total):
    import numpy as np

    best = (NO_KEY_R, 0, "major")
    for name, profile in (("major", MAJOR), ("minor", MINOR)):
        for tonic in range(PITCH_CLASSES):
            r = float(np.corrcoef(np.roll(profile, tonic), total)[0, 1])
            if r > best[0]:
                best = (r, tonic, name)
    return best[1], best[2], round(best[0], 3)


def chords_of(path, edges):
    """One (root, quality) a beat, or None where lv-chordia hears no chord
    (Jiang et al., ISMIR 2019)."""
    import numpy as np
    from lv_chordia.chord_recognition import chord_recognition

    spans = chord_recognition(audio_path=str(path), chord_dict_name="submission")
    starts = np.array([s["start_time"] for s in spans])
    out = []
    for at in (np.asarray(edges[:-1]) + np.asarray(edges[1:])) / 2:
        i = max(int(np.searchsorted(starts, at, side="right")) - 1, 0)
        root, _, rest = spans[i]["chord"].partition(":")
        out.append((PITCH[root], rest.split("/")[0] or "maj") if root in PITCH else None)
    return out


def harmony(chords, tonic, per_bar=BEATS_PER_BAR):
    """Chord tokens relative to the tonic: their share, what follows what, how
    long a chord lasts, and the shortest cycle the bars repeat on."""
    sounding = [c for c in chords if c]
    bars = [chords[i] for i in range(0, len(chords), per_bar)]
    tokens = [f"{(c[0] - tonic) % PITCH_CLASSES}:{c[1]}" if c else "N" for c in bars]
    roots = [c[0] if c else None for c in bars]
    changes = sum(a != b for a, b in zip(chords, chords[1:]))
    best_cycle, best_match = None, 0.0
    for cycle in CYCLE_BARS:
        if len(roots) < 2 * cycle:
            continue
        # by root: one voicing's added 7th isn't a new chord in the cycle
        match = sum(a == b for a, b in zip(roots, roots[cycle:])) / (len(roots) - cycle)
        if match > best_match + CYCLE_MARGIN:
            best_cycle, best_match = cycle, match
    return {
        "qualities": Counter(q for _, q in sounding),
        "roots": Counter((root - tonic) % PITCH_CLASSES for root, _ in sounding),
        "bar_tokens": Counter(tokens),
        "bigrams": Counter(f"{a}>{b}" for a, b in zip(tokens, tokens[1:]) if a != b),
        "beats_per_chord": round(len(chords) / (changes + 1), 2),
        "cycle_bars": best_cycle,
        "cycle_match": round(best_match, 3),
    }


def skyline(notes, edges):
    """The highest sounding note of each 16th that a note starts in."""
    top = {}
    import numpy as np

    loud = float(np.median([amp for _, _, _, amp in notes])) if notes else 0.0
    for start, end, pitch, amp in notes:
        if amp < loud or end - start < LEAD_MIN_S:
            continue
        step = int(np.searchsorted(edges, start, side="right")) - 1
        if step >= 0 and (step not in top or pitch > top[step][0]):
            top[step] = (pitch, amp)
    return sorted((step, pitch) for step, (pitch, _) in top.items())


def melody(line, tonic, mode, n_bars):
    import numpy as np

    if len(line) < MIN_LINE_NOTES:
        return None
    pitches = np.array([p for _, p in line])
    steps = np.array([s for s, _ in line])
    jumps = np.abs(np.diff(pitches))
    degrees = Counter(int((p - tonic) % PITCH_CLASSES) for p in pitches)
    pent = sum(n for d, n in degrees.items() if d in PENTATONIC[mode]) / sum(degrees.values())
    by_bar = {}
    for step, pitch in line:
        by_bar.setdefault(step // STEPS, []).append((step % STEPS, pitch))

    def alike(a, b):
        a, b = set(a), set(b)
        return len(a & b) / len(a | b)

    repeats = {
        str(lag): round(float(np.mean([alike(by_bar[b], by_bar[b - lag]) for b in by_bar if b - lag in by_bar] or [0.0])), 3)
        for lag in REPEAT_LAGS
    }
    return {
        "register": np.percentile(pitches, REGISTER_PERCENTILES).round(1).tolist(),
        "onsets_per_bar": round(len(line) / max(n_bars, 1), 2),
        "interval_hist": np.bincount(np.minimum(jumps, INTERVAL_CAP), minlength=INTERVAL_BINS).tolist(),
        "pentatonic_share": round(float(pent), 3),
        "step_share": round(float(np.mean(jumps <= STEP_SEMITONES)), 3),
        "repeat_at_bars": repeats,
        "eighth_grid_share": round(float(np.mean(steps % 2 == 0)), 3),
    }


def monophonic(notes):
    """`notes` as one line: of two that overlap, the stronger."""
    line = []
    for note in sorted(notes):
        if line and note[0] < line[-1][1] - OVERLAP_S:
            if note[3] > line[-1][3]:
                line[-1] = note
            continue
        line.append(note)
    return line


def bass_line(notes, n_bars, beats, kicks):
    """The bass's line, and where on the grid `beats` lays its notes start:
    each 16th's share of bars, the share on a beat and on an off-eighth, and
    the share with one of `kicks`, as near as `beat_grid` lands a hit."""
    import numpy as np

    line = monophonic(notes)
    if len(line) < MIN_LINE_NOTES:
        return None
    pitches = np.array([p for _, _, p, _ in line])
    starts = np.array([s for s, _, _, _ in line])
    lengths = np.array([e - s for s, e, _, _ in line])
    gaps = np.diff(starts)
    near = gaps < LEGATO_GAP_S
    steps = np.array([where[0] for where in (placed(t, beats) for t in starts) if where], dtype=int)
    with_kick = [kicks.size > 0 and float(np.abs(kicks - t).min()) < NEAR_S for t in starts]
    share = lambda held: round(float(np.mean(held)), 3) if steps.size else None
    return {
        "register": np.percentile(pitches, BASS_REGISTER_PERCENTILES).round(1).tolist(),
        "notes_per_bar": round(len(line) / max(n_bars, 1), 2),
        "legato": round(float(np.median(lengths[:-1][near] / gaps[near])), 3) if near.any() else None,
        "median_note_ms": round(float(np.median(lengths)) * 1000, 1),
        "steps": (np.bincount(steps, minlength=STEPS) / max(n_bars, 1)).round(3).tolist(),
        "on_beat_share": share(steps % STEPS_PER_BEAT == 0),
        "off_eighth_share": share(steps % STEPS_PER_BEAT == STEPS_PER_BEAT // 2),
        "with_kick_share": round(float(np.mean(with_kick)), 3),
    }


def bounce(root, names, out):
    """`names`' stems summed into a WAV at `out`."""
    import soundfile as sf

    rate = 0
    total = 0.0
    for name in names:
        audio, rate = sf.read(root / f"{name}.flac", dtype="float32")
        total = total + audio
    sf.write(out, total, rate, subtype="PCM_24")
    return out


@app.function(
    cpu=CPU, memory=MEMORY_MIB, timeout=TIMEOUT_S, max_containers=CONCURRENT,
    volumes={STUDY: study.with_mount_options(read_only=True)},
)
def one(job):
    import numpy as np

    key, hint = job
    root = Path(STUDY) / "stems" / key
    powers = json.loads((root / "done").read_text())
    mix = bounce(root, ("drums", "bass", "other", "vocals"), f"/tmp/{key}-mix.wav")
    pitched = bounce(root, ("bass", "other"), f"/tmp/{key}-pitched.wav")
    drums = stem(root, "drums")
    times, voices, levels = drum_hits(drums)
    beats, tempo, downbeat, four_four, grid_from = beat_grid(mix, drums, times, voices, hint)
    bar_times = beats[downbeat:]
    sixteenths = np.interp(np.arange(0, (bar_times.size - 1) * STEPS_PER_BEAT + 1) / STEPS_PER_BEAT, np.arange(bar_times.size), bar_times)
    bass_notes = transcribe(root / "bass.flac", BASS_LOW_HZ, BASS_HIGH_HZ)
    other_notes = transcribe(root / "other.flac", OTHER_LOW_HZ, OTHER_HIGH_HZ)
    beat_edges = bar_times.tolist()
    tonic, mode, confidence = key_of(chroma_of(other_notes + bass_notes, beat_edges).sum(axis=0))
    chords = chords_of(pitched, beat_edges)
    n_bars = max(1, (bar_times.size - 1) // BEATS_PER_BAR)
    return {
        "key": key,
        "powers": powers,
        "tempo": round(tempo, 1),
        "beats": int(beats.size),
        "downbeat": downbeat,
        "four_four": four_four,
        "grid_from": grid_from,
        "drums": drum_grid(times, voices, levels, beats, downbeat),
        "kick": drum_sound(drums, times, voices, levels, KICK, KICK_LENGTH_S),
        "snare": drum_sound(drums, times, voices, levels, SNARE, SNARE_LENGTH_S),
        "hat": drum_sound(drums, times, voices, levels, HAT, HAT_LENGTH_S),
        "bass": bass_line(bass_notes, n_bars, bar_times, times[voices == KICK]),
        "mode": mode,
        "tonic": tonic,
        "key_r": confidence,
        "harmony": harmony(chords, tonic),
        "other_register": np.percentile([p for _, _, p, _ in other_notes] or [0], REGISTER_PERCENTILES).round(1).tolist(),
        "polyphony": round(sum(e - s for s, e, _, _ in other_notes) / max(float(bar_times[-1] - bar_times[0]), 1.0), 2),
        "melody": melody(skyline(other_notes, sixteenths), tonic, mode, n_bars),
    }



@app.local_entrypoint()
def run(out: str, prefix: str = "", hints: str = ""):
    # a reference's beat tracker misreads swing without its hint, so its grid and
    # every per-bar row would move in silence
    if not prefix and not hints:
        sys.exit("analyze.py: the references need --hints, a JSON of each stem key's tempo")
    keys = sorted(e.path.rsplit("/", 1)[1] for e in study.listdir("/stems"))
    keys = [k for k in keys if k.startswith(prefix)] if prefix else keys
    # a reference's tempo hint, keyed as its stems are; a take's grid needs none
    hints = json.loads(Path(hints).read_text()) if hints else {}
    rows = list(one.map([(k, hints.get(k)) for k in keys], return_exceptions=True))
    good = [r for r in rows if isinstance(r, dict)]
    print(f"{len(good)} of {len(keys)} analysed")
    for key, row in zip(keys, rows):
        if not isinstance(row, dict):
            print("failed:", key, repr(row)[:ERROR_CHARS])
    path = Path(out, f"study{'-' + prefix.strip('-') if prefix else ''}.json")
    previous = json.loads(path.read_text()) if path.exists() else {}
    previous.update({r["key"]: r for r in good})
    path.write_text(json.dumps(previous, indent=0) + "\n")
    print("wrote", path.name, len(previous))
    # the good rows are kept, but a card over them would read like one over every take
    if len(good) != len(keys):
        sys.exit(f"analyze.py: {len(keys) - len(good)} of {len(keys)} failed")
