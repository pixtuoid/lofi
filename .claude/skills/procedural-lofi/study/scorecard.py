#!/usr/bin/env python3
"""Lofi Girl's sound against a set of our takes, row by row, in the order the
signal chain makes them, so what to work on next reads off the top of the card.

  scorecard.py targets OUT.json STUDY TIMBRE BASS STEMFAD
  scorecard.py score TAKES_DIR PREFIX

`targets` turns the references' per-track measures into `targets.json`, each
row's quartiles and nothing else, so no track is in this public repo. `score`
reads a commit's measures, as `score.sh` leaves them in TAKES_DIR, against
them. A row is off when our median leaves the tracks' interquartile range, and
its gap counts those ranges from their median.

The judge reads a mono fold, so the stereo rows are the card's alone. That it
reads harmony poorly is opinion, from #65's factor runs.
"""

import json
import sys
from pathlib import Path
from statistics import median, quantiles

HERE = Path(__file__).resolve().parent
TARGETS = HERE / "targets.json"
STUDY, TIMBRE, BASS, STEMFAD = "study", "timbre", "bass", "stemfad"
TRIADS = ("maj", "min")
MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)
MINOR_SCALE = (0, 2, 3, 5, 7, 8, 10)


def path(*keys):
    def read(row):
        for key in keys:
            if not isinstance(row, dict) or key not in row:
                return None
            row = row[key]
        return row if isinstance(row, (int, float)) else None

    return read


def item(index, *keys):
    def read(row):
        for key in keys:
            row = row.get(key) if isinstance(row, dict) else None
        return row[index] if isinstance(row, list) and len(row) > index else None

    return read


def percent_of(counts, picked):
    total = sum(counts.values())
    return 100 * sum(n for k, n in counts.items() if picked(k)) / total if total else None


def quality_share(*names):
    def read(row):
        return percent_of((row.get("harmony") or {}).get("qualities") or {}, lambda k: k in names)

    return read


def outside_key(row):
    scale = MAJOR_SCALE if row.get("mode") == "major" else MINOR_SCALE
    return percent_of((row.get("harmony") or {}).get("roots") or {}, lambda k: int(k) not in scale)


def stem_share(name):
    def read(row):
        return percent_of(row.get("powers") or {}, lambda k: k == name)

    return read


# (layer, [(label, source, read)]), upstream first
LAYERS = [
    ("1 harmony", [
        ("triads %", STUDY, quality_share(*TRIADS)),
        ("maj7 %", STUDY, quality_share("maj7", "maj9")),
        ("min7 %", STUDY, quality_share("min7", "min9")),
        ("dom7 %", STUDY, quality_share("7", "9")),
        ("roots off the key %", STUDY, outside_key),
        ("beats a chord", STUDY, path("harmony", "beats_per_chord")),
        ("loop bars", STUDY, path("harmony", "cycle_bars")),
    ]),
    ("2 rhythm", [
        ("tempo bpm", STUDY, path("tempo")),
        ("onsets /s", TIMBRE, path("onsets_per_s")),
        ("kicks /bar", STUDY, path("drums", "kick", "per_bar")),
        ("snares /bar", STUDY, path("drums", "snare", "per_bar")),
        ("hats /bar", STUDY, path("drums", "hat", "per_bar")),
        ("off-eighth", STUDY, path("drums", "off_eighth")),
        ("snare behind kick ms", STUDY, path("drums", "snare_behind_kick_ms")),
        ("kick level sd dB", STUDY, path("drums", "kick", "level_sd_db")),
        ("hat level sd dB", STUDY, path("drums", "hat", "level_sd_db")),
        ("bass notes /bar", STUDY, path("bass", "notes_per_bar")),
        ("bass legato", STUDY, path("bass", "legato")),
    ]),
    ("3 melody", [
        ("pentatonic share", STUDY, path("melody", "pentatonic_share")),
        ("step share", STUDY, path("melody", "step_share")),
        ("lead onsets /bar", STUDY, path("melody", "onsets_per_bar")),
        ("on eighth grid", STUDY, path("melody", "eighth_grid_share")),
        ("lead median MIDI", STUDY, item(1, "melody", "register")),
        ("repeat at 4 bars", STUDY, path("melody", "repeat_at_bars", "4")),
        ("repeat at 8 bars", STUDY, path("melody", "repeat_at_bars", "8")),
        ("polyphony", STUDY, path("polyphony")),
        ("other median MIDI", STUDY, item(1, "other_register")),
    ]),
    ("4 source", [
        ("drums FAD", STEMFAD, path("drums")),
        ("bass FAD", STEMFAD, path("bass")),
        ("other FAD", STEMFAD, path("other")),
        ("kick decay ms", STUDY, path("kick", "decay_ms")),
        ("snare centroid Hz", STUDY, path("snare", "centroid_hz")),
        ("snare decay ms", STUDY, path("snare", "decay_ms")),
        ("hat centroid Hz", STUDY, path("hat", "centroid_hz")),
        ("hat decay ms", STUDY, path("hat", "decay_ms")),
        ("bass centroid Hz", BASS, path("centroid_hz")),
        ("bass overtones dB", BASS, path("overtone_db")),
        ("bass under 45 Hz %", BASS, path("shares", "20-45")),
        ("bass 62-90 Hz %", BASS, path("shares", "62-90")),
        ("bass occupancy", BASS, path("occupancy")),
    ]),
    ("5 balance", [
        ("drums %", STUDY, stem_share("drums")),
        ("bass %", STUDY, stem_share("bass")),
        ("other %", STUDY, stem_share("other")),
    ]),
    ("6 dynamics", [
        ("crest dB", TIMBRE, path("peak_rms_db")),
        ("body swing dB", TIMBRE, path("body_swing_db")),
        ("low punch dB", TIMBRE, path("low_punch_db")),
        ("perc share %", TIMBRE, path("perc_share_pct")),
    ]),
    ("7 space", [
        ("side <125 dB", TIMBRE, path("side_mid_db <125")),
        ("side 125-500 dB", TIMBRE, path("side_mid_db 125-500")),
        ("side 500-2k dB", TIMBRE, path("side_mid_db 500-2k")),
        ("side 2-8k dB", TIMBRE, path("side_mid_db 2k-8k")),
        ("LR correlation", TIMBRE, path("lr_correlation")),
    ]),
    ("8 texture", [
        ("clicks /min", TIMBRE, path("clicks_per_min")),
        ("hiss floor dB", TIMBRE, path("hiss_floor_db_re_rms")),
    ]),
]


def quartiles(values):
    return quantiles(values, n=4, method="inclusive")


def targets(sources):
    out = {}
    for layer, rows in LAYERS:
        out[layer] = {}
        for label, source, read in rows:
            values = [v for v in map(read, sources[source]) if v is not None]
            if values:
                out[layer][label] = [round(q, 3) for q in quartiles(values)]
    return out


def row(quartile, values):
    """Our median, whether it holds, and its gap: None where the tracks' range
    is flat, which has no width to count in."""
    low, mid, high = quartile
    ours = median(values)
    return ours, low <= ours <= high, abs(ours - mid) / (high - low) if high > low else None


def card(target, sources):
    """`(layer, label, quartiles, ours)` a row, `ours` None where the takes
    hold no such measure."""
    for layer, rows in LAYERS:
        for label, source, read in rows:
            if label in target.get(layer, {}):
                values = [v for v in map(read, sources[source]) if v is not None]
                quartile = target[layer][label]
                yield layer, label, quartile, row(quartile, values) if values else None


def show(lines):
    held = total = 0
    shown = None
    for layer, label, (low, mid, high), ours in lines:
        if layer != shown:
            print(f"\n{layer}\n{'':24}{'P25':>9}{'P50':>9}{'P75':>9}{'ours':>9}")
            shown = layer
        if ours is None:
            print(f"{label:24}{low:9.2f}{mid:9.2f}{high:9.2f}  not measured")
            continue
        value, holds, gap = ours
        total += 1
        held += holds
        width = "flat" if gap is None else f"{gap:4.1f}"
        print(f"{label:24}{low:9.2f}{mid:9.2f}{high:9.2f}{value:9.2f}  {'ok ' if holds else 'OFF'} {width}")
    print(f"\n{held} of {total} rows inside the tracks' interquartile range")


def load(file):
    return json.loads(Path(file).read_text())


def main(args):
    if args[:1] == ["targets"] and len(args) == 6:
        study, timbre, bass, stemfad = map(load, args[2:])
        refs = lambda rows: [v for k, v in rows.items() if not k.startswith("take")]
        made = targets({
            STUDY: refs(study), TIMBRE: refs(timbre), BASS: refs(bass), STEMFAD: refs(stemfad),
        })
        Path(args[1]).write_text(json.dumps(made, indent=1) + "\n")
    elif args[:1] == ["score"] and len(args) == 3:
        taken, prefix = Path(args[1]), args[2]
        mine = lambda rows: [v for k, v in rows.items() if k.startswith(prefix)]
        # a commit's own: `timbre.json` keys its rows by path, not by prefix
        show(card(load(TARGETS), {
            STUDY: list(load(taken / f"study-{prefix.strip('-')}.json").values()),
            TIMBRE: list(load(taken / "timbre.json").values()),
            BASS: mine(load(taken / "basstone.json")),
            STEMFAD: mine(load(taken / "stemfad.json")),
        }))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
