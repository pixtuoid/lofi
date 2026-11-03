"""Where a time falls on a track's beat grid, how alike its bars are, and
what a track without a kit has no grid for, in the standard library alone so
`test_grid.py` runs it without the study's image."""

from bisect import bisect_right

BEATS_PER_BAR = 4
STEPS_PER_BEAT = 4
STEPS = BEATS_PER_BAR * STEPS_PER_BEAT


def placed(t, beats):
    """Where `t` falls on the grid `beats` lays from a downbeat: its 16th in
    its bar, its phase in its beat and that beat's length; None off the grid."""
    at = bisect_right(beats, t) - 1
    if at < 0 or at + 1 >= len(beats):
        return None
    span = beats[at + 1] - beats[at]
    phase = (t - beats[at]) / span
    return int(round((at % BEATS_PER_BAR) * STEPS_PER_BEAT + phase * STEPS_PER_BEAT)) % STEPS, phase, span


def placement(starts, beats, kicks, near_s, n_bars):
    """Where `starts` fall on the grid `beats` lays: each 16th's onsets a bar
    over `n_bars`, the share on a beat and on an off-eighth, and the share
    nearer one of `kicks` than `near_s`."""
    steps = [where[0] for where in (placed(t, beats) for t in starts) if where]
    with_kick = [len(kicks) > 0 and min(abs(k - t) for k in kicks) < near_s for t in starts]

    def share(held):
        return round(sum(held) / len(steps), 3) if steps else None

    return {
        "steps": [round(steps.count(step) / max(n_bars, 1), 3) for step in range(STEPS)],
        "on_beat_share": share(step % STEPS_PER_BEAT == 0 for step in steps),
        "off_eighth_share": share(step % STEPS_PER_BEAT == STEPS_PER_BEAT // 2 for step in steps),
        "with_kick_share": round(sum(with_kick) / len(starts), 3) if starts else None,
    }


def likeness(line, lag):
    """How alike each bar of `line`, `(16th from the first downbeat, pitch)`
    onsets, is to the bar `lag` before it: the share of the onsets either
    holds that both do, one a bar with a bar `lag` before it."""
    by_bar = {}
    for step, pitch in line:
        by_bar.setdefault(step // STEPS, set()).add((step % STEPS, pitch))
    return [
        len(by_bar[bar] & by_bar[bar - lag]) / len(by_bar[bar] | by_bar[bar - lag])
        for bar in by_bar
        if bar - lag in by_bar
    ]


# a drums stem under this share of the mix's power holds the separation's
# residue, no kit: Lofi Girl's sleep playlist peaks at 0.07 % and its main one
# starts at 10 % (measured 2026-11-03)
MIN_KIT_SHARE = 0.01
# the measures a line keeps without a grid: where its notes sit and how they
# move, never how many fall in a bar or where
UNGRIDDED_BASS = ("register", "legato", "median_note_ms")
UNGRIDDED_MELODY = ("register", "interval_hist", "pentatonic_share", "step_share")
GRIDDED_HARMONY = ("bar_tokens", "bigrams", "beats_per_chord", "cycle_bars", "cycle_match")


def has_kit(powers):
    """Whether a track's stems' `powers` hold a drum kit."""
    return powers["drums"] / sum(powers.values()) >= MIN_KIT_SHARE


def without_grid(row):
    """`row` with every measure read off a beat grid or a drum hit unset: a
    track without a kit has neither, and its tracker's grid is a convention,
    not the track's."""
    def keep(line, fields):
        return line and {k: (v if k in fields else None) for k, v in line.items()}

    return {
        **row,
        "tempo": None,
        "drums": None,
        "kick": None,
        "snare": None,
        "hat": None,
        "harmony": {k: (None if k in GRIDDED_HARMONY else v) for k, v in row["harmony"].items()},
        "bass": keep(row["bass"], UNGRIDDED_BASS),
        "melody": keep(row["melody"], UNGRIDDED_MELODY),
    }
