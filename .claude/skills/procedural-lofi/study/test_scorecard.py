#!/usr/bin/env python3
"""The scorecard's tests: `python3 -I test_scorecard.py`, with no venv."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from scorecard import BASS, LAYERS, STEMFAD, STUDY, TARGETS, TIMBRE, card, main, outside_key, path, row, targets

# one row of each source, every value distinct, so a reader of the wrong field
# reads the wrong number
FIXTURE = {
    STUDY: {
        "mode": "major",
        "tempo": 101.0,
        "powers": {"drums": 2, "bass": 3, "other": 5},
        "harmony": {
            "qualities": {"maj": 2, "min": 1, "maj7": 3, "min7": 4, "7": 10},
            "roots": {"0": 3, "3": 1},
            "beats_per_chord": 102.0,
            "cycle_bars": 103.0,
        },
        "drums": {
            "kick": {"per_bar": 104.0, "level_sd_db": 105.0},
            "snare": {"per_bar": 106.0},
            "hat": {"per_bar": 107.0, "level_sd_db": 108.0},
            "off_eighth": 109.0,
            "snare_behind_kick_ms": 110.0,
        },
        "bass": {
            "notes_per_bar": 111.0,
            "legato": 112.0,
            "on_beat_share": 127.0,
            "off_eighth_share": 128.0,
            "with_kick_share": 129.0,
        },
        "melody": {
            "pentatonic_share": 113.0,
            "step_share": 114.0,
            "onsets_per_bar": 115.0,
            "eighth_grid_share": 116.0,
            "register": [1.0, 117.0, 2.0],
            "repeat_at_bars": {"4": 118.0, "8": 119.0},
        },
        "polyphony": 120.0,
        "other_register": [1.0, 121.0, 2.0],
        "kick": {"decay_ms": 122.0},
        "snare": {"centroid_hz": 123.0, "decay_ms": 124.0},
        "hat": {"centroid_hz": 125.0, "decay_ms": 126.0},
    },
    TIMBRE: {
        "onsets_per_s": 201.0,
        "peak_rms_db": 202.0,
        "body_swing_db": 203.0,
        "low_punch_db": 204.0,
        "perc_share_pct": 205.0,
        "side_mid_db <125": 206.0,
        "side_mid_db 125-500": 207.0,
        "side_mid_db 500-2k": 208.0,
        "side_mid_db 2k-8k": 209.0,
        "lr_correlation": 210.0,
        "clicks_per_min": 211.0,
        "hiss_floor_db_re_rms": 212.0,
    },
    BASS: {
        "centroid_hz": 301.0,
        "overtone_db": 302.0,
        "occupancy": 303.0,
        "shares": {"20-45": 304.0, "62-90": 305.0},
    },
    STEMFAD: {"drums": 401.0, "bass": 402.0, "other": 403.0},
}
WANT = {
    "triads %": 15.0,
    "maj7 %": 15.0,
    "min7 %": 20.0,
    "dom7 %": 50.0,
    "roots off the key %": 25.0,
    "beats a chord": 102.0,
    "loop bars": 103.0,
    "tempo bpm": 101.0,
    "onsets /s": 201.0,
    "kicks /bar": 104.0,
    "snares /bar": 106.0,
    "hats /bar": 107.0,
    "off-eighth": 109.0,
    "snare behind kick ms": 110.0,
    "kick level sd dB": 105.0,
    "hat level sd dB": 108.0,
    "bass notes /bar": 111.0,
    "bass legato": 112.0,
    "bass on a beat": 127.0,
    "bass on an off-eighth": 128.0,
    "bass with a kick": 129.0,
    "pentatonic share": 113.0,
    "step share": 114.0,
    "top-line onsets /bar": 115.0,
    "on eighth grid": 116.0,
    "top-line median MIDI": 117.0,
    "repeat at 4 bars": 118.0,
    "repeat at 8 bars": 119.0,
    "polyphony": 120.0,
    "other median MIDI": 121.0,
    "drums FAD": 401.0,
    "bass FAD": 402.0,
    "other FAD": 403.0,
    "kick decay ms": 122.0,
    "snare centroid Hz": 123.0,
    "snare decay ms": 124.0,
    "hat centroid Hz": 125.0,
    "hat decay ms": 126.0,
    "bass centroid Hz": 301.0,
    "bass overtones dB": 302.0,
    "bass under 45 Hz %": 304.0,
    "bass 62-90 Hz %": 305.0,
    "bass occupancy": 303.0,
    "drums %": 20.0,
    "bass %": 30.0,
    "other %": 50.0,
    "crest dB": 202.0,
    "body swing dB": 203.0,
    "low punch dB": 204.0,
    "perc share %": 205.0,
    "side <125 dB": 206.0,
    "side 125-500 dB": 207.0,
    "side 500-2k dB": 208.0,
    "side 2-8k dB": 209.0,
    "LR correlation": 210.0,
    "clicks /min": 211.0,
    "hiss floor dB": 212.0,
}


class Scorecard(unittest.TestCase):
    def test_every_row_reads_its_own_field(self):
        got = {label: read(FIXTURE[source]) for _, rows in LAYERS for label, source, read in rows}
        self.assertEqual(got, WANT)

    def test_a_median_on_either_edge_of_the_range_holds(self):
        self.assertTrue(row([1, 2, 3], [1])[1])
        self.assertTrue(row([1, 2, 3], [3])[1])

    def test_a_median_outside_the_range_is_off_by_ranges_from_the_middle(self):
        self.assertEqual(row([1, 2, 3], [2, 2, 3]), (2, True, 0))
        self.assertEqual(row([1, 2, 3], [5]), (5, False, 1.5))

    def test_a_flat_range_has_no_gap_to_rank(self):
        self.assertEqual(row([2, 2, 2], [3]), (3, False, None))

    def test_a_row_the_takes_never_measured_says_so(self):
        target = {"2 rhythm": {"tempo bpm": [73, 77, 81]}}
        lines = list(card(target, {STUDY: [{}]}))
        self.assertEqual(lines, [("2 rhythm", "tempo bpm", [73, 77, 81], None)])

    def test_a_chord_root_outside_a_tracks_mode_counts_against_that_mode(self):
        harmony = {"roots": {"0": 3, "3": 1}}
        self.assertEqual(outside_key({"mode": "major", "harmony": harmony}), 25)
        self.assertEqual(outside_key({"mode": "minor", "harmony": harmony}), 0)

    def test_a_missing_or_null_measure_reads_none_not_zero(self):
        self.assertIsNone(path("drums", "kick")({"drums": {}}))
        self.assertIsNone(path("decay_ms")({"decay_ms": None}))

    def test_targets_are_true_quartiles_and_hold_nothing_else(self):
        rows = [{"tempo": t} for t in (70, 75, 80, 85)]
        made = targets({STUDY: rows, TIMBRE: [], BASS: [], STEMFAD: []})
        self.assertEqual(made["2 rhythm"], {"tempo bpm": [73.75, 77.5, 81.25]})

    def test_the_committed_targets_cover_every_row_in_order(self):
        committed = json.loads(TARGETS.read_text())
        self.assertEqual(
            {layer: sorted(rows) for layer, rows in committed.items()},
            {layer: sorted(label for label, _, _ in rows) for layer, rows in LAYERS},
        )
        for rows in committed.values():
            self.assertTrue(all(low <= mid <= high for low, mid, high in rows.values()))

    def test_a_card_reads_only_the_prefixs_takes(self):
        def take(tempo, fad):
            return {"tempo": tempo}, {"drums": fad}

        with tempfile.TemporaryDirectory() as tmp:
            mine, other = take(77.0, 0.4), take(500.0, 9.0)
            files = {
                "study-take-a_.json": {"take-a_0": mine[0]},
                "timbre.json": {"in/a_0.wav": FIXTURE[TIMBRE]},
                "basstone.json": {"take-a_0": FIXTURE[BASS], "take-b_0": {"centroid_hz": 9e9}},
                "stemfad.json": {"take-a_0": mine[1], "take-b_0": other[1]},
            }
            for name, rows in files.items():
                Path(tmp, name).write_text(json.dumps(rows))
            printed = io.StringIO()
            with contextlib.redirect_stdout(printed):
                main(["score", tmp, "take-a_"])
        card_text = printed.getvalue()
        self.assertRegex(card_text, r"drums FAD\s.*\s0\.40\s")
        self.assertRegex(card_text, r"bass centroid Hz\s.*\s301\.00\s")
        self.assertRegex(card_text, r"tempo bpm\s.*\s77\.00\s")
        self.assertNotIn("9000000000", card_text)


if __name__ == "__main__":
    unittest.main()
