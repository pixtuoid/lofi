#!/usr/bin/env python3
"""The beat grid's tests: `python3 -I test_grid.py`, with no venv."""

import sys
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

from grid import STEPS, has_kit, likeness, placed, placement, without_grid

# beats a second apart from a downbeat at 0, two bars of them
BEATS = [float(t) for t in range(9)]
NEAR_S = 0.05


class Placed(unittest.TestCase):
    def test_a_time_takes_its_nearest_16th_in_its_bar(self):
        self.assertEqual(placed(1.5, BEATS)[0], 6)
        self.assertEqual(placed(2.0, BEATS)[0], 8)
        self.assertEqual(placed(4.0, BEATS)[0], 0)
        self.assertEqual(placed(0.3, BEATS)[0], 1)

    def test_a_time_just_before_a_downbeat_wraps_onto_it(self):
        self.assertEqual(placed(3.9, BEATS)[0], 0)

    def test_its_phase_and_beat_length_come_back_with_it(self):
        step, phase, span = placed(5.25, [0.0, 2.0, 4.0, 6.0, 8.0])
        self.assertEqual((step, phase, span), (10, 0.625, 2.0))

    def test_a_time_off_the_grid_has_no_place(self):
        self.assertIsNone(placed(-0.1, BEATS))
        self.assertIsNone(placed(8.0, BEATS))


class Placement(unittest.TestCase):
    def test_each_start_counts_on_a_beat_an_off_eighth_or_neither(self):
        made = placement([0.0, 1.5, 2.25, 3.0], BEATS, [], NEAR_S, 1)
        self.assertEqual((made["on_beat_share"], made["off_eighth_share"]), (0.5, 0.25))

    def test_the_steps_are_onsets_a_bar_on_each_16th(self):
        made = placement([0.0, 4.0, 1.5], BEATS, [], NEAR_S, 2)
        self.assertEqual(len(made["steps"]), STEPS)
        self.assertEqual((made["steps"][0], made["steps"][6], sum(made["steps"])), (1.0, 0.5, 1.5))

    def test_a_start_is_with_a_kick_only_nearer_than_near_s(self):
        made = placement([0.0, 1.0, 2.0, 3.0], BEATS, [0.01, 1.2, 2.049], NEAR_S, 1)
        self.assertEqual(made["with_kick_share"], 0.5)

    def test_no_kick_puts_no_start_with_one(self):
        self.assertEqual(placement([0.0, 1.0], BEATS, [], NEAR_S, 1)["with_kick_share"], 0.0)

    def test_starts_all_off_the_grid_have_no_shares(self):
        made = placement([-1.0, 9.0], BEATS, [], NEAR_S, 1)
        self.assertIsNone(made["on_beat_share"])
        self.assertIsNone(made["off_eighth_share"])



class Likeness(unittest.TestCase):
    def test_a_bar_matched_lag_bars_later_reads_whole(self):
        line = [(0, 60), (6, 62), (16 * STEPS, 60), (16 * STEPS + 6, 62)]
        self.assertEqual(likeness(line, 16), [1.0])

    def test_two_bars_share_the_onsets_both_hold_over_those_either_holds(self):
        line = [(0, 60), (6, 62), (8 * STEPS, 60), (8 * STEPS + 4, 64)]
        self.assertEqual(likeness(line, 8), [1 / 3])

    def test_a_pitch_or_a_16th_moved_is_another_onset(self):
        line = [(0, 60), (STEPS, 61), (2 * STEPS, 60), (2 * STEPS + 1, 60)]
        self.assertEqual(likeness(line, 1), [0.0, 0.0])
        self.assertEqual(likeness(line, 2), [0.5])

    def test_a_bar_with_none_lag_before_counts_for_nothing(self):
        self.assertEqual(likeness([(0, 60), (STEPS, 60)], 16), [])



class WithoutGrid(unittest.TestCase):
    ROW = {
        "powers": {"drums": 0.0005, "bass": 0.5, "other": 0.4995},
        "tempo": 79.5,
        "beats": 200,
        "downbeat": 1,
        "four_four": 0.9,
        "grid_from": "beat-this",
        "drums": {"kick": {"per_bar": 8.3}},
        "kick": {"decay_ms": 168.0},
        "snare": {"centroid_hz": 184.0},
        "hat": {"centroid_hz": 187.0},
        "harmony": {"qualities": {"maj": 3}, "roots": {"0": 3}, "beats_per_chord": 30.2, "cycle_bars": 1, "bar_tokens": {"0:maj": 3}},
        "bass": {"register": [40.0], "legato": 0.8, "notes_per_bar": 0.7, "on_beat_share": 0.3},
        "melody": {"register": [50.0, 54.5, 59.0], "pentatonic_share": 0.96, "onsets_per_bar": 6.9, "repeat_at_bars": {"4": 0.05}},
        "polyphony": 3.2,
    }

    def test_a_drums_stem_of_residue_holds_no_kit_and_a_beat_holds_one(self):
        self.assertFalse(has_kit(self.ROW["powers"]))
        self.assertTrue(has_kit({"drums": 0.1, "bass": 0.4, "other": 0.5}))

    def test_a_track_without_a_kit_counts_nothing_by_its_grid_or_its_hits(self):
        row = without_grid(self.ROW)
        for field in ("tempo", "beats", "downbeat", "four_four", "grid_from", "drums", "kick", "snare", "hat"):
            self.assertIsNone(row[field], field)
        self.assertIsNone(row["harmony"]["beats_per_chord"])
        self.assertIsNone(row["harmony"]["cycle_bars"])
        self.assertIsNone(row["harmony"]["bar_tokens"])
        self.assertIsNone(row["bass"]["notes_per_bar"])
        self.assertIsNone(row["bass"]["on_beat_share"])
        self.assertIsNone(row["melody"]["onsets_per_bar"])
        self.assertIsNone(row["melody"]["repeat_at_bars"])

    def test_what_needs_no_grid_stays(self):
        row = without_grid(self.ROW)
        self.assertEqual(row["harmony"]["qualities"], {"maj": 3})
        self.assertEqual(row["harmony"]["roots"], {"0": 3})
        self.assertEqual(row["bass"]["legato"], 0.8)
        self.assertEqual(row["melody"]["pentatonic_share"], 0.96)
        self.assertEqual(row["melody"]["register"], [50.0, 54.5, 59.0])
        self.assertEqual(row["polyphony"], 3.2)

    def test_a_missing_line_stays_missing(self):
        row = without_grid({**self.ROW, "bass": None, "melody": None})
        self.assertIsNone(row["bass"])
        self.assertIsNone(row["melody"])


if __name__ == "__main__":
    unittest.main()
