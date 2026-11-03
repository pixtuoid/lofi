#!/usr/bin/env python3
"""The beat grid's tests: `python3 -I test_grid.py`, with no venv."""

import sys
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

from grid import STEPS, likeness, placed, placement

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
    # a bar 16 steps long: onsets (16th from the first downbeat, pitch)
    def test_a_bar_matched_lag_bars_later_reads_whole(self):
        line = [(0, 60), (6, 62), (16 * 16, 60), (16 * 16 + 6, 62)]
        self.assertEqual(likeness(line, 16), [1.0])

    def test_two_bars_share_the_onsets_both_hold_over_those_either_holds(self):
        line = [(0, 60), (6, 62), (8 * 16, 60), (8 * 16 + 4, 64)]
        self.assertEqual(likeness(line, 8), [1 / 3])

    def test_a_pitch_or_a_16th_moved_is_another_onset(self):
        line = [(0, 60), (16, 61), (32, 60), (32 + 1, 60)]
        self.assertEqual(likeness(line, 1), [0.0, 0.0])
        self.assertEqual(likeness(line, 2), [0.5])

    def test_a_bar_with_none_lag_before_counts_for_nothing(self):
        self.assertEqual(likeness([(0, 60), (16, 60)], 16), [])


if __name__ == "__main__":
    unittest.main()
