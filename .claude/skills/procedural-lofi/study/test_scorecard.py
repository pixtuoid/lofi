#!/usr/bin/env python3
"""The scorecard's tests: `python3 -I test_scorecard.py`, with no venv."""

import json
import sys
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from scorecard import LAYERS, STUDY, TARGETS, by_track, card, outside_key, path, quartiles, row, targets


class Scorecard(unittest.TestCase):
    def test_a_median_inside_the_range_holds_and_one_outside_is_off(self):
        self.assertEqual(row([1, 2, 3], [2, 2, 3]), (2, True, 0))
        self.assertEqual(row([1, 2, 3], [5]), (5, False, 1.5))

    def test_a_flat_range_reads_a_gap_rather_than_dividing_by_nothing(self):
        self.assertEqual(row([2, 2, 2], [3]), (3, False, 1))

    def test_quartiles_come_from_the_values_in_order(self):
        self.assertEqual(quartiles([5, 1, 3, 2, 4]), (2, 3, 4))

    def test_a_row_the_takes_never_measured_says_so(self):
        target = {"2 rhythm": {"tempo bpm": [73, 77, 81]}}
        lines = list(card(target, {STUDY: [{}]}))
        self.assertEqual(lines, [("2 rhythm", "tempo bpm", [73, 77, 81], None)])

    def test_a_chord_root_outside_a_tracks_mode_counts_against_that_mode(self):
        harmony = {"roots": {"0": 3, "3": 1}}
        self.assertEqual(outside_key({"mode": "major", "harmony": harmony}), 25)
        self.assertEqual(outside_key({"mode": "minor", "harmony": harmony}), 0)

    def test_a_missing_measure_reads_none_not_zero(self):
        self.assertIsNone(path("drums", "kick")({"drums": {}}))

    def test_stem_distances_turn_to_rows_by_track(self):
        self.assertEqual(by_track({"bass": {"a": 1}, "other": {"a": 2}}), {"a": {"bass": 1, "other": 2}})

    def test_targets_keep_each_rows_quartiles_only(self):
        rows = [{"tempo": t} for t in (70, 75, 80)]
        made = targets({STUDY: rows, "timbre": [], "bass": [], "stemfad": []})
        self.assertEqual(made["2 rhythm"], {"tempo bpm": [70, 75, 80]})

    def test_the_committed_targets_cover_every_row_and_name_no_track(self):
        committed = json.loads(TARGETS.read_text())
        self.assertEqual(
            {layer: sorted(rows) for layer, rows in committed.items()},
            {layer: sorted(label for label, _, _ in rows) for layer, rows in LAYERS},
        )
        for rows in committed.values():
            self.assertTrue(all(low <= mid <= high for low, mid, high in rows.values()))


if __name__ == "__main__":
    unittest.main()
