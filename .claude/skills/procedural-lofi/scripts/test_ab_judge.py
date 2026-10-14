#!/usr/bin/env python3
"""ab_judge.py's tests that need its venv: `$J/bin/python -I test_ab_judge.py`,
$J set up as SKILL.md's GENERATOR loop step 5 says."""

import sys
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from ab_judge import FLOOR_CALIBRATED_DBFS, FLOOR_CALIBRATED_LUFS, floored, mono_lufs

SR = 44100
NAME = "gen_day_empty_000.wav"


def take(seconds=10):
    return (0.1 * np.random.default_rng(1).standard_normal((seconds * SR, 2))).astype(np.float32)


class Floor(unittest.TestCase):
    def test_a_gain_scales_the_take_and_its_floor_together(self):
        x = take()
        gain = np.float32(10 ** (-0.5 / 20))
        quiet = gain * x
        want = gain * floored(x, NAME, mono_lufs(x, SR))
        # an absolute floor misses this by its own level times 1 − gain
        np.testing.assert_allclose(floored(quiet, NAME, mono_lufs(quiet, SR)), want, rtol=0, atol=1e-6)

    def test_a_take_at_the_calibration_loudness_gets_the_calibrated_floor(self):
        x = take()
        floor = floored(x, NAME, FLOOR_CALIBRATED_LUFS) - x
        rms_dbfs = 20 * np.log10(np.sqrt(np.mean(floor.astype(np.float64) ** 2)))
        self.assertAlmostEqual(rms_dbfs, FLOOR_CALIBRATED_DBFS, places=1)

    def test_both_sides_of_a_pair_draw_the_same_floor(self):
        x = take()
        self.assertTrue(np.array_equal(floored(x, NAME, -20.0), floored(x, NAME, -20.0)))
        self.assertFalse(np.array_equal(floored(x, NAME, -20.0), floored(x, "gen_day_empty_001.wav", -20.0)))


if __name__ == "__main__":
    unittest.main()
