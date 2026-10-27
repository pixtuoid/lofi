#!/usr/bin/env python3
"""ab_judge.py's tests that need its venv: `$J/bin/python -I test_ab_judge.py`,
$J set up as SKILL.md's GENERATOR loop step 5 says."""

import contextlib
import errno
import json
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from ab_judge import (
    FLOOR_CALIBRATED_DBFS,
    FLOOR_CALIBRATED_LUFS,
    ANCHOR_RIPPLE_DB,
    SCREEN_SHARE,
    WINDOW_S,
    anchor_filter,
    cached_quietest,
    cached_screen,
    environment,
    floored,
    mono_lufs,
    quietest_file,
    screened,
    screen_file,
    screened_out,
    smoke,
    stats_file,
    windows,
)
from scipy import signal
from verdict import JUDGES

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


class Windows(unittest.TestCase):
    def test_two_rates_cut_a_take_into_as_many_windows_weighted_by_its_length(self):
        # a window's edge either side, and a take under a sample of one window
        for n in (WINDOW_S * SR - 1, WINDOW_S * SR, WINDOW_S * SR + 1, 3 * WINDOW_S * SR + 7, 1):
            x = np.zeros((n, 2), dtype=np.float32)
            (high, weights), (low, _) = windows(x, SR, 48000), windows(x, SR, 24000)
            self.assertEqual(len(high), len(low), n)
            self.assertAlmostEqual(float(weights.sum()), n / SR / WINDOW_S, delta=1 / 24000, msg=n)


class Anchors(unittest.TestCase):
    def response_db(self, sos, hz):
        _, h = signal.sosfreqz(sos, worN=np.atleast_1d(hz), fs=SR)
        return 20 * np.log10(np.abs(h))

    def test_the_standard_anchor_meets_bs1534s_filter(self):
        sos = anchor_filter(SR, 3500)
        passband = self.response_db(sos, np.linspace(20, 3500, 400))
        self.assertLessEqual(passband.max() - passband.min(), ANCHOR_RIPPLE_DB + 1e-6)
        self.assertLessEqual(self.response_db(sos, 4000)[0], -25)
        self.assertLessEqual(self.response_db(sos, 4500)[0], -50)


class Dull:
    """A judge that hears only what an anchor keeps."""

    def score(self, x, sr):
        return float(np.sum(signal.sosfilt(signal.butter(4, 1000, "lowpass", fs=sr, output="sos"), x[:, 0]) ** 2))


class Bright:
    """A judge that hears only what an anchor cuts."""

    def score(self, x, sr):
        return float(np.sum(signal.sosfilt(signal.butter(4, 8000, "highpass", fs=sr, output="sos"), x[:, 0]) ** 2))


class Screening(unittest.TestCase):
    def test_a_judge_that_rates_every_anchor_unimpaired_is_screened_out_and_one_that_hears_them_is_kept(self):
        items = [(gain * take(1), SR) for gain in (1.0, 2.0, 4.0)]
        screen = screened({"dull": Dull(), "bright": Bright()}, items)
        self.assertEqual(set(screen["dull"]["anchors"].values()), {1.0})
        self.assertEqual(set(screen["bright"]["anchors"].values()), {0.0})
        # so only the anchors screen it out
        self.assertGreater(screen["dull"]["scale"], 0)
        self.assertEqual(screened_out(screen), ["dull"])

    def test_only_the_mid_range_anchor_past_bs1534s_share_screens_out(self):
        screen = {
            "at": {"anchors": {3500: 0.0, 7000: SCREEN_SHARE}, "scale": 1.0},
            "low": {"anchors": {3500: 0.5, 7000: 0.0}, "scale": 1.0},
            "past": {"anchors": {3500: 0.0, 7000: 0.2}, "scale": 1.0},
        }
        self.assertEqual(screened_out(screen), ["past"])

    def test_a_judge_that_scores_every_reference_alike_is_screened_out(self):
        screen = {"flat": {"anchors": {3500: 0.0, 7000: 0.0}, "scale": 0.0}}
        self.assertEqual(screened_out(screen), ["flat"])

    def test_a_judges_scale_is_its_scores_spread_over_the_items(self):
        rng = np.random.default_rng(3)
        items = [((0.1 * rng.standard_normal((SR, 2))).astype(np.float32) * gain, SR) for gain in (1.0, 2.0, 4.0)]
        screen = screened({"bright": Bright()}, items)
        energies = [Bright().score(x, sr) for x, sr in items]
        self.assertAlmostEqual(screen["bright"]["scale"], float(np.std(energies, ddof=1)), places=6)


class CorpusCache(unittest.TestCase):
    SHA = "c" * 64
    QUIETEST = -20.0

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.caches = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def cache_quietest(self):
        quietest_file(self.caches, self.SHA).write_text(json.dumps({"lufs": self.QUIETEST}) + "\n")

    def refused(self, takes_lufs):
        """`cached_quietest`'s exit message over a cache that refuses every write."""
        with mock.patch("tempfile.TemporaryFile", side_effect=OSError(errno.EROFS, "Read-only file system")):
            with self.assertRaises(SystemExit) as refused:
                cached_quietest(self.caches, self.SHA, takes_lufs)
        return str(refused.exception.code)

    def test_a_read_only_cache_without_the_corpus_loudness_asks_for_provision(self):
        self.assertIn("run provision", self.refused(self.QUIETEST + 1))

    def test_a_read_only_cache_without_the_corpus_statistics_asks_for_provision(self):
        self.cache_quietest()
        self.assertIn("run provision", self.refused(self.QUIETEST + 1))

    def test_a_read_only_cache_refuses_a_batch_quieter_than_the_corpus_locally(self):
        self.cache_quietest()
        stats_file(self.caches, self.SHA, self.QUIETEST).write_bytes(b"")
        self.assertIn("quieter than the corpus's quietest reference, so judge it locally", self.refused(self.QUIETEST - 1))

    def test_a_cache_holding_the_batch_level_is_read_without_a_probe(self):
        self.cache_quietest()
        stats_file(self.caches, self.SHA, self.QUIETEST).write_bytes(b"")
        with mock.patch("tempfile.TemporaryFile") as probe:
            self.assertEqual(cached_quietest(self.caches, self.SHA, self.QUIETEST + 1), self.QUIETEST)
        probe.assert_not_called()

    def test_a_writable_cache_is_let_through_and_keeps_no_probe(self):
        self.assertIsNone(cached_quietest(self.caches, self.SHA, self.QUIETEST))
        self.assertEqual(list(self.caches.iterdir()), [])

    def test_a_read_only_cache_without_the_panels_screening_asks_for_provision(self):
        fad = mock.Mock(manifest_sha=self.SHA)
        with mock.patch("tempfile.TemporaryFile", side_effect=OSError(errno.EROFS, "Read-only file system")):
            with self.assertRaises(SystemExit) as refused:
                cached_screen(self.caches, fad, {})
        self.assertIn("run provision", str(refused.exception.code))

    def test_a_cached_screening_reads_back_as_built(self):
        screen = {"deaf": {"anchors": {3500: 1.0, 7000: 1.0}, "scale": 0.5}}
        screen_file(self.caches, self.SHA).write_text(json.dumps(screen))
        self.assertEqual(cached_screen(self.caches, mock.Mock(manifest_sha=self.SHA), {}), screen)

    def test_a_cache_not_yet_made_is_made_and_keeps_no_probe(self):
        caches = self.caches / "fad-cache"
        self.assertIsNone(cached_quietest(caches, self.SHA, self.QUIETEST))
        self.assertEqual(list(caches.iterdir()), [])

    def test_no_cache_built_under_other_packages_is_read(self):
        def files():
            return (
                quietest_file(self.caches, self.SHA),
                stats_file(self.caches, self.SHA, self.QUIETEST),
                screen_file(self.caches, self.SHA),
            )

        def installed(version):
            torch = mock.Mock(version=version, metadata={"Name": "Torch"})
            return mock.patch("ab_judge.distributions", return_value=[torch])

        with installed("1"):
            self.assertIn("torch==1", environment())
            here = files()
        with installed("2"):
            elsewhere = files()
        self.assertTrue(all(a != b for a, b in zip(here, elsewhere, strict=True)), elsewhere)



class Smoke(unittest.TestCase):
    RATE = 48000

    def smoke(self, audiobox=1.0, distance=None):
        """`smoke` over fake models: Audiobox scoring `audiobox`, TuneJury
        0.5, and fadtk's distance `distance` (its own when None)."""
        frames = np.random.default_rng(0).standard_normal((8, 4))
        model = mock.Mock(sr=self.RATE, get_embedding=mock.Mock(return_value=frames))
        patches = [
            mock.patch("ab_judge.clap", return_value=model),
            mock.patch("ab_judge.judge_device", return_value="cpu"),
            mock.patch("ab_judge.Audiobox", return_value=mock.Mock(score=mock.Mock(return_value=audiobox))),
            mock.patch("ab_judge.TuneJury", return_value=mock.Mock(score=mock.Mock(return_value=0.5))),
        ]
        if distance is not None:
            patches.append(mock.patch("ab_judge.calc_frechet_distance", return_value=distance))
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            return smoke(np.zeros((self.RATE, 2), dtype=np.float32), self.RATE, 0.5)

    def test_every_judge_scores_the_smoke_take(self):
        self.assertEqual(set(self.smoke()), set(JUDGES))

    def test_a_panel_score_that_isnt_finite_fails_the_smoke(self):
        with self.assertRaises(ValueError):
            self.smoke(audiobox=float("nan"))

    def test_a_distance_that_isnt_finite_fails_the_smoke(self):
        with self.assertRaises(ValueError):
            self.smoke(distance=float("nan"))


if __name__ == "__main__":
    unittest.main()
