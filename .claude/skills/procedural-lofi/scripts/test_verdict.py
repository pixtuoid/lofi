#!/usr/bin/env python3
"""The verdict gate's tests: `python3 -I test_verdict.py`, with no venv."""

import sys
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from verdict import ALPHA, MIN_UNTIED, batch_verdict, exit_message, group_verdict, sign_test


def losses(n):
    return [-1.0] * n


class SignTest(unittest.TestCase):
    def test_p_is_the_exact_one_sided_binomial_tail(self):
        # scipy.stats.binomtest(k, n, 0.5, alternative="greater").pvalue, as
        # the judge used before this module: 3 of 12, 9 of 12, 4 of 4, 5 of 5
        for gains, p in [
            ([1.0] * 9 + [-1.0] * 3, 0.980712890625),
            ([1.0] * 3 + [-1.0] * 9, 0.072998046875),
            (losses(4), 0.0625),
            (losses(5), 0.03125),
        ]:
            self.assertAlmostEqual(sign_test(gains)["p"], p, places=12)

    def test_ties_are_dropped(self):
        t = sign_test([0.0, 0.0, 1.0, -1.0])
        self.assertEqual((t["wins"], t["losses"], t["ties"]), (1, 1, 2))
        # five losses out of five untied pairs, however many tie
        t = sign_test([0.0] * 7 + losses(5))
        self.assertEqual(t["p"], 0.03125)
        self.assertTrue(t["significant"])

    def test_min_untied_is_the_fewest_pairs_that_can_reject(self):
        self.assertFalse(sign_test(losses(MIN_UNTIED - 1))["p"] < ALPHA)
        self.assertTrue(sign_test(losses(MIN_UNTIED))["p"] < ALPHA)


class Verdicts(unittest.TestCase):
    def test_too_few_untied_pairs_is_insufficient_never_pass(self):
        self.assertEqual(group_verdict({"t": sign_test(losses(MIN_UNTIED - 1))}), "INSUFFICIENT")
        self.assertEqual(group_verdict({"t": sign_test([0.0] * 12)}), "INSUFFICIENT")

    def test_all_losses_at_the_floor_regress(self):
        self.assertEqual(group_verdict({"t": sign_test(losses(MIN_UNTIED))}), "REGRESSION")

    def test_one_rejecting_test_regresses_the_group(self):
        tests = {"a": sign_test([1.0] * 12), "b": sign_test(losses(12))}
        self.assertEqual(group_verdict(tests), "REGRESSION")
        tests = {"a": sign_test(losses(MIN_UNTIED)), "b": sign_test(losses(MIN_UNTIED - 1))}
        self.assertEqual(group_verdict(tests), "REGRESSION")

    def test_a_detected_regression_outranks_too_few_pairs(self):
        self.assertEqual(batch_verdict(["PASS", "INSUFFICIENT", "REGRESSION"]), "REGRESSION")
        self.assertEqual(batch_verdict(["PASS", "INSUFFICIENT"]), "INSUFFICIENT")
        self.assertEqual(batch_verdict(["PASS", "PASS"]), "PASS")

    def test_only_pass_exits_zero(self):
        self.assertIsNone(exit_message("PASS"))
        self.assertIsNotNone(exit_message("REGRESSION"))
        self.assertIsNotNone(exit_message("INSUFFICIENT"))


if __name__ == "__main__":
    unittest.main()
