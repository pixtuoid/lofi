#!/usr/bin/env python3
"""The verdict gate's tests: `python3 -I test_verdict.py`, with no venv."""

import re
import sys
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from verdict import ALPHA, GROUPS, JUDGES, MIN_UNTIED, exit_message, grouped, judged, min_untied, sign_test

# where the judge batch's matrix and its groups are spelled in Rust
BATCH_RS = HERE.parents[3] / "examples" / "common" / "batch.rs"


def losses(n):
    return [-1.0] * n


def pair(mood, gain=None):
    """A pair every judge scores `gain` on, or an identical one."""
    if gain is None:
        return {"mood": mood, "identical": True}
    return {"mood": mood, "identical": False, "gains": dict.fromkeys(JUDGES, gain)}


def pairs(mood, gains):
    return [pair(mood, g) for g in gains]


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
            self.assertAlmostEqual(sign_test(gains, ALPHA)["p"], p, places=12)

    def test_ties_are_dropped(self):
        t = sign_test([0.0, 0.0, 1.0, -1.0], ALPHA)
        self.assertEqual((t["wins"], t["losses"], t["ties"]), (1, 1, 2))
        # five losses out of five untied pairs, however many tie
        t = sign_test([0.0] * 7 + losses(5), ALPHA)
        self.assertEqual(t["p"], 0.03125)
        self.assertTrue(t["significant"])

    def test_min_untied_is_the_fewest_pairs_that_can_reject(self):
        for alpha in (ALPHA, ALPHA / 3, ALPHA / 6):
            n = min_untied(alpha)
            self.assertFalse(sign_test(losses(n - 1), alpha)["significant"])
            self.assertTrue(sign_test(losses(n), alpha)["significant"])

    def test_min_untied_is_the_fewest_at_every_gating_test(self):
        self.assertEqual(MIN_UNTIED, min_untied(ALPHA / (len(JUDGES) * len(GROUPS))))


class Grouping(unittest.TestCase):
    def test_moods_then_all_with_identical_pairs_counted_apart(self):
        batch = [pair("night", 1.0), pair("day"), pair("day", -1.0), pair("night")]
        groups = grouped(batch)
        self.assertEqual(list(groups), ["day", "night", "all"])
        self.assertEqual([len(g["changed"]) for g in groups.values()], [1, 1, 2])
        self.assertEqual([g["identical"] for g in groups.values()], [1, 1, 2])


class Verdicts(unittest.TestCase):
    def test_a_group_of_only_identical_pairs_is_unchanged_and_ranks_below_pass(self):
        batch = [pair("day")] * 12 + pairs("night", [1.0] * MIN_UNTIED)
        result = judged(batch)
        self.assertEqual(result["groups"]["day"]["verdict"], "UNCHANGED")
        self.assertEqual(result["groups"]["night"]["verdict"], "PASS")
        self.assertEqual(result["overall"], "PASS")

    def test_a_batch_of_only_identical_pairs_is_unchanged_and_exits_zero(self):
        result = judged([pair("day")] * 3 + [pair("night")] * 3)
        self.assertEqual(result["overall"], "UNCHANGED")
        self.assertIsNone(exit_message(result))

    def test_alpha_splits_over_the_moods_that_changed(self):
        # one changed mood tests 3 judges, so its fewest is the 3-way split's
        one = min_untied(ALPHA / len(JUDGES))
        self.assertEqual(judged(pairs("day", losses(one)))["overall"], "REGRESSION")
        self.assertEqual(judged(pairs("day", losses(one - 1)))["overall"], "INSUFFICIENT")
        # two changed moods test 6, which a mood's one-fewer losses can't reach
        both = pairs("day", losses(one)) + pairs("night", [1.0] * MIN_UNTIED)
        self.assertEqual(judged(both)["groups"]["day"]["verdict"], "INSUFFICIENT")
        both = pairs("day", losses(MIN_UNTIED)) + pairs("night", [1.0] * MIN_UNTIED)
        self.assertEqual(judged(both)["overall"], "REGRESSION")

    def test_the_pooled_group_is_reported_but_never_gates(self):
        # six losses of seven in each mood: not enough for either mood, enough
        # pooled
        mood = losses(6) + [1.0]
        result = judged(pairs("day", mood) + pairs("night", mood))
        self.assertEqual(result["groups"]["all"]["verdict"], "REGRESSION")
        self.assertEqual(result["overall"], "PASS")

    def test_too_few_untied_pairs_is_insufficient_never_pass(self):
        self.assertEqual(judged(pairs("day", [0.0] * 12))["overall"], "INSUFFICIENT")

    def test_a_detected_regression_outranks_too_few_pairs(self):
        # two changed groups test six
        two = min_untied(ALPHA / (2 * len(JUDGES)))
        batch = pairs("day", losses(two)) + pairs("night", [1.0] * (two - 1))
        result = judged(batch)
        self.assertEqual(result["groups"]["night"]["verdict"], "INSUFFICIENT")
        self.assertEqual(result["overall"], "REGRESSION")

    def test_only_regression_and_insufficient_exit_non_zero(self):
        self.assertIsNotNone(exit_message(judged(pairs("day", losses(MIN_UNTIED)))))
        self.assertIsNotNone(exit_message(judged(pairs("day", [1.0]))))
        self.assertIsNone(exit_message(judged(pairs("day", [1.0] * MIN_UNTIED))))


class RustCopies(unittest.TestCase):
    """The values `examples/common/batch.rs` spells for the judge batch."""

    def test_the_judge_batch_is_built_to_this_min_untied(self):
        rust = re.search(r"const JUDGE_MIN_UNTIED: usize = (\d+);", BATCH_RS.read_text())
        self.assertIsNotNone(rust, f"no JUDGE_MIN_UNTIED in {BATCH_RS}")
        self.assertEqual(int(rust[1]), MIN_UNTIED)

    def test_the_groups_are_the_judge_batchs_moods_then_sweeps(self):
        text = BATCH_RS.read_text()
        moods = re.search(r"const MOODS: \[\(&str, f32\); \d+\] = \[(.*)\];", text)
        sweeps = re.search(r"const JUDGE_SWEEPS: \[Sweep; \d+\] = \[(.*)\];", text)
        self.assertIsNotNone(moods, f"no MOODS in {BATCH_RS}")
        self.assertIsNotNone(sweeps, f"no JUDGE_SWEEPS in {BATCH_RS}")
        names = re.findall(r'"([a-z]+)"', moods[1])
        for variant in re.findall(r"Sweep::(\w+)", sweeps[1]):
            name = re.search(rf'Sweep::{variant} => "([a-z]+)"', text)
            self.assertIsNotNone(name, f"Sweep::{variant} has no name in {BATCH_RS}")
            names.append(name[1])
        self.assertEqual(tuple(names), GROUPS)


if __name__ == "__main__":
    unittest.main()
