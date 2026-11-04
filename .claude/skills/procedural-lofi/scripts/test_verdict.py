#!/usr/bin/env python3
"""The verdict gate's tests: `python3 -I test_verdict.py`, with no venv."""

import re
import sys
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from statistics import stdev

from verdict import (
    ALPHA,
    FAILING,
    GATES,
    GROUPS,
    GUARD,
    JUDGES,
    MIN_UNTIED,
    PANEL,
    RANKS,
    SMALL_EFFECT,
    STYLES,
    exit_message,
    grouped,
    judged,
    margin,
    min_untied,
    sign_test,
    pooled,
)

# where the judge batch's matrix and its groups are spelled in Rust
BATCH_RS = HERE.parents[3] / "examples" / "common" / "batch.rs"
# where the PR comment reads the verdicts back
JUDGE_YML = HERE.parents[3] / ".github" / "workflows" / "judge.yml"


# held-out FADs whose spread sets a margin far under a unit gain
HELD_ONE = [0.5, 0.6]
# each style's, alike
HELD = dict.fromkeys(STYLES, HELD_ONE)
# each panel judge in its own unit, in every style
SCALES = {style: dict.fromkeys(PANEL, 1.0) for style in STYLES}


def losses(n):
    return [-1.0] * n


def pair(group, gain=None, style=None):
    """A pair every judge scores `gain` on, or an identical one, of `style`,
    or of the style `group` names."""
    if gain is None:
        return {"group": group, "style": style or group, "identical": True}
    return {"group": group, "style": style or group, "identical": False, "gains": dict.fromkeys(JUDGES, gain)}


def pairs(group, gains):
    return [pair(group, g) for g in gains]


def judge_pair(group, gains, style=None):
    """A pair each judge scores its own gain on, in `JUDGES`' order."""
    return {"group": group, "style": style or group, "identical": False, "gains": dict(zip(JUDGES, gains, strict=True))}


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

    def test_favours_cand_needs_min_untied_wins_at_its_alpha(self):
        for alpha in (ALPHA, ALPHA / 3, ALPHA / 6):
            n = min_untied(alpha)
            self.assertFalse(sign_test([1.0] * (n - 1), alpha)["favours_cand"])
            self.assertTrue(sign_test([1.0] * n, alpha)["favours_cand"])


class Grouping(unittest.TestCase):
    def test_groups_then_all_with_identical_pairs_counted_apart(self):
        batch = [pair("sleep", 1.0), pair("hiphop"), pair("hiphop", -1.0), pair("sleep")]
        groups = grouped(batch)
        self.assertEqual(list(groups), ["hiphop", "sleep", "all"])
        self.assertEqual([len(g["changed"]) for g in groups.values()], [1, 1, 2])
        self.assertEqual([g["identical"] for g in groups.values()], [1, 1, 2])


class Verdicts(unittest.TestCase):
    def test_a_group_of_only_identical_pairs_is_unchanged_and_ranks_below_pass(self):
        batch = [pair("hiphop")] * 12 + pairs("sleep", [1.0] * MIN_UNTIED)
        result = judged(batch, HELD, SCALES)
        self.assertEqual(result["groups"]["hiphop"]["verdict"], "UNCHANGED")
        self.assertEqual(result["groups"]["sleep"]["verdict"], "PASS")
        self.assertEqual(result["overall"], "PASS")

    def test_a_batch_of_only_identical_pairs_is_unchanged_and_exits_zero(self):
        result = judged([pair("hiphop")] * 3 + [pair("sleep")] * 3, HELD, SCALES)
        self.assertEqual(result["overall"], "UNCHANGED")
        self.assertIsNone(exit_message(result))

    def test_alpha_splits_over_the_groups_that_changed(self):
        # one changed group tests its 2 gates, so its fewest is the 2-way split's
        one = min_untied(ALPHA / len(GATES))
        self.assertEqual(judged(pairs("hiphop", losses(one)), HELD, SCALES)["overall"], "REGRESSION")
        self.assertEqual(judged(pairs("hiphop", losses(one - 1)), HELD, SCALES)["overall"], "INSUFFICIENT")
        # two changed groups test 4, which a group's one-fewer losses can't reach
        both = pairs("hiphop", losses(one)) + pairs("sleep", [1.0] * MIN_UNTIED)
        self.assertEqual(judged(both, HELD, SCALES)["groups"]["hiphop"]["verdict"], "INSUFFICIENT")
        both = pairs("hiphop", losses(MIN_UNTIED)) + pairs("sleep", [1.0] * MIN_UNTIED)
        self.assertEqual(judged(both, HELD, SCALES)["overall"], "REGRESSION")

    def test_the_pooled_group_is_reported_but_never_gates(self):
        # six losses of seven in each group: not enough for either group,
        # enough pooled
        each = losses(6) + [1.0]
        result = judged(pairs("hiphop", each) + pairs("sleep", each), HELD, SCALES)
        self.assertEqual(result["groups"]["all"]["verdict"], "REGRESSION")
        self.assertEqual(result["overall"], "PASS")

    def test_too_few_untied_pairs_is_insufficient_never_pass(self):
        self.assertEqual(judged(pairs("hiphop", [0.0] * 12), HELD, SCALES)["overall"], "INSUFFICIENT")

    def test_a_detected_regression_outranks_too_few_pairs(self):
        # two changed groups test four
        two = min_untied(ALPHA / (2 * len(GATES)))
        batch = pairs("hiphop", losses(two)) + pairs("sleep", [1.0] * (two - 1))
        result = judged(batch, HELD, SCALES)
        self.assertEqual(result["groups"]["sleep"]["verdict"], "INSUFFICIENT")
        self.assertEqual(result["overall"], "REGRESSION")

    def test_only_regression_and_insufficient_exit_non_zero(self):
        self.assertIsNotNone(exit_message(judged(pairs("hiphop", losses(MIN_UNTIED)), HELD, SCALES)))
        self.assertIsNotNone(exit_message(judged(pairs("hiphop", [1.0]), HELD, SCALES)))
        self.assertIsNone(exit_message(judged(pairs("hiphop", [1.0] * MIN_UNTIED), HELD, SCALES)))


class Panel(unittest.TestCase):
    N = MIN_UNTIED * 2

    def test_judges_far_for_outweigh_one_near_against(self):
        for against in PANEL:
            gains = tuple(-0.5 if j == against else 2.0 if j in PANEL else 0.0 for j in JUDGES)
            result = judged([judge_pair("hiphop", gains) for _ in range(self.N)], HELD, SCALES)
            self.assertEqual(result["groups"]["hiphop"]["tests"]["panel"]["wins"], self.N, against)
            self.assertEqual(result["groups"]["hiphop"]["verdict"], "PASS", against)

    def test_of_three_judges_one_far_against_moves_no_majority(self):
        self.assertGreater(pooled({"a": 1.0, "b": 1.0, "c": -5.0}, dict.fromkeys("abc", 1.0)), 0)

    def test_two_judges_weigh_each_in_its_own_spread(self):
        gains = {"a": 1.0, "b": -2.0}
        self.assertLess(pooled(gains, {"a": 1.0, "b": 1.0}), 0)
        # b's scores spread four times wider, so its -2 is a half
        self.assertGreater(pooled(gains, {"a": 1.0, "b": 4.0}), 0)

    def test_a_judge_outweighed_on_every_pair_is_a_dissent_that_passes(self):
        gains = tuple(-0.5 if j == "tunejury" else 2.0 if j in PANEL else 0.0 for j in JUDGES)
        result = judged([judge_pair("hiphop", gains) for _ in range(self.N)], HELD, SCALES)
        self.assertEqual(result["overall"], "PASS")
        self.assertEqual(result["dissent"], {"hiphop": ["tunejury"]})

    def test_a_regression_names_no_dissent(self):
        result = judged(pairs("hiphop", losses(self.N)), HELD, SCALES)
        self.assertEqual(result["overall"], "REGRESSION")
        self.assertEqual(result["dissent"], {})

    def test_a_group_names_every_judge_that_dissents(self):
        gains = tuple(-0.5 if j == "tunejury" else -margin(HELD_ONE) / 2 if j == GUARD else 2.0 for j in JUDGES)
        result = judged([judge_pair("hiphop", gains) for _ in range(self.N)], HELD, SCALES)
        self.assertEqual(result["overall"], "PASS")
        self.assertEqual(result["dissent"], {"hiphop": ["tunejury", GUARD]})

    def test_a_group_runs_every_gate_its_alpha_counts(self):
        result = judged(pairs("hiphop", [1.0] * self.N), HELD, SCALES)
        self.assertEqual(set(result["groups"]["hiphop"]["tests"]), set(GATES))


class Guard(unittest.TestCase):
    N = MIN_UNTIED * 2

    def guarded(self, fad_gain):
        """The verdict on a group the panel favours on every pair, its FAD
        gain `fad_gain` on each."""
        return judged([judge_pair("hiphop", (*[1.0] * len(PANEL), fad_gain)) for _ in range(self.N)], HELD, SCALES)

    def test_the_margin_is_a_small_effect_in_the_held_out_spread(self):
        self.assertAlmostEqual(margin(HELD_ONE), SMALL_EFFECT * stdev(HELD_ONE), places=12)

    def test_a_move_under_the_margin_passes_and_is_a_dissent(self):
        result = self.guarded(-margin(HELD_ONE) / 2)
        self.assertEqual(result["overall"], "PASS")
        self.assertEqual(result["dissent"], {"hiphop": [GUARD]})

    def test_a_move_past_the_margin_is_a_regression_however_the_panel_votes(self):
        self.assertEqual(self.guarded(-2 * margin(HELD_ONE))["overall"], "REGRESSION")

    def test_a_move_toward_the_corpus_passes(self):
        result = self.guarded(margin(HELD_ONE))
        self.assertEqual(result["overall"], "PASS")
        self.assertEqual(result["dissent"], {})


class PerStyle(unittest.TestCase):
    """A group may hold several styles' takes, each judged in its own
    corpus's units."""

    N = MIN_UNTIED * 2

    def test_a_pairs_guard_reads_its_own_styles_margin(self):
        # sleep's references spread ten times hiphop's, so a FAD loss between
        # their margins is past hiphop's and under sleep's
        held = {"hiphop": HELD_ONE, "sleep": [10 * h for h in HELD_ONE]}
        loss = -2 * margin(HELD_ONE)
        for style, verdict in (("hiphop", "REGRESSION"), ("sleep", "PASS")):
            batch = [judge_pair("twilight", (*[1.0] * len(PANEL), loss), style) for _ in range(self.N)]
            self.assertEqual(judged(batch, held, SCALES)["overall"], verdict, style)

    def test_a_pairs_panel_reads_its_own_styles_scales(self):
        # tunejury's -2 is a loss in hiphop's units and a half in sleep's,
        # whose tunejury scores spread four times wider
        scales = {"hiphop": dict.fromkeys(PANEL, 1.0), "sleep": {**dict.fromkeys(PANEL, 1.0), "tunejury": 4.0}}
        gains = tuple(-2.0 if j == "tunejury" else 1.0 if j in PANEL else 0.0 for j in JUDGES)
        for style, verdict in (("hiphop", "REGRESSION"), ("sleep", "PASS")):
            batch = [judge_pair("twilight", gains, style) for _ in range(self.N)]
            self.assertEqual(judged(batch, HELD, scales)["overall"], verdict, style)

    def test_a_judge_its_corpus_leaves_unseated_weighs_in_on_none_of_its_pairs(self):
        # sleep's corpus seats tunejury alone, so audiobox scored no sleep take
        scales = {**SCALES, "sleep": {"tunejury": 1.0}}
        seated = {"group": "sleep", "style": "sleep", "identical": False, "gains": {"tunejury": 1.0, GUARD: 0.0}}
        result = judged([seated] * self.N, HELD, scales)
        self.assertEqual(result["overall"], "PASS")
        audiobox = result["groups"]["sleep"]["judges"]["audiobox_ce"]
        self.assertEqual((audiobox["wins"], audiobox["losses"], audiobox["ties"]), (0, 0, 0))
        self.assertEqual(result["dissent"], {})


class RustCopies(unittest.TestCase):
    """The values `examples/common/batch.rs` spells for the judge batch."""

    def test_the_judge_batch_is_built_to_this_min_untied(self):
        rust = re.search(r"const JUDGE_MIN_UNTIED: usize = (\d+);", BATCH_RS.read_text())
        self.assertIsNotNone(rust, f"no JUDGE_MIN_UNTIED in {BATCH_RS}")
        self.assertEqual(int(rust[1]), MIN_UNTIED)

    def test_the_styles_then_the_sweeps_are_the_judge_batchs_groups(self):
        text = BATCH_RS.read_text()
        styles = re.search(r"const STYLES: \[\(&str, f32\); \d+\] = \[(.*)\];", text)
        sweeps = re.search(r"const JUDGE_SWEEPS: \[Sweep; \d+\] = \[(.*)\];", text)
        self.assertIsNotNone(styles, f"no STYLES in {BATCH_RS}")
        self.assertIsNotNone(sweeps, f"no JUDGE_SWEEPS in {BATCH_RS}")
        self.assertEqual(tuple(re.findall(r'"([a-z]+)"', styles[1])), STYLES)
        names = list(STYLES)
        for variant in re.findall(r"Sweep::(\w+)", sweeps[1]):
            name = re.search(rf'Sweep::{variant} => "([a-z]+)"', text)
            self.assertIsNotNone(name, f"Sweep::{variant} has no name in {BATCH_RS}")
            names.append(name[1])
        self.assertEqual(tuple(names), GROUPS)


class WorkflowCopies(unittest.TestCase):
    """The verdicts `.github/workflows/judge.yml` spells in its shell."""

    def test_the_comment_takes_every_verdict_and_fails_the_failing_ones(self):
        text = JUDGE_YML.read_text()
        taken = re.search(r"^\s*((?:[A-Z]+ \| )+[A-Z]+)\)$", text, re.M)
        failed = re.search(r"^\s*((?:[A-Z]+ \| )+)\"infra failure\"\) exit 1", text, re.M)
        self.assertIsNotNone(taken, f"no verdict case in {JUDGE_YML}")
        self.assertIsNotNone(failed, f"no failing case in {JUDGE_YML}")
        self.assertEqual(set(taken[1].split(" | ")), set(RANKS))
        self.assertEqual(set(failed[1].strip(" |").split(" | ")), set(FAILING))


if __name__ == "__main__":
    unittest.main()
