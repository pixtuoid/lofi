#!/usr/bin/env python3
"""The remote judge's local logic's tests: `python3 -I test_judge_local.py`, with no venv."""

import contextlib
import io
import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge_local import NOT_A_SCORE, SHARDS, dealt, exit_code, merged, published, redact, replaced, select
from test_validate_batch import write_wav
from validate_batch import Takes, stage, validate

# the takes `Selection` writes on both sides
PAIRS = ["gen_hiphop_empty_hiphop_000.wav", "gen_sleep_busy_sleep_001.wav"]
BASE, CAND = "a" * 40, "b" * 40
TAKES = Takes(PAIRS, [], [], PAIRS[1:], PAIRS[:1])


class Selection(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        for side, sha in (("base", BASE), ("cand", CAND)):
            (self.dir / side).mkdir()
            (self.dir / side / "COMMIT").write_text(sha + "\n")
            write_wav(self.dir / side / "gen_hiphop_empty_hiphop_000.wav")
            write_wav(self.dir / side / "gen_sleep_busy_sleep_001.wav", frames=8 if side == "base" else 9)

    def tearDown(self):
        self.tmp.cleanup()

    def assertRefused(self, base_sha, cand_sha, reason):
        """`select` ends the run as no score at all, never as a verdict, and
        says why on stderr."""
        with self.assertRaises(SystemExit) as caught, contextlib.redirect_stderr(io.StringIO()) as err:
            select(self.dir, base_sha, cand_sha, force_score=False)
        self.assertEqual(caught.exception.code, NOT_A_SCORE)
        self.assertIn(f"invalid batch: {reason}", err.getvalue())

    def test_only_the_pairs_that_differ_are_sent(self):
        self.assertEqual(select(self.dir, BASE, CAND, force_score=False), (TAKES, PAIRS[1:]))

    def test_force_score_sends_every_pair(self):
        self.assertEqual(select(self.dir, BASE, CAND, force_score=True), (TAKES, PAIRS))

    def test_a_staged_batch_sends_only_the_pairs_it_holds(self):
        up = self.dir / "up"
        stage(self.dir, validate(self.dir, BASE, CAND), up)
        for force_score in (False, True):
            self.assertEqual(select(up, BASE, CAND, force_score), (TAKES, PAIRS[1:]))

    def test_a_batch_of_other_commits_is_refused(self):
        self.assertRefused(CAND, BASE, "base/COMMIT doesn't name")

    def test_a_batch_with_no_pair_is_refused(self):
        for side in ("base", "cand"):
            for p in (self.dir / side).glob("*.wav"):
                p.unlink()
        self.assertRefused(BASE, CAND, "no take on both sides")

    def test_a_take_on_one_side_only_is_left_out_and_the_pairs_still_sent(self):
        (self.dir / "cand" / "gen_hiphop_empty_hiphop_000.wav").unlink()
        write_wav(self.dir / "cand" / "gen_dusk_moderate_hiphop_000.wav")
        takes = Takes(PAIRS[1:], ["gen_dusk_moderate_hiphop_000.wav"], ["gen_hiphop_empty_hiphop_000.wav"], PAIRS[1:], [])
        self.assertEqual(select(self.dir, BASE, CAND, force_score=True), (takes, PAIRS[1:]))

    def test_a_batch_whose_takes_are_each_on_one_side_is_refused(self):
        (self.dir / "cand" / "gen_hiphop_empty_hiphop_000.wav").unlink()
        (self.dir / "base" / "gen_sleep_busy_sleep_001.wav").unlink()
        self.assertRefused(BASE, CAND, "no take on both sides")

    def test_a_missing_commit_is_refused(self):
        (self.dir / "cand" / "COMMIT").unlink()
        self.assertRefused(BASE, CAND, "cand/COMMIT doesn't name")


class Outcomes(unittest.TestCase):
    def test_only_a_score_that_fails_exits_one_and_anything_else_two(self):
        self.assertEqual([exit_code(o) for o in ("PASS", "UNCHANGED")], [0, 0])
        self.assertEqual([exit_code(o) for o in ("REGRESSION", "INSUFFICIENT")], [1, 1])
        self.assertEqual(exit_code("infra failure"), 2)


class Redaction(unittest.TestCase):
    CORPORA = {
        "hiphop": [{"file": "Some-Artist-Some-Title.mp3"}, {"file": "Some-Artist-Some-Title-Extended.mp3"}],
        "sleep": [{"file": "Another-Artist-Slow.mp3"}],
    }

    def test_full_names_and_stems_become_styles_and_manifest_indices(self):
        text = (
            "/refs/hiphop/audio/Some-Artist-Some-Title-Extended.mp3 failed;"
            " Some-Artist-Some-Title decoded; Some-Artist-Some-Title.mp3 too; Another-Artist-Slow too"
        )
        self.assertEqual(
            redact(text, self.CORPORA),
            "/refs/hiphop/audio/hiphop#1 failed; hiphop#0 decoded; hiphop#0 too; sleep#0 too",
        )


class Publishing(unittest.TestCase):
    CORPORA = Redaction.CORPORA

    def test_the_record_publishes_its_own_table_masked_and_no_corpus_name(self):
        record = {
            "dir": "/batch",
            "judges": {"fad": {"embedding": "clap"}},
            "corpora": {
                "hiphop": {
                    "refs": "/refs/hiphop",
                    "lufs": -20.0,
                    "fad_calibration": {"Some-Artist-Some-Title.mp3": 0.4, "Some-Artist-Some-Title-Extended.mp3": 0.2},
                },
                "sleep": {"refs": "/refs/sleep", "lufs": -30.0, "fad_calibration": {"Another-Artist-Slow.mp3": 0.3}},
            },
            "table": "take dCE\nhiphop_empty_hiphop_000 +0.01\nnearest Some-Artist-Some-Title, Another-Artist-Slow",
            "overall": "PASS",
        }
        got = published(record, self.CORPORA)
        self.assertEqual(got["outcome"], "PASS")
        self.assertEqual(got["table"], "take dCE\nhiphop_empty_hiphop_000 +0.01\nnearest hiphop#0, sleep#0")
        self.assertEqual(
            got["record"],
            {
                "judges": {"fad": {"embedding": "clap"}},
                "corpora": {
                    "hiphop": {"lufs": -20.0, "fad_calibration": [0.2, 0.4]},
                    "sleep": {"lufs": -30.0, "fad_calibration": [0.3]},
                },
                "overall": "PASS",
            },
        )


class Dealing(unittest.TestCase):
    def test_every_pair_goes_to_one_shard_of_the_most_a_batch_takes(self):
        pairs = [f"gen_hiphop_empty_hiphop_{i:03}.wav" for i in range(95)]
        shards = dealt(pairs)
        self.assertEqual(len(shards), SHARDS)
        self.assertEqual(sorted(name for shard in shards for name in shard), pairs)

    def test_the_shards_of_a_batch_differ_by_at_most_one_pair(self):
        sizes = [len(shard) for shard in dealt(list(range(95)))]
        self.assertLessEqual(max(sizes) - min(sizes), 1)

    def test_a_batch_smaller_than_the_shards_takes_one_each_and_no_empty_one(self):
        self.assertEqual(dealt(PAIRS), [PAIRS[:1], PAIRS[1:]])


class Merging(unittest.TestCase):
    JUDGES = {"fad": {"embedding": "clap"}}
    HIPHOP = {"lufs": -20.0}

    def shard(self, *names, judges=JUDGES, corpora=None):
        corpora = corpora or {"hiphop": self.HIPHOP}
        return {"seat": {"judges": judges, "corpora": corpora}, "rows": {name: {"take": name} for name in names}}

    def test_the_rows_join_under_the_seat_the_shards_share(self):
        seat, rows = merged([self.shard(PAIRS[1]), self.shard(PAIRS[0])], PAIRS)
        self.assertEqual(seat, {"judges": self.JUDGES, "corpora": {"hiphop": self.HIPHOP}})
        self.assertEqual(set(rows), set(PAIRS))

    def test_each_corpus_a_shard_read_joins_the_seat(self):
        sleep = {"lufs": -30.0}
        shards = [self.shard(PAIRS[0]), self.shard(PAIRS[1], corpora={"hiphop": self.HIPHOP, "sleep": sleep})]
        seat, _ = merged(shards, PAIRS)
        self.assertEqual(seat["corpora"], {"hiphop": self.HIPHOP, "sleep": sleep})

    def test_a_shard_that_dropped_a_pair_is_refused(self):
        with self.assertRaises(SystemExit):
            merged([self.shard(PAIRS[0])], PAIRS)

    def test_a_pair_two_shards_scored_is_refused(self):
        with self.assertRaises(SystemExit):
            merged([self.shard(*PAIRS), self.shard(PAIRS[0])], PAIRS)

    def test_a_shard_seated_at_another_level_in_a_corpus_is_refused(self):
        other = {"hiphop": {"lufs": -21.0}}
        with self.assertRaises(SystemExit):
            merged([self.shard(PAIRS[0]), self.shard(PAIRS[1], corpora=other)], PAIRS)

    def test_a_shard_that_sat_another_panel_is_refused(self):
        other = {"fad": {"embedding": "other"}}
        with self.assertRaises(SystemExit):
            merged([self.shard(PAIRS[0]), self.shard(PAIRS[1], judges=other)], PAIRS)

    def test_a_batch_with_no_pair_to_score_has_no_seat_and_no_rows(self):
        self.assertEqual(merged([], []), ({}, {}))


class RustCopies(unittest.TestCase):
    def test_the_warm_up_take_is_at_lofis_rate(self):
        root = Path(__file__).resolve().parents[4]
        rust = re.search(r"pub const SAMPLE_RATE: u32 = ([\d_]+);", (root / "src" / "dsp.rs").read_text())
        py = re.search(r"^WARMUP_RATE = (\d+)$", (root / ".claude/skills/procedural-lofi/judge/modal_judge.py").read_text(), re.M)
        self.assertIsNotNone(rust, "no SAMPLE_RATE in src/dsp.rs")
        self.assertIsNotNone(py, "no WARMUP_RATE in modal_judge.py")
        self.assertEqual(int(py[1]), int(rust[1].replace("_", "")))




class Replaced(unittest.TestCase):
    def test_a_corpus_the_volume_lacks_or_holds_alike_uploads(self):
        self.assertIsNone(replaced("sleep", None, b"{}"))
        self.assertIsNone(replaced("sleep", b"{}", b"{}"))

    def test_a_corpus_the_volume_holds_otherwise_is_refused_and_says_what_to_clear(self):
        refusal = replaced("sleep", b'{"tracks": [1]}', b'{"tracks": [2]}')
        self.assertIn("/sleep", refusal)
        self.assertIn("/stems/sleep-*", refusal)


if __name__ == "__main__":
    unittest.main()
