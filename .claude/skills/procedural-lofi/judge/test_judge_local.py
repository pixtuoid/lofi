#!/usr/bin/env python3
"""The remote judge's local logic's tests: `python3 -I test_judge_local.py`, with no venv."""

import contextlib
import io
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge_local import (
    NOT_A_SCORE,
    SHARDS,
    batch_key,
    dealt,
    exit_code,
    merged,
    published,
    redact,
    replaced,
    select,
    verdict_key,
)
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


class Keys(unittest.TestCase):
    """A verdict is kept under the bytes that decide it, so a push that
    renders them again reads it instead of scoring them again."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.batch = self.dir / "batch"
        for side, sha in (("base", BASE), ("cand", CAND)):
            (self.batch / side).mkdir(parents=True)
            (self.batch / side / "COMMIT").write_text(sha + "\n")
            for i, name in enumerate(PAIRS):
                write_wav(self.batch / side / name, frames=8 + i + (side == "cand"))
        self.code = [self.dir / "scripts", self.dir / "judge"]
        for d in self.code:
            d.mkdir()
            (d / f"{d.name}.py").write_text("x = 1\n")

    def tearDown(self):
        self.tmp.cleanup()

    def key(self, batch=None, paired=PAIRS, sent=PAIRS):
        return batch_key(batch or self.batch, paired, sent, self.code)

    def test_the_same_bytes_anywhere_keep_one_key(self):
        moved = self.dir / "moved"
        shutil.copytree(self.batch, moved)
        self.assertEqual(self.key(), self.key(moved))

    def test_a_byte_of_either_side_of_a_sent_take_moves_the_key(self):
        before = self.key()
        for side in ("base", "cand"):
            take = self.batch / side / PAIRS[1]
            data = take.read_bytes()
            take.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
            self.assertNotEqual(self.key(), before, side)
            take.write_bytes(data)

    def test_a_commit_the_bytes_rendered_at_keeps_the_key(self):
        before = self.key()
        (self.batch / "cand" / "COMMIT").write_text("c" * 40 + "\n")
        self.assertEqual(self.key(), before)

    def test_another_pairing_or_another_set_sent_moves_the_key(self):
        before = self.key()
        self.assertNotEqual(self.key(paired=[*PAIRS, "gen_rain_busy_hiphop_002.wav"]), before)
        self.assertNotEqual(self.key(sent=PAIRS[1:]), before)

    def test_an_edit_to_the_judge_moves_the_key_and_its_bytecode_doesnt(self):
        before = self.key()
        cache = self.code[0] / "__pycache__"
        cache.mkdir()
        (cache / "scripts.cpython-312.pyc").write_bytes(b"\0")
        self.assertEqual(self.key(), before)
        (self.code[1] / "judge.py").write_text("x = 2\n")
        self.assertNotEqual(self.key(), before)

    def test_a_file_moved_between_the_judges_directories_moves_the_key(self):
        before = self.key()
        (self.code[0] / "scripts.py").rename(self.code[1] / "scripts.py")
        self.assertNotEqual(self.key(), before)

    def test_another_selection_of_a_corpus_or_environment_moves_the_key_and_their_order_doesnt(self):
        key = self.key()
        manifests = {"hiphop": b'{"tracks": [1]}', "sleep": b'{"tracks": [2]}'}
        installed = ["3.12.15", "torch==2.14.1"]
        before = verdict_key(key, manifests, installed)
        self.assertEqual(verdict_key(key, dict(reversed(manifests.items())), installed), before)
        self.assertNotEqual(verdict_key(key, {**manifests, "sleep": b'{"tracks": [3]}'}, installed), before)
        self.assertNotEqual(verdict_key(key, manifests, ["3.12.16", "torch==2.14.1"]), before)
        self.assertNotEqual(verdict_key("0" * 64, manifests, installed), before)


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

    def test_a_label_is_never_masked_again(self):
        # a corpus holding a reference whose stem is a label's own word
        corpora = {"sleep": [{"file": "sleep.mp3"}]}
        self.assertEqual(redact("sleep.mp3 failed; sleep decoded", corpora), "sleep#0 failed; sleep#0 decoded")


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

    def test_a_shard_that_read_a_corpus_at_another_level_is_refused(self):
        other = {"hiphop": {"lufs": -21.0}}
        with self.assertRaises(SystemExit):
            merged([self.shard(PAIRS[0]), self.shard(PAIRS[1], corpora=other)], PAIRS)

    def test_a_shard_run_in_another_environment_is_refused(self):
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
