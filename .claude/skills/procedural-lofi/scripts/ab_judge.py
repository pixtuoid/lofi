#!/usr/bin/env python3
"""Judges an A/B batch of a sound change by measurement. Pairs the takes a
baseline batch and a candidate batch share by name (rendered as SKILL.md's
GENERATOR loop step 5 says, which also sets up the venv this runs in),
scores both sides with two learned judges, describes each pair, and writes
the verdict.

Usage: $J/bin/python ab_judge.py DIR
  DIR holds base/ and cand/; writes DIR/judge.json and prints a table.

The verdict is the controller's proposal (opinion): per mood and over every
take, an exact one-sided sign test that cand scores LOWER than base, ties
dropped, on each judge's headline score. Either judge significant at ALPHA
is REGRESSION; anything else is PASS. Every third-octave band whose
difference passes TO_EXPLAIN_DB is listed TO-EXPLAIN: a PR names its cause,
but it fails nothing.

A score is a guard against regression between two engines or two commits,
never an objective to maximize: lofi's character (hiss, crackle, roll-off) is
held by the descriptors and LOFI-BIBLE.md. Each judge's blind spots:

Audiobox Aesthetics (https://arxiv.org/abs/2502.05139):
- It hears 16 kHz mono ("we re-sample all input waveform to 16kHz and
  single-channel", §3; `sample_rate = 16000` at
  https://github.com/facebookresearch/audiobox-aesthetics/blob/2618e9d451b456e9328b39495b5e6234678aa550/src/audiobox_aesthetics/infer.py#L85),
  so nothing above 8 kHz and no stereo.
- "we did not use any synthetic data to train" it (§4.2.2), and on
  PAM-music its CE correlates r = 0.661 with human CE (Table 4).
- Its PQ axis scores "clarity & fidelity, dynamics, frequencies and
  spatialization" (§2.1), a polish lofi trades away on purpose (opinion).

MuQ-Eval (https://arxiv.org/abs/2603.22677):
- A v1 preprint with no journal reference (its abs page).
- Trained on MusicEval alone: "Cross-dataset generalization ... is
  untested" (§VI-F).
- "insensitive to musical-structural distortions (pitch shift ..., tempo
  stretch ...)" (§VI-F).
- Its concordance on MILD Gaussian noise is 0.39 and on mild MP3 0.52, at
  chance 0.50 (Table V, measured on its A3b variant, not the A1 scored
  here), so it can miss a subtle change; the descriptors and Audiobox cover
  that.
"""

import argparse
import json
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from audiobox_aesthetics.infer import initialize_predictor
from huggingface_hub import hf_hub_download, snapshot_download
from omegaconf import OmegaConf
from scipy import signal
from scipy.stats import binomtest

# no __pycache__ beside it, since the next batch renders from a clean tree
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ab_listen import NAME

# the clone of https://github.com/dgtql/MuQ-Eval that SKILL.md's setup puts
# in the venv: its model code is no package
MUQ_EVAL = Path(sys.prefix) / "MuQ-Eval"
sys.path.insert(0, str(MUQ_EVAL))
from src.model import MusicQualityModel

# each checkpoint at the revision this judge was ratified on
AUDIOBOX = ("facebook/audiobox-aesthetics", "9b1dd8e5df9af7216e836a98974fe3b82c56ded6")
MUQ_EVAL_A1 = ("zhudi2825/MuQ-Eval-A1", "b082e238827df51d99ac66bdd1b6f0d2406bc90e")
MUQ_310M = ("OpenMuQ/MuQ-large-msd-iter", "0562a57814f6f8bbd9fdea0a25921a2fce1a841a")

# Audiobox's own window, `make_inference_batch(wavs, 10, 10, ...)` in its
# infer.py: both judges score a take as consecutive windows this long
WINDOW_S = 10
# windows per forward pass, which bounds its device memory (opinion)
MUQ_BATCH = 6

# the verdict's significance and the TO-EXPLAIN threshold (opinion)
ALPHA = 0.05
TO_EXPLAIN_DB = 3.0

# third octaves on base-2 centres, 1000·2^(k/3) Hz: 20 Hz to 16 kHz
BANDS_HZ = [1000 * 2 ** (k / 3) for k in range(-17, 13)]
# so the 20 Hz band, under 5 Hz wide, holds several bins
WELCH_NPERSEG = 1 << 15
FLOOR = 1e-30

# the hat band's flux, frame to frame (the controller's analysis, opinion)
HAT_HZ = (6000, 12000)
FLUX_NFFT = 4096
FLUX_HOP = 1024

# a click: the 4 kHz high band standing CLICK_MADS median absolute
# deviations out of its own floor, events closer than CLICK_GAP_S one click
# (the controller's analysis, opinion)
CLICK_HZ = 4000
CLICK_ORDER = 4
CLICK_MADS = 12
CLICK_GAP_S = 0.005


def channels(x):
    """`(left, right)` of a frames × channels take; a mono take is both."""
    return (x, x) if x.ndim == 1 else (x[:, 0], x[:, 1])


def describe(x, sr):
    left, right = channels(x)
    mid = (left + right) / 2
    f, psd = signal.welch(mid, sr, nperseg=WELCH_NPERSEG)
    df = f[1] - f[0]
    bands = [
        10 * np.log10(psd[(f >= c * 2 ** (-1 / 6)) & (f < c * 2 ** (1 / 6))].sum() * df + FLOOR)
        for c in BANDS_HZ
    ]
    rms = np.sqrt(np.mean(mid**2))
    freqs, _, z = signal.stft(mid, sr, nperseg=FLUX_NFFT, noverlap=FLUX_NFFT - FLUX_HOP, boundary=None)
    hat = np.abs(z[(freqs >= HAT_HZ[0]) & (freqs < HAT_HZ[1])])
    flux = np.diff(hat, axis=1).clip(min=0).sum(axis=0)
    return {
        "bands_db": bands,
        "centroid_hz": float((f * psd).sum() / psd.sum()),
        "crest_db": float(20 * np.log10(np.abs(mid).max() / rms)),
        "lr_corr": float(np.corrcoef(left, right)[0, 1]),
        "clicks_per_min": clicks_per_min(mid, sr),
        "hat_flux_p95_over_median": float(np.percentile(flux, 95) / np.median(flux)),
    }


def clicks_per_min(mid, sr):
    sos = signal.butter(CLICK_ORDER, CLICK_HZ, "highpass", fs=sr, output="sos")
    high = signal.sosfilt(sos, mid)
    mad = np.median(np.abs(high - np.median(high)))
    over = np.flatnonzero(np.abs(high) > CLICK_MADS * mad)
    starts = over[np.insert(np.diff(over) > CLICK_GAP_S * sr, 0, True)] if over.size else over
    return starts.size / (mid.size / sr / 60)


class MuqEval:
    """MuQ-Eval's A1 checkpoint, its MI head on a 1–5 scale."""

    def __init__(self, device):
        repo, rev = MUQ_EVAL_A1
        config = hf_hub_download(repo, "config.yaml", revision=rev)
        base = hf_hub_download(repo, "base.yaml", revision=rev)
        # merged as MuQ-Eval's own `train.py:load_config` merges an
        # experiment over the config its `defaults` names
        cfg = OmegaConf.merge(OmegaConf.load(base), OmegaConf.load(config))
        del cfg["defaults"]
        if cfg.model.encoder_id != MUQ_310M[0]:
            sys.exit(f"MuQ-Eval's A1 config names encoder {cfg.model.encoder_id}, not {MUQ_310M[0]}")
        # a local snapshot, so the encoder loads at its pinned revision
        cfg.model.encoder_id = snapshot_download(
            MUQ_310M[0], revision=MUQ_310M[1], allow_patterns=["config.json", "model.safetensors"]
        )
        self.model = MusicQualityModel(cfg)
        weights = hf_hub_download(repo, "model_state_dict.pt", revision=rev)
        self.model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        self.model.to(device).eval()
        self.sr = cfg.data.sample_rate
        if cfg.data.clip_samples != WINDOW_S * self.sr:
            sys.exit(f"MuQ-Eval's clip is {cfg.data.clip_samples} samples, not {WINDOW_S} s")
        self.device = device

    def score(self, x, sr):
        """The take's windows' MI, averaged as Audiobox averages its own:
        the last window zero-padded and weighted by the share it fills."""
        left, right = channels(x)
        mono = torch.from_numpy(np.ascontiguousarray((left + right) / 2, dtype=np.float32))
        # as MuQ-Eval's `AudioProcessor.process` resamples
        wav = torchaudio.functional.resample(mono, sr, self.sr)
        clip = WINDOW_S * self.sr
        windows = [wav[i : i + clip] for i in range(0, wav.shape[-1], clip)]
        weights = torch.tensor([w.shape[-1] / clip for w in windows])
        padded = torch.stack([F.pad(w, (0, clip - w.shape[-1])) for w in windows])
        with torch.inference_mode():
            mi = torch.cat(
                [
                    self.model(padded[i : i + MUQ_BATCH].to(self.device))["MI"].float().cpu()
                    for i in range(0, len(windows), MUQ_BATCH)
                ]
            )
        return float((mi * weights).sum() / weights.sum())


def sign_test(deltas):
    """Whether cand scores lower than base more often than chance allows."""
    wins = sum(d > 0 for d in deltas)
    losses = sum(d < 0 for d in deltas)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html:
    # losses as successes out of the untied pairs, H1 a success rate over 1/2
    p = binomtest(losses, wins + losses, 0.5, alternative="greater").pvalue if wins + losses else 1.0
    return {
        "wins": wins,
        "losses": losses,
        "ties": len(deltas) - wins - losses,
        "mean_delta": float(np.mean(deltas)),
        "p": float(p),
        "significant": bool(p < ALPHA),
    }


def to_explain(pairs):
    """Every band some pair moves past TO_EXPLAIN_DB, with how many pairs do
    and the band's median and range over all of them."""
    deltas = np.array([p["bands_delta_db"] for p in pairs])
    rows = []
    for i, hz in enumerate(BANDS_HZ):
        over = int((np.abs(deltas[:, i]) > TO_EXPLAIN_DB).sum())
        if over:
            col = deltas[:, i]
            rows.append(
                {
                    "band_hz": round(hz),
                    "pairs_over": over,
                    "median_db": float(np.median(col)),
                    "min_db": float(col.min()),
                    "max_db": float(col.max()),
                }
            )
    return rows


def read_commit(side):
    commit = side / "COMMIT"
    return commit.read_text().strip() if commit.exists() else None


def main():
    ap = argparse.ArgumentParser(description="Judge an A/B batch by measurement.")
    ap.add_argument("dir", type=Path, help="the batch: base/ and cand/ of same-named takes")
    args = ap.parse_args()
    base_dir, cand_dir = args.dir / "base", args.dir / "cand"

    base = {p.name for p in base_dir.glob("*.wav")}
    cand = {p.name for p in cand_dir.glob("*.wav")}
    if base ^ cand:
        print(f"unpaired, left out: {', '.join(sorted(base ^ cand))}")
    # a --shuffle run draws its own take_XX order, so equal names there are
    # different takes
    opaque = sorted(n for n in base & cand if not NAME.fullmatch(n))
    if opaque:
        sys.exit(f"not unshuffled lofi_audition names, so a pair may hold two takes: {', '.join(opaque)}")
    names = sorted(base & cand)
    if not names:
        sys.exit("no .wav name is in both directories")

    # the device Audiobox's own `setup_model` picks
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    audiobox = initialize_predictor(hf_hub_download(AUDIOBOX[0], "checkpoint.pt", revision=AUDIOBOX[1]))
    muq = MuqEval(device)

    pairs = []
    for name in names:
        row = {"take": name.removesuffix(".wav"), "mood": NAME.fullmatch(name)["mood"]}
        takes = {}
        for side, d in (("base", base_dir), ("cand", cand_dir)):
            x, sr = sf.read(d / name, dtype="float32")
            takes[side] = (x, sr)
            row[side] = describe(x, sr)
            row[side]["muq_mi"] = muq.score(x, sr)
        # Audiobox's `torchaudio.load` path needs torchcodec, so it gets tensors
        scores = audiobox.forward(
            [{"path": torch.from_numpy(np.atleast_2d(x.T).copy()), "sample_rate": sr} for x, sr in takes.values()]
        )
        for side, s in zip(("base", "cand"), scores):
            row[side]["audiobox"] = s
        row["bands_delta_db"] = [c - b for b, c in zip(row["base"]["bands_db"], row["cand"]["bands_db"])]
        pairs.append(row)
        print(f"scored {row['take']}", flush=True)

    groups = {mood: [p for p in pairs if p["mood"] == mood] for mood in sorted({p["mood"] for p in pairs})}
    groups["all"] = pairs
    verdict = {}
    for group, members in groups.items():
        ce = sign_test([p["cand"]["audiobox"]["CE"] - p["base"]["audiobox"]["CE"] for p in members])
        mi = sign_test([p["cand"]["muq_mi"] - p["base"]["muq_mi"] for p in members])
        verdict[group] = {
            "pairs": len(members),
            "audiobox_ce": ce,
            "audiobox_mean_delta_pq": float(
                np.mean([p["cand"]["audiobox"]["PQ"] - p["base"]["audiobox"]["PQ"] for p in members])
            ),
            "muq_eval_mi": mi,
            "verdict": "REGRESSION" if ce["significant"] or mi["significant"] else "PASS",
            "to_explain": to_explain(members),
        }
    overall = "REGRESSION" if any(v["verdict"] == "REGRESSION" for v in verdict.values()) else "PASS"

    record = {
        "dir": str(args.dir.resolve()),
        "commits": {"base": read_commit(base_dir), "cand": read_commit(cand_dir)},
        "judges": {
            "audiobox": {"package": version("audiobox_aesthetics"), "checkpoint": AUDIOBOX},
            "muq_eval": {"code": str(MUQ_EVAL), "checkpoint": MUQ_EVAL_A1, "encoder": MUQ_310M},
            "torch": version("torch"),
            "device": str(device),
        },
        "alpha": ALPHA,
        "bands_hz": BANDS_HZ,
        "pairs": pairs,
        "verdict": verdict,
        "overall": overall,
    }
    (args.dir / "judge.json").write_text(json.dumps(record, indent=1) + "\n")

    print(f"\n{'take':<24} {'dCE':>6} {'dPQ':>6} {'dMI':>6} {'max|dB|':>8} {'@Hz':>6}"
          f" {'centroid b/c':>13} {'crest b/c':>11} {'LR b/c':>11} {'clk/min b/c':>11} {'hat b/c':>11}")
    for p in pairs:
        b, c = p["base"], p["cand"]
        worst = int(np.argmax(np.abs(p["bands_delta_db"])))
        print(
            f"{p['take'].removeprefix('gen_'):<24}"
            f" {c['audiobox']['CE'] - b['audiobox']['CE']:+6.2f}"
            f" {c['audiobox']['PQ'] - b['audiobox']['PQ']:+6.2f}"
            f" {c['muq_mi'] - b['muq_mi']:+6.2f}"
            f" {p['bands_delta_db'][worst]:+8.1f} {round(BANDS_HZ[worst]):>6}"
            f" {b['centroid_hz']:6.0f}/{c['centroid_hz']:<6.0f}"
            f" {b['crest_db']:5.1f}/{c['crest_db']:<5.1f}"
            f" {b['lr_corr']:5.2f}/{c['lr_corr']:<5.2f}"
            f" {b['clicks_per_min']:5.0f}/{c['clicks_per_min']:<5.0f}"
            f" {b['hat_flux_p95_over_median']:5.2f}/{c['hat_flux_p95_over_median']:<5.2f}"
        )
    print(f"\n{'group':<6} {'pairs':>5}  {'Audiobox CE w/l/t':>17} {'p':>7} {'mean dCE':>8} {'mean dPQ':>8}"
          f"  {'MuQ-Eval MI w/l/t':>17} {'p':>7} {'mean dMI':>8}  verdict")
    for group, v in verdict.items():
        ce, mi = v["audiobox_ce"], v["muq_eval_mi"]
        print(
            f"{group:<6} {v['pairs']:>5}  {ce['wins']:>7}/{ce['losses']}/{ce['ties']:<5} {ce['p']:7.4f}"
            f" {ce['mean_delta']:+8.3f} {v['audiobox_mean_delta_pq']:+8.3f}"
            f"  {mi['wins']:>7}/{mi['losses']}/{mi['ties']:<5} {mi['p']:7.4f} {mi['mean_delta']:+8.3f}  {v['verdict']}"
        )
    for group, v in verdict.items():
        for t in v["to_explain"]:
            print(
                f"TO-EXPLAIN {group}: {t['band_hz']} Hz, {t['pairs_over']} of {v['pairs']} pairs past"
                f" ±{TO_EXPLAIN_DB:g} dB, median {t['median_db']:+.1f} ({t['min_db']:+.1f}..{t['max_db']:+.1f})"
            )
    print(f"\nverdict: {overall}  ({args.dir / 'judge.json'})")


if __name__ == "__main__":
    main()
