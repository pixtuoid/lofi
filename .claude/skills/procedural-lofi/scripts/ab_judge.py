#!/usr/bin/env python3
"""Judges an A/B batch of a sound change by measurement. Pairs the takes a
baseline batch and a candidate batch share by name (rendered as SKILL.md's
GENERATOR loop step 5 says, which also sets up the venv this runs in),
scores both sides with a panel of learned judges and their distance from
real lofi, describes each pair, and writes the verdict.

Usage: $J/bin/python ab_judge.py DIR --refs REFS [--corpus-cache CACHE] [--score-identical]
  DIR holds base/ and cand/, each with the COMMIT it rendered at; writes
  DIR/judge.json and prints its table.
  REFS is the reference corpus: audio/ and its manifest.json of sha256s.
  CACHE holds the corpus's FAD statistics and the panel's screening, built on
  a miss.
  --score-identical scores byte-identical pairs too: the proof that they tie.

The verdict is the controller's proposal (opinion), `verdict.judged`'s, and
this exits non-zero on REGRESSION or INSUFFICIENT. The owner's blind listen,
when given, overrides it. Every third-octave band whose difference passes
TO_EXPLAIN_DB is listed TO-EXPLAIN: a PR names its cause, but it fails
nothing.

A score is a guard against regression from a base batch to a candidate,
never an objective to maximize: lofi's character (hiss, crackle, roll-off) is
held by the descriptors and LOFI-BIBLE.md. A judge sits on the panel only
while ITU-R BS.1534-3's post-screening seats it on the held-out references
(`screened`), and it weighs in its own spread over them. Each measure's blind
spots:

Audiobox Aesthetics (https://arxiv.org/abs/2502.05139):
- It hears 16 kHz mono ("we re-sample all input waveform to 16kHz and
  single-channel", §3; `sample_rate = 16000` at
  https://github.com/facebookresearch/audiobox-aesthetics/blob/2618e9d451b456e9328b39495b5e6234678aa550/src/audiobox_aesthetics/infer.py#L85),
  so nothing above 8 kHz and no stereo.
- "we did not use any synthetic data to train" it (§4.2.2), and on
  PAM-music its CE correlates r = 0.661 with human CE (Table 4).

TuneJury (https://arxiv.org/abs/2606.17006):
- A v1 preprint (its abs page), trained on "~17.5K human-rated A vs. B
  pairs" of text-to-music, at "0.7086 pairwise accuracy" on its held-out
  pairs (the README at the commit `TuneJury` cites).
- It hears mono, its encoders at 48 and 24 kHz (its score.py).
- Its text branch is zeroed: lofi's takes have no prompt.

Per-song FAD over fadtk (https://github.com/microsoft/fadtk, MIT), the
method of https://arxiv.org/abs/2311.01616 (ICASSP 2024):
- A distance no listener trained, so it guards the genre and never votes
  (`verdict.judged`).
- The corpus is 88 of 100 tracks from one netlabel, Dusted Wax Kingdom (its
  manifest), so it guards against drifting away from real lofi and is never
  a target to chase.
- FAD is biased by its sample size (§3.3 of
  https://arxiv.org/abs/2311.01616), so both takes of a pair are scored on
  equally many frames.

The weights' licences: Audiobox Aesthetics CC-BY 4.0, LAION-CLAP CC0 1.0,
and TuneJury and MERT-v1-330M CC-BY-NC 4.0 (each Hugging Face card at the
revision pinned below).
"""

import argparse
import hashlib
import json
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

import fadtk.model_loader
import numpy as np
import pyloudnorm
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from audiobox_aesthetics.infer import initialize_predictor
from fadtk.fad import calc_embd_statistics, calc_frechet_distance
from huggingface_hub import hf_hub_download
from scipy import signal
from transformers import AutoModel, Wav2Vec2FeatureExtractor

# no __pycache__ beside it, since the next batch renders from a clean tree
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
# `judge/` sits beside `scripts/` here and in the Modal image alike
sys.path[:0] = [str(HERE), str(HERE.parent / "judge")]
from validate_batch import NAME, Invalid, scored, validate
from verdict import GUARD, JUDGES, PANEL, exit_message, judged

# each checkpoint at the revision this judge was ratified on
AUDIOBOX = ("facebook/audiobox-aesthetics", "9b1dd8e5df9af7216e836a98974fe3b82c56ded6")
LAION_CLAP = (
    "lukewys/laion_clap",
    "b3708341862f581175dba5c356a4ebf74a9b6651",
    "music_audioset_epoch_15_esc_90.14.pt",
)
TUNEJURY = ("TuneJury/tunejury", "a36a6adaad4a9786a12c3e5fd82501f2f2dc54ff", "tunejury.pt")
MERT = ("m-a-p/MERT-v1-330M", "5240c2708a5acaee1007f43fb9735c7dcd0b78c9")

# fadtk's LAION-CLAP music embedding ("L-CLAP mus"): "CLAP and L-CLAP
# outperform other embeddings for AQ and MQ MOS prediction" (§4.4 of
# https://arxiv.org/abs/2311.01616)
EMBEDDING = "music"
# reference tracks left out of the statistics, so their own per-song FAD
# shows what real lofi scores (the controller's proposal)
HOLDOUT = 12
# fadtk's own resampler to a model's rate, `load_audio` in its fad.py
RESAMPLE = {
    "lowpass_filter_width": 64,
    "rolloff": 0.9475937167399596,
    "resampling_method": "sinc_interp_kaiser",
    "beta": 14.769656459379492,
}

# the floor every take gets before anything scores it: white noise over both
# engines' numerical floors, which a learned judge scores and a listener
# can't hear (inaudible is opinion). It was calibrated at this RMS in dBFS
# under takes at this integrated loudness, lofi_audition's `TARGET_LUFS` then,
# and keeps that distance under each take's own loudness: the engines'
# numerical floors scale with the music, and a pure gain change must move a
# take and its floor together, or FAD's loudness normalization hears the
# floor move
FLOOR_CALIBRATED_DBFS = -70.0
FLOOR_CALIBRATED_LUFS = -18.0
FLOOR_RE_TAKE_DB = FLOOR_CALIBRATED_DBFS - FLOOR_CALIBRATED_LUFS

# Audiobox's own window, `make_inference_batch(wavs, 10, 10, ...)` in its
# infer.py: every judge scores a take as consecutive windows this long
WINDOW_S = 10
# windows per forward pass, which bounds a judge's device memory (opinion)
JUDGE_BATCH = 6
# TuneJury's hidden layers, its model.py's default
TUNEJURY_HIDDEN = (1024, 512, 256, 128)

# ITU-R BS.1534-3's anchors (§5.1): "The standard anchor is a low-pass
# filtered version of the original signal with a cut-off frequency of 3.5 kHz;
# the mid quality anchor has a cut-off frequency of 7 kHz", the first with
# "Maximum pass band ripple = 0.1 dB, Minimum attenuation at 4 kHz = 25 dB,
# Minimum attenuation at 4.5 kHz = 50 dB". Both are cut 50 dB down by 4 kHz's
# ratio to 3.5, which meets both edges; the 7 kHz one's is opinion
ANCHORS_HZ = (3500, 7000)
ANCHOR_RIPPLE_DB = 0.1
ANCHOR_STOP_DB = 50
ANCHOR_STOP_RATIO = 4000 / 3500
# §4.1.2's post-screening: "an assessor should be excluded from the
# aggregated responses if he or she rates the mid-range anchor for more than
# 15% of the test items higher than a score of 90". A judge is an assessor,
# and an anchor it scores at or over its reference is one it rates as
# unimpaired, since a judge grades on no scale of 100 (opinion). The 3.5 kHz
# anchor's share is kept beside it, screening nothing
SCREEN_ANCHOR_HZ = ANCHORS_HZ[1]
SCREEN_SHARE = 0.15

# the TO-EXPLAIN threshold (opinion)
TO_EXPLAIN_DB = 3.0
# each judge's column in the table
LABELS = {"audiobox_ce": "CE", "tunejury": "TJ", "fad": "FAD"}

# third octaves on base-2 centres, 1000·2^(k/3) Hz: 20 Hz to 16 kHz
BANDS_HZ = [1000 * 2 ** (k / 3) for k in range(-17, 13)]
# so the 20 Hz band, under 5 Hz wide, holds several bins
WELCH_NPERSEG = 1 << 15
POWER_EPS = 1e-30

# the hat band's flux, frame to frame (the controller's analysis, opinion)
HAT_HZ = (6000, 12000)
FLUX_NFFT = 4096
FLUX_HOP = 1024
# the flux's peak, as this percentile of its frames over their median
HAT_FLUX_PERCENTILE = 95
HAT_FLUX_KEY = f"hat_flux_p{HAT_FLUX_PERCENTILE}_over_median"

# a click: the CLICK_HZ high band standing CLICK_MADS median absolute
# deviations out of its own floor, events closer than CLICK_GAP_S one click
# (the controller's analysis, opinion)
CLICK_HZ = 4000
CLICK_ORDER = 4
CLICK_MADS = 12
CLICK_GAP_S = 0.005


def channels(x):
    """`(left, right)` of a frames × channels take; a mono take is both."""
    return (x, x) if x.ndim == 1 else (x[:, 0], x[:, 1])


def mono(x):
    left, right = channels(x)
    return (left + right) / 2


def floored(x, name, lufs):
    """`x`, whose integrated loudness is `lufs`, over white noise
    `FLOOR_RE_TAKE_DB` under it, the same in each channel and seeded by the
    take's name: both sides of a pair share the draw, so two identical takes
    still tie and the floor adds no variance to the pair's difference."""
    seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")
    noise = np.random.default_rng(seed).standard_normal(len(x)).astype(np.float32)
    noise *= np.float32(10 ** ((lufs + FLOOR_RE_TAKE_DB) / 20))
    return x + (noise if x.ndim == 1 else noise[:, None])


def mono_lufs(x, sr):
    """The integrated loudness of the mono the embeddings hear."""
    return pyloudnorm.Meter(sr).integrated_loudness(mono(x).astype(np.float64))


def describe(x, sr):
    left, right = channels(x)
    mid = (left + right) / 2
    f, psd = signal.welch(mid, sr, nperseg=WELCH_NPERSEG)
    df = f[1] - f[0]
    bands = [
        10 * np.log10(psd[(f >= c * 2 ** (-1 / 6)) & (f < c * 2 ** (1 / 6))].sum() * df + POWER_EPS)
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
        HAT_FLUX_KEY: float(np.percentile(flux, HAT_FLUX_PERCENTILE) / np.median(flux)),
    }


def clicks_per_min(mid, sr):
    sos = signal.butter(CLICK_ORDER, CLICK_HZ, "highpass", fs=sr, output="sos")
    high = signal.sosfilt(sos, mid)
    mad = np.median(np.abs(high - np.median(high)))
    over = np.flatnonzero(np.abs(high) > CLICK_MADS * mad)
    starts = over[np.insert(np.diff(over) > CLICK_GAP_S * sr, 0, True)] if over.size else over
    return starts.size / (mid.size / sr / 60)


def windows(x, sr, rate):
    """The mono of `x`, at `sr`, resampled to `rate` and cut into `WINDOW_S`
    windows, the last zero-padded, with the share of each the take fills: a
    take's score is its windows' scores weighted so, as Audiobox averages its
    own. Any two rates cut a take into equally many windows, since each holds
    `WINDOW_S` and resampling rounds a length up."""
    heard = torch.from_numpy(np.ascontiguousarray(mono(x), dtype=np.float32))
    wav = torchaudio.functional.resample(heard, sr, rate)
    clip = WINDOW_S * rate
    cut = [wav[i : i + clip] for i in range(0, wav.shape[-1], clip)]
    weights = torch.tensor([w.shape[-1] / clip for w in cut])
    return torch.stack([F.pad(w, (0, clip - w.shape[-1])) for w in cut]), weights


def anchor_filter(sr, cutoff_hz):
    """The elliptic low-pass, as second-order sections at `sr`, that cuts
    BS.1534-3's anchor at `cutoff_hz`."""
    order, edge = signal.ellipord(cutoff_hz, cutoff_hz * ANCHOR_STOP_RATIO, ANCHOR_RIPPLE_DB, ANCHOR_STOP_DB, fs=sr)
    return signal.ellip(order, ANCHOR_RIPPLE_DB, ANCHOR_STOP_DB, edge, fs=sr, output="sos")


def anchor(x, sr, cutoff_hz):
    """BS.1534-3's low-pass anchor of `x` at `cutoff_hz`, as loud as `x`, so a
    judge hears the band it lost, not a level (opinion)."""
    y = signal.sosfilt(anchor_filter(sr, cutoff_hz), x, axis=0).astype(np.float32)
    return y * np.float32(10 ** ((mono_lufs(x, sr) - mono_lufs(y, sr)) / 20))


def screened(judges, items):
    """Each judge on `items`, each `(x, sr)`: its share of them whose anchor
    it scores at or over the item itself, an anchor of `ANCHORS_HZ` at a time,
    which is how often it rates an impaired take unimpaired; and its scores'
    spread over them, its unit in the panel's pooled gain."""
    unimpaired = {name: dict.fromkeys(ANCHORS_HZ, 0) for name in judges}
    heard = {name: [] for name in judges}
    for x, sr in items:
        anchors = {hz: anchor(x, sr, hz) for hz in ANCHORS_HZ}
        for name, judge in judges.items():
            heard[name].append(judge.score(x, sr))
            for hz, y in anchors.items():
                unimpaired[name][hz] += judge.score(y, sr) >= heard[name][-1]
    return {
        name: {"anchors": {hz: n / len(items) for hz, n in unimpaired[name].items()}, "scale": float(np.std(heard[name], ddof=1))}
        for name in judges
    }


def screened_out(screen):
    """The judges BS.1534-3's post-screening excludes, by `screened`'s shares
    on its mid-range anchor."""
    return [name for name, judge in screen.items() if judge["anchors"][SCREEN_ANCHOR_HZ] > SCREEN_SHARE]


class Audiobox:
    """Audiobox Aesthetics' CE axis, its content enjoyment."""

    def __init__(self):
        self.model = initialize_predictor(hf_hub_download(AUDIOBOX[0], "checkpoint.pt", revision=AUDIOBOX[1]))

    def score(self, x, sr):
        # Audiobox reads a path with `torchaudio.load`
        # (https://github.com/facebookresearch/audiobox-aesthetics/blob/2618e9d451b456e9328b39495b5e6234678aa550/src/audiobox_aesthetics/infer.py#L32-L45),
        # which "relies on TorchCodec" since TorchAudio 2.9
        # (https://docs.pytorch.org/audio/2.11.0/generated/torchaudio.load.html),
        # so it gets a tensor
        return self.model.forward([{"path": torch.from_numpy(np.atleast_2d(x.T).copy()), "sample_rate": sr}])[0]["CE"]


class TuneJury:
    """TuneJury's reward from its audio branches alone. Its scorer zeroes the
    text branch for an empty prompt, the protocol §4.2 of
    https://arxiv.org/abs/2606.17006 gives audio its prompts don't describe."""

    def __init__(self, clap, device):
        # the head as `TuneJury.__init__` builds it, over its encoders'
        # embeddings in its order: CLAP's audio, MERT's, CLAP's text
        # (https://github.com/yonghyunk1m/TuneJury/blob/9dd8ea3e4a72f358ce311a6a856194ca6932c89e/tunejury/model.py,
        # its score.py's `_extract_features`)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(MERT[0], revision=MERT[1])
        self.mert = AutoModel.from_pretrained(MERT[0], revision=MERT[1], trust_remote_code=True).to(device).eval()
        layers, width = [], 2 * clap.num_features + self.mert.config.hidden_size
        for hidden in TUNEJURY_HIDDEN:
            layers += [torch.nn.Linear(width, hidden), torch.nn.BatchNorm1d(hidden), torch.nn.ReLU(), torch.nn.Dropout()]
            width = hidden
        self.head = torch.nn.Sequential(*layers, torch.nn.Linear(width, 1))
        repo, rev, file = TUNEJURY
        state = torch.load(hf_hub_download(repo, file, revision=rev), map_location="cpu", weights_only=True)
        self.head.load_state_dict({k.removeprefix("net."): v for k, v in state.items()})
        self.head.to(device).eval()
        self.clap = clap
        self.device = device

    def score(self, x, sr):
        """The take's windows' reward, averaged as `windows` weights them."""
        clap, weights = windows(x, sr, self.clap.sr)
        mert, _ = windows(x, sr, self.processor.sampling_rate)
        rewards = []
        with torch.inference_mode():
            for c, m in zip(clap.split(JUDGE_BATCH), mert.split(JUDGE_BATCH), strict=True):
                audio = torch.from_numpy(self.clap.model.get_audio_embedding_from_data(x=c.numpy(), use_tensor=False))
                heard = self.processor(list(m.numpy()), sampling_rate=self.processor.sampling_rate, return_tensors="pt")
                # MERT's last layer pooled over time, as its score.py pools it
                music = self.mert(**heard.to(self.device)).last_hidden_state.mean(dim=1).float().cpu()
                features = torch.cat([audio, music, torch.zeros_like(audio)], dim=1)
                rewards.append(self.head(features.to(self.device))[:, 0].float().cpu())
        rewards = torch.cat(rewards)
        return float((rewards * weights).sum() / weights.sum())


def quietest_file(caches, manifest_sha):
    # the corpus alone sets it, and measuring it decodes every track
    return caches / f"{manifest_sha}-quietest.json"


def stats_file(caches, manifest_sha, lufs):
    # keyed by everything the statistics are built from; the level is one any
    # batch louder than the quietest reference shares
    built = json.dumps([LAION_CLAP, EMBEDDING, HOLDOUT, RESAMPLE])
    return caches / f"{manifest_sha}-{hashlib.sha256(built.encode()).hexdigest()[:16]}-{lufs:.2f}.npz"


def cached_quietest(caches, manifest_sha, takes_lufs):
    """The quietest reference's loudness if `caches` holds it, else None.
    Exits first when `caches` lacks a file this run builds and can't take
    it, as the remote judge's read-only cache can't: with the corpus's own
    statistics missing, provisioning builds them; a batch quieter than the
    quietest reference needs its own level, which only a local run builds.
    Tried rather than asked, since Modal promises a read-only mount only as
    "read only from within containers" (modal 1.6.1 `volume.py`:598)."""
    cached = quietest_file(caches, manifest_sha)
    quietest = json.loads(cached.read_text())["lufs"] if cached.exists() else None
    if quietest is not None and stats_file(caches, manifest_sha, min(takes_lufs, quietest)).exists():
        return quietest
    try:
        caches.mkdir(parents=True, exist_ok=True)
        tempfile.TemporaryFile(dir=caches).close()
    except OSError:
        if quietest is not None and takes_lufs < quietest:
            sys.exit(
                "no FAD statistics at this batch's level, and the cache is read-only: the batch"
                " is quieter than the corpus's quietest reference, so judge it locally"
            )
        sys.exit("the corpus cache lacks this corpus's FAD statistics and is read-only: run provision")
    return quietest


def screen_file(caches, manifest_sha):
    # keyed by everything a screening is built from
    built = json.dumps([PANEL, AUDIOBOX, TUNEJURY, MERT, LAION_CLAP, HOLDOUT, ANCHORS_HZ, ANCHOR_RIPPLE_DB, ANCHOR_STOP_DB, ANCHOR_STOP_RATIO])
    return caches / f"{manifest_sha}-screen-{hashlib.sha256(built.encode()).hexdigest()[:16]}.json"


def cached_screen(caches, fad, panel):
    """`screened`'s shares for `panel` on the held-out references, read from
    `caches` or built into it. Building needs a cache this run can write, as
    `cached_quietest` tries one, and the remote judge's can't."""
    path = screen_file(caches, fad.manifest_sha)
    if path.exists():
        screen = json.loads(path.read_text())
        for judge in screen.values():
            judge["anchors"] = {int(hz): share for hz, share in judge["anchors"].items()}
        return screen
    try:
        tempfile.TemporaryFile(dir=caches).close()
    except OSError:
        sys.exit("the corpus cache lacks the panel's screening and is read-only: run provision")
    shares = screened(panel, [fad.read(fad.refs, {"file": f}) for f in sorted(fad.calibration)])
    path.write_text(json.dumps(shares) + "\n")
    return shares


class Fad:
    """Per-song FAD from a take to the reference corpus, over fadtk's
    LAION-CLAP embedding: the Fréchet distance between the corpus's frame
    statistics and the take's, as fadtk's `score_individual` takes it."""

    def __init__(self, refs, takes_lufs, caches):
        manifest = (refs / "manifest.json").read_bytes()
        manifest_sha = hashlib.sha256(manifest).hexdigest()
        quietest = cached_quietest(caches, manifest_sha, takes_lufs)
        tracks = json.loads(manifest)["tracks"]
        # by index: the corpus's names are private, and the CI's log is not
        for i, t in enumerate(tracks):
            path = refs / "audio" / t["file"]
            if not path.is_file():
                sys.exit(f"reference {i} of the manifest is missing; refusing the corpus")
            if hashlib.sha256(path.read_bytes()).hexdigest() != t["sha256"]:
                sys.exit(f"reference {i} doesn't match its manifest sha256; refusing the corpus")
        # fadtk loads whatever sits at its checkpoint path and downloads a
        # missing one from the repo's head
        # (https://github.com/microsoft/fadtk/blob/6f815282007288a7165d037a95dacd6d783b6f32/fadtk/model_loader.py#L303-L310),
        # so only a link to the pinned one may sit there
        repo, rev, file = LAION_CLAP
        pinned = Path(hf_hub_download(repo, file, revision=rev))
        seeded = Path(fadtk.model_loader.__file__).parent / ".model-checkpoints" / file
        if not (seeded.is_symlink() and seeded.resolve() == pinned.resolve()):
            seeded.unlink(missing_ok=True)
            seeded.parent.mkdir(parents=True, exist_ok=True)
            seeded.symlink_to(pinned)
        self.model = fadtk.model_loader.CLAPLaionModel(EMBEDDING)
        self.model.load_model()
        self.refs, self.manifest_sha = refs, manifest_sha
        if quietest is None:
            quietest = min(mono_lufs(*self.read(refs, t)) for t in tracks)
            quietest_file(caches, manifest_sha).write_text(json.dumps({"lufs": quietest}) + "\n")
        # one level for every input, the takes' unless a reference is
        # quieter, so no reference is ever raised and clipped
        self.lufs = min(takes_lufs, quietest)
        cache = stats_file(caches, manifest_sha, self.lufs)
        if cache.exists():
            with np.load(cache) as stats:
                self.mu, self.cov = stats["mu"], stats["cov"]
                self.calibration = dict(zip(stats["held"].tolist(), stats["held_fad"].tolist()))
            return
        # the lowest sha256s, a choice no file order moves
        held = {t["file"] for t in sorted(tracks, key=lambda t: t["sha256"])[:HOLDOUT]}
        frames = np.concatenate([self.embed(*self.read(refs, t)) for t in tracks if t["file"] not in held])
        self.mu, self.cov = calc_embd_statistics(frames)
        self.calibration = {f: self.score(*self.read(refs, {"file": f})) for f in sorted(held)}
        np.savez(
            cache,
            lufs=self.lufs,
            mu=self.mu,
            cov=self.cov,
            held=np.array(list(self.calibration)),
            held_fad=np.array(list(self.calibration.values())),
        )

    @staticmethod
    def read(refs, track):
        return sf.read(refs / "audio" / track["file"], dtype="float32")

    def embed(self, x, sr):
        """`x`'s frames of the embedding, at the one level every input shares."""
        heard = mono(x).astype(np.float64)
        heard *= 10 ** ((self.lufs - mono_lufs(x, sr)) / 20)
        wav = torch.from_numpy(heard.astype(np.float32))[None]
        wav = torchaudio.transforms.Resample(sr, self.model.sr, **RESAMPLE)(wav)[0]
        return self.model.get_embedding(wav.numpy())

    def distance(self, frames):
        mu, cov = calc_embd_statistics(frames)
        return float(calc_frechet_distance(self.mu, self.cov, mu, cov))

    def score(self, x, sr):
        return self.distance(self.embed(x, sr))

    def score_pair(self, base, cand):
        """Both takes' FAD on equally many embedding frames, the longer's
        thinned evenly across its length."""
        frames = [self.embed(*take) for take in (base, cand)]
        n = min(len(f) for f in frames)
        return [self.distance(f[np.linspace(0, len(f) - 1, n).round().astype(int)]) for f in frames]


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
    """The commit `side` rendered at, as SKILL.md's step 5 records it."""
    commit = side / "COMMIT"
    if not commit.exists():
        sys.exit(f"{commit} is missing, and a verdict names the commits it compared")
    return commit.read_text().strip()


def main():
    ap = argparse.ArgumentParser(description="Judge an A/B batch by measurement.")
    ap.add_argument("dir", type=Path, help="the batch: base/ and cand/ of same-named takes")
    ap.add_argument("--refs", type=Path, required=True, help="the reference corpus: audio/ and manifest.json")
    ap.add_argument(
        "--corpus-cache",
        type=Path,
        default=Path(sys.prefix) / "corpus-cache",
        help="the corpus's FAD statistics and the panel's screening, built here on a miss (default: the venv's)",
    )
    ap.add_argument("--score-identical", action="store_true", help="score byte-identical pairs too")
    args = ap.parse_args()
    base_dir, cand_dir = args.dir / "base", args.dir / "cand"

    try:
        takes = validate(args.dir)
    except Invalid as e:
        sys.exit(f"invalid batch: {e}")
    if takes.new or takes.gone:
        print(f"unpaired, left out: {', '.join(takes.new + takes.gone)}")
    commits = {"base": read_commit(base_dir), "cand": read_commit(cand_dir)}
    sides = (("base", base_dir), ("cand", cand_dir))
    changed = scored(args.dir, takes, args.score_identical)
    pairs = [
        {"take": n.removesuffix(".wav"), "mood": NAME.fullmatch(n)["mood"], "identical": n not in changed}
        for n in takes.paired
    ]
    record = {"dir": str(args.dir.resolve()), "commits": commits, "floor_re_take_db": FLOOR_RE_TAKE_DB}
    held, scales = [], {}

    if changed:
        # the device Audiobox's own `setup_model` picks
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        if device.type == "cuda":
            # cuDNN "can be a source of nondeterminism", and the algorithm it
            # picks "may be nondeterministic, unless ... cudnn.deterministic =
            # True" (https://docs.pytorch.org/docs/2.14/notes/randomness.html),
            # which would let two identical takes' scores differ
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        takes_lufs = float(
            np.median([mono_lufs(*sf.read(d / n, dtype="float32")) for n in changed for _, d in sides])
        )
        fad = Fad(args.refs, takes_lufs, args.corpus_cache)
        panel = dict(zip(PANEL, (Audiobox(), TuneJury(fad.model, device)), strict=True))
        record["screen"] = cached_screen(args.corpus_cache, fad, panel)
        if out := screened_out(record["screen"]):
            sys.exit(
                f"{', '.join(out)} rated BS.1534-3's mid-range anchor of the held-out references unimpaired past"
                f" §4.1.2's {SCREEN_SHARE:.0%} ({record['screen']}), so the panel can't seat it"
            )
        scales = {name: judge["scale"] for name, judge in record["screen"].items()}
        record["judges"] = {
            "audiobox": {"package": version("audiobox_aesthetics"), "checkpoint": AUDIOBOX},
            "tunejury": {"checkpoint": TUNEJURY, "encoders": [LAION_CLAP, MERT]},
            "fad": {
                "package": version("fadtk"),
                "embedding": fad.model.name,
                "checkpoint": LAION_CLAP,
                "refs": str(args.refs.resolve()),
                "lufs": fad.lufs,
            },
            "torch": version("torch"),
            "device": str(device),
        }
        record["fad_calibration"] = fad.calibration
        held = list(fad.calibration.values())
        for row in pairs:
            if row["identical"]:
                continue
            name = f"{row['take']}.wav"
            takes = {}
            for side, d in sides:
                x, sr = sf.read(d / name, dtype="float32")
                # the descriptors measure the take as rendered, the judges hear it floored
                row[side] = describe(x, sr)
                takes[side] = floored(x, name, mono_lufs(x, sr)), sr
                row[side]["scores"] = {j: judge.score(*takes[side]) for j, judge in panel.items()}
            row["base"]["scores"][GUARD], row["cand"]["scores"][GUARD] = fad.score_pair(takes["base"], takes["cand"])
            row["bands_delta_db"] = [c - b for b, c in zip(row["base"]["bands_db"], row["cand"]["bands_db"])]
            b, c = row["base"]["scores"], row["cand"]["scores"]
            row["deltas"] = {j: c[j] - b[j] for j in JUDGES}
            # nearer the corpus is better, so a gain is a fall in FAD
            row["gains"] = {j: -d if j == GUARD else d for j, d in row["deltas"].items()}
            print(f"scored {row['take']}", flush=True)

    result = judged(pairs, held, scales)
    verdict = {}
    for group, g in result["groups"].items():
        members = g["changed"]
        v = {"pairs": len(members), "identical": g["identical"], "gates": group != "all", "verdict": g["verdict"]}
        if members:
            for j, test in g["judges"].items():
                v[j] = {**test, "mean_delta": float(np.mean([p["deltas"][j] for p in members]))}
            v.update(g["tests"])
            v["to_explain"] = to_explain(members)
        verdict[group] = v
    record.update(
        alpha=result["alpha"],
        min_untied=result["min_untied"],
        margin=result["margin"],
        bands_hz=BANDS_HZ,
        pairs=pairs,
        verdict=verdict,
        dissent=result["dissent"],
        overall=result["overall"],
    )
    # built once, so judge.json's table is the one printed
    table = []
    table.append(f"\n{'take':<24}" + "".join(f" {'d' + LABELS[j]:>6}" for j in JUDGES) + f" {'max|dB|':>8} {'@Hz':>6}"
          f" {'centroid b/c':>13} {'crest b/c':>11} {'LR b/c':>11} {'clk/min b/c':>11} {'hat b/c':>11}")
    for p in pairs:
        if p["identical"]:
            table.append(f"{p['take'].removeprefix('gen_'):<24} identical")
            continue
        b, c, d = p["base"], p["cand"], p["deltas"]
        worst = int(np.argmax(np.abs(p["bands_delta_db"])))
        table.append(
            f"{p['take'].removeprefix('gen_'):<24}"
            + "".join(f" {d[j]:+6.3f}" for j in JUDGES)
            + f" {p['bands_delta_db'][worst]:+8.1f} {round(BANDS_HZ[worst]):>6}"
            f" {b['centroid_hz']:6.0f}/{c['centroid_hz']:<6.0f}"
            f" {b['crest_db']:5.1f}/{c['crest_db']:<5.1f}"
            f" {b['lr_corr']:5.2f}/{c['lr_corr']:<5.2f}"
            f" {b['clicks_per_min']:5.0f}/{c['clicks_per_min']:<5.0f}"
            f" {b[HAT_FLUX_KEY]:5.2f}/{c[HAT_FLUX_KEY]:<5.2f}"
        )
    table.append(f"\n{'group':<6} {'pairs':>5} {'same':>4} " + "".join(f" {LABELS[j] + ' w/l':>7}" for j in JUDGES)
          + f" {'mean d' + LABELS[GUARD]:>9}  {'panel w/l/t':>11} {'p':>7}  {'guard w/l':>9} {'p':>7}  verdict")
    for group, v in verdict.items():
        label = v["verdict"] if v["gates"] else f"({v['verdict']}, not gating)"
        if not v["pairs"]:
            table.append(f"{group:<6} {0:>5} {v['identical']:>4}  {label}")
            continue
        judges = " ".join(f"{v[j]['wins']:>3}/{v[j]['losses']:<3}" for j in JUDGES)
        panel, guard = v["panel"], v["guard"]
        table.append(
            f"{group:<6} {v['pairs']:>5} {v['identical']:>4}  {judges} {v[GUARD]['mean_delta']:+9.4f}"
            f"  {panel['wins']:>5}/{panel['losses']}/{panel['ties']:<3} {panel['p']:7.4f}"
            f"  {guard['wins']:>5}/{guard['losses']:<3} {guard['p']:7.4f}  {label}"
        )
    for group, against in result["dissent"].items():
        table.append(
            f"DISSENT {group}: {', '.join(against)} significantly favour base on their own tests, and no gate fails"
            + (f" ({GUARD} under the guard's margin)" if GUARD in against else "")
        )
    if changed:
        table.append(f"alpha {result['alpha']:.4g} a test, so one needs {result['min_untied']} untied pairs to ever reject")
        table.append(
            f"\n{LABELS[GUARD]} calibration, {len(held)} held-out references: median {np.median(held):.4f}"
            f" ({min(held):.4f}..{max(held):.4f}), so the guard's margin is {result['margin']:.4f};"
            f" takes: base median {np.median([p['base']['scores'][GUARD] for p in pairs if not p['identical']]):.4f},"
            f" cand median {np.median([p['cand']['scores'][GUARD] for p in pairs if not p['identical']]):.4f}"
        )
    for group, v in verdict.items():
        for t in v.get("to_explain", []):
            table.append(
                f"TO-EXPLAIN {group}: {t['band_hz']} Hz, {t['pairs_over']} of {v['pairs']} pairs past"
                f" ±{TO_EXPLAIN_DB:g} dB, median {t['median_db']:+.1f} ({t['min_db']:+.1f}..{t['max_db']:+.1f})"
            )
    record["table"] = "\n".join(table)
    (args.dir / "judge.json").write_text(json.dumps(record, indent=1) + "\n")
    print(record["table"])
    print(f"\nverdict: {result['overall']}  ({args.dir / 'judge.json'})")
    if failure := exit_message(result):
        sys.exit(failure)


if __name__ == "__main__":
    main()
