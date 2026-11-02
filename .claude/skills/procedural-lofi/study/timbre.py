"""What a track's spectrum doesn't show: stereo, transients, drum weight, the
noise floor, crackle and swing, the same way for a reference and for a take.

usage: timbre.py OUT.json FILE...   in the judge's venv
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ab_judge import clicks_per_min

SR = 44100
HOP = 441  # 10 ms
WIDTH_BANDS = {"<125": (20, 125), "125-500": (125, 500), "500-2k": (500, 2000), "2k-8k": (2000, 8000)}
# ab_judge.py's click meter
CLICK_HZ, CLICK_ORDER, CLICK_MADS, CLICK_GAP_S = 4000, 4, 12, 0.005
SILENT_DB = -50.0
WORKERS = 8
BAND_ORDER = 4
POWER_FLOOR = 1e-20
LOG_FLOOR = 1e-12
EDGE_S = 10
EDGES_MIN = 3
LOW_LO_HZ, LOW_HI_HZ = 40, 125
LOW_WIN_S = 0.03
PUNCH_HI_PCT, PUNCH_MID_PCT = 95, 50
BODY_LO_HZ, BODY_HI_HZ = 250, 2000
BODY_WIN_S = 0.05
SWING_HI_PCT, SWING_LO_PCT = 90, 10
N_FFT = 2048
FFT_HOP = 512
PERC_BANDS = {"<200": (0, 200), ">2k": (2000, 22050)}
HISS_LO_HZ, HISS_HI_HZ = 4000, 12000
HISS_WIN_S = 0.05
HISS_FLOOR_PCT = 10


def band(x, lo, hi):
    sos = signal.butter(BAND_ORDER, [lo, hi], "bandpass", fs=SR, output="sos")
    return signal.sosfiltfilt(sos, x)


def power(x):
    return float(np.mean(x**2)) + POWER_FLOOR


def env_db(x, win_s):
    win = int(win_s * SR)
    smooth = signal.fftconvolve(x**2, np.ones(win) / win, mode="same")
    return 10 * np.log10(np.maximum(smooth[::HOP], LOG_FLOOR))


def features(path):
    y, _ = librosa.load(path, sr=SR, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y])
    # the fades aren't the track's sound
    edge = int(EDGE_S * SR)
    if y.shape[1] > EDGES_MIN * edge:
        y = y[:, edge:-edge]
    left, right = y[0].astype(np.float64), y[1].astype(np.float64)
    mid, side = (left + right) / 2, (left - right) / 2
    rms = np.sqrt(power(mid))
    out = {"rms_db": 20 * np.log10(rms), "peak_rms_db": 20 * np.log10(np.abs(mid).max() / rms)}
    for name, (lo, hi) in WIDTH_BANDS.items():
        out[f"side_mid_db {name}"] = 10 * np.log10(power(band(side, lo, hi)) / power(band(mid, lo, hi)))
    out["lr_correlation"] = float(np.corrcoef(left, right)[0, 1])

    low = env_db(band(mid, LOW_LO_HZ, LOW_HI_HZ), LOW_WIN_S)
    low = low[low > low.max() + SILENT_DB]
    out["low_punch_db"] = float(np.percentile(low, PUNCH_HI_PCT) - np.percentile(low, PUNCH_MID_PCT))
    body = env_db(band(mid, BODY_LO_HZ, BODY_HI_HZ), BODY_WIN_S)
    body = body[body > body.max() + SILENT_DB]
    out["body_swing_db"] = float(np.percentile(body, SWING_HI_PCT) - np.percentile(body, SWING_LO_PCT))

    stft = librosa.stft(mid.astype(np.float32), n_fft=N_FFT, hop_length=FFT_HOP)
    harm, perc = librosa.decompose.hpss(stft)
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    p_all, p_perc = np.abs(stft) ** 2, np.abs(perc) ** 2
    out["perc_share_pct"] = 100 * float(p_perc.sum() / p_all.sum())
    for name, (lo, hi) in PERC_BANDS.items():
        keep = (freqs >= lo) & (freqs < hi)
        out[f"perc_share_pct {name}"] = 100 * float(p_perc[keep].sum() / (p_all[keep].sum() + POWER_FLOOR))

    hiss = env_db(band(mid, HISS_LO_HZ, HISS_HI_HZ), HISS_WIN_S)
    out["hiss_floor_db_re_rms"] = float(np.percentile(hiss, HISS_FLOOR_PCT) - 20 * np.log10(rms))
    out["clicks_per_min"] = float(clicks_per_min(mid, SR))
    out["onsets_per_s"] = float(
        librosa.onset.onset_detect(y=mid.astype(np.float32), sr=SR, hop_length=FFT_HOP, units="time").size / (mid.size / SR)
    )
    return {k: float(v) for k, v in out.items()}


def one(path):
    try:
        return str(path), features(path)
    except Exception as err:  # one bad file mustn't lose the run
        return str(path), {"error": repr(err)}


def main():
    out, files = Path(sys.argv[1]), sys.argv[2:]
    with ProcessPoolExecutor(WORKERS) as pool:
        rows = dict(pool.map(one, files))
    out.write_text(json.dumps(rows, indent=0) + "\n")
    print(len(rows), "read,", sum("error" in r for r in rows.values()), "failed")


if __name__ == "__main__":
    main()
