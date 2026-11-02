"""Splits the private corpus into drums, bass, other and vocals on Modal's T4s,
into a Volume of its own, so what a mix hides can be measured stem by stem.

  modal run stems.py::split [--limit N]
  modal run stems.py::split_takes --takes DIR   the control: our own takes

A reference is named by its manifest index on the Volume and in everything
this prints, as modal_judge.py names it: the corpus's names stay private.
Stems are derived from copyrighted audio: they stay on the Volume.
"""

import json
import sys
from pathlib import Path

import modal

REFS, STUDY = "/refs", "/study"
# Rouard et al., ICASSP 2023; "achieves a SDR of 9.00 dB on the MUSDB HQ test
# set" (https://github.com/facebookresearch/demucs)
MODEL = "htdemucs_ft"
# `judge_local.py`'s SHARDS, for its reason: a job this GPU-bound queues past it
CONCURRENT = 8
TIMEOUT_S = 1800
CPU = 4.0
MEMORY_MIB = 16384
SHIFTS = 1
OVERLAP = 0.25

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install("torch==2.0.1", "torchaudio==2.0.2", "demucs==4.0.1", "soundfile==0.12.1", "numpy<2")
    .env({"TORCH_HOME": f"{STUDY}/torch"})
)
app = modal.App("lofi-study", image=image)
refs = modal.Volume.from_name("lofi-judge-refs")
study = modal.Volume.from_name("lofi-study", create_if_missing=True)


@app.function(
    gpu="T4", cpu=CPU, memory=MEMORY_MIB, timeout=TIMEOUT_S, max_containers=CONCURRENT,
    volumes={REFS: refs.with_mount_options(read_only=True), STUDY: study},
)
def split_one(job):
    import numpy as np
    import soundfile as sf
    import torch
    from demucs.apply import apply_model
    from demucs.audio import convert_audio
    from demucs.pretrained import get_model

    key, source = job
    out = Path(STUDY) / "stems" / key
    if (out / "done").exists():
        return {"key": key, "skipped": True}
    model = get_model(MODEL)
    model.cuda().eval()
    data, rate = sf.read(source, dtype="float32", always_2d=True)
    wav = convert_audio(torch.from_numpy(data.T), rate, model.samplerate, model.audio_channels)
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / ref.std()
    with torch.no_grad():
        sources = apply_model(model, wav[None], device="cuda", shifts=SHIFTS, split=True, overlap=OVERLAP)[0]
    sources = (sources * ref.std() + ref.mean()).cpu().numpy()
    out.mkdir(parents=True, exist_ok=True)
    powers = {}
    for stem, audio in zip(model.sources, sources):
        sf.write(out / f"{stem}.flac", audio.T, model.samplerate, subtype="PCM_24")
        powers[stem] = float(np.mean(audio**2))
    (out / "done").write_text(json.dumps(powers) + "\n")
    study.commit()
    return {"key": key, "powers": powers}


def report(jobs):
    results = list(split_one.map(jobs, return_exceptions=True))
    good = [r for r in results if isinstance(r, dict)]
    print(f"{len(good)} of {len(jobs)} split")
    for job, r in zip(jobs, results):
        if not isinstance(r, dict):
            print("failed:", job[0], type(r).__name__)
    # a card over what survived would read like one over every take
    if len(good) != len(jobs):
        sys.exit(f"stems.py: {len(jobs) - len(good)} of {len(jobs)} failed")


@app.local_entrypoint()
def split(limit: int = 0):
    manifest = json.loads(b"".join(refs.read_file("/manifest.json")))["tracks"]
    jobs = [(f"{i:03d}", f"{REFS}/audio/{t['file']}") for i, t in enumerate(manifest)]
    report(jobs[:limit] if limit else jobs)


@app.local_entrypoint()
def split_takes(takes: str):
    files = sorted(Path(takes).glob("*.wav"))
    with study.batch_upload(force=True) as upload:
        for f in files:
            upload.put_file(str(f), f"/takes-in/{f.name}")
    report([(f"take-{f.stem}", f"{STUDY}/takes-in/{f.name}") for f in files])
