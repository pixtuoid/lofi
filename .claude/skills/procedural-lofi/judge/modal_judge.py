"""`ab_judge.py` on Modal's T4s, a shard of the pairs on each, over the
reference corpus and the models two Volumes cache, for a batch this machine
uploads.

Usage, with modal-requirements.txt's client and a configured token:
  modal run modal_judge.py::upload_refs --refs-dir REFS --corpus STYLE
                                          STYLE's corpus, from REFS to the
                                          Volume's /STYLE
  modal run modal_judge.py::provision     build the caches over it
  modal run modal_judge.py::check         run every model in this lock's image
                                          on the warm-up take, offline
  modal run modal_judge.py::judge --batch DIR --base SHA --cand SHA [--out OUT] [--force-score]
                                          judge DIR's base/ and cand/, rendered
                                          at the commits their COMMIT files name

REFS is the private corpus: its manifest's names and its audio never leave
Modal in anything this prints or returns, which names a reference by its
manifest index.
"""

import hashlib
import json
import os
import re
import resource
import subprocess
import sys
import time
from pathlib import Path

import modal

# where the image keeps ab_judge.py and the scripts it imports, and this
# directory beside them, where ab_judge finds validate_batch
REMOTE_SCRIPTS, REMOTE_JUDGE = "/root/scripts", "/root/judge"
HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent / "scripts" if modal.is_local() else Path(REMOTE_SCRIPTS)
JUDGE = HERE if modal.is_local() else Path(REMOTE_JUDGE)
MODELS, REFS, BATCH = "/models", "/refs", "/batch"
# the style whose corpus judges every take until #93's step 4 splits the batch
# by style
JUDGED = "hiphop"
sys.path.insert(0, str(JUDGE))
from judge_local import dealt, exit_code, published, redact, select

# the uv the image and SKILL.md's install run, which modal-requirements.in
# pins where Dependabot moves it
UV_VERSION = re.search(r"^uv==(\S+)", (JUDGE / "modal-requirements.txt").read_text(), re.M)[1]
# the lock, baked in where uv reads it a second time as the build constraints:
# encodec, wget and progressbar publish only an sdist
# (https://pypi.org/pypi/encodec/0.1.1/json and wget's and progressbar's), and
# uv's `--require-build-hashes`, a preview feature, "only applies to build
# dependencies" (https://docs.astral.sh/uv/pip/compile/,
# /uv/concepts/preview/): under `--require-hashes` alone, uv 0.13.0 built
# them with the newest setuptools, unpinned
LOCK = HERE / "judge-requirements.txt"
REMOTE_LOCK = "/root/judge-requirements.txt"

# over what the cache build and the full judge batches used, which the
# commit that set them records; the timeout is twice the slowest batch
CPU = 2.0
MEMORY_MIB = 12288
JUDGE_TIMEOUT_S = 4500

# the rehearsal's limit, ten times its wall time on a T4, which the commit
# that set it records, so a hung model ends it, not JUDGE_TIMEOUT_S later
# (opinion)
REHEARSAL_TIMEOUT_S = 650

# the warm-up pair `build` judges: a take at lofi's own rate (`SAMPLE_RATE`
# in src/dsp.rs), the one a batch's takes arrive at
WARMUP_RATE = 44100
# a few of ab_judge.py's `WINDOW_S` windows, so every judge scores whole ones
# (opinion)
WARMUP_S = 30
# the noise's RMS: the pair sits louder than any reference, so `Fad` builds
# its statistics at the quietest reference's level, which every real batch's
# takes are over
WARMUP_LEVEL = 0.25
# the cand side's gain, so the pair differs: an identical one is never scored
WARMUP_CAND_GAIN = 0.5


image = (
    modal.Image.debian_slim(python_version="3.12")
    .add_local_file(LOCK, REMOTE_LOCK, copy=True)
    .uv_pip_install(
        requirements=[str(LOCK)],
        extra_options=f"--require-hashes --build-constraints {REMOTE_LOCK} --require-build-hashes",
        uv_version=UV_VERSION,
    )
    .env({"HF_HOME": MODELS})
    .add_local_dir(SCRIPTS, REMOTE_SCRIPTS, ignore=["**/__pycache__"])
    .add_local_dir(JUDGE, REMOTE_JUDGE, ignore=["**/__pycache__"])
)
app = modal.App("lofi-judge", image=image)
models = modal.Volume.from_name("lofi-judge-models", create_if_missing=True)
refs = modal.Volume.from_name("lofi-judge-refs", create_if_missing=True)


def tracks():
    return json.loads((Path(REFS) / JUDGED / "manifest.json").read_text())["tracks"]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def took(started, shards=()):
    """What the calling function's children and `shards` have taken since
    `started`: wall time, their CPU and the highest peak RSS."""
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {
        "wall_s": time.time() - started,
        "cpu_s": usage.ru_utime + usage.ru_stime + sum(shard["cpu_s"] for shard in shards),
        "peak_rss_mib": max([usage.ru_maxrss / 1024, *(shard["peak_rss_mib"] for shard in shards)]),
    }


def offline():
    return {**os.environ, "HF_HUB_OFFLINE": "1"}


def run_judge(batch, *flags, **options):
    """`ab_judge.py` over `batch` and the corpus, in a child whose output
    goes nowhere until a caller masks it: nothing a model or a library prints
    reaches a public log."""
    command = [sys.executable, f"{REMOTE_SCRIPTS}/ab_judge.py", str(batch), "--refs", f"{REFS}/{JUDGED}", "--corpus-cache", f"{REFS}/corpus-cache"]
    return subprocess.run([*command, *flags], capture_output=True, text=True, **options)


def mounted(batch):
    """A scoring function's Volumes: `batch`, and the caches read-only."""
    return {
        BATCH: batch,
        MODELS: models.with_mount_options(read_only=True),
        REFS: refs.with_mount_options(read_only=True),
    }


def write_warm(path, gain):
    """Writes the warm-up take, seeded noise at `gain`, to `path`."""
    import numpy as np
    import soundfile as sf

    take = WARMUP_LEVEL * np.random.default_rng(0).standard_normal((WARMUP_S * WARMUP_RATE, 2))
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, take * gain, WARMUP_RATE, subtype="PCM_16")


@app.function(gpu="T4", cpu=CPU, memory=MEMORY_MIB, volumes={MODELS: models, REFS: refs}, timeout=JUDGE_TIMEOUT_S)
def build():
    """Caches every model, the corpus's FAD statistics and the panel's
    screening by judging a warm-up pair as a batch would be judged, online,
    then measures what it took."""
    warm = Path("/tmp/warm")
    for side, gain in (("base", 1.0), ("cand", WARMUP_CAND_GAIN)):
        write_warm(warm / side / "gen_day_empty_000.wav", gain)
        (warm / side / "COMMIT").write_text("0" * 40 + "\n")
    started = time.time()
    run = run_judge(warm)
    models.commit()
    refs.commit()
    built = (warm / "judge.json").is_file()
    if not built:
        print(redact(run.stdout + run.stderr, tracks()))
    return {"built": built, **took(started)}


# offline, its network blocked, and mounting the models alone, read-only: a
# lock not yet merged writes nothing a judge reads, and the corpus isn't there
# for it to print into a public log
@app.function(gpu="T4", cpu=CPU, memory=MEMORY_MIB, timeout=REHEARSAL_TIMEOUT_S, block_network=True, restrict_modal_access=True)
def rehearse():
    """The judge's environment as this image builds it: `ab_judge.py`'s
    tests, then its `smoke`, every model on the warm-up take, each a child
    process `took` counts; a failure raises out of the call."""
    started = time.time()
    take = Path("/tmp/smoke.wav")
    write_warm(take, 1.0)
    smoke = (
        "import sys; sys.path.insert(0, sys.argv[1]); import ab_judge, soundfile as sf;"
        " print('smoke scores:', ab_judge.smoke(*sf.read(sys.argv[2], dtype='float32'), float(sys.argv[3])))"
    )
    subprocess.run([sys.executable, "-I", f"{REMOTE_SCRIPTS}/test_ab_judge.py"], env=offline(), check=True)
    subprocess.run([sys.executable, "-I", "-c", smoke, REMOTE_SCRIPTS, str(take), str(WARMUP_CAND_GAIN)], env=offline(), check=True)
    return took(started)


# one shard to a container, since a ready one "will be reused"
# (https://modal.com/docs/guide/cold-start): the child's usage `took` reads and
# the rows it leaves are then this shard's alone
@app.function(gpu="T4", cpu=CPU, memory=MEMORY_MIB, timeout=JUDGE_TIMEOUT_S, single_use_containers=True)
def score_shard(names):
    """Scores the pairs `names` of the batch the caller mounts at BATCH,
    offline, over the read-only caches, and returns their rows and seat with
    what it took, or None with its log, names masked, Modal's."""
    started = time.time()
    rows = Path("/tmp/rows.json")
    run = run_judge(BATCH, "--shard", str(rows), *names, env=offline())
    if not rows.is_file():
        print(redact(run.stdout + run.stderr, tracks()))
        return None
    return {**json.loads(rows.read_text()), **took(started)}


# no T4 of its own: it runs the tests before any shard takes one, then waits
@app.function(cpu=CPU, memory=MEMORY_MIB, timeout=JUDGE_TIMEOUT_S)
def score(batch, paired, changed):
    """Judges the pairs `changed` of `paired` on the Volume `batch`, which
    the caller mounts here too, a shard to each container, and returns what's
    safe to publish: the table, the record without the corpus's names, and
    what the run took."""
    started = time.time()
    known = tracks()
    # the tests that need this venv, which runs nowhere else: a broken floor
    # or corpus cache still prints a verdict
    tests = subprocess.run(
        [sys.executable, "-I", f"{REMOTE_SCRIPTS}/test_ab_judge.py"], capture_output=True, text=True, env=offline()
    )
    if tests.returncode:
        print(redact(tests.stdout + tests.stderr, known))
        return {"outcome": "infra failure", "message": "ab_judge.py's tests failed; their log, names masked, is Modal's", **took(started)}
    shards = list(score_shard.with_options(volumes=mounted(batch)).map(dealt(changed), return_exceptions=True))
    if not all(isinstance(shard, dict) for shard in shards):
        return {"outcome": "infra failure", "message": "a shard stopped before its rows; its log, names masked, is Modal's", **took(started)}
    plan = Path("/tmp/plan.json")
    plan.write_text(json.dumps({"paired": paired, "changed": changed, "shards": shards}))
    run = run_judge(BATCH, "--merge", str(plan), env=offline())
    written = Path(BATCH) / "judge.json"
    if not written.is_file():
        print(redact(run.stdout + run.stderr, known))
        return {"outcome": "infra failure", "message": "the judge stopped before a verdict; its log, names masked, is Modal's", **took(started, shards)}
    return {**published(json.loads(written.read_text()), known), **took(started, shards)}


@app.local_entrypoint()
def provision():
    """Builds the caches over the judged corpus in the refs Volume, which
    `Fad` refuses unless every reference matches the manifest."""
    report = build.remote()
    print(f"caches: {report}")
    if not report["built"]:
        raise SystemExit(1)


@app.local_entrypoint()
def check():
    """Rehearses the judge in an image built from this checkout's lock, over
    the cached models, read-only; a failed test or model ends `modal run`
    non-zero."""
    result = rehearse.with_options(volumes={MODELS: models.with_mount_options(read_only=True)}).remote()
    print(f"rehearsed: every model ran; wall {result['wall_s']:.0f} s,"
          f" judge CPU {result['cpu_s']:.0f} s, peak RSS {result['peak_rss_mib']:.0f} MiB")


@app.local_entrypoint()
def judge(batch: str, base: str, cand: str, out: str = "", force_score: bool = False):
    """Judges `batch`'s base/ and cand/, rendered at commits `base` and
    `cand`, through a Volume that lives as long as the run, prints the table,
    and writes the result to `out`. Only the pairs `judge_local.select` picks
    go up, so a batch with none makes no remote call; `force_score` sends and
    scores every pair, the proof that identical ones tie."""
    root = Path(batch)
    takes, differ = select(root, base, cand, force_score)
    if differ:
        with modal.Volume.ephemeral() as vol:
            with vol.batch_upload() as upload:
                for side in ("base", "cand"):
                    upload.put_file(str(root / side / "COMMIT"), f"/{side}/COMMIT")
                    for n in differ:
                        upload.put_file(str(root / side / n), f"/{side}/{n}")
            result = score.with_options(volumes=mounted(vol)).remote(vol, takes.paired, differ)
        print(result.get("table", result.get("message")))
        print(f"\nverdict: {result['outcome']}")
        print(f"\nwall {result['wall_s']:.0f} s, judge CPU {result['cpu_s']:.0f} s, peak RSS {result['peak_rss_mib']:.0f} MiB")
    else:
        result = {"outcome": "UNCHANGED"}
        print(f"verdict: UNCHANGED, all {len(takes.paired)} pairs byte-identical; nothing was sent or scored")
    result["identical_not_sent"] = len(takes.paired) - len(differ)
    for side in ("new", "gone"):
        result[side] = getattr(takes, side)
        if result[side]:
            print(f"{side} at cand, unscored: {' '.join(result[side])}")
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / "verdict.json").write_text(json.dumps(result, indent=1) + "\n")
    code = exit_code(result["outcome"])
    if code:
        raise SystemExit(code)


@app.local_entrypoint()
def upload_refs(refs_dir: str, corpus: str):
    """Uploads from `refs_dir` every reference of `corpus`, once its sha256
    matches the manifest, over any the refs Volume holds under its name, then
    the manifest, last: a judge never reads a manifest whose audio isn't
    there. The corpora share `corpus-cache`, whose files their manifests'
    sha256s name."""
    local = Path(refs_dir)
    manifest = json.loads((local / "manifest.json").read_text())["tracks"]
    with refs.batch_upload(force=True) as upload:
        for i, t in enumerate(manifest):
            path = local / "audio" / t["file"]
            if sha256(path.read_bytes()) != t["sha256"]:
                raise SystemExit(f"reference {i} doesn't match its manifest sha256 here")
            upload.put_file(str(path), f"/{corpus}/audio/{t['file']}")
    with refs.batch_upload(force=True) as upload:
        upload.put_file(str(local / "manifest.json"), f"/{corpus}/manifest.json")
    print(f"uploaded {corpus}'s {len(manifest)} references and its manifest")
