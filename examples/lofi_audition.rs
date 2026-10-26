//! The generator's STATISTICAL gate: render N random-seed takes to wav so
//! `ab_judge.py` judges the COMPOSER, not one take. A take is the first
//! `--songs` songs of a seed's run at one of pixtuoid's tiers, or under a named
//! `--sweep` of control calls, through either engine (`batch::Engine`): a
//! stream take carries the plate on its two sides, a whole-lane one is dry and
//! centred.
//!
//! Usage:
//!   cargo run --release --example lofi_audition -- \
//!     [--engine whole|stream] [--mood day|night] [--seeds N] [--start S] [--out DIR] \
//!     [--tier empty|moderate|busy | --solo pad|sparkle|keys|drums|texture|bass] \
//!     [--sweep intensity|dusk|twilight|rain] [--songs N] [--batch judge] [--shuffle KEYFILE]
//!   `--solo` renders a lane whole, so it takes `--engine whole`. `--batch judge`
//!   renders what `ab_judge.py` scores in place of the mood, tier and seeds.

#[path = "common/batch.rs"]
mod batch;
#[path = "common/loudness.rs"]
mod loudness;

use std::f64::consts::PI;
use std::fs::File;
use std::hash::{BuildHasher as _, RandomState};
use std::io::{self, BufWriter, Write as _};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use pixtuoid_lofi::Frame;
use pixtuoid_lofi::audition::Lane;
use pixtuoid_lofi::dsp::{NoiseStream, SAMPLE_RATE};

use batch::{Batch, Engine, lane_gains, parse_choice, usage_error};
use loudness::{fold, gated_lufs};

/// Every take renders at ONE gated loudness, so a blind A/B compares takes, not
/// gain.
const TARGET_LUFS: f64 = -18.0;

/// What a solo take renders at instead, since a lane alone peaks past full
/// scale at [`TARGET_LUFS`]: the loudest level at which every lane's solo,
/// both moods and seeds 0–23, kept its peak under full scale, as the commit
/// that set it records, less [`SOLO_MARGIN_DB`].
const SOLO_TARGET_LUFS: f64 = -31.6 - SOLO_MARGIN_DB;

/// Room for a seed past that sweep to peak higher (opinion).
const SOLO_MARGIN_DB: f64 = 1.0;

/// How long, in seconds, a take fades in and out under a raised cosine: cut
/// from a stream or a loop, it starts and ends mid-waveform, and the click
/// there differs by engine, so it could tell a blind A/B's sides apart. The
/// length is opinion.
const EDGE_FADE_S: f64 = 0.01;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("lofi_audition: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> io::Result<()> {
    let args = Args::parse(std::env::args().skip(1))?;
    std::fs::create_dir_all(&args.out)?;
    let takes = args.takes()?;
    let files = match args.key {
        Some(_) => shuffled_names(takes.len()),
        None => takes.iter().map(|take| take.name.clone()).collect(),
    };
    let mut order: Vec<(&String, &Take)> = files.iter().zip(&takes).collect();
    // rendered in file-name order, so the files' times don't give the seeds away
    order.sort_by_key(|&(file, _)| file);
    for &(file, take) in &order {
        let path = args.out.join(file);
        write_wav(&path, &args.render(&take.run, take.seed)?)?;
        if args.key.is_none() {
            let (seed, score) = (take.seed, take.run.score(take.seed));
            println!(
                "  seed {seed:3}  {:>3.0} bpm  lead={:5}  {}",
                score.bpm(),
                score.lead_voice_name(),
                path.display()
            );
        }
    }
    match &args.key {
        Some(key) => {
            let mut w = BufWriter::new(File::create(key)?);
            for (file, take) in &order {
                writeln!(w, "{file} {}", take.name)?;
            }
            w.flush()?;
            println!("\nblind-audition:");
        }
        None => println!("\nblind-audition (--shuffle KEYFILE hides the seeds):"),
    }
    for (file, ..) in &order {
        println!("  afplay {}", args.out.join(file).display());
    }
    Ok(())
}

/// One take a run renders: its seed in one of the run's batches, and its
/// file name.
struct Take {
    run: Batch,
    seed: u64,
    name: String,
}

/// What a run renders, and where.
struct Args {
    batch: Batch,
    out: PathBuf,
    /// The lane `--solo` plays alone, as its index in [`Lane::NAMES`].
    solo: Option<usize>,
    key: Option<PathBuf>,
}

impl Args {
    fn parse(mut args: impl Iterator<Item = String>) -> io::Result<Self> {
        let mut parsed = Self {
            batch: Batch::default(),
            out: PathBuf::from("audio-demos"),
            solo: None,
            key: None,
        };
        while let Some(a) = args.next() {
            if parsed.batch.parse_flag(&a, &mut args)? {
                continue;
            }
            match a.as_str() {
                "--out" => match args.next() {
                    Some(v) => parsed.out = PathBuf::from(v),
                    None => return Err(usage_error("--out needs a directory".into())),
                },
                "--solo" => parsed.solo = Some(parse_choice(args.next(), "--solo", &Lane::NAMES)?),
                "--shuffle" => match args.next() {
                    Some(v) => parsed.key = Some(PathBuf::from(v)),
                    None => return Err(usage_error("--shuffle needs a key file".into())),
                },
                other => {
                    return Err(usage_error(format!(
                        "unknown argument {other:?}; valid: --engine --mood --seeds --start --out --tier --sweep --songs --batch --solo --shuffle"
                    )));
                }
            }
        }
        if parsed.solo.is_some() {
            if parsed.batch.tier.is_some()
                || parsed.batch.sweep.is_some()
                || parsed.batch.named.is_some()
            {
                return Err(usage_error(
                    "--solo plays one lane, so it takes no --tier, --sweep or --batch".into(),
                ));
            }
            if parsed.batch.engine != Engine::WholeLane {
                return Err(usage_error(
                    "--solo renders one lane whole, so it takes --engine whole".into(),
                ));
            }
        }
        Ok(parsed)
    }

    /// Every take of every run. No engine changes a name, so `ab_listen.py`
    /// pairs a whole-lane batch with a stream batch; an error if two share
    /// one, since the second would overwrite the first's file.
    fn takes(&self) -> io::Result<Vec<Take>> {
        let runs = self.batch.runs();
        let takes = runs.iter().flat_map(|run| {
            let mix = match self.solo {
                Some(s) => Lane::NAMES[s],
                None => run.mix_name(),
            };
            let mood = run.mood_name();
            (run.seeds()).map(move |seed| Take {
                run: *run,
                seed,
                name: format!("gen_{mood}_{mix}_{seed:03}.wav"),
            })
        });
        let takes: Vec<Take> = takes.collect();
        let mut names: Vec<&str> = takes.iter().map(|take| take.name.as_str()).collect();
        names.sort_unstable();
        match names.windows(2).find(|pair| pair[0] == pair[1]) {
            Some(pair) => Err(usage_error(format!(
                "two takes would both write {}",
                pair[0]
            ))),
            None => Ok(takes),
        }
    }

    /// `seed`'s take in `run` at its target loudness; an error if that clips
    /// it, which would write a different take.
    fn render(&self, run: &Batch, seed: u64) -> io::Result<Vec<Frame>> {
        let mut take = match run.engine {
            Engine::WholeLane => {
                let gains = |controls| match self.solo {
                    Some(s) => {
                        let mut gains = [0.0; Lane::ALL.len()];
                        gains[s] = 1.0;
                        gains
                    }
                    None => lane_gains(controls),
                };
                (run.whole(seed, gains).into_iter())
                    .flat_map(|(.., mix)| mix)
                    .map(|x| [x; 2])
                    .collect()
            }
            Engine::Stream => run.perform(seed),
        };
        fade_edges(&mut take);
        let target = self.target_lufs();
        let gain = 10f64.powf((target - gated_lufs(&fold(&take), SAMPLE_RATE)) / 20.0) as f32;
        for v in take.iter_mut().flatten() {
            *v *= gain;
        }
        if take.iter().flatten().any(|v| v.abs() > 1.0) {
            return Err(io::Error::other(format!(
                "seed {seed} clips at {target} LUFS"
            )));
        }
        Ok(take)
    }

    /// The gated loudness every take of the run renders at.
    fn target_lufs(&self) -> f64 {
        match self.solo {
            Some(_) => SOLO_TARGET_LUFS,
            None => TARGET_LUFS,
        }
    }
}

/// `take` faded in from silence and out to it over [`EDGE_FADE_S`].
fn fade_edges(take: &mut [Frame]) {
    let n = (EDGE_FADE_S * f64::from(SAMPLE_RATE)) as usize;
    let last = take.len() - 1;
    for i in 0..n {
        let w = (0.5 * (1.0 - (PI * i as f64 / n as f64).cos())) as f32;
        for at in [i, last - i] {
            take[at] = take[at].map(|v| v * w);
        }
    }
}

/// `count` opaque file names in a random order, seeded from a [`RandomState`],
/// which is "initialized with random keys"
/// (<https://doc.rust-lang.org/std/hash/struct.RandomState.html#method.new>).
fn shuffled_names(count: usize) -> Vec<String> {
    shuffle_names(
        count,
        &mut NoiseStream::new(RandomState::new().hash_one(count)),
    )
}

/// `count` opaque file names in the order `rng` shuffles them into.
fn shuffle_names(count: usize, rng: &mut NoiseStream) -> Vec<String> {
    let mut order: Vec<usize> = (0..count).collect();
    for i in (1..count).rev() {
        let j = ((rng.unit() * (i + 1) as f32) as usize).min(i);
        order.swap(i, j);
    }
    order.iter().map(|i| format!("take_{i:02}.wav")).collect()
}

/// `x` as the 16-bit sample the file holds, which the loudness test reads too.
fn pcm(x: f32) -> i16 {
    (x * f32::from(i16::MAX)) as i16
}

/// 16-bit stereo RIFF/WAVE.
fn write_wav(path: &Path, take: &[Frame]) -> io::Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    let n = take.len() as u32;
    let data_len = n * 4; // 2 channels × i16
    let byte_rate = SAMPLE_RATE * 2 * 2;
    w.write_all(b"RIFF")?;
    w.write_all(&(36 + data_len).to_le_bytes())?;
    w.write_all(b"WAVEfmt ")?;
    w.write_all(&16u32.to_le_bytes())?;
    w.write_all(&1u16.to_le_bytes())?; // PCM
    w.write_all(&2u16.to_le_bytes())?; // stereo
    w.write_all(&SAMPLE_RATE.to_le_bytes())?;
    w.write_all(&byte_rate.to_le_bytes())?;
    w.write_all(&4u16.to_le_bytes())?; // block align
    w.write_all(&16u16.to_le_bytes())?;
    w.write_all(b"data")?;
    w.write_all(&data_len.to_le_bytes())?;
    for s in take.iter().flatten() {
        w.write_all(&pcm(*s).to_le_bytes())?;
    }
    w.flush()
}

#[cfg(test)]
mod tests {
    use super::*;
    use batch::Sweep;
    use pixtuoid_lofi::audition;

    fn parsed(args: &[&str]) -> Args {
        Args::parse(args.iter().map(|&a| a.to_owned())).expect("the arguments parse")
    }

    /// `seed`'s take in the run's one batch.
    fn rendered(args: &[&str], seed: u64) -> io::Result<Vec<Frame>> {
        let args = parsed(args);
        args.render(&args.batch, seed)
    }

    #[test]
    fn shuffled_names_are_each_take_once_and_not_kept_in_order() {
        for count in [1, 2, 7, 12] {
            let want: Vec<String> = (0..count).map(|i| format!("take_{i:02}.wav")).collect();
            let mut moved = false;
            for seed in 0..16 {
                let mut names = shuffle_names(count, &mut NoiseStream::new(seed));
                moved |= names != want;
                names.sort();
                assert_eq!(names, want, "{count} names, seed {seed}");
            }
            // one name has only one order
            assert!(
                count == 1 || moved,
                "{count} names stayed in order for every seed"
            );
        }
    }

    #[test]
    fn a_take_keeps_its_name_under_either_engine() {
        let [whole, stream] = ["whole", "stream"].map(|engine| {
            parsed(&["--engine", engine, "--mood", "night", "--tier", "empty"])
                .takes()
                .expect("the takes' names are distinct")
        });
        let names = |takes: Vec<Take>| {
            takes
                .into_iter()
                .map(|t| (t.seed, t.name))
                .collect::<Vec<_>>()
        };
        let (whole, stream) = (names(whole), names(stream));
        assert_eq!(whole, stream);
        // the shape `ab_listen.py`'s NAME pairs
        assert_eq!(whole[0], (0, "gen_night_empty_000.wav".to_owned()));
    }

    #[test]
    fn the_judge_batch_names_each_take_once_by_its_group_mix_and_seed() {
        let takes = parsed(&["--batch", "judge"])
            .takes()
            .expect("the takes' names are distinct");
        for take in &takes {
            // the group, apart from `mood_name`, since `ab_judge.py` groups by it
            let group = match take.run.sweep {
                Some(sweep) => sweep.name(),
                None => take.run.mood_name(),
            };
            let mix = match take.run.sweep {
                Some(_) => take.run.mix_name(),
                None => take.run.tier().name(),
            };
            assert_eq!(take.name, format!("gen_{group}_{mix}_{:03}.wav", take.seed));
        }
        assert!(
            Args::parse(
                ["--batch", "judge", "--solo", "pad"]
                    .map(str::to_owned)
                    .into_iter()
            )
            .is_err()
        );
    }

    #[test]
    fn both_engines_write_a_take_as_long_and_at_the_target_loudness() {
        let seed = 1;
        let [whole, stream] = ["whole", "stream"].map(|engine| {
            let take = rendered(&["--engine", engine], seed);
            let written: Vec<Frame> = (take.expect("the take renders").iter())
                .map(|frame| frame.map(|x| f32::from(pcm(x)) / f32::from(i16::MAX)))
                .collect();
            let lufs = gated_lufs(&fold(&written), SAMPLE_RATE);
            assert!(
                (lufs - TARGET_LUFS).abs() <= 0.1,
                "{engine} writes {lufs:.3} LUFS"
            );
            written.len()
        });
        assert_eq!(whole, stream, "the engines' takes last apart");
    }

    #[test]
    fn both_engines_write_a_take_from_silence_to_silence() {
        for engine in ["whole", "stream"] {
            let take = rendered(&["--engine", engine], 2);
            let edges = take
                .map(|t| [t[0], t[t.len() - 1]].map(|frame| frame.map(pcm)))
                .ok();
            assert_eq!(edges, Some([[0; 2]; 2]), "{engine}'s first and last frames");
        }
    }

    #[test]
    fn every_sweep_renders_alike_twice_and_as_long_under_either_engine() {
        for sweep in Sweep::ALL {
            for engine in ["whole", "stream"] {
                let args = parsed(&["--engine", engine, "--sweep", sweep.name(), "--seeds", "1"]);
                for run in args.batch.runs() {
                    let take = args.render(&run, 0).expect("the sweep renders");
                    let again = args.render(&run, 0).expect("the sweep renders");
                    assert!(take == again, "{sweep:?} under {engine} renders apart");
                    let songs = run.timeline(0).songs;
                    let len: usize = songs.iter().map(|&(_, len)| len).sum();
                    assert_eq!(take.len(), len, "{sweep:?} under {engine}");
                }
            }
        }
    }

    #[test]
    fn the_intensity_sweeps_calls_reach_either_engine() {
        let start = |args: &[&str]| parsed(args).batch.controls();
        assert_eq!(
            start(&["--sweep", "intensity"]),
            start(&["--tier", "empty"])
        );
        for engine in ["whole", "stream"] {
            // the second song's first bar latches the busy call under any latch
            let swept = rendered(
                &["--engine", engine, "--sweep", "intensity", "--songs", "2"],
                0,
            );
            let held = rendered(&["--engine", engine, "--tier", "empty", "--songs", "2"], 0);
            assert!(
                swept.is_ok() && swept.ok() != held.ok(),
                "{engine} played no call"
            );
        }
    }

    #[test]
    fn songs_writes_that_many_songs_under_either_engine() {
        let c = Batch::default().controls();
        let want: usize = (0..2).map(|k| audition::song_len(3, k, c)).sum();
        for engine in ["whole", "stream"] {
            let take = rendered(&["--engine", engine, "--songs", "2"], 3);
            assert_eq!(take.map(|t| t.len()).ok(), Some(want), "{engine}");
        }
    }

    #[test]
    fn a_solo_takes_no_sweep() {
        let solo = ["--engine", "whole", "--solo", "pad", "--sweep", "twilight"];
        assert!(Args::parse(solo.map(str::to_owned).into_iter()).is_err());
    }

    #[test]
    fn a_run_whose_takes_share_a_file_name_is_refused() {
        // the twilight takes, both named for the lane past the parse's guard
        let mut args = parsed(&["--engine", "whole", "--sweep", "twilight"]);
        args.solo = Some(0);
        assert!(args.takes().is_err());
        assert!(parsed(&["--sweep", "twilight"]).takes().is_ok());
    }

    #[test]
    fn every_lane_writes_a_solo_take() {
        for mood in ["day", "night"] {
            for lane in Lane::NAMES {
                let take = rendered(&["--engine", "whole", "--mood", mood, "--solo", lane], 0);
                assert!(take.is_ok(), "{mood} {lane}: {:?}", take.err());
            }
        }
    }
}
