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

use std::f64::consts::PI;
use std::fs::File;
use std::hash::{BuildHasher as _, RandomState};
use std::io::{self, BufWriter, Write as _};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use pixtuoid_lofi::Frame;
use pixtuoid_lofi::audition::Lane;
use pixtuoid_lofi::dsp::{NoiseStream, SAMPLE_RATE};

use batch::{Batch, Engine, lane_gains, parse_choice, usage_error, valid_flags};
use pixtuoid_lofi::audition::{TARGET_LUFS, channels, gated_lufs};

/// What a solo take renders at instead, since a lane alone peaks past full
/// scale at [`TARGET_LUFS`]: the loudest level at which every lane's solo,
/// both moods and seeds 0–23, kept its peak under full scale, as the commit
/// that set it records, less [`SOLO_MARGIN_DB`].
const SOLO_TARGET_LUFS: f64 = -29.1 - SOLO_MARGIN_DB;

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

/// The flags lofi_audition takes beyond the batch's.
const OWN_FLAGS: [&str; 3] = ["--out", "--solo", "--shuffle"];

/// What a run renders, and where.
struct Args {
    batch: Batch,
    out: PathBuf,
    /// The lane `--solo` plays alone.
    solo: Option<Lane>,
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
                "--solo" => {
                    parsed.solo =
                        Some(Lane::ALL[parse_choice(args.next(), "--solo", &Lane::NAMES)?]);
                }
                "--shuffle" => match args.next() {
                    Some(v) => parsed.key = Some(PathBuf::from(v)),
                    None => return Err(usage_error("--shuffle needs a key file".into())),
                },
                other => {
                    return Err(usage_error(format!(
                        "unknown argument {other:?}; valid: {}",
                        valid_flags(&OWN_FLAGS)
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
                Some(lane) => lane.name(),
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
                    Some(lane) => Lane::ALL.map(|l| f32::from(u8::from(l == lane))),
                    None => lane_gains(controls),
                };
                (run.whole(seed, gains).into_iter())
                    .flat_map(|song| song.mix())
                    .map(|x| [x; 2])
                    .collect()
            }
            Engine::Stream => run.perform(seed),
        };
        fade_edges(&mut take);
        let target = self.target_lufs();
        let [left, right] = channels(&take);
        let gain = 10f64.powf((target - gated_lufs(&[&left, &right], SAMPLE_RATE)) / 20.0) as f32;
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

/// A take's channels and each sample's bits, as `write_wav` writes them.
const CHANNELS: u16 = 2;
const BITS: u16 = 16;
/// The fmt chunk's size and its format code for PCM, "0x0001
/// WAVE_FORMAT_PCM" (https://www.mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/WAVE.html).
const FMT_SIZE: u32 = 16;
const WAVE_FORMAT_PCM: u16 = 1;
/// The header's bytes after the RIFF size: "WAVE", the fmt chunk and the data
/// chunk's header.
const HEADER_AFTER_SIZE: u32 = 4 + 8 + FMT_SIZE + 8;

/// The data chunk's size for a take of `frames`, or `None` past what the
/// header's RIFF size can count.
fn data_len(frames: usize) -> Option<u32> {
    (u32::try_from(frames).ok())
        .and_then(|n| n.checked_mul(u32::from(CHANNELS * BITS / 8)))
        .filter(|n| n.checked_add(HEADER_AFTER_SIZE).is_some())
}

/// 16-bit stereo RIFF/WAVE; an error if `take` holds more data than a RIFF
/// size can count.
fn write_wav(path: &Path, take: &[Frame]) -> io::Result<()> {
    let frame = CHANNELS * BITS / 8;
    let data_len = data_len(take.len()).ok_or_else(|| {
        usage_error(format!(
            "a take of {} frames is too long for a wav",
            take.len()
        ))
    })?;
    let mut w = BufWriter::new(File::create(path)?);
    w.write_all(b"RIFF")?;
    w.write_all(&(HEADER_AFTER_SIZE + data_len).to_le_bytes())?;
    w.write_all(b"WAVEfmt ")?;
    w.write_all(&FMT_SIZE.to_le_bytes())?;
    w.write_all(&WAVE_FORMAT_PCM.to_le_bytes())?;
    w.write_all(&CHANNELS.to_le_bytes())?;
    w.write_all(&SAMPLE_RATE.to_le_bytes())?;
    w.write_all(&(SAMPLE_RATE * u32::from(frame)).to_le_bytes())?;
    w.write_all(&frame.to_le_bytes())?;
    w.write_all(&BITS.to_le_bytes())?;
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
    use pixtuoid_lofi::{Controls, audition};

    #[test]
    fn the_usage_names_every_flag() {
        let usage: String = (include_str!("lofi_audition.rs").lines())
            .take_while(|line| line.starts_with("//!"))
            .collect();
        for flag in valid_flags(&OWN_FLAGS).split(' ') {
            assert!(
                usage.contains(&format!("[{flag}")) || usage.contains(&format!("| {flag}")),
                "the usage leaves out {flag}"
            );
        }
    }

    #[test]
    fn a_wav_counts_every_take_its_riff_size_can_and_no_longer_one() {
        let frame = u32::from(CHANNELS * BITS / 8);
        let longest = ((u32::MAX - HEADER_AFTER_SIZE) / frame) as usize;
        assert_eq!(data_len(0), Some(0));
        assert_eq!(data_len(longest), Some(longest as u32 * frame));
        assert_eq!(data_len(longest + 1), None);
        // past what the multiply and the conversion count, each alone
        assert_eq!(data_len((u32::MAX / frame) as usize + 1), None);
        #[cfg(target_pointer_width = "64")]
        assert_eq!(data_len(u32::MAX as usize + 1), None);
    }

    #[test]
    fn a_wav_header_counts_every_byte_after_it() {
        let path = std::env::temp_dir().join(format!("lofi-wav-{}.wav", std::process::id()));
        write_wav(&path, &[[0.5, -0.5], [0.0, 1.0], [-1.0, 0.25]]).expect("the take writes");
        let raw = std::fs::read(&path).expect("the take reads back");
        std::fs::remove_file(&path).expect("the take is removed");
        let u32_at = |at: usize| u32::from_le_bytes(raw[at..at + 4].try_into().expect("4 bytes"));
        // RIFF's size is the file's after its first 8 bytes, and data's its samples'
        assert_eq!(u32_at(4) as usize, raw.len() - 8);
        assert_eq!(u32_at(40) as usize, 3 * 2 * 2);
        assert_eq!(&raw[36..40], b"data");
    }

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
        // `ab_judge.py` groups a take by its name's first word, and pairs it
        // by the rest
        let mut named: Vec<(String, String)> = (takes.iter())
            .map(|take| {
                let [group, mix, seed] = (take.name.strip_prefix("gen_"))
                    .and_then(|n| n.strip_suffix(".wav"))
                    .map(|n| n.splitn(3, '_').collect::<Vec<_>>())
                    .and_then(|parts| parts.try_into().ok())
                    .expect("gen_GROUP_MIX_SEED.wav");
                assert_eq!(seed, format!("{:03}", take.seed), "{}", take.name);
                (group.to_owned(), mix.to_owned())
            })
            .collect();
        named.sort();
        named.dedup();
        let want = [
            ("day", "busy"),
            ("day", "empty"),
            ("day", "moderate"),
            ("night", "busy"),
            ("night", "empty"),
            ("night", "moderate"),
            ("rain", "moderate"),
            ("twilight", "quarter"),
            ("twilight", "threequarters"),
        ];
        assert_eq!(named, want.map(|(g, m)| (g.to_owned(), m.to_owned())));
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
            let [left, right] = channels(&written);
            let lufs = gated_lufs(&[&left, &right], SAMPLE_RATE);
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
            let runs = parsed(&["--sweep", sweep.name(), "--seeds", "1"])
                .batch
                .runs();
            let lens: Vec<usize> = (runs.iter())
                .map(|run| take_len(run, 0).unwrap_or_else(|| panic!("{sweep:?} ends mid-song")))
                .collect();
            for engine in ["whole", "stream"] {
                let args = parsed(&["--engine", engine, "--sweep", sweep.name(), "--seeds", "1"]);
                for (run, &len) in args.batch.runs().iter().zip(&lens) {
                    let take = args.render(run, 0).expect("the sweep renders");
                    let again = args.render(run, 0).expect("the sweep renders");
                    assert!(take == again, "{sweep:?} under {engine} renders apart");
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
        let want = take_len(&parsed(&["--songs", "2"]).batch, 3);
        for engine in ["whole", "stream"] {
            let take = rendered(&["--engine", engine, "--songs", "2"], 3);
            assert_eq!(take.map(|t| t.len()).ok(), want, "{engine}");
        }
    }

    /// How long `run`'s take of `seed` lasts, if it ends as its last song plays
    /// alone, past the first loop the song before rings into: song lengths the
    /// timeline got wrong would end it mid-song, or in the next.
    fn take_len(run: &Batch, seed: u64) -> Option<usize> {
        let (take, timeline) = (run.perform(seed), run.timeline(seed));
        let k = timeline.songs.len() - 1;
        let starts: usize = timeline.songs[..k].iter().map(|&(_, len)| len).sum();
        let calls: Vec<(usize, Controls)> = (timeline.calls.iter())
            .filter(|&&(at, _)| at >= starts)
            .map(|&(at, controls)| (at - starts, controls))
            .collect();
        let alone = audition::song(seed, k as u64, timeline.songs[k].0, &calls);
        let rung = if k == 0 {
            0
        } else {
            audition::samples_at(
                audition::song_score(seed, k as u64, timeline.songs[k].0).loop_secs(),
                SAMPLE_RATE,
            )
        };
        (take.len() >= alone.len() && take[take.len() - alone.len() + rung..] == alone[rung..])
            .then_some(take.len())
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
        args.solo = Some(Lane::Pad);
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
