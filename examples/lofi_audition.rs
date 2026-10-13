//! The generator's STATISTICAL listen gate: render N random-seed takes to wav so
//! the owner blind-auditions the COMPOSER, not one take. A take is the first
//! song of a seed's run at one of pixtuoid's tiers. `--engine v2`, the default,
//! renders it as a `Performance` plays it, so what you hear is what lofi ships;
//! `--engine v1` renders the same score's lanes whole and loops their mix, as
//! pixtuoid's office played lofi before the stream. Either take is mono and a
//! song long.
//!
//! Usage:
//!   cargo run --release --example lofi_audition -- \
//!     [--engine v1|v2] [--mood day|night] [--seeds N] [--start S] [--out DIR] \
//!     [--tier empty|moderate|busy | --solo pad|sparkle|keys|drums|texture|bass] \
//!     [--shuffle KEYFILE]
//!   `--solo` renders a lane whole, so it takes `--engine v1`.

#[path = "common/batch.rs"]
mod batch;
#[path = "common/loudness.rs"]
mod loudness;

use std::fs::File;
use std::hash::{BuildHasher as _, RandomState};
use std::io::{self, BufWriter, Write as _};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use pixtuoid_lofi::audition::Lane;
use pixtuoid_lofi::dsp::{NoiseStream, SAMPLE_RATE};

use batch::{Batch, Engine, LANE_NAMES, parse_choice, render_take, soak_mix, usage_error};
use loudness::gated_lufs;

/// Every take renders at ONE gated loudness, so a blind A/B compares takes, not
/// gain.
const TARGET_LUFS: f64 = -18.0;

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
    let takes = args.takes();
    let files = match args.key {
        Some(_) => shuffled_names(takes.len()),
        None => takes.iter().map(|(_, take)| take.clone()).collect(),
    };
    let mut order: Vec<(&String, &String, u64)> = files
        .iter()
        .zip(&takes)
        .map(|(file, (seed, take))| (file, take, *seed))
        .collect();
    // rendered in file-name order, so the files' times don't give the seeds away
    order.sort();
    for &(file, _, seed) in &order {
        let path = args.out.join(file);
        write_wav(&path, &args.render(seed)?)?;
        if args.key.is_none() {
            let score = args.batch.score(seed);
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
            for (file, take, _) in &order {
                writeln!(w, "{file} {take}")?;
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

/// What a run renders, and where.
struct Args {
    batch: Batch,
    out: PathBuf,
    /// The lane `--solo` plays alone, as its index in [`LANE_NAMES`].
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
                "--solo" => parsed.solo = Some(parse_choice(args.next(), "--solo", &LANE_NAMES)?),
                "--shuffle" => match args.next() {
                    Some(v) => parsed.key = Some(PathBuf::from(v)),
                    None => return Err(usage_error("--shuffle needs a key file".into())),
                },
                other => {
                    return Err(usage_error(format!(
                        "unknown argument {other:?}; valid: --engine --mood --seeds --start --out --tier --solo --shuffle"
                    )));
                }
            }
        }
        if parsed.solo.is_some() {
            if parsed.batch.tier.is_some() {
                return Err(usage_error(
                    "--solo plays one lane, so it takes no --tier".into(),
                ));
            }
            if parsed.batch.engine != Engine::V1 {
                return Err(usage_error(
                    "--solo renders one lane whole, so it takes --engine v1".into(),
                ));
            }
        }
        Ok(parsed)
    }

    /// Each seed with its take's file name. No engine changes a name, so
    /// `ab_listen.py` pairs a v1 batch with a v2 batch.
    fn takes(&self) -> Vec<(u64, String)> {
        let mix = match self.solo {
            Some(s) => LANE_NAMES[s],
            None => self.batch.tier().name(),
        };
        let mood = self.batch.mood_name();
        (self.batch.seeds())
            .map(|seed| (seed, format!("gen_{mood}_{mix}_{seed:03}.wav")))
            .collect()
    }

    /// `seed`'s take at [`TARGET_LUFS`]; an error if that clips it, which
    /// would write a different take.
    fn render(&self, seed: u64) -> io::Result<Vec<f32>> {
        let mut take = match self.batch.engine {
            Engine::V1 => {
                let gains = match self.solo {
                    Some(s) => {
                        let mut gains = [0.0; Lane::ALL.len()];
                        gains[s] = 1.0;
                        gains
                    }
                    None => self.batch.tier().lane_gains(),
                };
                let (beds, _) = render_take(&self.batch.score(seed));
                soak_mix(&beds, &gains)
            }
            Engine::V2 => self.batch.perform(seed)?,
        };
        let gain = 10f64.powf((TARGET_LUFS - gated_lufs(&take, SAMPLE_RATE)) / 20.0) as f32;
        for v in &mut take {
            *v *= gain;
        }
        if take.iter().any(|v| v.abs() > 1.0) {
            return Err(io::Error::other(format!(
                "seed {seed} clips at {TARGET_LUFS} LUFS"
            )));
        }
        Ok(take)
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

/// 16-bit stereo RIFF/WAVE, the take in both channels.
fn write_wav(path: &Path, mono: &[f32]) -> io::Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    let n = mono.len() as u32;
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
    for &s in mono {
        let pcm = pcm(s);
        w.write_all(&pcm.to_le_bytes())?;
        w.write_all(&pcm.to_le_bytes())?;
    }
    w.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parsed(args: &[&str]) -> Args {
        Args::parse(args.iter().map(|&a| a.to_owned())).expect("the arguments parse")
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
        let [v1, v2] = ["v1", "v2"].map(|engine| {
            parsed(&["--engine", engine, "--mood", "night", "--tier", "empty"]).takes()
        });
        assert_eq!(v1, v2);
        // the shape `ab_listen.py`'s NAME pairs
        assert_eq!(v1[0], (0, "gen_night_empty_000.wav".to_owned()));
    }

    #[test]
    fn both_engines_write_a_take_as_long_and_at_the_target_loudness() {
        let seed = 1;
        let [v1, v2] = ["v1", "v2"].map(|engine| {
            let take = parsed(&["--engine", engine]).render(seed);
            let written: Vec<f32> = (take.expect("the take renders").iter())
                .map(|&x| f32::from(pcm(x)) / f32::from(i16::MAX))
                .collect();
            let lufs = gated_lufs(&written, SAMPLE_RATE);
            assert!(
                (lufs - TARGET_LUFS).abs() <= 0.1,
                "{engine} writes {lufs:.3} LUFS"
            );
            written.len()
        });
        assert_eq!(v1, v2, "the engines' takes last apart");
    }
}
