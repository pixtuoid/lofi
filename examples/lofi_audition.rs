//! The generator's STATISTICAL gate: render N random-seed takes to wav so
//! `ab_judge.py` judges the COMPOSER, not one take. Renders through the REAL
//! `render_lane` chain and mixes it at one of pixtuoid's tiers, so what you hear
//! is what ships.
//!
//! Usage:
//!   cargo run --release --example lofi_audition -- \
//!     [--mood day|night] [--seeds N] [--start S] [--out DIR] \
//!     [--tier empty|moderate|busy | --solo pad|sparkle|keys|drums|texture|bass] \
//!     [--shuffle KEYFILE]

#[path = "common/batch.rs"]
mod batch;
#[path = "common/loudness.rs"]
mod loudness;

use std::fs::File;
use std::hash::{BuildHasher as _, RandomState};
use std::io::{BufWriter, Write as _};
use std::path::PathBuf;
use std::process::ExitCode;

use pixtuoid_lofi::dsp::{NoiseStream, SAMPLE_RATE};
use pixtuoid_lofi::{Lane, compose};

use batch::{Batch, LANE_NAMES, parse_choice, render_take, soak_mix, tier_gain, usage_error};
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

fn run() -> std::io::Result<()> {
    let mut batch = Batch::default();
    let mut out = PathBuf::from("audio-demos");
    let mut solo: Option<usize> = None;
    let mut key: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        if batch.parse_flag(&a, &mut args)? {
            continue;
        }
        match a.as_str() {
            "--out" => match args.next() {
                Some(v) => out = PathBuf::from(v),
                None => return Err(usage_error("--out needs a directory".into())),
            },
            "--solo" => solo = Some(parse_choice(args.next(), "--solo", &LANE_NAMES)?),
            "--shuffle" => match args.next() {
                Some(v) => key = Some(PathBuf::from(v)),
                None => return Err(usage_error("--shuffle needs a key file".into())),
            },
            other => {
                return Err(usage_error(format!(
                    "unknown argument {other:?}; valid: --mood --seeds --start --out --tier --solo --shuffle"
                )));
            }
        }
    }
    if solo.is_some() && batch.tier.is_some() {
        return Err(usage_error(
            "--solo plays one lane, so it takes no --tier".into(),
        ));
    }
    std::fs::create_dir_all(&out)?;

    let (mix_name, gains) = match solo {
        Some(s) => {
            let mut gains = [0.0; Lane::ALL.len()];
            gains[s] = 1.0;
            (LANE_NAMES[s], gains)
        }
        None => {
            let tier = batch.tier();
            (tier.name(), Lane::ALL.map(|lane| tier_gain(lane, tier)))
        }
    };
    let seeds: Vec<u64> = batch.seeds().collect();
    let takes: Vec<String> = seeds
        .iter()
        .map(|seed| format!("gen_{}_{mix_name}_{seed:03}.wav", batch.mood_name()))
        .collect();
    let files = match key {
        Some(_) => shuffled_names(takes.len()),
        None => takes.clone(),
    };
    let mut order: Vec<(&String, &String, u64)> = files
        .iter()
        .zip(&takes)
        .zip(&seeds)
        .map(|((file, take), &seed)| (file, take, seed))
        .collect();
    // rendered in file-name order, so the files' times don't give the seeds away
    order.sort();
    for &(file, _, seed) in &order {
        let score = compose(batch.mood(), seed);
        let beds = render_take(&score).map(|(bed, _)| bed);
        let mut mixdown = soak_mix(&beds, &gains);
        loudness_normalize(&mut mixdown);
        let path = out.join(file);
        write_wav(&path, &mixdown)?;
        if key.is_none() {
            println!(
                "  seed {seed:3}  {:>3.0} bpm  lead={:5}  {}",
                score.bpm(),
                score.lead_voice_name(),
                path.display()
            );
        }
    }
    match &key {
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
        println!("  afplay {}", out.join(file).display());
    }
    Ok(())
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

/// Scale the mixdown to [`TARGET_LUFS`].
fn loudness_normalize(x: &mut [f32]) {
    let gain = 10f64.powf((TARGET_LUFS - gated_lufs(x)) / 20.0) as f32;
    for v in x.iter_mut() {
        *v *= gain;
    }
}

/// 16-bit stereo RIFF/WAVE (mono mixdown duplicated L/R) with the audition soft clip.
fn write_wav(path: &PathBuf, mono: &[f32]) -> std::io::Result<()> {
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
        let clipped = (s * 1.1).tanh() * 0.85;
        let pcm = (clipped * 32767.0) as i16;
        w.write_all(&pcm.to_le_bytes())?;
        w.write_all(&pcm.to_le_bytes())?;
    }
    w.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
