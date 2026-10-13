//! Score access and per-lane renders for lofi's own tools (`examples/`) to
//! solo, mute and measure, and the loudness bounds lofi's gates share. Not a
//! contract: any release may change it.

use std::ops::Range;

pub use crate::compose::{Mood, Score, compose};
pub use crate::stream::master::{TIER_INTENSITIES, lane_gain};
pub use crate::synth::{Lane, render_lane};

use crate::{Controls, Frame, Performance};

/// Song `k` of `seed`'s run, from its first bar to the next song's, played
/// under `controls` with no earlier song ringing into it.
pub fn song(seed: u64, k: u64, controls: Controls) -> Vec<Frame> {
    Performance::from_song(seed, k, controls).finish_song()
}

/// How many frames song `k` of `seed`'s run lasts under `controls`: where
/// the next song's first bar falls, without rendering a frame.
pub fn song_len(seed: u64, k: u64, controls: Controls) -> usize {
    Performance::song_len(seed, k, controls)
}

/// The seeds [`WHOLE_LANE_LUFS_SD`] was measured over.
pub const WHOLE_LANE_SEEDS: Range<u64> = 0..24;

/// How much the whole-lane mix's gated loudness varied across
/// [`WHOLE_LANE_SEEDS`], as [`lufs_sd`] reads it, at an intensity and a
/// daylight: `(intensity, daylight, LU)`. What lofi played before it
/// streamed, which the commit that set it records; a run's songs may vary up
/// to [`LUFS_SD_TOLERANCE`] times as much, a non-regression, not a target.
pub const WHOLE_LANE_LUFS_SD: [(f32, f32, f64); 4] = [
    (TIER_INTENSITIES[1], 1.0, 0.4471),
    (TIER_INTENSITIES[1], 0.0, 0.2791),
    (TIER_INTENSITIES[0], 1.0, 0.4655),
    (TIER_INTENSITIES[2], 1.0, 0.4571),
];

/// How many times [`WHOLE_LANE_LUFS_SD`] a run's songs may vary: room for
/// sampling, which is opinion.
pub const LUFS_SD_TOLERANCE: f64 = 1.1;

/// How much `lufs` vary: their sample standard deviation, in LU.
pub fn lufs_sd(lufs: &[f64]) -> f64 {
    let n = lufs.len() as f64;
    let mean = lufs.iter().sum::<f64>() / n;
    (lufs.iter().map(|l| (l - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
}
