//! Score access and per-lane renders for lofi's own tools (`examples/`) to
//! solo, mute and measure, and the loudness bounds lofi's gates share. Not a
//! contract: any release may change it.

use std::ops::Range;

pub use crate::compose::Score;
pub use crate::stream::master::{MASTER_GAIN, TIER_INTENSITIES, lane_gain};
pub use crate::stream::schedule::{LOOPS_PER_SONG, song_score};
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
/// streamed, on the songs a `Performance` plays: `lofi_measure --engine whole`
/// prints it over those seeds at the row's `--tier` and `--mood`. A run's
/// songs may vary up to [`lufs_sd_bound`], a non-regression, not a target.
pub const WHOLE_LANE_LUFS_SD: [(f32, f32, f64); 4] = [
    (TIER_INTENSITIES[1], 1.0, 0.3336),
    (TIER_INTENSITIES[1], 0.0, 0.2277),
    (TIER_INTENSITIES[0], 1.0, 0.3795),
    (TIER_INTENSITIES[2], 1.0, 0.3427),
];

/// The most a run's songs may vary against `whole_lane`, a
/// [`WHOLE_LANE_LUFS_SD`] row: two standard errors over it, a sample
/// standard deviation's standard error over [`WHOLE_LANE_SEEDS`] being about
/// σ/√(2(n−1)) when the sample is normal (eq. 3 of
/// <https://web.eecs.umich.edu/~fessler/papers/files/tr/stderr.pdf>).
/// That the songs' loudness is normal is assumed, not measured.
pub fn lufs_sd_bound(whole_lane: f64) -> f64 {
    let n = (WHOLE_LANE_SEEDS.end - WHOLE_LANE_SEEDS.start) as f64;
    whole_lane * (1.0 + 2.0 / (2.0 * (n - 1.0)).sqrt())
}

/// How much `lufs` vary: their sample standard deviation, in LU.
pub fn lufs_sd(lufs: &[f64]) -> f64 {
    let n = lufs.len() as f64;
    let mean = lufs.iter().sum::<f64>() / n;
    (lufs.iter().map(|l| (l - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
}
