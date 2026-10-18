//! Score access and per-lane renders for lofi's own tools (`examples/`) to
//! solo, mute and measure, and the loudness bounds lofi's gates share. Not a
//! contract: any release may change it.

use std::ops::Range;

pub use crate::compose::Score;
pub use crate::stream::master::{MASTER_GAIN, TIER_INTENSITIES, lane_gain};
pub use crate::stream::schedule::{LOOPS_PER_SONG, bar_frames, song_score};
pub use crate::synth::{Lane, render_lane};

use crate::{Controls, Frame, Performance};

/// Song `k` of `seed`'s run, from its first bar to the next song's, played
/// from `controls` with no earlier song ringing into it, each of `calls`
/// made to [`Performance::set_controls`] once its frame of the song has
/// played.
///
/// # Panics
///
/// If `calls` aren't in order, or one falls at or past the song's end, which
/// would render a frame twice or another song's.
pub fn song(seed: u64, k: u64, controls: Controls, calls: &[(usize, Controls)]) -> Vec<Frame> {
    assert!(calls.is_sorted_by_key(|&(at, _)| at), "calls out of order");
    // a call before the first render picks the song, as `set_controls` says
    let first = calls.iter().take_while(|&&(at, _)| at == 0).count();
    let (controls, calls) = (
        calls[..first].last().map_or(controls, |&(_, c)| c),
        &calls[first..],
    );
    let len = Performance::song_len(seed, k, controls);
    assert!(
        calls.iter().all(|&(at, _)| at < len),
        "a call at or past the song's {len} frames"
    );
    let mut performance = Performance::from_song(seed, k, controls);
    let mut played = Vec::new();
    for &(at, controls) in calls {
        let mut frames = vec![Frame::default(); at - played.len()];
        performance.render(&mut frames);
        played.extend(frames);
        performance.set_controls(controls);
    }
    played.extend(performance.finish_song());
    played
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "calls out of order")]
    fn a_song_refuses_calls_out_of_order() {
        let c = Controls::default();
        song(1, 0, c, &[(2000, c), (1000, c)]);
    }

    #[test]
    fn a_call_at_frame_0_picks_the_song_and_bounds_its_calls() {
        let (day, night) = (Controls::default(), Controls::default().with_daylight(0.0));
        let (seed, night_len) = (1, song_len(1, 0, night));
        assert!(
            song_len(seed, 0, day) < night_len,
            "seed 1's night song is no longer"
        );
        // the last of the frame-0 calls wins, as before any latch
        let late = [(0, day), (0, night), (night_len - 1, night)];
        assert_eq!(song(seed, 0, day, &late), song(seed, 0, night, &late[2..]));
    }

    #[test]
    #[should_panic(expected = "a call at or past the song's")]
    fn a_song_refuses_a_call_past_its_end() {
        let c = Controls::default();
        song(1, 0, c, &[(song_len(1, 0, c), c)]);
    }
}
