//! Score access and per-lane renders for lofi's own tools (`examples/`) to
//! solo, mute and measure, and the loudness bounds lofi's gates share. Not a
//! contract: any release may change it.

use std::collections::VecDeque;
use std::ops::Range;

pub use crate::compose::{Edge, Form, Score};
pub use crate::stream::master::{TIER_INTENSITIES, lane_gain};
pub use crate::stream::schedule::{bar_frames, song_bars, song_score};
pub use crate::synth::{Lane, render_lane};

use crate::dsp::SAMPLE_RATE;
use crate::perform::Probe;
use crate::stream::master::lookahead;
use crate::stream::schedule::LANES;
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

/// A performance's mix as the sum it is, on the fold every reading hears:
/// each lane's dry sum and the plate's return, as the ceiling played them,
/// so a frame's stems sum to its fold but for the rounding each takes.
#[derive(Debug)]
pub struct Stems {
    /// Each stem of the mix the ceiling has yet to play, the lanes then the
    /// plate, starting with the silence it plays first.
    pending: VecDeque<([f64; LANES], f64)>,
    lanes: [Vec<f32>; LANES],
    plate: Vec<f32>,
}

impl Default for Stems {
    /// The stems of a performance yet to play a frame.
    fn default() -> Self {
        Self {
            pending: VecDeque::from(vec![([0.0; LANES], 0.0); lookahead(SAMPLE_RATE)]),
            lanes: std::array::from_fn(|_| Vec::new()),
            plate: Vec::new(),
        }
    }
}

impl Stems {
    /// `performance.render(out)`, each frame's stems added after the ones
    /// before.
    ///
    /// # Panics
    ///
    /// If `performance` rendered frames these stems didn't, which they would
    /// misplace.
    pub fn render(&mut self, performance: &mut Performance, out: &mut [Frame]) {
        assert_eq!(
            performance.position(),
            self.plate.len() as u64,
            "the performance rendered frames these stems didn't"
        );
        performance.render_probed(out, self);
    }

    /// Each lane's stem, in [`Lane::ALL`] order.
    pub fn lanes(&self) -> &[Vec<f32>; LANES] {
        &self.lanes
    }

    /// The plate's return.
    pub fn plate(&self) -> &[f32] {
        &self.plate
    }
}

impl Probe for Stems {
    fn step(&mut self, lanes: &[Vec<f64>; LANES], plate: &[Vec<f64>; 2]) {
        let [left, right] = plate;
        self.pending.extend(
            (left.iter().zip(right).enumerate())
                .map(|(i, (l, r))| (lanes.each_ref().map(|lane| lane[i]), l.midpoint(*r))),
        );
    }

    fn gain(&mut self, gain: f64) {
        // the ceiling plays no frame it hasn't been given
        let (lanes, plate) = self.pending.pop_front().unwrap_or_default();
        let heard = |x: f64| (x * Performance::MASTER_GAIN * gain) as f32;
        for (stem, x) in self.lanes.iter_mut().zip(lanes) {
            stem.push(heard(x));
        }
        self.plate.push(heard(plate));
    }
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
    fn a_takes_stems_sum_to_its_fold() {
        // f32's rounding, of each stem and of the frame it sums to
        let rounding = 8.0 * f64::from(f32::EPSILON);
        let (seed, busy) = (8, Controls::default().with_intensity(1.0));
        let frames = 20 * SAMPLE_RATE as usize;
        // in two renders, night asked for between them
        let take = |render: &mut dyn FnMut(&mut Performance, &mut [Frame])| {
            let mut performance = Performance::new(seed, busy);
            let mut out = vec![Frame::default(); frames];
            let (head, tail) = out.split_at_mut(frames / 2);
            render(&mut performance, head);
            performance.set_controls(busy.with_daylight(0.0));
            render(&mut performance, tail);
            out
        };
        let mut stems = Stems::default();
        let got = take(&mut |p, out| stems.render(p, out));
        let want = take(&mut |p, out| p.render(out));
        let first_diff =
            (got.iter().zip(&want)).position(|(a, b)| a.map(f32::to_bits) != b.map(f32::to_bits));
        assert_eq!(first_diff, None, "the stems' render played other frames");
        assert!(stems.plate().iter().any(|&x| x != 0.0), "no plate plays");
        let fold = crate::loudness::fold(&want);
        let off = (0..frames).position(|i| {
            let sum = stems
                .lanes()
                .iter()
                .map(|stem| f64::from(stem[i]))
                .sum::<f64>()
                + f64::from(stems.plate()[i]);
            (sum - f64::from(fold[i])).abs() > rounding
        });
        assert_eq!(off, None);
    }

    #[test]
    fn a_stem_plays_as_late_as_the_ceiling_and_at_its_gain() {
        // no take turns the ceiling down soon enough to test in one
        let (late, gain) = (lookahead(SAMPLE_RATE), 0.5);
        let lanes: [Vec<f64>; LANES] = std::array::from_fn(|k| vec![k as f64 + 1.0]);
        let mut stems = Stems::default();
        stems.step(&lanes, &[vec![1.0], vec![3.0]]);
        for _ in 0..late {
            stems.gain(1.0);
        }
        stems.gain(gain);
        let heard = |x: f64| (x * Performance::MASTER_GAIN * gain) as f32;
        let silent = |stem: &[f32]| stem[..late].iter().all(|&x| x == 0.0);
        assert!(stems.lanes().iter().all(|stem| silent(stem)) && silent(stems.plate()));
        assert_eq!(
            stems.lanes().each_ref().map(|stem| stem[late]),
            std::array::from_fn(|k| heard(k as f64 + 1.0))
        );
        assert_eq!(stems.plate()[late], heard(2.0));
    }

    #[test]
    #[should_panic(expected = "rendered frames these stems didn't")]
    fn stems_refuse_a_performance_already_playing() {
        let mut performance = Performance::new(1, Controls::default());
        performance.render(&mut [Frame::default(); 1]);
        Stems::default().render(&mut performance, &mut [Frame::default(); 1]);
    }

    #[test]
    #[should_panic(expected = "a call at or past the song's")]
    fn a_song_refuses_a_call_past_its_end() {
        let c = Controls::default();
        song(1, 0, c, &[(song_len(1, 0, c), c)]);
    }
}
