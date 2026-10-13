//! The public engine: a [`Performance`] streams a seed's songs as [`Frame`]s
//! under the listener's [`Controls`].

use std::fmt;

use crate::dsp::SAMPLE_RATE;
use crate::stream::schedule::{BLOCK, Schedule};

/// One stereo frame, `[left, right]`, at [`SAMPLE_RATE`].
pub type Frame = [f32; 2];

/// What the listener's world asks of the music; how it sounds is lofi's call.
/// Out-of-range values clamp; NaN reads as the default.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct Controls {
    /// 0 = night … 1 = full day.
    pub daylight: f32,
    /// 0 = an empty room … 1 = a busy one.
    pub intensity: f32,
    /// 0 = dry … 1 = a storm.
    pub rain: f32,
}

impl Default for Controls {
    /// Full day, a moderately busy room, no rain.
    fn default() -> Self {
        Self {
            daylight: 1.0,
            intensity: 0.5,
            rain: 0.0,
        }
    }
}

impl Controls {
    /// These controls with `daylight` replaced.
    #[must_use]
    pub const fn with_daylight(self, daylight: f32) -> Self {
        Self { daylight, ..self }
    }

    /// These controls with `intensity` replaced.
    #[must_use]
    pub const fn with_intensity(self, intensity: f32) -> Self {
        Self { intensity, ..self }
    }

    /// These controls with `rain` replaced.
    #[must_use]
    pub const fn with_rain(self, rain: f32) -> Self {
        Self { rain, ..self }
    }

    fn clamped(self) -> Self {
        let default = Self::default();
        let unit = |v: f32, default: f32| {
            if v.is_nan() {
                default
            } else {
                v.clamp(0.0, 1.0)
            }
        };
        Self {
            daylight: unit(self.daylight, default.daylight),
            intensity: unit(self.intensity, default.intensity),
            rain: unit(self.rain, default.rain),
        }
    }
}

/// An endless run of the songs a seed composes. Its output is a pure function
/// of the seed, the starting controls, and each
/// [`set_controls`](Self::set_controls) call with the
/// [`position`](Self::position) it is made at.
pub struct Performance {
    /// The latest controls asked for, which the next song's first bar latches.
    asked: Controls,
    position: u64,
    schedule: Schedule,
}

// Losing `Send` is silent here and breaks only a consumer that renders off-thread.
const _: fn() = || {
    fn s<T: Send>() {}
    s::<Performance>();
};

// each lane's ring holds seconds of samples, which a derived `Debug` would print
impl fmt::Debug for Performance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Performance")
            .field("asked", &self.asked)
            .field("position", &self.position)
            .finish_non_exhaustive()
    }
}

impl Performance {
    /// `seed` picks the whole sequence of songs; lofi reads no clock, so the
    /// caller supplies it.
    pub fn new(seed: u64, controls: Controls) -> Self {
        Self::from_song(seed, 0, controls)
    }

    /// A performance of `seed`'s run from song `first` on.
    pub(crate) fn from_song(seed: u64, first: u64, controls: Controls) -> Self {
        Self {
            asked: controls,
            position: 0,
            schedule: Schedule::new(seed, first, SAMPLE_RATE),
        }
    }

    /// Applies at the next song boundary, the incoming song's first bar, at or
    /// after [`position`](Self::position); the first song's is position 0. Of
    /// several calls before one boundary, the last wins.
    pub fn set_controls(&mut self, controls: Controls) {
        self.asked = controls;
    }

    /// Fills `out` with the next frames, identical however the calls chunk it.
    /// May allocate, so callers render ahead, off the real-time thread.
    pub fn render(&mut self, out: &mut [Frame]) {
        let mut done = 0;
        while done < out.len() {
            self.start_due_song();
            let n = (out.len() - done).min(BLOCK).min(self.schedule.song_left());
            let lanes = self.schedule.play(n);
            for (i, frame) in out[done..done + n].iter_mut().enumerate() {
                let mix: f64 = lanes.iter().map(|lane| lane[i]).sum();
                *frame = [mix as f32; 2];
            }
            done += n;
        }
        self.position += out.len() as u64;
    }

    /// Frames rendered so far.
    pub fn position(&self) -> u64 {
        self.position
    }

    /// Starts the next song if [`position`](Self::position) is its first bar,
    /// under the controls asked for last.
    fn start_due_song(&mut self) {
        if self.schedule.song_left() == 0 {
            self.schedule.start_song(self.asked.clamped());
        }
    }

    /// The frames from [`position`](Self::position) to the next song's first
    /// bar, the song at `position` started first if it is due.
    pub(crate) fn finish_song(&mut self) -> Vec<Frame> {
        self.start_due_song();
        let mut out = vec![Frame::default(); self.schedule.song_left()];
        self.render(&mut out);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audition;
    use crate::dsp::NoiseStream;
    use crate::stream::schedule::LOOPS_PER_SONG;

    fn night() -> Controls {
        Controls::default().with_daylight(0.0)
    }

    /// `p`'s next `frames` frames, in one call.
    fn next(p: &mut Performance, frames: usize) -> Vec<Frame> {
        let mut out = vec![Frame::default(); frames];
        p.render(&mut out);
        out
    }

    /// How many frames song `k` of `seed`'s run lasts under `controls`.
    fn song_len(seed: u64, k: u64, controls: Controls) -> usize {
        let mut p = Performance::from_song(seed, k, controls);
        p.start_due_song();
        p.schedule.song_left()
    }

    /// The first frame at which `a` and `b` differ in a bit.
    fn first_diff(a: &[Frame], b: &[Frame]) -> Option<usize> {
        a.iter()
            .zip(b)
            .position(|(x, y)| x.map(f32::to_bits) != y.map(f32::to_bits))
    }

    fn audible(frames: &[Frame]) -> bool {
        frames.iter().any(|&[l, _]| l != 0.0)
    }

    #[test]
    fn render_plays_the_same_frames_however_the_calls_chunk_them() {
        let (seed, c) = (3, Controls::default());
        let boundary = song_len(seed, 0, c);
        // through song 1's first loop, past every tail of song 0
        let span = boundary + song_len(seed, 1, c) / LOOPS_PER_SONG;
        let whole = next(&mut Performance::new(seed, c), span);
        assert!(
            audible(&whole[..boundary]) && audible(&whole[boundary..]),
            "an invariance over silence proves nothing"
        );
        let mut rng = NoiseStream::new(1);
        let (mut random, mut total) = (Vec::new(), 0);
        while total < span {
            let size = match (rng.unit() * 4.0) as u32 {
                0 => 1,
                1 => 127,
                2 => 4096,
                _ => 1 + (rng.unit() * 20_000.0) as usize,
            };
            total += size;
            random.push(size);
        }
        let chunkings = [
            ("random sizes", random),
            ("one frame at the boundary", vec![boundary, 1]),
            ("a call across the boundary", vec![boundary - 63, 127]),
        ];
        for (name, chunks) in chunkings {
            let mut p = Performance::new(seed, c);
            let mut at = 0;
            for size in chunks.into_iter().chain([span]) {
                let n = size.min(span - at);
                let got = next(&mut p, n);
                assert_eq!(
                    first_diff(&got, &whole[at..at + n]).map(|i| at + i),
                    None,
                    "{name}: the first frame that differs"
                );
                at += n;
            }
            assert_eq!(p.position(), span as u64, "{name}");
        }
    }

    #[test]
    fn a_song_plays_the_same_reached_by_playback_or_alone() {
        let (seed, c) = (5, Controls::default());
        let mut playing = Performance::new(seed, c);
        playing.finish_song();
        let reached = playing.finish_song();
        let alone = audition::song(seed, 1, c);
        assert!(audible(&alone));
        assert_eq!(reached.len(), alone.len());
        let first_loop = alone.len() / LOOPS_PER_SONG;
        assert!(
            first_diff(&reached[..first_loop], &alone[..first_loop]).is_some(),
            "song 0's tails ring into song 1"
        );
        assert_eq!(
            first_diff(&reached[first_loop..], &alone[first_loop..]),
            None,
            "song 1 after the tails, from its second loop"
        );
    }

    #[test]
    fn a_seed_always_plays_the_same_and_seeds_play_apart() {
        let play = |seed| next(&mut Performance::new(seed, Controls::default()), 1 << 17);
        let one = play(1);
        assert_eq!(first_diff(&one, &play(1)), None);
        assert!(first_diff(&one, &play(2)).is_some());
    }

    #[test]
    fn controls_clamp_into_range_and_read_nan_as_the_default() {
        let wild = Controls {
            daylight: f32::NAN,
            intensity: 7.0,
            rain: f32::NEG_INFINITY,
        };
        assert_eq!(
            wild.clamped(),
            Controls {
                daylight: 1.0,
                intensity: 1.0,
                rain: 0.0,
            }
        );
        let wild = Controls {
            daylight: -0.5,
            intensity: f32::NAN,
            rain: f32::INFINITY,
        };
        assert_eq!(
            wild.clamped(),
            Controls {
                daylight: 0.0,
                intensity: 0.5,
                rain: 1.0,
            }
        );
    }

    #[test]
    fn a_performance_hears_nan_controls_as_the_default() {
        let play = |c| next(&mut Performance::new(4, c), SAMPLE_RATE as usize);
        let day = play(Controls::default());
        let nan = Controls::default()
            .with_daylight(f32::NAN)
            .with_intensity(f32::NAN)
            .with_rain(f32::NAN);
        assert_eq!(first_diff(&play(nan), &day), None);
        assert!(first_diff(&play(night()), &day).is_some());
    }

    #[test]
    fn set_controls_takes_effect_at_the_next_song_and_the_last_call_wins() {
        let seed = 6;
        let boundary = song_len(seed, 0, Controls::default());
        let mut plain = Performance::new(seed, Controls::default());
        let unasked = [plain.finish_song(), plain.finish_song()].concat();
        // `frames` frames, each of `calls` made once its frame count has played
        let play = |calls: &[(usize, Controls)], frames: usize| {
            let mut p = Performance::new(seed, Controls::default());
            let mut out = Vec::with_capacity(frames);
            for &(at, c) in calls {
                out.extend(next(&mut p, at - out.len()));
                p.set_controls(c);
            }
            out.extend(next(&mut p, frames - out.len()));
            out
        };
        let second = SAMPLE_RATE as usize;
        let mut early = Performance::new(seed, Controls::default());
        early.set_controls(night());
        let early = next(&mut early, second);
        assert!(first_diff(&early, &unasked[..second]).is_some());
        assert_eq!(
            first_diff(&early, &next(&mut Performance::new(seed, night()), second)),
            None,
            "a call before the first render missed the first song"
        );
        let mid = boundary / 2;
        let night_song = audition::song(seed, 1, night());
        let second_loop = night_song.len() / LOOPS_PER_SONG;
        for (name, at) in [("mid-song", mid), ("at the boundary", boundary)] {
            let asked = play(&[(at, night())], boundary + night_song.len());
            assert_eq!(
                first_diff(&asked[..boundary], &unasked[..boundary]),
                None,
                "{name}: the playing song changed"
            );
            assert!(
                first_diff(&asked[boundary..], &unasked[boundary..]).is_some(),
                "{name}: the call changed nothing"
            );
            assert_eq!(
                first_diff(&asked[boundary + second_loop..], &night_song[second_loop..]),
                None,
                "{name}: the next song isn't the night one"
            );
        }
        assert_eq!(
            first_diff(&play(&[(boundary + 1, night())], unasked.len()), &unasked),
            None,
            "a call past the boundary changed the song it passed"
        );
        let calls = [(mid, night()), (mid + 1, Controls::default())];
        assert_eq!(
            first_diff(&play(&calls, unasked.len()), &unasked),
            None,
            "the last call didn't win"
        );
    }
}
