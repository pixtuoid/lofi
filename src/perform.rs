//! The public engine: a [`Performance`] streams a seed's songs as [`Frame`]s
//! under the listener's [`Controls`].

use std::fmt;

use crate::dsp::SAMPLE_RATE;
use crate::stream::master::{Ceiling, MASTER_GAIN, TIER_INTENSITIES, lane_gain};
use crate::stream::schedule::{BLOCK, Schedule};
use crate::synth::Lane;

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
}

impl Default for Controls {
    /// Full day, pixtuoid's Moderate office (its taste anchor).
    fn default() -> Self {
        let [_, moderate, _] = TIER_INTENSITIES;
        Self {
            daylight: 1.0,
            intensity: moderate,
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
    /// The lanes' mix over a step.
    mix: Vec<f64>,
    ceiling: Ceiling,
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
            mix: Vec::with_capacity(BLOCK),
            ceiling: Ceiling::new(SAMPLE_RATE),
        }
    }

    /// Applies at the next song boundary, the incoming song's first bar, at or
    /// after [`position`](Self::position); the first song's is position 0. Of
    /// several calls before one boundary, the last wins.
    pub fn set_controls(&mut self, controls: Controls) {
        self.asked = controls;
    }

    /// Fills `out` with the next frames, identical however the calls chunk it.
    /// The stream is mastered as a background bed, under lofi's own ceiling:
    /// play it at the listener's volume, with no further trim. May allocate,
    /// so callers render ahead, off the real-time thread.
    pub fn render(&mut self, out: &mut [Frame]) {
        let mut done = 0;
        while done < out.len() {
            self.mix_next(out.len() - done);
            self.ceiling.limit(&mut self.mix);
            for (frame, &x) in out[done..].iter_mut().zip(&self.mix) {
                *frame = [self.ceiling.clamp(x); 2];
            }
            done += self.mix.len();
        }
        self.position += out.len() as u64;
    }

    /// Mixes the lanes' next frames into `mix` at [`MASTER_GAIN`]: at most
    /// `max`, [`BLOCK`] and the rest of the playing song.
    fn mix_next(&mut self, max: usize) {
        self.start_due_song();
        let n = max.min(BLOCK).min(self.schedule.song_left());
        let lanes = self.schedule.play(n);
        self.mix.clear();
        self.mix
            .extend((0..n).map(|i| lanes.iter().map(|lane| lane[i]).sum::<f64>() * MASTER_GAIN));
    }

    /// Frames rendered so far.
    pub fn position(&self) -> u64 {
        self.position
    }

    /// Starts the next song if [`position`](Self::position) is its first bar,
    /// under the controls asked for last.
    fn start_due_song(&mut self) {
        if self.schedule.song_left() == 0 {
            let controls = self.asked.clamped();
            let levels = Lane::ALL.map(|lane| f64::from(lane_gain(lane, controls.intensity)));
            self.schedule.start_song(controls, levels);
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

    /// How many frames song `k` of `seed`'s run lasts under `controls`.
    pub(crate) fn song_len(seed: u64, k: u64, controls: Controls) -> usize {
        let mut p = Self::from_song(seed, k, controls);
        p.start_due_song();
        p.schedule.song_left()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audition;
    use crate::dsp::NoiseStream;
    use crate::stream::schedule::{LOOPS_PER_SONG, UNITY};

    fn night() -> Controls {
        Controls::default().with_daylight(0.0)
    }

    /// `p`'s next `frames` frames, in one call.
    fn next(p: &mut Performance, frames: usize) -> Vec<Frame> {
        let mut out = vec![Frame::default(); frames];
        p.render(&mut out);
        out
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

    /// `p`'s next `frames` frames of the lanes' mix, before the ceiling.
    fn mixed(p: &mut Performance, frames: usize) -> Vec<f64> {
        let mut out = Vec::with_capacity(frames);
        while out.len() < frames {
            p.mix_next(frames - out.len());
            out.extend_from_slice(&p.mix);
        }
        out
    }

    #[test]
    fn the_intensity_anchors_play_pixtuoids_tiers_lane_for_lane() {
        // pixtuoid's `TierGain`s (crates/pixtuoid-scene/src/audio/mod.rs at
        // a63f178), each tier's lanes in `Lane::ALL` order
        let tiers: [(f32, [f32; Lane::ALL.len()]); 3] = [
            (0.0, [0.75, 0.70, 0.0, 0.0, 0.78, 0.60]),
            (0.5, [0.70, 0.0, 0.60, 0.35, 0.84, 0.70]),
            (1.0, [0.65, 0.0, 0.70, 0.60, 0.78, 0.75]),
        ];
        let seed = 2;
        let mut whole = Schedule::new(seed, 0, SAMPLE_RATE);
        whole.start_song(Controls::default(), UNITY);
        // the first loop, before any song but the first plays
        let frames = whole.song_left() / LOOPS_PER_SONG;
        let mut unity = vec![Vec::with_capacity(frames); Lane::ALL.len()];
        while unity[0].len() < frames {
            let n = BLOCK.min(frames - unity[0].len());
            for (lane, sum) in unity.iter_mut().zip(whole.play(n)) {
                lane.extend_from_slice(sum);
            }
        }
        for (intensity, gains) in tiers {
            assert_eq!(
                Lane::ALL.map(|lane| audition::lane_gain(lane, intensity)),
                gains,
                "intensity {intensity}"
            );
            let c = Controls::default().with_intensity(intensity);
            let got = mixed(&mut Performance::new(seed, c), frames);
            let first_diff = (0..frames).position(|i| {
                let lanes: f64 = (unity.iter().zip(gains))
                    .map(|(lane, gain)| f64::from(gain) * lane[i])
                    .sum();
                got[i] != lanes * MASTER_GAIN
            });
            assert_eq!(
                first_diff, None,
                "intensity {intensity}: lanes off their gains"
            );
        }
        for ((below, low), (above, high)) in tiers.iter().zip(&tiers[1..]) {
            let between = f32::midpoint(*below, *above);
            assert_eq!(
                Lane::ALL.map(|lane| audition::lane_gain(lane, between)),
                std::array::from_fn(|i| f32::midpoint(low[i], high[i])),
                "intensity {between} isn't halfway between its anchors' gains"
            );
        }
    }

    #[test]
    fn the_limiter_is_a_rare_backstop_at_every_corner_across_a_song_boundary() {
        // opinion: the ceiling is a rare backstop, as pixtuoid's bus limiter
        // has been, so the limiter turns down few frames, and not far
        const MOST_LIMITED_SHARE: f64 = 0.001;
        const DEEPEST_REDUCTION_DB: f64 = 1.0;
        let corners: Vec<Controls> = (0..4u8)
            .map(|bits| {
                let on = |bit: u8| f32::from((bits >> bit) & 1);
                Controls::default()
                    .with_daylight(on(0))
                    .with_intensity(on(1))
            })
            .collect();
        // each corner's song into the opposite corner's, whose first loop the
        // first one's tails ring over
        let runs: Vec<(u64, Controls, Controls)> = (0..2)
            .flat_map(|seed| {
                (corners.iter().zip(corners.iter().rev())).map(move |(&a, &b)| (seed, a, b))
            })
            .collect();
        let heard: Vec<Heard> = std::thread::scope(|scope| {
            let runs: Vec<_> = (runs.iter())
                .map(|&(seed, a, b)| {
                    scope.spawn(move || {
                        let span = audition::song_len(seed, 0, a)
                            + audition::song_len(seed, 1, b) / LOOPS_PER_SONG;
                        let mut p = Performance::new(seed, a);
                        let mut ceiling = Ceiling::new(SAMPLE_RATE);
                        let mut heard = Heard::default();
                        while heard.played < span {
                            p.mix_next(span - heard.played);
                            if heard.played == 0 {
                                heard.first = p.schedule.playing().cloned();
                                // a song latches the controls asked for
                                // before its first frame, so the opposite
                                // corner is asked for once this one plays
                                p.set_controls(b);
                            }
                            // a frame at a time, which plays the same frames,
                            // to read the gain each one plays at
                            for i in 0..p.mix.len() {
                                heard.loudest = heard.loudest.max(p.mix[i].abs());
                                ceiling.limit(&mut p.mix[i..=i]);
                                heard.hear(ceiling.gain(), ceiling.clamp(p.mix[i]), p.mix[i]);
                            }
                        }
                        heard.second = p.schedule.playing().cloned();
                        heard
                    })
                })
                .collect();
            runs.into_iter()
                .map(|run| run.join().expect("a run panicked"))
                .collect()
        });
        assert!(
            heard.iter().all(|h| h.loudest > 0.0),
            "a run plays silence, which no ceiling limits"
        );
        let unpaired: Vec<String> = (runs.iter().zip(&heard))
            .filter(|((seed, a, b), h)| {
                h.first.as_ref() != Some(&audition::song_score(*seed, 0, *a))
                    || h.second.as_ref() != Some(&audition::song_score(*seed, 1, *b))
            })
            .map(|((seed, a, b), _)| format!("seed {seed}, {a:?} into {b:?}"))
            .collect();
        assert!(
            unpaired.is_empty(),
            "runs that didn't play their corner, then the opposite one: {unpaired:#?}"
        );
        let off: Vec<String> = (runs.iter().zip(&heard))
            .filter(|(_, h)| {
                h.limited_share() > MOST_LIMITED_SHARE
                    || h.deepest_db() > DEEPEST_REDUCTION_DB
                    || h.caught > 0
            })
            .map(|((seed, a, b), h)| format!("seed {seed}, {a:?} into {b:?}: {h:?}"))
            .collect();
        assert!(
            off.is_empty(),
            "the ceiling did more than back the mix up: {off:#?}"
        );
    }

    /// What one run's ceiling did.
    #[derive(Debug, Default)]
    struct Heard {
        played: usize,
        /// The loudest sample into the limiter.
        loudest: f64,
        /// Frames the limiter turned down, and the least gain it played.
        limited: usize,
        least_gain: Option<f64>,
        /// Samples the clamp changed after the limiter.
        caught: usize,
        /// The first song's score, and the second's.
        first: Option<audition::Score>,
        second: Option<audition::Score>,
    }

    impl Heard {
        fn hear(&mut self, gain: f64, clamped: f32, limited: f64) {
            self.played += 1;
            if gain < 1.0 {
                self.limited += 1;
                self.least_gain = Some(self.least_gain.map_or(gain, |g| g.min(gain)));
            }
            self.caught += usize::from(clamped != limited as f32);
        }

        fn limited_share(&self) -> f64 {
            self.limited as f64 / self.played as f64
        }

        fn deepest_db(&self) -> f64 {
            self.least_gain.map_or(0.0, |g| -20.0 * g.log10())
        }
    }

    #[test]
    fn render_plays_the_mix_limited_clamped_and_alike_in_both_channels() {
        let (seed, c) = (8, Controls::default().with_intensity(1.0));
        let frames = 10 * SAMPLE_RATE as usize;
        let got = next(&mut Performance::new(seed, c), frames);
        let mut mix = mixed(&mut Performance::new(seed, c), frames);
        let mut ceiling = Ceiling::new(SAMPLE_RATE);
        ceiling.limit(&mut mix);
        let first_diff = (got.iter().zip(&mix)).position(|(&[l, r], &x)| {
            l.to_bits() != ceiling.clamp(x).to_bits() || l.to_bits() != r.to_bits()
        });
        assert_eq!(first_diff, None);
    }

    /// `frames` frames of `seed` from `controls`, in calls of `sizes` and then
    /// one for the rest, each of `asks` made once its frame count has played,
    /// which splits the call it falls in.
    fn play(
        seed: u64,
        controls: Controls,
        asks: &[(usize, Controls)],
        sizes: &[usize],
        frames: usize,
    ) -> Vec<Frame> {
        let mut p = Performance::new(seed, controls);
        let mut out = Vec::with_capacity(frames);
        let mut asks = asks.iter().peekable();
        let mut sizes = sizes.iter();
        let mut left = 0;
        while out.len() < frames {
            while let Some(&&(at, c)) = asks.peek()
                && at == out.len()
            {
                p.set_controls(c);
                asks.next();
            }
            if left == 0 {
                left = sizes.next().copied().unwrap_or(frames);
            }
            let until = asks.peek().map_or(frames, |&&(at, _)| at);
            let n = left.min(until - out.len());
            out.extend(next(&mut p, n));
            left -= n;
        }
        assert_eq!(p.position(), frames as u64);
        out
    }

    #[test]
    fn render_plays_the_same_frames_however_the_calls_chunk_them() {
        let seed = 3;
        let busy = Controls::default().with_intensity(1.0);
        let order = [Controls::default(), night(), busy];
        let lens = [0, 1, 2].map(|k| audition::song_len(seed, k, order[k as usize]));
        let bounds = [lens[0], lens[0] + lens[1]];
        // each song asks for the next one's controls halfway through
        let asks = [(lens[0] / 2, order[1]), (bounds[0] + lens[1] / 2, order[2])];
        // through song 2's first loop, past every tail of song 1
        let span = bounds[1] + lens[2] / LOOPS_PER_SONG;
        let whole = play(seed, order[0], &asks, &[], span);
        assert!(
            [0, bounds[0], bounds[1]]
                .windows(2)
                .all(|w| audible(&whole[w[0]..w[1]]))
                && audible(&whole[bounds[1]..]),
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
        let [b0, b1] = bounds;
        let chunkings = [
            ("random sizes", random),
            ("one frame at each boundary", vec![b0, 1, b1 - b0 - 1, 1]),
            (
                "a call across each boundary",
                vec![b0 - 63, 127, b1 - b0 - 127, 127],
            ),
        ];
        let off: Vec<String> = std::thread::scope(|scope| {
            let runs: Vec<_> = (chunkings.iter())
                .map(|(name, sizes)| {
                    let got = scope.spawn(|| play(seed, order[0], &asks, sizes, span));
                    (name, got)
                })
                .collect();
            runs.into_iter()
                .filter_map(|(name, got)| {
                    let got = got.join().expect("a run panicked");
                    first_diff(&got, &whole).map(|at| format!("{name}: frame {at}"))
                })
                .collect()
        });
        assert!(off.is_empty(), "the first frame that differs: {off:#?}");
    }

    #[test]
    fn a_song_plays_the_same_reached_by_playback_or_alone() {
        let (seed, c) = (5, Controls::default());
        // songs before it at other controls, which it must not remember
        let mut playing = Performance::new(seed, night().with_intensity(0.0));
        playing.finish_song();
        playing.set_controls(c.with_intensity(1.0));
        playing.finish_song();
        playing.set_controls(c);
        let reached = playing.finish_song();
        let alone = audition::song(seed, 2, c);
        assert!(audible(&alone));
        assert_eq!(reached.len(), alone.len());
        let first_loop = alone.len() / LOOPS_PER_SONG;
        assert!(
            first_diff(&reached[..first_loop], &alone[..first_loop]).is_some(),
            "song 1's tails ring into song 2"
        );
        assert_eq!(
            first_diff(&reached[first_loop..], &alone[first_loop..]),
            None,
            "song 2 after the tails, from its second loop"
        );
    }

    /// `song`'s gated loudness, in LUFS, which must be finite: a silent or a
    /// NaN song's would make [`audition::lufs_sd`] NaN, which no bound fails.
    fn lufs(song: &[Frame]) -> f64 {
        let left: Vec<f32> = song.iter().map(|&[l, _]| l).collect();
        let lufs = crate::loudness::gated_lufs(&left, SAMPLE_RATE);
        assert!(lufs.is_finite(), "a song reads {lufs} LUFS");
        lufs
    }

    #[test]
    fn songs_vary_in_loudness_no_more_than_the_whole_lane_mix() {
        let off: Vec<String> = (audition::WHOLE_LANE_LUFS_SD.iter())
            .filter(|&&(intensity, ..)| intensity == Controls::default().intensity)
            .filter_map(|&(intensity, daylight, whole_lane)| {
                let c = Controls::default()
                    .with_intensity(intensity)
                    .with_daylight(daylight);
                let songs: Vec<f64> = std::thread::scope(|scope| {
                    let runs: Vec<_> = audition::WHOLE_LANE_SEEDS
                        .map(|seed| {
                            scope.spawn(move || {
                                // song 0's second loop, the first its own
                                // tails ring into, as they do every later one
                                let loop_len = audition::song_len(seed, 0, c) / LOOPS_PER_SONG;
                                let mut p = Performance::new(seed, c);
                                next(&mut p, loop_len);
                                lufs(&next(&mut p, loop_len))
                            })
                        })
                        .collect();
                    (runs.into_iter())
                        .map(|run| run.join().expect("a song panicked"))
                        .collect()
                });
                let (sd, bound) = (
                    audition::lufs_sd(&songs),
                    audition::lufs_sd_bound(whole_lane),
                );
                (sd > bound).then(|| format!("daylight {daylight}: {sd:.4} LU over {bound:.4}"))
            })
            .collect();
        assert!(off.is_empty(), "songs vary more: {off:#?}");
    }

    #[test]
    fn ten_minutes_of_changing_controls_play_finite() {
        // every song asks the next for other controls, so each boundary
        // changes the mood or the room
        let asks = [
            Controls::default(),
            night().with_intensity(0.0),
            Controls::default().with_intensity(1.0),
        ];
        let mut p = Performance::new(7, asks[0]);
        let mut songs = 0;
        while p.position() < 10 * 60 * u64::from(SAMPLE_RATE) {
            let from = p.position();
            let song = p.finish_song();
            songs += 1;
            p.set_controls(asks[songs % asks.len()]);
            if let Some(i) = song.iter().position(|f| !f.iter().all(|v| v.is_finite())) {
                panic!("frame {} plays {:?}", from + i as u64, song[i]);
            }
            assert!(
                audible(&song),
                "song {songs} is silent: finite proves nothing"
            );
        }
        assert!(
            songs > 3,
            "ten minutes crossed only {} songs' boundaries",
            songs - 1
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
        };
        assert_eq!(
            wild.clamped(),
            Controls {
                daylight: 1.0,
                intensity: 1.0,
            }
        );
        let wild = Controls {
            daylight: -0.5,
            intensity: f32::NAN,
        };
        assert_eq!(
            wild.clamped(),
            Controls {
                daylight: 0.0,
                intensity: 0.5,
            }
        );
    }

    #[test]
    fn a_performance_hears_nan_controls_as_the_default() {
        let play = |c| next(&mut Performance::new(4, c), SAMPLE_RATE as usize);
        let day = play(Controls::default());
        let nan = Controls::default()
            .with_daylight(f32::NAN)
            .with_intensity(f32::NAN);
        assert_eq!(first_diff(&play(nan), &day), None);
        assert!(first_diff(&play(night()), &day).is_some());
    }

    #[test]
    fn set_controls_takes_effect_at_the_next_song_and_the_last_call_wins() {
        let seed = 6;
        let boundary = audition::song_len(seed, 0, Controls::default());
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
