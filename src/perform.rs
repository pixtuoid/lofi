//! The public engine: a [`Performance`] streams a seed's songs as [`Frame`]s
//! under the listener's [`Controls`].

use std::fmt;

use crate::dsp::SAMPLE_RATE;
use crate::stream::master::{Ceiling, MASTER_GAIN, TIER_INTENSITIES};
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
    /// 0 = dry … 1 = raining: the playing song darkens and softens, its
    /// tempo, key, harmony and melody kept. lofi plays no rain itself.
    pub rain: f32,
}

impl Default for Controls {
    /// Full day, pixtuoid's Moderate office (its taste anchor).
    fn default() -> Self {
        let [_, moderate, _] = TIER_INTENSITIES;
        Self {
            daylight: 1.0,
            intensity: moderate,
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
    /// The latest controls asked for, which the next bar latches.
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

    /// Applies from the next bar on, latched a little before its line so a
    /// note played early for it plays under it; a song's first bar latches on
    /// its line, the first song's what was asked last before the first render.
    /// It is heard from the latch plus the ceiling's lookahead, `LOOKAHEAD_S`
    /// in `stream::master`. The intensity enters a lane at its first note in
    /// the bar and moves its level within a beat; the daylight grades the
    /// colour within a bar and picks the next song's day or night family. Of
    /// several calls before one latch, the last wins.
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
    /// `max`, [`BLOCK`] and the frames to the playing song's next latch, so
    /// each bar latches on its own frame however the calls chunk the render.
    fn mix_next(&mut self, max: usize) {
        self.start_due_song();
        self.schedule.latch_due(self.asked.clamped());
        let n = max.min(BLOCK).min(self.schedule.until_latch());
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
    use crate::stream::schedule::{ANTICIPATION_S, LOOPS_PER_SONG, UNITY};
    use crate::synth::Lane;

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
        whole.start_song_at(Controls::default(), UNITY);
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
                        // past song 0's last latch, so it plays its corner to
                        // its end and the next song latches the opposite one
                        let bars = audition::bar_frames(seed, 0, a);
                        let last = bars[bars.len() - 1]
                            - crate::synth::samples_at(ANTICIPATION_S, SAMPLE_RATE);
                        let mut p = Performance::new(seed, a);
                        let mut ceiling = Ceiling::new(SAMPLE_RATE);
                        let mut heard = Heard::default();
                        while heard.played < span {
                            if heard.played == last + 1 {
                                p.set_controls(b);
                            }
                            let until = if heard.played <= last { last + 1 } else { span };
                            p.mix_next(until - heard.played);
                            if heard.played == 0 {
                                heard.first = p.schedule.playing().cloned();
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
        // rain asked mid-song, so its phrase-long ramps cross a seam
        let order = [
            Controls::default(),
            night().with_rain(1.0),
            busy.with_rain(0.5),
        ];
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
        // song 0's latch frames, the night's grade ramping from the first
        // past its ask
        let anticipation = crate::synth::samples_at(ANTICIPATION_S, SAMPLE_RATE);
        let latches: Vec<usize> = (audition::bar_frames(seed, 0, order[0]).iter().skip(1))
            .map(|bar| bar - anticipation)
            .collect();
        let cut_at = |cuts: &[usize]| -> Vec<usize> {
            (cuts.iter().scan(0, |at, &cut| {
                let size = cut - *at;
                *at = cut;
                Some(size)
            }))
            .filter(|&size| size > 0)
            .collect()
        };
        let beside: Vec<usize> = latches.iter().flat_map(|&l| [l - 1, l + 1]).collect();
        let block = crate::stream::lanes::COEFF_BLOCK;
        let chunkings = [
            ("random sizes", random),
            ("one frame at each boundary", vec![b0, 1, b1 - b0 - 1, 1]),
            (
                "a call across each boundary",
                vec![b0 - 63, 127, b1 - b0 - 127, 127],
            ),
            ("cut at each latch", cut_at(&latches)),
            ("a frame either side of each latch", cut_at(&beside)),
            (
                "a frame off each coefficient block",
                [block - 1, block + 1].repeat(span / (2 * block) + 1),
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
        let alone = audition::song(seed, 2, c, &[]);
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
            rain: f32::NAN,
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
            rain: 3.0,
        };
        assert_eq!(
            wild.clamped(),
            Controls {
                daylight: 0.0,
                intensity: 0.5,
                rain: 1.0,
            }
        );
        assert_eq!(Controls::default().rain, 0.0, "the default rains");
        assert_eq!(Controls::default().with_rain(0.3).rain, 0.3);
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
        // and a bar latches them alike
        let bars = audition::bar_frames(4, 0, Controls::default());
        let busy = Controls::default().with_intensity(1.0);
        let asked = |c| self::play(4, busy, &[(bars[2], c)], &[], bars[5]);
        assert_eq!(first_diff(&asked(nan), &asked(Controls::default())), None);
    }

    /// How late the ceiling plays the mix, in frames.
    fn lookahead() -> usize {
        (crate::stream::master::LOOKAHEAD_S * f64::from(SAMPLE_RATE)) as usize
    }

    #[test]
    fn a_call_is_heard_from_the_next_bars_latch_and_the_last_call_before_it_wins() {
        let (seed, c) = (6, Controls::default());
        let busy = c.with_intensity(1.0);
        let bars = audition::bar_frames(seed, 0, c);
        let anticipation = crate::synth::samples_at(ANTICIPATION_S, SAMPLE_RATE);
        let [previous, latch, after] = [3, 4, 5].map(|bar| bars[bar] - anticipation);
        let frames = bars[6];
        let unasked = play(seed, c, &[], &[], frames);
        let at_latch = play(seed, c, &[(latch, busy)], &[], frames);
        let heard = first_diff(&at_latch, &unasked);
        assert!(
            heard.is_some_and(|at| at >= latch + lookahead()),
            "heard at {heard:?}, latched at {latch}"
        );
        for at in [previous + 1, previous.midpoint(latch)] {
            let early = play(seed, c, &[(at, busy)], &[], frames);
            assert_eq!(
                first_diff(&early, &at_latch),
                None,
                "a call at {at} isn't the bar's"
            );
        }
        let late = play(seed, c, &[(latch + 1, busy)], &[], frames);
        let heard = first_diff(&late, &unasked);
        assert!(
            heard.is_some_and(|at| at >= after + lookahead()),
            "a call past the latch is heard at {heard:?}, before the next bar's"
        );
        let calls = [(previous + 1, night().with_intensity(0.0)), (latch, busy)];
        assert_eq!(
            first_diff(&play(seed, c, &calls, &[], frames), &at_latch),
            None,
            "the last call didn't win"
        );
        // the first bar latches the controls a call before the first render asks for
        let mut first = Performance::new(seed, c);
        first.set_controls(night());
        let second = SAMPLE_RATE as usize;
        assert_eq!(
            first_diff(
                &next(&mut first, second),
                &next(&mut Performance::new(seed, night()), second)
            ),
            None,
            "a call before the first render missed the first bar"
        );
    }

    #[test]
    fn the_daylight_grades_the_playing_song_and_picks_the_next_songs_family() {
        let seed = 6;
        let boundary = audition::song_len(seed, 0, Controls::default());
        let bars = audition::bar_frames(seed, 0, Controls::default());
        let anticipation = crate::synth::samples_at(ANTICIPATION_S, SAMPLE_RATE);
        let latch = bars[8] - anticipation;
        let mut p = Performance::new(seed, Controls::default());
        let mut heard = next(&mut p, latch);
        p.set_controls(night());
        heard.extend(next(&mut p, boundary - latch));
        assert_eq!(
            p.schedule.playing(),
            Some(&audition::song_score(seed, 0, Controls::default()))
        );
        let unasked = next(&mut Performance::new(seed, Controls::default()), boundary);
        let graded = first_diff(&heard, &unasked);
        assert!(
            graded.is_some_and(|at| at >= latch + lookahead()),
            "the colour moved at {graded:?}, latched at {latch}"
        );
        next(&mut p, 1);
        assert_eq!(
            p.schedule.playing(),
            Some(&audition::song_score(seed, 1, night())),
            "the next song isn't the night's"
        );
    }

    #[test]
    fn a_song_plays_its_timeline_the_same_reached_by_playback_or_alone() {
        let (seed, c) = (5, Controls::default());
        let k0 = audition::bar_frames(seed, 0, c);
        let anticipation = crate::synth::samples_at(ANTICIPATION_S, SAMPLE_RATE);
        let boundary = audition::song_len(seed, 0, c);
        // asked at song 0's last latch, so its last bar plays `into` and song
        // 1 starts under it, while song 0's phrase of rain still ramps
        let last = k0[k0.len() - 1] - anticipation;
        let into = night().with_intensity(0.2).with_rain(1.0);
        let mut playing = Performance::new(seed, c);
        next(&mut playing, last);
        playing.set_controls(into);
        next(&mut playing, boundary - last);
        let k1 = audition::bar_frames(seed, 1, into);
        let calls = [
            (k1[3], into.with_intensity(1.0)),
            (k1[9], c.with_daylight(0.3)),
            (k1[12], c.with_rain(0.4)),
        ];
        let mut reached = Vec::new();
        for &(at, controls) in &calls {
            reached.extend(next(&mut playing, at - reached.len()));
            playing.set_controls(controls);
        }
        reached.extend(playing.finish_song());
        let alone = audition::song(seed, 1, into, &calls);
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
    fn the_ceiling_stays_a_backstop_under_random_timelines_across_seams() {
        // opinion, as the corners' row: few frames turned down, and not far
        const MOST_LIMITED_SHARE: f64 = 0.001;
        const DEEPEST_REDUCTION_DB: f64 = 1.0;
        let runs: Vec<Heard> = std::thread::scope(|scope| {
            let runs: Vec<_> = (0..4u64)
                .map(|seed| {
                    scope.spawn(move || {
                        let mut rng = NoiseStream::new(seed + 40);
                        let span = (0..3)
                            .map(|k| audition::song_len(seed, k, Controls::default()))
                            .sum::<usize>();
                        let mut p = Performance::new(seed, Controls::default());
                        let mut ceiling = Ceiling::new(SAMPLE_RATE);
                        let mut heard = Heard::default();
                        let mut next_call = 0;
                        while heard.played < span {
                            if heard.played >= next_call {
                                p.set_controls(
                                    Controls::default()
                                        .with_daylight(rng.unit())
                                        .with_intensity(rng.unit())
                                        .with_rain(rng.unit()),
                                );
                                next_call += 1 + (rng.unit() * 4.0 * SAMPLE_RATE as f32) as usize;
                            }
                            p.mix_next((span - heard.played).min(next_call - heard.played));
                            for i in 0..p.mix.len() {
                                heard.loudest = heard.loudest.max(p.mix[i].abs());
                                ceiling.limit(&mut p.mix[i..=i]);
                                heard.hear(ceiling.gain(), ceiling.clamp(p.mix[i]), p.mix[i]);
                            }
                        }
                        heard
                    })
                })
                .collect();
            runs.into_iter()
                .map(|run| run.join().expect("a run panicked"))
                .collect()
        });
        let off: Vec<String> = (runs.iter().enumerate())
            .filter(|(_, h)| {
                h.loudest == 0.0
                    || h.limited_share() > MOST_LIMITED_SHARE
                    || h.deepest_db() > DEEPEST_REDUCTION_DB
                    || h.caught > 0
            })
            .map(|(seed, h)| format!("seed {seed}: {h:?}"))
            .collect();
        assert!(
            off.is_empty(),
            "the ceiling did more than back the mix up: {off:#?}"
        );
    }
}
