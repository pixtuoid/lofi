//! The public engine: a [`Performance`] streams a seed's songs as [`Frame`]s
//! under the listener's [`Controls`].

use std::fmt;

use crate::dsp::SAMPLE_RATE;
use crate::stream::master::{Ceiling, TIER_INTENSITIES};
use crate::stream::schedule::Schedule;
use crate::stream::{BLOCK, LANES};

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
    /// The daylight below which a song is composed for the night, picked at
    /// the next song's first bar; within a song, any daylight grades its colour.
    pub const DUSK: f32 = 0.5;

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

/// What [`Performance::render_probed`] shows of the mix as it plays.
pub(crate) trait Probe {
    /// The lanes' sums, in [`Lane::ALL`](crate::synth::Lane::ALL) order,
    /// and the plate's stereo return over a step, as they enter the mix:
    /// before [`Performance::MASTER_GAIN`] and the ceiling.
    fn step(&mut self, lanes: &[Vec<f64>; LANES], plate: &[Vec<f64>; 2]);

    /// The gain the ceiling plays the next frame out at, a step's frames each
    /// heard after the step.
    fn gain(&mut self, gain: f64);

    /// A step's frames as they enter the ceiling.
    fn mixed(&mut self, _frames: &[[f64; 2]]) {}

    /// A step's frames as the ceiling's limiter plays them, before its clamp.
    fn limited(&mut self, _frames: &[[f64; 2]]) {}
}

/// What [`Performance::render`] shows: nothing.
impl Probe for () {
    fn step(&mut self, _: &[Vec<f64>; LANES], _: &[Vec<f64>; 2]) {}

    fn gain(&mut self, _: f64) {}
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
    /// The mix over a step, a stereo frame each.
    mix: Vec<[f64; 2]>,
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
    /// The mix's gain into the ceiling: lofi's mastering level for a background
    /// bed, the level pixtuoid's office has played the lanes at
    /// (<https://github.com/pixtuoid/pixtuoid/blob/a63f1782f477c357666e71a9c1147090b47fe17d/crates/pixtuoid-scene/src/audio/mixer.rs#L63-L65>),
    /// which leaves the ceiling a rare backstop; a host that stages its own
    /// sounds alike keeps the balance lofi masters at.
    pub const MASTER_GAIN: f64 = 0.35;

    /// `seed` picks the whole sequence of songs; lofi reads no clock, so the
    /// caller supplies it.
    pub fn new(seed: u64, controls: Controls) -> Self {
        Self::from_song(seed, 0, controls)
    }

    /// [`new`](Self::new), every song playing each of its loop's lanes in
    /// every bar: what a test of a mechanism a form would mute plays.
    #[cfg(test)]
    pub(crate) fn plain(seed: u64, controls: Controls) -> Self {
        let mut performance = Self::new(seed, controls);
        performance.schedule = Schedule::plain(seed, 0, SAMPLE_RATE);
        performance
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
        self.render_probed(out, &mut ());
    }

    /// [`render`](Self::render), showing `probe` the mix as it plays.
    pub(crate) fn render_probed(&mut self, out: &mut [Frame], probe: &mut impl Probe) {
        let mut done = 0;
        while done < out.len() {
            self.mix_next(out.len() - done);
            probe.step(self.schedule.sums(), self.schedule.returned());
            probe.mixed(&self.mix);
            self.ceiling
                .limit_heard(&mut self.mix, |gain| probe.gain(gain));
            probe.limited(&self.mix);
            for (frame, x) in out[done..].iter_mut().zip(&self.mix) {
                *frame = x.map(|side| self.ceiling.clamp(side));
            }
            done += self.mix.len();
        }
        self.position += out.len() as u64;
    }

    /// Mixes the lanes' next frames into `mix` at [`Self::MASTER_GAIN`]: at most
    /// `max`, [`BLOCK`] and the frames to the playing song's next latch, so
    /// each bar latches on its own frame however the calls chunk the render.
    fn mix_next(&mut self, max: usize) {
        let n = self.schedule.step(self.asked.clamped(), max);
        let (lanes, plate) = (self.schedule.sums(), self.schedule.returned());
        self.mix.clear();
        self.mix.extend((0..n).map(|i| {
            let dry = lanes.iter().map(|lane| lane[i]).sum::<f64>();
            // the dry lanes centred, the plate's return on its own sides
            [0, 1].map(|side| (dry + plate[side][i]) * Self::MASTER_GAIN)
        }));
    }

    /// Frames rendered so far.
    pub fn position(&self) -> u64 {
        self.position
    }

    /// The frames from [`position`](Self::position) to the next song's first
    /// bar, the song at `position` started first if it is due.
    pub(crate) fn finish_song(&mut self) -> Vec<Frame> {
        self.schedule.start_if_due(self.asked.clamped());
        let mut out = vec![Frame::default(); self.schedule.song_left()];
        self.render(&mut out);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audition;
    use crate::compose::{Edge, Form};
    use crate::dsp::NoiseStream;
    use crate::stream::schedule::{ANTICIPATION_S, UNITY, bars_and_len};
    use crate::synth::Lane;

    /// The bars' first frames of song `k` of `seed`'s run under `c` playing
    /// [`Form::plain`], and its length.
    fn plain_bars(seed: u64, k: u64, c: Controls) -> (Vec<usize>, usize) {
        bars_and_len(
            &audition::song_score(seed, k, c),
            &Form::plain(),
            SAMPLE_RATE,
        )
    }

    /// Song `k`'s loop length under `c`.
    fn loop_frames(seed: u64, k: u64, c: Controls) -> usize {
        audition::samples_at(audition::song_score(seed, k, c).loop_secs(), SAMPLE_RATE)
    }

    /// Asserts `songs`, `(seed, k, the controls k's first bar latches)`, span
    /// a form, so an invariance proven over them is proven over one.
    fn assert_spans_a_form(songs: &[(u64, u64, Controls)]) {
        let forms: Vec<Form> = (songs.iter())
            .map(|&(seed, k, c)| audition::song_form(seed, k, c))
            .collect();
        audition::spans_a_form(&forms).unwrap();
    }

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
        crate::loudness::fold(frames).iter().any(|&x| x != 0.0)
    }

    /// `p`'s next `frames` frames: its mix before the ceiling, and the two
    /// parts it is made of, the lanes' dry sum and the plate's return.
    fn mixed(p: &mut Performance, frames: usize) -> (Vec<[f64; 2]>, Vec<f64>, Vec<[f64; 2]>) {
        let (mut mix, mut dry, mut plate) = (Vec::new(), Vec::new(), Vec::new());
        while mix.len() < frames {
            p.mix_next(frames - mix.len());
            mix.extend_from_slice(&p.mix);
            let (lanes, wet) = (p.schedule.sums(), p.schedule.returned());
            dry.extend((0..p.mix.len()).map(|i| lanes.iter().map(|lane| lane[i]).sum::<f64>()));
            plate.extend((0..p.mix.len()).map(|i| [wet[0][i], wet[1][i]]));
        }
        (mix, dry, plate)
    }

    #[test]
    fn the_intensity_anchors_play_their_tiers_lane_for_lane() {
        // pixtuoid's `TierGain`s (crates/pixtuoid-scene/src/audio/mod.rs at
        // a63f178), but the lead's at the working tiers, each tier's lanes in
        // `Lane::ALL` order
        let tiers: [(f32, [f32; Lane::ALL.len()]); 3] = [
            (0.0, [0.75, 0.70, 0.0, 0.0, 0.78, 0.60]),
            (0.5, [0.70, 0.50, 0.60, 0.35, 0.84, 0.70]),
            (1.0, [0.65, 0.55, 0.70, 0.60, 0.78, 0.75]),
        ];
        let seed = 2;
        let mut whole = Schedule::plain(seed, 0, SAMPLE_RATE);
        whole.start_song_at(Controls::default(), UNITY);
        // the first loop, before any song but the first plays
        let frames = loop_frames(seed, 0, Controls::default());
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
            let (_, got, _) = mixed(&mut Performance::plain(seed, c), frames);
            let first_diff = (0..frames).position(|i| {
                let lanes: f64 = (unity.iter().zip(gains))
                    .map(|(lane, gain)| f64::from(gain) * lane[i])
                    .sum();
                got[i] != lanes
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
        let songs: Vec<(u64, u64, Controls)> = runs
            .iter()
            .flat_map(|&(seed, a, b)| [(seed, 0, a), (seed, 1, b)])
            .collect();
        assert_spans_a_form(&songs);
        let heard: Vec<Heard> = std::thread::scope(|scope| {
            let runs: Vec<_> = (runs.iter())
                .map(|&(seed, a, b)| {
                    scope.spawn(move || {
                        let span = audition::song_len(seed, 0, a) + loop_frames(seed, 1, b);
                        // past song 0's last latch, so it plays its corner to
                        // its end and the next song latches the opposite one
                        let bars = audition::bar_frames(seed, 0, a);
                        let last = bars[bars.len() - 1]
                            - crate::synth::samples_at(ANTICIPATION_S, SAMPLE_RATE);
                        let mut p = Performance::new(seed, a);
                        let mut heard = Heard::default();
                        while heard.played < span {
                            if heard.played == last + 1 {
                                p.set_controls(b);
                            }
                            let until = if heard.played <= last { last + 1 } else { span };
                            heard.play(&mut p, until - heard.played);
                            if heard.first.is_none() {
                                heard.first = p.schedule.playing().cloned();
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

    /// What the ceiling did to a stretch [`Performance::render_probed`]
    /// played.
    #[derive(Default)]
    struct Ceilinged {
        loudest: f64,
        gains: Vec<f64>,
        limited: Vec<[f64; 2]>,
    }

    impl Probe for Ceilinged {
        fn step(&mut self, _: &[Vec<f64>; LANES], _: &[Vec<f64>; 2]) {}

        fn gain(&mut self, gain: f64) {
            self.gains.push(gain);
        }

        fn mixed(&mut self, frames: &[[f64; 2]]) {
            self.loudest = (frames.iter().flatten()).fold(self.loudest, |m, x| m.max(x.abs()));
        }

        fn limited(&mut self, frames: &[[f64; 2]]) {
            self.limited.extend_from_slice(frames);
        }
    }

    impl Heard {
        /// Renders `n` frames of `p`, a step at most at a time, hearing what
        /// its ceiling did to each.
        fn play(&mut self, p: &mut Performance, n: usize) {
            let mut out = vec![Frame::default(); BLOCK];
            for k in (0..n).step_by(BLOCK).map(|at| BLOCK.min(n - at)) {
                let mut probe = Ceilinged::default();
                p.render_probed(&mut out[..k], &mut probe);
                self.loudest = self.loudest.max(probe.loudest);
                for ((&gain, &limited), &clamped) in
                    probe.gains.iter().zip(&probe.limited).zip(&out[..k])
                {
                    self.hear(gain, clamped, limited);
                }
            }
        }

        fn hear(&mut self, gain: f64, clamped: [f32; 2], limited: [f64; 2]) {
            self.played += 1;
            if gain < 1.0 {
                self.limited += 1;
                self.least_gain = Some(self.least_gain.map_or(gain, |g| g.min(gain)));
            }
            self.caught += usize::from(clamped != limited.map(|side| side as f32));
        }

        fn limited_share(&self) -> f64 {
            self.limited as f64 / self.played as f64
        }

        fn deepest_db(&self) -> f64 {
            self.least_gain.map_or(0.0, |g| -20.0 * g.log10())
        }
    }

    #[test]
    fn render_plays_the_dry_lanes_centred_under_the_plate_limited_and_clamped() {
        let (seed, c) = (8, Controls::default().with_intensity(1.0));
        let frames = 10 * SAMPLE_RATE as usize;
        let got = next(&mut Performance::new(seed, c), frames);
        let (mut mix, dry, plate) = mixed(&mut Performance::new(seed, c), frames);
        // one dry value on both sides, the plate's own on each
        let centred = (0..frames).position(|i| {
            mix[i] != plate[i].map(|side| (dry[i] + side) * Performance::MASTER_GAIN)
        });
        assert_eq!(centred, None, "the dry lanes play apart");
        assert!(
            plate.iter().any(|[l, r]| l != r),
            "no plate plays, so nothing proves the sides"
        );
        let mut ceiling = Ceiling::new(SAMPLE_RATE);
        ceiling.limit(&mut mix);
        let first_diff = (got.iter().zip(&mix)).position(|(frame, x)| {
            frame.map(f32::to_bits) != x.map(|side| ceiling.clamp(side).to_bits())
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
        play_from(Performance::new(seed, controls), asks, sizes, frames)
    }

    /// [`play`] from `p`.
    fn play_from(
        mut p: Performance,
        asks: &[(usize, Controls)],
        sizes: &[usize],
        frames: usize,
    ) -> Vec<Frame> {
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
        let span = bounds[1] + loop_frames(seed, 2, order[2]);
        assert_spans_a_form(&[0, 1, 2].map(|k| (seed, k, order[k as usize])));
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
    fn every_edge_plays_the_same_however_it_is_chunked() {
        let c = Controls::default();
        type Kind = (&'static str, fn(&Form) -> bool);
        let kinds: [Kind; 5] = [
            ("fade-in", |f| f.entry() == Some(Edge::Fade)),
            ("filtered intro", |f| f.entry() == Some(Edge::Filter)),
            ("dip", |f| f.dip().is_some()),
            ("fade-out", |f| f.exit() == Some(Edge::Fade)),
            ("filtered outro", |f| f.exit() == Some(Edge::Filter)),
        ];
        let mut rng = NoiseStream::new(2);
        for (name, kind) in kinds {
            let seed = (0..512)
                .find(|&seed| kind(&audition::song_form(seed, 0, c)))
                .expect("a seed draws the edge");
            let whole = audition::song(seed, 0, c, &[]);
            let mut p = Performance::from_song(seed, 0, c);
            let mut chunked = Vec::with_capacity(whole.len());
            while chunked.len() < whole.len() {
                let size = (1 + (rng.unit() * 6000.0) as usize).min(whole.len() - chunked.len());
                chunked.extend(next(&mut p, size));
            }
            assert_eq!(first_diff(&chunked, &whole), None, "{name}");
        }
    }

    #[test]
    fn a_cold_start_after_a_ringing_or_fading_close_plays_as_it_does_alone() {
        type Close = (&'static str, fn(&Form) -> bool);
        let c = Controls::default();
        let closes: [Close; 2] = [
            ("ring-out", Form::rings_out),
            ("fade-out", |f| f.exit() == Some(Edge::Fade)),
        ];
        for (name, closes) in closes {
            let (seed, k) = (0..64)
                .flat_map(|seed| (0..6).map(move |k| (seed, k)))
                .find(|&(seed, k)| {
                    closes(&audition::song_form(seed, k, c))
                        && audition::song_form(seed, k + 1, c).intro() == 0
                })
                .expect("a seed plays the pair");
            let mut playing = Performance::from_song(seed, k, c);
            let closing = playing.finish_song();
            let reached = playing.finish_song();
            let alone = audition::song(seed, k + 1, c, &[]);
            assert_eq!(reached.len(), alone.len(), "{name}");
            let first_loop = loop_frames(seed, k + 1, c);
            assert_eq!(
                first_diff(&reached[first_loop..], &alone[first_loop..]),
                None,
                "{name}: the cold start after its first loop"
            );
            // the join steps no further than the closing song itself does
            let step = |frames: &[Frame]| {
                (frames.windows(2))
                    .flat_map(|w| [0, 1].map(|side| (w[1][side] - w[0][side]).abs()))
                    .fold(0.0f32, f32::max)
            };
            let join: Vec<Frame> = (closing[closing.len() - BLOCK..].iter())
                .chain(&reached[..BLOCK])
                .copied()
                .collect();
            let (at_join, within) = (step(&join), step(&closing));
            assert!(
                at_join <= within,
                "{name}: the join steps {at_join}, the song {within}"
            );
        }
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
        assert_spans_a_form(&[
            (seed, 0, night().with_intensity(0.0)),
            (seed, 1, c.with_intensity(1.0)),
            (seed, 2, c),
        ]);
        assert_eq!(reached.len(), alone.len());
        let first_loop = loop_frames(seed, 2, c);
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
        let lufs = crate::loudness::gated_lufs(&crate::loudness::fold(song), SAMPLE_RATE);
        assert!(lufs.is_finite(), "a song reads {lufs} LUFS");
        lufs
    }

    #[test]
    fn songs_vary_in_loudness_no_more_than_the_whole_lane_mix() {
        let off: Vec<String> = (audition::WHOLE_LANE_LUFS_SD.iter())
            .filter_map(|&(intensity, daylight, whole_lane)| {
                let c = Controls::default()
                    .with_intensity(intensity)
                    .with_daylight(daylight);
                let songs: Vec<f64> = std::thread::scope(|scope| {
                    let runs: Vec<_> = audition::LUFS_SEEDS
                        .map(|seed| {
                            scope.spawn(move || {
                                // song 0's second loop, the first its own
                                // tails ring into, as they do every later one
                                let loop_len = loop_frames(seed, 0, c);
                                let mut p = Performance::plain(seed, c);
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
                    audition::lufs_sd_bound(whole_lane, songs.len()),
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
        // a run whose ten minutes fade or filter somewhere, as most seeds' do
        let seed = 2;
        let ten = 10 * 60 * u64::from(SAMPLE_RATE);
        // as many songs as start inside ten minutes at their own lengths
        let (mut want, mut start) = (0, 0);
        while start < ten {
            start += audition::song_len(seed, want, asks[want as usize % asks.len()]) as u64;
            want += 1;
        }
        let songs: Vec<_> = (0..want)
            .map(|k| (seed, k, asks[k as usize % asks.len()]))
            .collect();
        assert_spans_a_form(&songs);
        let mut p = Performance::new(seed, asks[0]);
        let mut songs = 0;
        while p.position() < ten {
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
            want > 2,
            "ten minutes crossed only {} songs' boundaries",
            want - 1
        );
        assert_eq!(songs as u64, want, "the songs ten minutes played");
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
        crate::stream::master::lookahead(SAMPLE_RATE)
    }

    #[test]
    fn a_call_is_heard_from_the_next_bars_latch_and_the_last_call_before_it_wins() {
        let (seed, c) = (6, Controls::default());
        let busy = c.with_intensity(1.0);
        let (bars, _) = plain_bars(seed, 0, c);
        let anticipation = crate::synth::samples_at(ANTICIPATION_S, SAMPLE_RATE);
        let [previous, latch, after] = [3, 4, 5].map(|bar| bars[bar] - anticipation);
        let frames = bars[6];
        let play = |seed, c, asks: &[(usize, Controls)], sizes: &[usize], frames| {
            play_from(Performance::plain(seed, c), asks, sizes, frames)
        };
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
        let mut first = Performance::plain(seed, c);
        first.set_controls(night());
        let second = SAMPLE_RATE as usize;
        assert_eq!(
            first_diff(
                &next(&mut first, second),
                &next(&mut Performance::plain(seed, night()), second)
            ),
            None,
            "a call before the first render missed the first bar"
        );
    }

    #[test]
    fn the_daylight_grades_the_playing_song_and_picks_the_next_songs_family() {
        let seed = 6;
        let (bars, boundary) = plain_bars(seed, 0, Controls::default());
        let anticipation = crate::synth::samples_at(ANTICIPATION_S, SAMPLE_RATE);
        let latch = bars[8] - anticipation;
        let mut p = Performance::plain(seed, Controls::default());
        let mut heard = next(&mut p, latch);
        p.set_controls(night());
        heard.extend(next(&mut p, boundary - latch));
        assert_eq!(
            p.schedule.playing(),
            Some(&audition::song_score(seed, 0, Controls::default()))
        );
        let unasked = next(&mut Performance::plain(seed, Controls::default()), boundary);
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
        assert_spans_a_form(&[(seed, 0, c), (seed, 1, into)]);
        assert_eq!(reached.len(), alone.len());
        let first_loop = loop_frames(seed, 1, into);
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
        // the timelines' random intensities are above 0, where every song
        // plays the form it draws
        assert_spans_a_form(
            &(0..4)
                .flat_map(|seed| (0..3).map(move |k| (seed, k, Controls::default())))
                .collect::<Vec<_>>(),
        );
        let runs: Vec<Heard> = std::thread::scope(|scope| {
            let runs: Vec<_> = (0..4u64)
                .map(|seed| {
                    scope.spawn(move || {
                        let mut rng = NoiseStream::new(seed + 40);
                        let span = (0..3)
                            .map(|k| audition::song_len(seed, k, Controls::default()))
                            .sum::<usize>();
                        let mut p = Performance::new(seed, Controls::default());
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
                            let n = (span - heard.played).min(next_call - heard.played);
                            heard.play(&mut p, n);
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
