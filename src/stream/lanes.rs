//! Each lane's chain for one song: the whole-lane recipe (`synth`'s lane
//! recipes, `lofi_post`, `master` and `brighten`) over causal filters,
//! with fixed gains where a whole lane normalized to its peak. A song's chains
//! start at rest on its first frame, so how it sounds never depends on the
//! songs before it.

use std::f64::consts::{FRAC_PI_2, TAU};
use std::ops::Range;

use super::biquad::{Cascade, Lowpass, butter_lowpass, cookbook_bandpass};
use super::delay::DelayLine;
use super::gain_of_db;
use super::hiss::Hiss;
use super::onepole::Tilt;
use super::rain;
use super::ramp::{Ramp, Shape};
use crate::Controls;
use crate::compose::{Lane, Mood};
use crate::dsp::NoiseStream;
use crate::synth::{
    DRIVE, DUCK_DB, DUCK_RELEASE_S, HEAD_BUMP_HZ, HEAD_BUMP_MIX, HISS_HZ, HISS_LEVEL, HUM_HZ,
    HUM_SWELL_HZ, PadTone, Part, ROLLOFF_HZ, TILT_HZ, TILT_LIFT, WARPS, room_tone, samples_at,
};

/// The order of each slope that stands in for a whole-lane brickwall.
const ORDER: usize = 4;

/// How often, in frames from the song's first, a grading lane recomputes its
/// tone's coefficients and its drive: each step moves the corner a small
/// fraction of an octave. Opinion.
pub(crate) const COEFF_BLOCK: usize = 32;

/// The first frame of frame `n`'s coefficient block.
const fn block_of(n: usize) -> usize {
    n - n % COEFF_BLOCK
}

/// The first frame of the coefficient block after frame `n`'s, or `end` if
/// that comes first.
fn next_block(n: usize, end: usize) -> usize {
    (block_of(n) + COEFF_BLOCK).min(end)
}

/// The corner a filtered edge or a dip low-passes a music lane at. Opinion: a
/// muffle like Neal's verse piano, "run through a lowpass filter with a
/// roll-off around 800 Hz" (Organised Sound 27(1), 2022, §3.1).
pub(crate) const FILTERED_HZ: f32 = 800.0;

/// How long a song's texture bed crossfades into the next song's, so the hum,
/// one sine that each song starts at its own phase, doesn't jump. The length
/// is opinion: no measurement or LISTEN set it.
const BED_CROSSFADE_S: f32 = 0.03;

/// The phase in radians, within one cycle, of a sine at `hz` `n` frames into a
/// song at `sr`: counted in f64 and reduced before it narrows, since `n` passes
/// f32's exact integers within minutes at a high rate.
fn phase(hz: f32, n: usize, sr: u32) -> f32 {
    let cycles = f64::from(hz) * n as f64 / f64::from(sr);
    (TAU * cycles.fract()) as f32
}

/// Each lane's fixed gains `[in, out]`, standing in for the whole-lane render's
/// peak normalizations: `in` for its first `normalize`, which sets the level
/// into the drive (the texture's, into the duck), and `out` for the product of
/// the rest. A gain sets its lane's level in the stream, so it moves as a
/// sound change the judge scores; `lofi_measure --staging` prints each beside
/// the median of its normalizations over `audition::STAGING_SEEDS`, and how
/// far apart they sit.
pub(crate) fn staging(mood: Mood, lane: Lane) -> [f64; 2] {
    match (mood, lane) {
        (Mood::Day, Lane::Pad) => [0.1376, 0.7424],
        (Mood::Day, Lane::Sparkle) => [1.1440, 0.5299],
        (Mood::Day, Lane::Keys) => [0.5419, 0.7848],
        (Mood::Day, Lane::Drums) => [0.6785, 0.6305],
        (Mood::Day, Lane::Texture) => [3.1570, 0.5869],
        (Mood::Day, Lane::Bass) => [0.8351, 0.6117],
        (Mood::Night, Lane::Pad) => [0.2049, 0.7833],
        (Mood::Night, Lane::Sparkle) => [1.2683, 0.5361],
        (Mood::Night, Lane::Keys) => [1.0453, 0.6983],
        (Mood::Night, Lane::Drums) => [1.0628, 0.7487],
        (Mood::Night, Lane::Texture) => [3.0763, 0.6012],
        (Mood::Night, Lane::Bass) => [1.3064, 0.6026],
    }
}

/// One lane's chain for one song, from the song's first frame.
#[derive(Debug)]
pub(crate) enum Chain {
    Tape(Tape),
    Room(Room),
}

impl Chain {
    /// `lane`'s chain in a `mood` song `len` frames long at `sr`, picked by
    /// its `part` as `render_lane` picks its whole-lane core, playing under
    /// `automation` from its first frame; the texture ducks under each onset
    /// in `kicks`.
    pub(crate) fn new(
        mood: Mood,
        lane: Lane,
        part: &Part<'_>,
        len: usize,
        kicks: &[usize],
        automation: &Automation,
        sr: u32,
    ) -> Self {
        let gains = staging(mood, lane);
        let darkens = rain::darkens(lane);
        let tape = |own, pad| Self::Tape(Tape::new(own, pad, darkens, gains, automation, sr));
        match part {
            Part::Chords { .. } => {
                let pad = PadTone::of(mood);
                let own = Tone::Fixed {
                    corner_hz: pad.cutoff_hz,
                    drive: f64::from(DRIVE),
                };
                tape(own, Some(pad))
            }
            Part::Notes { recipes: r, .. } => tape(
                Tone::Graded(Grade::new(
                    [r.night.cutoff_hz, r.day.cutoff_hz],
                    [DRIVE, DRIVE],
                )),
                None,
            ),
            Part::Hits { recipes: r, .. } => tape(
                Tone::Graded(Grade::new(
                    [r.night.cutoff_hz, r.day.cutoff_hz],
                    [r.night.drive, r.day.drive],
                )),
                None,
            ),
            Part::Bed { .. } => Self::Room(Room::new(len, kicks, gains, sr)),
        }
    }

    /// Whether the lane's colour grades with the daylight.
    #[cfg(test)]
    pub(crate) fn grades(&self) -> bool {
        matches!(
            self,
            Self::Tape(Tape {
                own: Tone::Graded(_),
                ..
            })
        )
    }

    /// Whether rain darkens the lane's tilt.
    #[cfg(test)]
    pub(crate) fn darkens(&self) -> bool {
        matches!(self, Self::Tape(Tape { darkens: true, .. }))
    }

    /// The corner the lane's tone is tuned to, if it has one.
    #[cfg(test)]
    pub(crate) fn tuned_hz(&self) -> Option<f32> {
        match self {
            Self::Tape(tape) => Some(tape.tuned),
            Self::Room(_) => None,
        }
    }

    /// Plays the lane's frames from song frame `from` in place, under
    /// `automation`, working in `work`; the texture draws its hiss from
    /// `noise`, and ducks under the kicks in them if `drums_present`.
    pub(crate) fn process(
        &mut self,
        buf: &mut [f64],
        from: usize,
        automation: &Automation,
        noise: &mut NoiseStream,
        drums_present: bool,
        work: &mut Vec<f64>,
    ) {
        match self {
            Self::Tape(tape) => tape.process(buf, from, automation, work),
            Self::Room(room) => room.process(buf, from, noise, drums_present, work),
        }
    }

    /// Whether silence in from song frame `at` on plays exact silence out,
    /// for good.
    pub(crate) fn at_rest(&self, at: usize) -> bool {
        match self {
            Self::Tape(tape) => tape.at_rest(),
            Self::Room(room) => room.at_rest(at),
        }
    }
}

/// `buf` through `tanh(x·drive)`, normalized so a full-scale input stays
/// full scale.
fn saturate(buf: &mut [f64], drive: f64) {
    let norm = drive.tanh();
    for x in buf {
        *x = (*x * drive).tanh() / norm;
    }
}

/// A lane's tone corner and drive between its night and day values, graded
/// with the daylight: `night + (day − night)·daylight`, the corner in octaves,
/// each exact at both ends.
#[derive(Debug)]
pub(crate) struct Grade {
    corners_hz: [f32; 2],
    drives: [f32; 2],
}

/// What a song's chains play under, each one timeline on the song's clock,
/// as an `AudioParam` "maintains a list of zero or more automation events"
/// (<https://webaudio.github.io/web-audio-api/>) however many nodes it
/// moves: the daylight the colours grade to and the form's low-pass sweep,
/// read once per coefficient block, and the rain the tilts darken by, read
/// on every frame.
#[derive(Clone, Debug)]
pub(crate) struct Automation {
    daylight: Ramp,
    rain: Ramp,
    sweep: Option<Sweep>,
}

impl Automation {
    /// Holding `controls`' daylight and rain from the song's first frame,
    /// with the form's `sweep`, if it has one.
    pub(crate) fn new(controls: Controls, sweep: Option<Sweep>) -> Self {
        Self {
            daylight: Ramp::held(f64::from(controls.daylight), Shape::Linear),
            rain: Ramp::held(f64::from(controls.rain), Shape::Linear),
            sweep,
        }
    }

    /// From frame `at`, grades toward `controls`' daylight over `grade_ramp`
    /// frames and darkens toward their rain over `rain_ramp` frames.
    pub(crate) fn latch(
        &mut self,
        at: usize,
        controls: Controls,
        grade_ramp: usize,
        rain_ramp: usize,
    ) {
        self.daylight
            .toward(at, f64::from(controls.daylight), grade_ramp);
        self.rain.toward(at, f64::from(controls.rain), rain_ramp);
    }

    /// Holds every timeline where it is on frame `at`, the song's end.
    pub(crate) fn freeze(&mut self, at: usize) {
        self.daylight.hold(at);
        self.rain.hold(at);
    }

    /// The rain, on every frame.
    pub(crate) fn rain(&self) -> &Ramp {
        &self.rain
    }

    /// The daylight frame `n` plays at: the ramp's at the start of `n`'s
    /// coefficient block, or its target exactly from its end on.
    fn daylight(&self, n: usize) -> f64 {
        if n >= self.daylight.end() {
            self.daylight.to()
        } else {
            self.daylight.at(block_of(n))
        }
    }

    /// The frame after `n` from which [`daylight`](Self::daylight) may
    /// differ, or `None` once it holds.
    fn daylight_changes(&self, n: usize) -> Option<usize> {
        let end = self.daylight.end();
        (n < end).then(|| next_block(n, end))
    }

    /// The frame the daylight holds from.
    #[cfg(test)]
    pub(crate) fn graded_until(&self) -> usize {
        self.daylight.end()
    }
}

/// A song's low-pass automation on its music lanes: how open each lane's tone
/// is, from 0 at [`FILTERED_HZ`] to 1 at its own corner, at breakpoints in
/// song frames, linear between them, open before the first and held past the
/// last. Read once per coefficient block, as a grade is.
#[derive(Clone, Debug)]
pub(crate) struct Sweep(Vec<Point>);

/// A sweep's breakpoint: `(frame in the song, openness)`.
type Point = (usize, f64);

impl Sweep {
    /// The sweep through `points`, `(frame, openness)` in frame order.
    pub(crate) fn new(points: Vec<Point>) -> Self {
        debug_assert!(points.is_sorted_by_key(|&(at, _)| at));
        Self(points)
    }

    /// The breakpoints on either side of frame `n`: the last at or before it
    /// and the first after it.
    fn around(&self, n: usize) -> (Option<Point>, Option<Point>) {
        let i = self.0.partition_point(|&(at, _)| at <= n);
        (i.checked_sub(1).map(|i| self.0[i]), self.0.get(i).copied())
    }

    /// How open frame `n` plays: its segment's at the start of `n`'s
    /// coefficient block or of the segment, whichever is later, so a
    /// breakpoint's openness is exact from its frame on.
    fn at(&self, n: usize) -> f64 {
        match self.around(n) {
            (None, _) => 1.0,
            (Some((_, x)), None) => x,
            (Some((a, xa)), Some((b, xb))) => {
                let m = block_of(n).max(a);
                xa + (xb - xa) * (m - a) as f64 / (b - a) as f64
            }
        }
    }

    /// The frame after `n` from which [`at`](Self::at) may differ, or `None`
    /// once it holds.
    fn next_change(&self, n: usize) -> Option<usize> {
        match self.around(n) {
            (_, None) => None,
            (None, Some((b, _))) => Some(b),
            (Some((_, xa)), Some((b, xb))) if xa == xb => Some(b),
            (Some(_), Some((b, _))) => Some(next_block(n, b)),
        }
    }
}

/// A lane's tone corner `own`, opened `x` of the way from [`FILTERED_HZ`] to
/// it in octaves: the lower of the lane's own corner and the sweep's, as FMOD
/// combines a cutoff automated twice ("The lowest curve value is used",
/// <https://fmod.com/docs/2.03/studio/advanced-topics.html#automation-curve-combination>).
/// Its ends return without a `powf`, whose precision "is non-deterministic"
/// (<https://doc.rust-lang.org/std/primitive.f64.html#method.powf>).
fn swept(own: f32, x: f64) -> f32 {
    if x >= 1.0 || FILTERED_HZ >= own {
        own
    } else if x <= 0.0 {
        FILTERED_HZ
    } else {
        (f64::from(own).powf(x) * f64::from(FILTERED_HZ).powf(1.0 - x)) as f32
    }
}

/// A tape's own tone corner and drive: set once, or graded with the daylight.
#[derive(Debug)]
enum Tone {
    Fixed { corner_hz: f32, drive: f64 },
    Graded(Grade),
}

impl Tone {
    /// The tone's own corner and drive on frame `n` under `automation`.
    fn at(&self, n: usize, automation: &Automation) -> (f32, f64) {
        match self {
            Self::Fixed { corner_hz, drive } => (*corner_hz, *drive),
            Self::Graded(grade) => {
                let t = automation.daylight(n);
                (grade.corner_hz(t), grade.drive(t))
            }
        }
    }

    /// The frame after `n` from which [`at`](Self::at) may differ, or `None`
    /// once it holds.
    fn next_change(&self, n: usize, automation: &Automation) -> Option<usize> {
        match self {
            Self::Fixed { .. } => None,
            Self::Graded(_) => automation.daylight_changes(n),
        }
    }
}

/// `len` frames from frame `from`, cut wherever `automation` may move
/// `tone`, by its grade or its sweep: each piece's range from `from`, and its
/// first frame, which both read the whole piece at.
fn pieces<'a>(
    tone: &'a Tone,
    automation: &'a Automation,
    from: usize,
    len: usize,
) -> impl Iterator<Item = (Range<usize>, usize)> + 'a {
    let mut at = 0;
    std::iter::from_fn(move || {
        (at < len).then(|| {
            let n = from + at;
            let next = [
                tone.next_change(n, automation),
                (automation.sweep.as_ref()).and_then(|sweep| sweep.next_change(n)),
            ];
            let next =
                (next.into_iter().flatten().min()).map_or(len, |next| (next - from).min(len));
            let piece = (at..next, n);
            at = next;
            piece
        })
    })
}

impl Grade {
    fn new(corners_hz: [f32; 2], drives: [f32; 2]) -> Self {
        Self { corners_hz, drives }
    }

    fn corner_hz(&self, t: f64) -> f32 {
        let [night, day] = self.corners_hz;
        if t <= 0.0 {
            night
        } else if t >= 1.0 {
            day
        } else {
            (f64::from(night) * (f64::from(day) / f64::from(night)).powf(t)) as f32
        }
    }

    fn drive(&self, t: f64) -> f64 {
        let [night, day] = self.drives.map(f64::from);
        if t <= 0.0 {
            night
        } else if t >= 1.0 {
            day
        } else {
            night * (1.0 - t) + day * t
        }
    }
}

/// A musical lane's chain: its own low-pass, then `lofi_post`'s tape, then
/// the tilt.
#[derive(Debug)]
pub(crate) struct Tape {
    tone: Lowpass,
    /// The tone's own corner and drive.
    own: Tone,
    /// Whether the song's rain darkens its tilt.
    darkens: bool,
    /// The pad's slow swell.
    pad: Option<PadTone>,
    /// The corner the tone is tuned to.
    tuned: f32,
    gains: [f64; 2],
    warp: DelayLine,
    /// Wow and flutter, each `(radians per frame, read-head excursion in
    /// frames)`.
    warps: [(f64, f64); WARPS.len()],
    /// The sum of the excursions: `warp_resample` reads that far either side
    /// of now, so a causal line reads that late, and the tape lags by it.
    reach: f64,
    bump: Cascade,
    rolloff: Lowpass,
    tilt: Tilt,
    sr: u32,
}

impl Tape {
    /// A tape whose tone starts at `own`'s first frame under `automation`.
    fn new(
        own: Tone,
        pad: Option<PadTone>,
        darkens: bool,
        gains: [f64; 2],
        automation: &Automation,
        sr: u32,
    ) -> Self {
        let (cutoff_hz, _) = own.at(0, automation);
        let rate = f64::from(sr);
        // the excursion `warp_resample` gives a `(hz, dev)` pair
        let warps = WARPS.map(|(hz, dev)| {
            let w = TAU * f64::from(hz);
            (w / rate, f64::from(dev) * rate / w)
        });
        let reach = warps.iter().map(|&(_, a)| a).sum();
        Self {
            tone: butter_lowpass(cutoff_hz, ORDER, sr),
            own,
            tuned: cutoff_hz,
            darkens,
            pad,
            gains,
            warp: DelayLine::new(2.0 * reach),
            warps,
            reach,
            bump: cookbook_bandpass(HEAD_BUMP_HZ.0, HEAD_BUMP_HZ.1, sr),
            rolloff: butter_lowpass(ROLLOFF_HZ, ORDER, sr),
            tilt: Tilt::new(TILT_HZ, f64::from(TILT_LIFT), sr),
            sr,
        }
    }

    /// The tape's grade, if its tone grades.
    #[cfg(test)]
    fn grade(&self) -> Option<&Grade> {
        match &self.own {
            Tone::Graded(grade) => Some(grade),
            Tone::Fixed { .. } => None,
        }
    }

    /// Plays `buf`, song frames from `from` on, under `automation`, its head
    /// bump's band in `band`.
    fn process(
        &mut self,
        buf: &mut [f64],
        from: usize,
        automation: &Automation,
        band: &mut Vec<f64>,
    ) {
        let [gain_in, gain_out] = self.gains;
        let len = buf.len();
        let sweep = automation.sweep.as_ref();
        for (range, n) in pieces(&self.own, automation, from, len) {
            let (own, _) = self.own.at(n, automation);
            let corner = swept(own, sweep.map_or(1.0, |sweep| sweep.at(n)));
            if corner != self.tuned {
                self.tone.retune(corner, self.sr);
                self.tuned = corner;
            }
            self.tone.process(&mut buf[range]);
        }
        for (n, x) in (from..).zip(buf.iter_mut()) {
            *x *= self.swell(n) * gain_in;
        }
        let (warps, reach) = (self.warps, self.reach);
        self.warp.process(buf, |n| {
            warps
                .iter()
                .fold(reach, |d, &(w, a)| d - a * (w * n as f64).sin())
        });
        for (range, n) in pieces(&self.own, automation, from, len) {
            let (_, drive) = self.own.at(n, automation);
            saturate(&mut buf[range], drive);
        }
        band.clear();
        band.extend_from_slice(buf);
        self.bump.process(band);
        for (x, b) in buf.iter_mut().zip(band.iter()) {
            *x += f64::from(HEAD_BUMP_MIX) * b;
        }
        self.rolloff.process(buf);
        let rain = automation.rain();
        match self.darkens {
            true if rain.end() > from => {
                self.tilt
                    .process_lifted(buf, |i| rain::lift(rain.at(from + i)));
            }
            true => {
                let lift = rain::lift(rain.to());
                self.tilt.process_lifted(buf, |_| lift);
            }
            false => self.tilt.process(buf),
        }
        for x in buf.iter_mut() {
            *x *= gain_out;
        }
    }

    /// The pad's swell on frame `n`; 1 on every other lane.
    fn swell(&self, n: usize) -> f64 {
        self.pad.map_or(1.0, |pad| {
            f64::from(pad.swell(phase(pad.swell_hz, n, self.sr)))
        })
    }

    fn at_rest(&self) -> bool {
        self.tone.at_rest()
            && self.warp.at_rest()
            && self.bump.at_rest()
            && self.rolloff.at_rest()
            && self.tilt.at_rest()
    }
}

/// The texture's chain: its bed of hiss, room tone and the lane's crackle,
/// crossfaded at the song's ends and ducked under the night kicks, then the
/// tilt.
#[derive(Debug)]
pub(crate) struct Room {
    hiss: Hiss,
    gains: [f64; 2],
    /// The kick onsets the form strikes, in order, each with whether it
    /// sounded once the bed reached it: a kick struck while the drums are
    /// absent is silent, and a duck under it would pump the room with no beat.
    kicks: Vec<(usize, bool)>,
    /// The first of `kicks` the bed hasn't reached.
    reached: usize,
    /// The first of `kicks` whose duck still releases.
    ducking: usize,
    /// The duck's floor gain, and its release in frames.
    floor: f64,
    release: usize,
    tilt: Tilt,
    /// The song's length, and the bed's crossfade past each end, in frames.
    len: usize,
    crossfade: usize,
    sr: u32,
}

impl Room {
    fn new(len: usize, kicks: &[usize], gains: [f64; 2], sr: u32) -> Self {
        Self {
            hiss: Hiss::new(HISS_HZ, sr),
            gains,
            kicks: kicks.iter().map(|&kick| (kick, false)).collect(),
            reached: 0,
            ducking: 0,
            floor: gain_of_db(f64::from(DUCK_DB)),
            release: samples_at(DUCK_RELEASE_S, sr),
            tilt: Tilt::new(TILT_HZ, f64::from(TILT_LIFT), sr),
            len,
            crossfade: samples_at(BED_CROSSFADE_S, sr),
            sr,
        }
    }

    /// The bed's frames end here, a crossfade into the next song.
    fn end(&self) -> usize {
        self.len + self.crossfade
    }

    /// Plays the bed over `buf`, song frames from `from` on, which holds the
    /// lane's crackle, its hiss drawn from `noise` into `hissed`: ducked
    /// under the kicks in them if `drums_present`, which a step holds from
    /// one latch to the next.
    fn process(
        &mut self,
        buf: &mut [f64],
        from: usize,
        noise: &mut NoiseStream,
        drums_present: bool,
        hissed: &mut Vec<f64>,
    ) {
        while let Some(kick) = self.kicks.get_mut(self.reached)
            && kick.0 < from + buf.len()
        {
            kick.1 = drums_present;
            self.reached += 1;
        }
        let [gain_in, gain_out] = self.gains;
        let end = self.end();
        let bed = end.saturating_sub(from).min(buf.len());
        hissed.clear();
        hissed.resize(buf.len(), 0.0);
        self.hiss.fill(&mut hissed[..bed], noise);
        for (i, n) in (from..from + buf.len()).enumerate() {
            buf[i] = if n < end {
                let sum = hissed[i] * f64::from(HISS_LEVEL) + self.hum(n) + buf[i];
                sum * self.crossfade_gain(n) * gain_in * self.duck(n)
            } else {
                0.0
            };
        }
        self.tilt.process(buf);
        for x in buf.iter_mut() {
            *x *= gain_out;
        }
    }

    /// The room tone on frame `n`.
    fn hum(&self, n: usize) -> f64 {
        let (hum, swell) = (phase(HUM_HZ, n, self.sr), phase(HUM_SWELL_HZ, n, self.sr));
        f64::from(room_tone(hum, swell))
    }

    /// The bed's gain on frame `n`: an equal-power fade in from the song's
    /// start and out past its end, since two songs' hiss is uncorrelated and
    /// its powers add.
    fn crossfade_gain(&self, n: usize) -> f64 {
        let (fade, len) = (self.crossfade as f64, self.len);
        let x = if n < self.crossfade {
            n as f64 / fade
        } else if n >= len {
            1.0 - (n - len) as f64 / fade
        } else {
            return 1.0;
        };
        (x * FRAC_PI_2).sin()
    }

    /// The duck's gain on frame `n`: each sounded kick's dip releasing
    /// linearly, multiplied where releases overlap, as the whole-lane night
    /// texture ducks.
    fn duck(&mut self, n: usize) -> f64 {
        while self
            .kicks
            .get(self.ducking)
            .is_some_and(|&(kick, _)| kick + self.release <= n)
        {
            self.ducking += 1;
        }
        let (floor, release) = (self.floor, self.release as f64);
        self.kicks[self.ducking..]
            .iter()
            .take_while(|&&(kick, _)| kick <= n)
            .filter(|&&(_, sounded)| sounded)
            .map(|&(kick, _)| floor + (1.0 - floor) * (n - kick) as f64 / release)
            .product()
    }

    /// Whether the bed is past its end at song frame `at` with its tilt at
    /// rest.
    fn at_rest(&self, at: usize) -> bool {
        at >= self.end() && self.tilt.at_rest()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Controls;
    use crate::compose::{DrumKind, Hit, spans_a_form};
    use crate::dsp::{SAMPLE_RATE, band_energy_share};
    use crate::stream::BLOCK;
    use crate::stream::schedule::{Schedule, UNITY, song_form};
    use crate::synth::{WOW, part};

    fn controls(mood: Mood) -> Controls {
        let daylight = match mood {
            Mood::Day => 1.0,
            Mood::Night => 0.0,
        };
        Controls::default().with_daylight(daylight)
    }

    /// `run`'s next `frames` frames, each lane's apart.
    fn lanes(run: &mut Schedule, frames: usize) -> Vec<Vec<f32>> {
        let mut out = vec![Vec::with_capacity(frames); Lane::ALL.len()];
        let mut left = frames;
        while left > 0 {
            let n = left.min(BLOCK);
            for (lane, sum) in out.iter_mut().zip(run.play(n)) {
                lane.extend(sum.iter().map(|&v| v as f32));
            }
            left -= n;
        }
        out
    }

    #[test]
    fn a_kick_never_pre_echoes() {
        let sr = SAMPLE_RATE;
        // a quarter of the wow's cycle in, where the warp lags least, within
        // the flutter's swing of a frame: a stage sounding even one frame
        // early sounds before the kick
        let onset = samples_at(0.25 / WOW.0, sr);
        let hits = [Hit {
            at: 0.0,
            kind: DrumKind::Kick,
            gain: 1.0,
        }];
        let part = Part::Hits {
            hits: &hits,
            recipes: crate::synth::DRUMS,
            mood: Mood::Day,
        };
        let (kick, gain) = part.note(0, &mut NoiseStream::new(9), sr);
        let mut echoes = Vec::new();
        for mood in [Mood::Day, Mood::Night] {
            let mut drums = vec![0.0; onset + kick.len()];
            for (slot, &s) in drums[onset..].iter_mut().zip(&kick) {
                *slot = f64::from(s * gain);
            }
            let automation = Automation::new(controls(mood), None);
            let mut chain = Chain::new(mood, Lane::Drums, &part, drums.len(), &[], &automation, sr);
            let (mut noise, mut work) = (NoiseStream::new(0), Vec::new());
            chain.process(&mut drums, 0, &automation, &mut noise, true, &mut work);
            // from rest, a causal chain is exactly silent until its input
            let first = drums.iter().position(|&v| v != 0.0);
            if first.is_none_or(|f| f < onset) {
                echoes.push(format!("{mood:?}: first sound at {first:?}, onset {onset}"));
            }
        }
        assert!(echoes.is_empty(), "kick pre-echoes: {echoes:?}");
    }

    /// A tape as a song plays it: under the song's automation, on its clock.
    struct Playing {
        tape: Tape,
        automation: Automation,
        at: usize,
        work: Vec<f64>,
    }

    impl Playing {
        fn process(&mut self, buf: &mut [f64]) {
            (self.tape).process(buf, self.at, &self.automation, &mut self.work);
            self.at += buf.len();
        }

        /// The tone's own corner and drive on the frame it plays next.
        fn own(&self) -> (f32, f64) {
            self.tape.own.at(self.at, &self.automation)
        }
    }

    /// `lane`'s tape in a `mood` song at `daylight`, a minute long.
    fn graded_tape(mood: Mood, lane: Lane, daylight: f32) -> Playing {
        tape(mood, lane, Controls::default().with_daylight(daylight))
    }

    /// `lane`'s tape in a `mood` song under `controls`, a minute long.
    fn tape(mood: Mood, lane: Lane, controls: Controls) -> Playing {
        let score = crate::compose::compose(mood, 0);
        let len = samples_at(60.0, SAMPLE_RATE);
        let automation = Automation::new(controls, None);
        match Chain::new(
            mood,
            lane,
            &part(&score, lane),
            len,
            &[],
            &automation,
            SAMPLE_RATE,
        ) {
            Chain::Tape(tape) => Playing {
                tape,
                automation,
                at: 0,
                work: Vec::new(),
            },
            Chain::Room(_) => panic!("{lane:?} plays through a room"),
        }
    }

    #[test]
    fn rain_darkens_the_sparkle_keys_and_drums_top_and_leaves_the_rest() {
        let n = samples_at(2.0, SAMPLE_RATE);
        let mut rng = NoiseStream::new(21);
        let noise: Vec<f64> = (0..n).map(|_| f64::from(rng.norm()) * 0.3).collect();
        for lane in [
            Lane::Pad,
            Lane::Sparkle,
            Lane::Keys,
            Lane::Drums,
            Lane::Bass,
        ] {
            let [dry, wet] = [0.0, 1.0].map(|rain| {
                let mut tape = tape(Mood::Day, lane, Controls::default().with_rain(rain));
                let mut buf = noise.clone();
                tape.process(&mut buf);
                buf.iter().map(|&v| v as f32).collect::<Vec<f32>>()
            });
            if !rain::DARKENED.contains(&lane) {
                assert_eq!(dry, wet, "rain tilts {lane:?}");
                continue;
            }
            // a ratio, as the tape's own tilt row reads its lift
            let top = |b: &[f32]| {
                band_energy_share(b, 2000.0, 4000.0) / band_energy_share(b, 300.0, 500.0)
            };
            let drop = 10.0 * (top(&wet) / top(&dry)).log10();
            assert!(
                drop < -3.0,
                "rain darkens {lane:?}'s top by only {drop:.2} dB"
            );
        }
    }

    #[test]
    fn a_rain_ramp_darkens_the_tilt_a_sample_at_a_time() {
        let n = samples_at(1.0, SAMPLE_RATE);
        let mut rng = NoiseStream::new(8);
        let noise: Vec<f64> = (0..n).map(|_| f64::from(rng.norm()) * 0.3).collect();
        let latch = n / 2;
        let play = |rain: f64, ramp: bool| {
            let mut tape = tape(
                Mood::Day,
                Lane::Drums,
                Controls::default().with_rain(rain as f32),
            );
            if ramp {
                (tape.automation.rain).toward(latch, 1.0, samples_at(4.0, SAMPLE_RATE));
            }
            let mut buf = noise.clone();
            tape.process(&mut buf);
            buf
        };
        let [dry, wet, ramped] = [play(0.0, false), play(1.0, false), play(0.0, true)];
        assert_eq!(
            &ramped[..latch],
            &dry[..latch],
            "the ramp moved before its latch"
        );
        let off = |window: Range<usize>, b: &[f64]| -> f64 {
            window.map(|i| (ramped[i] - b[i]).powi(2)).sum()
        };
        // a tenth of the way in, it has left dry but sits far nearer it
        let early = latch + samples_at(0.4, SAMPLE_RATE);
        let window = early..early + samples_at(0.01, SAMPLE_RATE);
        let (from_dry, from_wet) = (off(window.clone(), &dry), off(window, &wet));
        assert!(from_dry > 0.0, "the tilt never moved off dry");
        assert!(from_dry < from_wet, "the tilt jumped toward its target");
    }

    /// The lanes whose colour grades with the daylight.
    const GRADED: [Lane; 4] = [Lane::Sparkle, Lane::Keys, Lane::Bass, Lane::Drums];

    /// `lane`'s tone corner and drive in `mood`'s recipe.
    fn recipe(mood: Mood, lane: Lane) -> (f32, f64) {
        use crate::synth::{BASS, DRUMS, KEYS, SPARKLE};
        let note = |r: crate::synth::NoteRecipe| (r.cutoff_hz, f64::from(DRIVE));
        match lane {
            Lane::Drums => {
                let r = DRUMS.at(mood);
                (r.cutoff_hz, f64::from(r.drive))
            }
            Lane::Sparkle => note(SPARKLE.at(mood)),
            Lane::Keys => note(KEYS.at(mood)),
            Lane::Bass => note(BASS.at(mood)),
            Lane::Pad | Lane::Texture => panic!("{lane:?} doesn't grade"),
        }
    }

    #[test]
    fn a_grade_is_its_moods_recipe_at_each_end_and_brighter_by_day_between() {
        for lane in GRADED {
            for (mood, daylight) in [(Mood::Night, 0.0), (Mood::Day, 1.0)] {
                let tape = graded_tape(mood, lane, daylight);
                let grade = tape.tape.grade().expect("the lane grades");
                let got = (tape.tape.tuned, tape.own().1);
                assert_eq!(tape.tape.tuned, grade.corner_hz(f64::from(daylight)));
                assert_eq!(got, recipe(mood, lane), "{lane:?} at {daylight}");
            }
            let tape = graded_tape(Mood::Day, lane, 1.0);
            let grade = tape.tape.grade().expect("the lane grades");
            let steps: Vec<(f32, f64)> = (0..=16)
                .map(|i| f64::from(i) / 16.0)
                .map(|t| (grade.corner_hz(t), grade.drive(t)))
                .collect();
            assert!(
                steps
                    .windows(2)
                    .all(|w| w[1].0 > w[0].0 && w[1].1 >= w[0].1),
                "{lane:?} grades out of order: {steps:?}"
            );
            // halfway in daylight is halfway in octaves
            let [night, day] = grade.corners_hz.map(f64::from);
            let half = f64::from(grade.corner_hz(0.5));
            assert!(
                (half - (night * day).sqrt()).abs() < 1e-3,
                "{lane:?} halfway at {half} Hz"
            );
        }
        for lane in [Lane::Pad, Lane::Texture] {
            let score = crate::compose::compose(Mood::Day, 0);
            let automation = Automation::new(Controls::default().with_daylight(0.5), None);
            let chain = Chain::new(
                Mood::Day,
                lane,
                &part(&score, lane),
                1,
                &[],
                &automation,
                SAMPLE_RATE,
            );
            assert!(!chain.grades(), "{lane:?} grades");
        }
    }

    #[test]
    fn a_grade_lands_on_its_target_at_its_end_and_tunes_to_it_exactly() {
        let mut tape = graded_tape(Mood::Day, Lane::Drums, 0.6);
        let (latch, over) = (1000, samples_at(2.0, SAMPLE_RATE));
        tape.automation.daylight.toward(latch, 1.0, over);
        let mut rng = NoiseStream::new(4);
        let mut buf: Vec<f64> = (0..latch + over + BLOCK)
            .map(|_| f64::from(rng.norm()))
            .collect();
        for chunk in buf.chunks_mut(BLOCK) {
            tape.process(chunk);
        }
        let grade = tape.tape.grade().expect("the lane grades");
        assert!(
            tape.automation.daylight(latch + over - 1) < 1.0,
            "the grade ends early"
        );
        assert_eq!(tape.automation.daylight(latch + over), 1.0);
        assert_eq!(
            tape.tape.tuned,
            grade.corner_hz(1.0),
            "no recompute at the ramp's end"
        );
        assert_eq!(tape.own().1, recipe(Mood::Day, Lane::Drums).1);
    }

    #[test]
    fn a_swept_grading_tape_plays_the_same_however_it_is_chunked() {
        let make = || {
            let mut tape = graded_tape(Mood::Day, Lane::Drums, 1.0);
            tape.automation.daylight.toward(1001, 0.0, 5003);
            // segments that start and end between coefficient blocks, one
            // across the grade's ramp
            let points = vec![(0, 0.0), (3001, 1.0), (7013, 1.0), (9029, 0.0)];
            tape.automation.sweep = Some(Sweep::new(points));
            tape
        };
        crate::stream::assert_chunk_invariant(1, make, |tape, buf| tape.process(buf));
    }

    #[test]
    fn a_grading_tape_plays_the_same_however_it_is_chunked() {
        let make = || {
            let mut tape = graded_tape(Mood::Day, Lane::Drums, 1.0);
            // through odd lengths, so the ramp starts and ends between
            // coefficient blocks
            tape.automation.daylight.toward(1001, 0.0, 5003);
            tape
        };
        crate::stream::assert_chunk_invariant(1, make, |tape, buf| tape.process(buf));
    }

    #[test]
    fn a_grading_ramp_stays_within_its_ends_high_band_energy() {
        // opinion: how far over the louder end a ramp's top may sound
        const MARGIN_DB: f64 = 1.0;
        let window = samples_at(0.01, SAMPLE_RATE);
        let (latch, over) = (window * 10 + 7, samples_at(2.0, SAMPLE_RATE));
        let mut rng = NoiseStream::new(12);
        let noise: Vec<f64> = (0..latch + over + window * 10)
            .map(|_| f64::from(rng.norm()) * 0.3)
            .collect();
        let play = |from: f32, to: Option<f32>| {
            let mut tape = graded_tape(Mood::Day, Lane::Drums, from);
            if let Some(to) = to {
                tape.automation.daylight.toward(latch, f64::from(to), over);
            }
            let mut buf = noise.clone();
            for chunk in buf.chunks_mut(BLOCK) {
                tape.process(chunk);
            }
            buf
        };
        let tops = |buf: &[f64]| -> Vec<f64> {
            (buf.windows(2)
                .map(|w| (w[1] - w[0]).powi(2))
                .collect::<Vec<_>>())
            .chunks(window)
            .map(|w| w.iter().sum())
            .collect()
        };
        for (from, to) in [(0.0, 1.0), (1.0, 0.0)] {
            let [ramped, a, b] =
                [play(from, Some(to)), play(from, None), play(to, None)].map(|b| tops(&b));
            let off: Vec<String> = (ramped.iter().zip(a.iter().zip(&b)).enumerate())
                .filter(|&(_, (&r, (&a, &b)))| 10.0 * (r / a.max(b)).log10() > MARGIN_DB)
                .map(|(i, (r, (a, b)))| format!("window {i}: {r:.3e} over {a:.3e}, {b:.3e}"))
                .collect();
            assert!(off.is_empty(), "{from} to {to}: {off:#?}");
        }
    }

    #[test]
    fn a_tape_saturates_bumps_rolls_off_and_tilts() {
        let sr = SAMPLE_RATE;
        // the drums' tone, the brightest, so the roll-off is what shapes the top
        let crate::synth::HitRecipe {
            cutoff_hz: cutoff,
            drive,
            ..
        } = crate::synth::DRUMS.day;
        let play = |input: &[f32]| -> Vec<f32> {
            let own = Tone::Fixed {
                corner_hz: cutoff,
                drive: f64::from(drive),
            };
            let automation = Automation::new(Controls::default(), None);
            let mut tape = Tape::new(own, None, false, [1.0, 1.0], &automation, sr);
            let mut buf: Vec<f64> = input.iter().map(|&v| f64::from(v)).collect();
            tape.process(&mut buf, 0, &automation, &mut Vec::new());
            buf.iter().map(|&v| v as f32).collect()
        };
        let n = samples_at(2.0, sr);
        let tau = std::f32::consts::TAU;
        let tone: Vec<f32> = (0..n)
            .map(|i| (tau * 440.0 * i as f32 / sr as f32).sin() * 0.8)
            .collect();
        let third = band_energy_share(&play(&tone), 1200.0, 1450.0);
        assert!(
            third > 0.005,
            "tanh drive must write a 3rd harmonic: {third:.4}"
        );
        let mut rng = NoiseStream::new(21);
        let noise: Vec<f32> = (0..n).map(|_| rng.norm() * 0.3).collect();
        let out = play(&noise);
        // a ratio, not a share: the tilt lifts most of the spectrum, so a
        // share's total moves too
        let lift = |hi: (f32, f32), lo: (f32, f32)| {
            let ratio =
                |b: &[f32]| band_energy_share(b, hi.0, hi.1) / band_energy_share(b, lo.0, lo.1);
            10.0 * (ratio(&out) / ratio(&noise)).log10()
        };
        let bump = lift((60.0, 120.0), (300.0, 500.0));
        assert!(
            bump > 0.6,
            "the head bump must lift 60-120 Hz: {bump:.2} dB"
        );
        let top = lift((8000.0, 12000.0), (1000.0, 2000.0));
        assert!(
            top < -18.0,
            "the top must roll off past the corner: {top:.2} dB"
        );
        let tilt = lift((2000.0, 4000.0), (300.0, 500.0));
        assert!(tilt > 1.5, "the tilt must lift the top: {tilt:.2} dB");
    }

    #[test]
    fn the_song_clock_keeps_its_precision_past_2_24_frames() {
        // 2²⁴ frames is f32's last exact integer, under three minutes at 96 kHz
        let sr = 96_000;
        let exact = |hz: f32, n: usize| {
            (TAU * f64::from(hz) * n as f64 / f64::from(sr)).rem_euclid(TAU) as f32
        };
        let room = Room::new(0, &[], [1.0, 1.0], sr);
        let pad = PadTone::of(Mood::Day);
        let own = Tone::Fixed {
            corner_hz: pad.cutoff_hz,
            drive: f64::from(DRIVE),
        };
        let automation = Automation::new(Controls::default(), None);
        let tape = Tape::new(own, Some(pad), false, [1.0, 1.0], &automation, sr);
        let mut off = Vec::new();
        for n in [(1 << 24) + 1, (1 << 24) + 3, (1 << 26) + 5, (1 << 27) + 7] {
            let hum = f64::from(room_tone(exact(HUM_HZ, n), exact(HUM_SWELL_HZ, n)));
            let swell = f64::from(pad.swell(exact(pad.swell_hz, n)));
            for (name, got, want, tolerance) in [
                ("hum", room.hum(n), hum, 1e-8),
                // a step and a half of f32 near 1, so a phase that rounds
                // the other way still passes
                ("swell", tape.swell(n), swell, 1.5 * f64::from(f32::EPSILON)),
            ] {
                if (got - want).abs() >= tolerance {
                    off.push(format!("{name} at {n}: {:e} off", (got - want).abs()));
                }
            }
        }
        assert!(off.is_empty(), "the song clock slipped: {off:#?}");
    }

    #[test]
    fn every_lane_stays_finite_for_ten_minutes() {
        let sr = SAMPLE_RATE;
        const SEED: u64 = 11;
        let mut run = Schedule::new(SEED, 0, sr);
        let mut moods = [Mood::Day, Mood::Night].into_iter().cycle();
        let mut peaks = [0.0f64; Lane::ALL.len()];
        let (mut played, mut forms) = (0, Vec::new());
        while played < samples_at(600.0, sr) {
            if run.song_left() == 0 {
                let c = controls(moods.next().unwrap());
                run.start_song_at(c, UNITY);
                forms.push(song_form(SEED, forms.len() as u64, c));
            }
            let n = run.song_left().min(BLOCK);
            for ((lane, sum), peak) in Lane::ALL.iter().zip(run.play(n)).zip(&mut peaks) {
                if let Some(v) = sum.iter().find(|v| !v.is_finite()) {
                    panic!("{lane:?} plays {v} by frame {played}");
                }
                *peak = sum.iter().fold(*peak, |p, v| p.max(v.abs()));
            }
            played += n;
        }
        // finite over songs with no form proves nothing of one
        spans_a_form(&forms).unwrap();
        for (lane, peak) in Lane::ALL.iter().zip(peaks) {
            assert!(peak > 0.0, "{lane:?} was silent: finite proves nothing");
        }
    }

    #[test]
    fn every_lane_plays_at_its_whole_lane_level() {
        // `lofi_measure --staging`'s rows: each lane's median gated loudness
        // over the whole-lane renders whose peak normalizations the staging
        // stands in for; the tolerance is opinion, and a staging off by 2×
        // misses it by 4.5 LU
        const TOLERANCE_LU: f64 = 1.5;
        let whole_lane: [(Mood, [f64; Lane::ALL.len()]); 2] = [
            (Mood::Day, [-12.89, -18.63, -16.86, -17.87, -34.91, -11.83]),
            (
                Mood::Night,
                [-11.76, -19.97, -16.99, -17.20, -36.23, -10.22],
            ),
        ];
        let sr = SAMPLE_RATE;
        let mut off = Vec::new();
        for (mood, want) in whole_lane {
            let mut lufs = vec![Vec::new(); Lane::ALL.len()];
            for seed in 0..6 {
                let mut run = Schedule::plain(seed, 0, sr);
                run.start_song_at(controls(mood), UNITY);
                let loop_len = samples_at(run.playing().expect("a song plays").loop_secs(), sr);
                // the second loop, which the first's tails ring into as every
                // later one does
                lanes(&mut run, loop_len);
                for (lufs, lane) in lufs.iter_mut().zip(lanes(&mut run, loop_len)) {
                    lufs.push(crate::audition::gated_lufs(&[&lane], sr));
                }
            }
            for ((lane, mut got), want) in Lane::ALL.iter().zip(lufs).zip(want) {
                got.sort_by(f64::total_cmp);
                let median = f64::midpoint(got[got.len() / 2 - 1], got[got.len() / 2]);
                if (median - want).abs() > TOLERANCE_LU || median.is_nan() {
                    off.push(format!("{mood:?} {lane:?}: {median:.2} vs {want}"));
                }
            }
        }
        assert!(off.is_empty(), "lanes off their whole-lane level: {off:#?}");
    }
}
