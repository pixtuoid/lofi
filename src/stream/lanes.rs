//! Each lane's chain for one song: the whole-lane recipe (`synth`'s
//! `lane_recipe`, `lofi_post`, `master` and `brighten`) over causal filters,
//! with fixed gains where a whole lane normalized to its peak. A song's chains
//! start at rest on its first frame, so how it sounds never depends on the
//! songs before it.

use std::f64::consts::{FRAC_PI_2, TAU};

use super::biquad::{Cascade, butter_lowpass, cookbook_bandpass};
use super::delay::DelayLine;
use super::onepole::Tilt;
use super::schedule::BLOCK;
use crate::compose::Mood;
use crate::dsp::NoiseStream;
use crate::synth::{
    DRIVE, DUCK_DB, DUCK_RELEASE_S, HEAD_BUMP_HZ, HEAD_BUMP_MIX, HISS_HZ, HISS_LEVEL, Lane,
    PadTone, Part, ROLLOFF_HZ, TILT_HZ, TILT_LIFT, WARPS, lane_recipe, room_tone, samples_at,
};

/// The order of each slope that stands in for a whole-lane brickwall.
const ORDER: usize = 4;

/// How long a song's texture bed crossfades into the next song's: their hums
/// meet at unrelated phases.
const BED_CROSSFADE_S: f32 = 0.03;

/// Each lane's fixed gains `[in, out]`, standing in for the whole-lane render's
/// peak normalizations: `in` for its first `normalize`, which sets the level
/// into the drive (the texture's, into the duck), and `out` for the product of
/// the rest. Each is the median of what those normalizations gave over a seed
/// sweep, which the commit that set them records.
fn staging(mood: Mood, lane: Lane) -> [f64; 2] {
    match (mood, lane) {
        (Mood::Day, Lane::Pad) => [0.1376, 0.7424],
        (Mood::Day, Lane::Sparkle) => [1.1440, 0.5299],
        (Mood::Day, Lane::Keys) => [0.5419, 0.7848],
        (Mood::Day, Lane::Drums) => [0.6785, 0.6305],
        (Mood::Day, Lane::Texture) => [3.1570, 0.5869],
        (Mood::Day, Lane::Bass) => [0.8075, 0.7210],
        (Mood::Night, Lane::Pad) => [0.2049, 0.7833],
        (Mood::Night, Lane::Sparkle) => [1.2683, 0.5361],
        (Mood::Night, Lane::Keys) => [1.0453, 0.6983],
        (Mood::Night, Lane::Drums) => [1.0628, 0.7487],
        (Mood::Night, Lane::Texture) => [3.0763, 0.6012],
        (Mood::Night, Lane::Bass) => [1.2978, 0.6178],
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
    /// its `part` as `render_lane` picks its whole-lane core; the texture
    /// ducks under `kicks`, the song's kick onsets.
    pub(crate) fn new(
        mood: Mood,
        lane: Lane,
        part: &Part<'_>,
        len: usize,
        kicks: &[usize],
        sr: u32,
    ) -> Self {
        let gains = staging(mood, lane);
        let (r0, r1, _) = lane_recipe(mood, lane);
        match part {
            Part::Chords { .. } => {
                let pad = PadTone::of(mood);
                Self::Tape(Tape::new(pad.cutoff_hz, Some(pad), DRIVE, gains, sr))
            }
            Part::Notes { .. } => Self::Tape(Tape::new(r1, None, DRIVE, gains, sr)),
            Part::Hits(_) => Self::Tape(Tape::new(r0, None, r1, gains, sr)),
            Part::Bed => Self::Room(Room::new(mood, len, kicks, gains, sr)),
        }
    }

    /// Plays the lane's next frames in place; the texture draws its hiss
    /// from `noise`.
    pub(crate) fn process(&mut self, buf: &mut [f64], noise: &mut NoiseStream) {
        match self {
            Self::Tape(tape) => tape.process(buf),
            Self::Room(room) => room.process(buf, noise),
        }
    }

    /// Whether silence in now plays exact silence out, for good.
    pub(crate) fn at_rest(&self) -> bool {
        match self {
            Self::Tape(tape) => tape.at_rest(),
            Self::Room(room) => room.at_rest(),
        }
    }
}

/// A musical lane's chain: its own low-pass, then `lofi_post`'s tape, then
/// the tilt.
#[derive(Debug)]
pub(crate) struct Tape {
    tone: Cascade,
    /// The pad's slow swell.
    pad: Option<PadTone>,
    gains: [f64; 2],
    warp: DelayLine,
    /// Wow and flutter, each `(radians per frame, read-head excursion in
    /// frames)`.
    warps: [(f64, f64); WARPS.len()],
    /// The sum of the excursions: `warp_resample` reads that far either side
    /// of now, so a causal line reads that late, and the tape lags by it.
    reach: f64,
    drive: f64,
    bump: Cascade,
    /// The bump's band over a step.
    band: Vec<f64>,
    rolloff: Cascade,
    tilt: Tilt,
    sr: u32,
    /// Frames played: the song's clock.
    frame: usize,
}

impl Tape {
    fn new(cutoff_hz: f32, pad: Option<PadTone>, drive: f32, gains: [f64; 2], sr: u32) -> Self {
        let rate = f64::from(sr);
        // the excursion `warp_resample` gives a `(hz, dev)` pair
        let warps = WARPS.map(|(hz, dev)| {
            let w = TAU * f64::from(hz);
            (w / rate, f64::from(dev) * rate / w)
        });
        let reach = warps.iter().map(|&(_, a)| a).sum();
        Self {
            tone: butter_lowpass(cutoff_hz, ORDER, sr),
            pad,
            gains,
            warp: DelayLine::new(2.0 * reach),
            warps,
            reach,
            drive: f64::from(drive),
            bump: cookbook_bandpass(HEAD_BUMP_HZ.0, HEAD_BUMP_HZ.1, sr),
            band: Vec::with_capacity(BLOCK),
            rolloff: butter_lowpass(ROLLOFF_HZ, ORDER, sr),
            tilt: Tilt::new(TILT_HZ, f64::from(TILT_LIFT), sr),
            sr,
            frame: 0,
        }
    }

    fn process(&mut self, buf: &mut [f64]) {
        let [gain_in, gain_out] = self.gains;
        self.tone.process(buf);
        for (n, x) in (self.frame..).zip(buf.iter_mut()) {
            let swell = self.pad.map_or(1.0, |pad| f64::from(pad.swell(n, self.sr)));
            *x *= swell * gain_in;
        }
        let (warps, reach) = (self.warps, self.reach);
        self.warp.process(buf, |n| {
            warps
                .iter()
                .fold(reach, |d, &(w, a)| d - a * (w * n as f64).sin())
        });
        let norm = self.drive.tanh();
        for x in buf.iter_mut() {
            *x = (*x * self.drive).tanh() / norm;
        }
        self.band.clear();
        self.band.extend_from_slice(buf);
        self.bump.process(&mut self.band);
        for (x, b) in buf.iter_mut().zip(&self.band) {
            *x += f64::from(HEAD_BUMP_MIX) * b;
        }
        self.rolloff.process(buf);
        self.tilt.process(buf);
        for x in buf.iter_mut() {
            *x *= gain_out;
        }
        self.frame += buf.len();
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
    hiss: Cascade,
    /// The hiss over a step.
    hissed: Vec<f64>,
    gains: [f64; 2],
    /// The kick onsets the bed ducks under, in order; none by day.
    kicks: Vec<usize>,
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
    /// Frames played: the song's clock.
    frame: usize,
}

impl Room {
    fn new(mood: Mood, len: usize, kicks: &[usize], gains: [f64; 2], sr: u32) -> Self {
        Self {
            hiss: butter_lowpass(HISS_HZ, ORDER, sr),
            hissed: Vec::with_capacity(BLOCK),
            gains,
            kicks: match mood {
                Mood::Day => Vec::new(),
                Mood::Night => kicks.to_vec(),
            },
            ducking: 0,
            floor: 10f64.powf(f64::from(DUCK_DB) / 20.0),
            release: samples_at(DUCK_RELEASE_S, sr),
            tilt: Tilt::new(TILT_HZ, f64::from(TILT_LIFT), sr),
            len,
            crossfade: samples_at(BED_CROSSFADE_S, sr),
            sr,
            frame: 0,
        }
    }

    /// The bed's frames end here, a crossfade into the next song.
    fn end(&self) -> usize {
        self.len + self.crossfade
    }

    /// Plays the bed over `buf`, which holds the lane's crackle, drawing one
    /// hiss sample from `noise` per frame of bed.
    fn process(&mut self, buf: &mut [f64], noise: &mut NoiseStream) {
        let [gain_in, gain_out] = self.gains;
        let end = self.end();
        let bed = end.saturating_sub(self.frame).min(buf.len());
        self.hissed.clear();
        self.hissed
            .extend((0..bed).map(|_| f64::from(noise.norm())));
        self.hissed.resize(buf.len(), 0.0);
        self.hiss.process(&mut self.hissed);
        for (i, n) in (self.frame..self.frame + buf.len()).enumerate() {
            buf[i] = if n < end {
                let room = f64::from(room_tone(n as f32 / self.sr as f32));
                let sum = self.hissed[i] * f64::from(HISS_LEVEL) + room + buf[i];
                sum * self.crossfade_gain(n) * gain_in * self.duck(n)
            } else {
                0.0
            };
        }
        self.tilt.process(buf);
        for x in buf.iter_mut() {
            *x *= gain_out;
        }
        self.frame += buf.len();
    }

    /// The bed's gain on frame `n`: an equal-power fade in from the song's
    /// start and out past its end, since two songs' beds are uncorrelated and
    /// their powers add.
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

    /// The duck's gain on frame `n`: each kick's dip releasing linearly,
    /// multiplied where releases overlap, as the whole-lane night texture ducks.
    fn duck(&mut self, n: usize) -> f64 {
        while self
            .kicks
            .get(self.ducking)
            .is_some_and(|&kick| kick + self.release <= n)
        {
            self.ducking += 1;
        }
        let (floor, release) = (self.floor, self.release as f64);
        self.kicks[self.ducking..]
            .iter()
            .take_while(|&&kick| kick <= n)
            .map(|&kick| floor + (1.0 - floor) * (n - kick) as f64 / release)
            .product()
    }

    fn at_rest(&self) -> bool {
        self.frame >= self.end() && self.hiss.at_rest() && self.tilt.at_rest()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Controls;
    use crate::compose::DrumKind;
    use crate::dsp::{SAMPLE_RATE, band_energy_share};
    use crate::stream::schedule::{LOOPS_PER_SONG, Schedule};

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
        let onset = samples_at(0.5, sr);
        let hits = [(0.0, DrumKind::Kick, 1.0)];
        let part = Part::Hits(&hits);
        let (kick, gain) = part.note(0, &mut NoiseStream::new(9), sr);
        let mut echoes = Vec::new();
        for mood in [Mood::Day, Mood::Night] {
            let mut drums = vec![0.0; 2 * onset];
            for (slot, &s) in drums[onset..].iter_mut().zip(&kick) {
                *slot = f64::from(s * gain);
            }
            let mut chain = Chain::new(mood, Lane::Drums, &part, drums.len(), &[], sr);
            chain.process(&mut drums, &mut NoiseStream::new(0));
            let at = |s: f32| (onset as f32 + s * sr as f32) as usize;
            let energy = |from_s, to_s| {
                drums[at(from_s)..at(to_s)]
                    .iter()
                    .map(|v| v * v)
                    .sum::<f64>()
            };
            let pre_db = 10.0 * (energy(-0.040, -0.005) / energy(0.0, 0.035)).log10();
            if pre_db.is_nan() || pre_db > -60.0 {
                echoes.push(format!("{mood:?}: {pre_db:.1} dB"));
            }
        }
        assert!(echoes.is_empty(), "kick pre-echoes: {echoes:?}");
    }

    #[test]
    fn a_tape_saturates_bumps_rolls_off_and_tilts() {
        let sr = SAMPLE_RATE;
        // the drums' tone, the brightest, so the roll-off is what shapes the top
        let (cutoff, drive, _) = lane_recipe(Mood::Day, Lane::Drums);
        let play = |input: &[f32]| -> Vec<f32> {
            let mut tape = Tape::new(cutoff, None, drive, [1.0, 1.0], sr);
            let mut buf: Vec<f64> = input.iter().map(|&v| f64::from(v)).collect();
            tape.process(&mut buf);
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
    fn every_lane_stays_finite_for_ten_minutes() {
        let sr = SAMPLE_RATE;
        let mut run = Schedule::new(11, 0, sr);
        let mut moods = [Mood::Day, Mood::Night].into_iter().cycle();
        let mut peaks = [0.0f64; Lane::ALL.len()];
        let mut played = 0;
        while played < samples_at(600.0, sr) {
            if run.song_left() == 0 {
                run.start_song(controls(moods.next().unwrap()));
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
        for (lane, peak) in Lane::ALL.iter().zip(peaks) {
            assert!(peak > 0.0, "{lane:?} was silent: finite proves nothing");
        }
    }

    #[test]
    fn every_lane_plays_at_its_whole_lane_level() {
        // each lane's median gated loudness over seeds 0..24 of `render_lane`,
        // whose peak normalizations the staging stands in for; the tolerance
        // is opinion, and a staging off by 2× misses it by 4.5 LU
        const TOLERANCE_LU: f64 = 1.5;
        let whole_lane: [(Mood, [f64; Lane::ALL.len()]); 2] = [
            (Mood::Day, [-12.92, -18.55, -16.84, -17.98, -35.41, -12.75]),
            (
                Mood::Night,
                [-11.75, -19.75, -16.84, -16.75, -36.06, -12.44],
            ),
        ];
        let sr = SAMPLE_RATE;
        let mut off = Vec::new();
        for (mood, want) in whole_lane {
            let mut lufs = vec![Vec::new(); Lane::ALL.len()];
            for seed in 0..6 {
                let mut run = Schedule::new(seed, 0, sr);
                run.start_song(controls(mood));
                let loop_len = run.song_left() / LOOPS_PER_SONG;
                // the second loop, which the first's tails ring into as every
                // later one does
                lanes(&mut run, loop_len);
                for (lufs, lane) in lufs.iter_mut().zip(lanes(&mut run, loop_len)) {
                    lufs.push(crate::loudness::gated_lufs(&lane, sr));
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
