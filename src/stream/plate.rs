//! The plate: the reverb across the melodic lanes, Dattorro's plate-class
//! reverberator ("Effect Design Part 1", JAES 45(9), 1997, Fig. 1,
//! <https://ccrma.stanford.edu/~dattorro/EffectDesignPart1.pdf>). Each song
//! owns one, built empty, which rings out with the song's tails.

use std::f64::consts::TAU;

use super::biquad::{Cascade, butter_highpass};
use super::delay::DelayLine;
use super::flush;
use super::onepole::OnePole;
use crate::compose::Mood;
use crate::synth::{Lane, samples_at};

/// How far under its dry lane each wet lane's return reads on the fold,
/// `[pad, keys, sparkle]`, by family. Opinion: the pad wettest, then the
/// keys, then the lead, which stays nearest; the day 3 LU drier than the
/// night, since #23's judge heard the day's faster, brighter songs smear
/// (MuQ-Eval's loss there alone).
const NIGHT_RETURN_LU: [f64; 3] = [12.0, 14.0, 16.0];
const DAY_RETURN_LU: [f64; 3] = [15.0, 17.0, 19.0];

/// A `mood` song's `lane` send into the plate, as a gain on its mix level;
/// none for a lane lofi keeps dry: the drums and bass ground the centre, and
/// the texture is a bed already.
pub(crate) fn send(mood: Mood, lane: Lane) -> Option<f64> {
    // each wet lane's return at a 0 dB send, in LU on its dry lane: the
    // median over seeds the commit that set it measured
    let (k, unity) = match (mood, lane) {
        (Mood::Day, Lane::Pad) => (0, -6.08),
        (Mood::Day, Lane::Keys) => (1, -7.24),
        (Mood::Day, Lane::Sparkle) => (2, -3.02),
        (Mood::Night, Lane::Pad) => (0, -5.16),
        (Mood::Night, Lane::Keys) => (1, -6.85),
        (Mood::Night, Lane::Sparkle) => (2, -3.39),
        (_, Lane::Drums | Lane::Bass | Lane::Texture) => return None,
    };
    let under = match mood {
        Mood::Day => DAY_RETURN_LU,
        Mood::Night => NIGHT_RETURN_LU,
    }[k];
    Some(10f64.powf((-under - unity) / 20.0))
}

/// The rate Table 1's lengths and Table 2's taps are counted at: "Sample
/// rate F s = 29761 Hz". Every length and tap scales from it to the song's
/// rate, as the paper's ports do.
const DATTORRO_HZ: f64 = 29_761.0;

/// How long the plate takes to fall 60 dB, from which its decay comes.
/// Opinion: inside the 1.2–2.0 s a plate on lofi's keys takes, and short
/// enough that a song's tail rests within the next song's first loop.
pub(crate) const PLATE_T60_S: f64 = 1.5;

/// The send's delay into the plate. Opinion: the short end of a lofi
/// plate's 20–40 ms, so the reverb trails the note without parting from it.
pub(crate) const PLATE_PREDELAY_S: f32 = 0.02;

/// Table 1: "decay diffusion 2 = decay + 0.15, floor = 0.25, ceiling = 0.50".
const DECAY_DIFFUSION_2_OVER_DECAY: f64 = 0.15;
const DECAY_DIFFUSION_2_RANGE: (f64, f64) = (0.25, 0.5);

/// Where the tank's damping one-pole sits: Dattorro's `damping` of about
/// 0.5 at 44.1 kHz, so the tail darkens as it rings. Opinion.
const PLATE_DAMPING_HZ: f32 = 5000.0;

/// The return's high-pass: "Any reverb return HPF ≥300 Hz" (LOFI-BIBLE.md,
/// Sparkle), so the plate leaves the low end to the bass and pad.
const PLATE_HPF_HZ: f32 = 300.0;
/// Its order. Opinion: a gentle 12 dB an octave.
const PLATE_HPF_ORDER: usize = 2;

/// The tank's modulation: "a rate on the order of 1 Hz, and … a peak
/// excursion of about 8 samples for a sample rate of about 29.8 kHz" (the
/// paper's footnote 14), about 3 cents.
const LFO_HZ: f64 = 1.0;
const EXCURSION: f64 = 8.0;

/// Table 1's diffusions: "input diffusion 1 = 0.750", "input diffusion 2 =
/// 0.625" and "decay diffusion 1 = 0.70".
const INPUT_DIFFUSION: [f64; 2] = [0.75, 0.625];
const DECAY_DIFFUSION_1: f64 = 0.70;

/// Fig. 1's four input lattices' lengths, at [`DATTORRO_HZ`].
const INPUT_DIFFUSERS: [f64; 4] = [142.0, 107.0, 379.0, 277.0];

/// Fig. 1's tank halves' lengths at [`DATTORRO_HZ`], left then right: the
/// modulated decay diffuser, the long delay, the second decay diffuser,
/// the second delay.
const HALVES: [[f64; 4]; 2] = [
    [672.0, 4453.0, 1800.0, 3720.0],
    [908.0, 4217.0, 2656.0, 3163.0],
];

/// Each half multiplies by `decay` twice, so one trip round the tank's
/// figure eight takes it four times.
const DECAYS_PER_TRIP: f64 = 4.0;

/// The level a T60 falls by, in dB.
const T60_DB: f64 = 60.0;

/// The tank's lines that Table 2 taps, by their nodes in Fig. 1.
#[derive(Clone, Copy, Debug, PartialEq)]
enum TankLine {
    /// 24_30, 31_33 and 33_39.
    LeftDelay1,
    LeftDiffuser2,
    LeftDelay2,
    /// 48_54, 55_59 and 59_63.
    RightDelay1,
    RightDiffuser2,
    RightDelay2,
}

/// Each output's taps, `(line, index at DATTORRO_HZ, sign)`, as Table 2
/// spells them, each at [`TAP_GAIN`].
const TAPS: [[(TankLine, u32, f64); 7]; 2] = {
    use TankLine::*;
    [
        [
            (RightDelay1, 266, 1.0),
            (RightDelay1, 2974, 1.0),
            (RightDiffuser2, 1913, -1.0),
            (RightDelay2, 1996, 1.0),
            (LeftDelay1, 1990, -1.0),
            (LeftDiffuser2, 187, -1.0),
            (LeftDelay2, 1066, -1.0),
        ],
        [
            (LeftDelay1, 353, 1.0),
            (LeftDelay1, 3627, 1.0),
            (LeftDiffuser2, 1228, -1.0),
            (LeftDelay2, 2673, 1.0),
            (RightDelay1, 2111, -1.0),
            (RightDiffuser2, 335, -1.0),
            (RightDelay2, 121, -1.0),
        ],
    ]
};
const TAP_GAIN: f64 = 0.6;

/// `at`, a count at [`DATTORRO_HZ`], at `sr`.
fn scaled(at: f64, sr: u32) -> usize {
    (at * f64::from(sr) / DATTORRO_HZ).round() as usize
}

/// A lattice all-pass, `w = x + g·d`, `y = d − g·w`, `d` being `w` delayed:
/// flat in magnitude, so it diffuses without colouring.
#[derive(Debug)]
struct Allpass {
    line: DelayLine,
    delay: usize,
}

impl Allpass {
    /// A lattice `delay` frames long, which may stretch by up to `reach`.
    fn new(delay: usize, reach: f64) -> Self {
        Self {
            line: DelayLine::new(delay as f64 + reach),
            delay,
        }
    }

    /// The next output, `x` in, at coefficient `g`, its line `d` delayed;
    /// `kick` adds to what the line stores.
    fn step_from(&mut self, x: f64, g: f64, d: f64, kick: f64) -> f64 {
        let w = flush(x + g * d) + kick;
        self.line.push(w);
        d - g * w
    }

    /// [`step_from`](Self::step_from) at the lattice's own length.
    fn step(&mut self, x: f64, g: f64, kick: f64) -> f64 {
        let d = self.line.at(self.delay - 1);
        self.step_from(x, g, d, kick)
    }

    /// [`step_from`](Self::step_from) `stretch` frames longer, between frames.
    fn step_stretched(&mut self, x: f64, g: f64, stretch: f64) -> f64 {
        let d = self.line.tap(self.delay as f64 + stretch - 1.0);
        self.step_from(x, g, d, 0.0)
    }
}

/// One half of the tank's figure eight.
#[derive(Debug)]
struct Half {
    diffuser_1: Allpass,
    delay_1: DelayLine,
    delay_1_len: usize,
    damping: OnePole,
    diffuser_2: Allpass,
    delay_2: DelayLine,
    delay_2_len: usize,
}

impl Half {
    fn new([diffuser_1, delay_1, diffuser_2, delay_2]: [f64; 4], excursion: f64, sr: u32) -> Self {
        let [delay_1, delay_2] = [delay_1, delay_2].map(|len| scaled(len, sr));
        Self {
            diffuser_1: Allpass::new(scaled(diffuser_1, sr), excursion),
            delay_1: DelayLine::new(delay_1 as f64),
            delay_1_len: delay_1,
            damping: OnePole::new(PLATE_DAMPING_HZ, sr),
            diffuser_2: Allpass::new(scaled(diffuser_2, sr), 0.0),
            delay_2: DelayLine::new(delay_2 as f64),
            delay_2_len: delay_2,
        }
    }

    /// The sample leaving the half on this frame, before its pushes.
    fn leaving(&self) -> f64 {
        self.delay_2.at(self.delay_2_len - 1)
    }

    fn at_rest(&self) -> bool {
        self.diffuser_1.line.at_rest()
            && self.delay_1.at_rest()
            && self.damping.at_rest()
            && self.diffuser_2.line.at_rest()
            && self.delay_2.at_rest()
    }
}

/// A song's plate: a mono send in, a stereo return out.
#[derive(Debug)]
pub(crate) struct Plate {
    predelay: DelayLine,
    predelay_len: usize,
    diffusers: [Allpass; 4],
    halves: [Half; 2],
    decay: f64,
    decay_diffusion_2: f64,
    excursion: f64,
    return_hpf: [Cascade; 2],
    /// Table 2's taps, each `(line, frames back, sign)` at the song's rate.
    taps: [[(TankLine, usize, f64); 7]; 2],
    sr: u32,
    /// Frames played: the LFO's clock, which starts with the song.
    frame: u64,
    /// A test's unit impulse into one tank line on frame 0.
    #[cfg(test)]
    impulse_into: Option<TankLine>,
}

impl Plate {
    pub(crate) fn new(sr: u32) -> Self {
        let excursion = EXCURSION * f64::from(sr) / DATTORRO_HZ;
        let halves = HALVES.map(|lens| Half::new(lens, excursion, sr));
        // one trip round the tank, at the song's rate
        let trip = HALVES
            .iter()
            .flatten()
            .map(|&len| scaled(len, sr))
            .sum::<usize>();
        let trip_s = trip as f64 / f64::from(sr);
        let decay = 10f64.powf(-T60_DB / 20.0 * trip_s / (DECAYS_PER_TRIP * PLATE_T60_S));
        let predelay_len = samples_at(PLATE_PREDELAY_S, sr);
        Self {
            predelay: DelayLine::new(predelay_len as f64),
            predelay_len,
            diffusers: INPUT_DIFFUSERS.map(|len| Allpass::new(scaled(len, sr), 0.0)),
            halves,
            decay,
            decay_diffusion_2: (decay + DECAY_DIFFUSION_2_OVER_DECAY)
                .clamp(DECAY_DIFFUSION_2_RANGE.0, DECAY_DIFFUSION_2_RANGE.1),
            excursion,
            return_hpf: [(); 2].map(|_| butter_highpass(PLATE_HPF_HZ, PLATE_HPF_ORDER, sr)),
            taps: TAPS
                .map(|side| side.map(|(line, at, sign)| (line, scaled(f64::from(at), sr), sign))),
            sr,
            frame: 0,
            #[cfg(test)]
            impulse_into: None,
        }
    }

    /// One frame's return, before the high-pass, `x` sent in.
    fn step(&mut self, x: f64) -> [f64; 2] {
        #[cfg(test)]
        let kick = |line: TankLine| -> f64 {
            f64::from(u8::from(self.frame == 0 && self.impulse_into == Some(line)))
        };
        #[cfg(not(test))]
        let kick = |_: TankLine| 0.0;
        let kicks = [
            [
                TankLine::LeftDelay1,
                TankLine::LeftDiffuser2,
                TankLine::LeftDelay2,
            ]
            .map(kick),
            [
                TankLine::RightDelay1,
                TankLine::RightDiffuser2,
                TankLine::RightDelay2,
            ]
            .map(kick),
        ];
        self.predelay.push(x);
        let mut x = self.predelay.at(self.predelay_len);
        for (i, diffuser) in self.diffusers.iter_mut().enumerate() {
            x = diffuser.step(x, INPUT_DIFFUSION[i / 2], 0.0);
        }
        let leaving = self.halves.each_ref().map(Half::leaving);
        // a quadrature pair, "to decrease the correlation" between the halves
        let turn = TAU * (LFO_HZ * self.frame as f64 / f64::from(self.sr)).fract();
        let lfo = [turn.sin(), turn.cos()];
        let (decay, dd2) = (self.decay, self.decay_diffusion_2);
        for (h, half) in self.halves.iter_mut().enumerate() {
            let fed = x + decay * leaving[1 - h];
            // "note sign": decay diffusion 1's lattice is negated
            let a =
                (half.diffuser_1).step_stretched(fed, -DECAY_DIFFUSION_1, self.excursion * lfo[h]);
            half.delay_1.push(flush(a) + kicks[h][0]);
            let b = half.delay_1.at(half.delay_1_len);
            let c = half.damping.lowpass(b) * decay;
            let d = half.diffuser_2.step(c, dd2, kicks[h][1]);
            half.delay_2.push(flush(d) + kicks[h][2]);
        }
        self.frame += 1;
        self.taps.map(|taps| {
            let sum: f64 = (taps.iter())
                .map(|&(line, back, sign)| sign * self.line(line).at(back))
                .sum();
            TAP_GAIN * sum
        })
    }

    /// The tank line Table 2 names.
    fn line(&self, line: TankLine) -> &DelayLine {
        let [left, right] = &self.halves;
        match line {
            TankLine::LeftDelay1 => &left.delay_1,
            TankLine::LeftDiffuser2 => &left.diffuser_2.line,
            TankLine::LeftDelay2 => &left.delay_2,
            TankLine::RightDelay1 => &right.delay_1,
            TankLine::RightDiffuser2 => &right.diffuser_2.line,
            TankLine::RightDelay2 => &right.delay_2,
        }
    }

    /// The return to `send`, high-passed, into `out`'s two sides.
    pub(crate) fn process(&mut self, send: &[f64], out: &mut [Vec<f64>; 2]) {
        for side in out.iter_mut() {
            side.clear();
        }
        for &x in send {
            let [l, r] = self.step(x);
            out[0].push(l);
            out[1].push(r);
        }
        for (side, hpf) in out.iter_mut().zip(&mut self.return_hpf) {
            hpf.process(side);
        }
    }

    /// Whether every line and filter holds only zeros, so silence in plays
    /// silence out.
    pub(crate) fn at_rest(&self) -> bool {
        self.predelay.at_rest()
            && self.diffusers.iter().all(|d| d.line.at_rest())
            && self.halves.iter().all(Half::at_rest)
            && self.return_hpf.iter().all(Cascade::at_rest)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compose::{BEATS_PER_BAR, DAY_BPM, GEN_LOOP_BARS};
    use crate::dsp::{NoiseStream, SAMPLE_RATE};
    use crate::stream::biquad::{butter_highpass, butter_lowpass};

    #[test]
    fn a_day_song_sends_drier_than_a_night_song_on_every_wet_lane() {
        for lane in Lane::ALL {
            match [Mood::Day, Mood::Night].map(|mood| send(mood, lane)) {
                [Some(day), Some(night)] => {
                    assert!(day < night, "{lane:?}: {day} by day, {night} by night")
                }
                [None, None] => {}
                sends => panic!("{lane:?} is wet in one family only: {sends:?}"),
            }
        }
    }

    /// `frames` of the plate's return, `send` in from frame 0 then silence.
    fn returned(plate: &mut Plate, send: &[f64], frames: usize) -> [Vec<f64>; 2] {
        let mut input = send.to_vec();
        input.resize(frames, 0.0);
        let mut out = [Vec::new(), Vec::new()];
        plate.process(&input, &mut out);
        out
    }

    #[test]
    fn each_output_tap_reads_its_line_at_table_2s_index_and_sign() {
        use TankLine::*;
        // Table 2 as the paper prints it, spelled apart from TAPS, so a
        // slip in either shows: `(side, line, index, sign)`
        let table_2 = [
            (0, RightDelay1, 266, 1.0),
            (0, RightDelay1, 2974, 1.0),
            (0, RightDiffuser2, 1913, -1.0),
            (0, RightDelay2, 1996, 1.0),
            (0, LeftDelay1, 1990, -1.0),
            (0, LeftDiffuser2, 187, -1.0),
            (0, LeftDelay2, 1066, -1.0),
            (1, LeftDelay1, 353, 1.0),
            (1, LeftDelay1, 3627, 1.0),
            (1, LeftDiffuser2, 1228, -1.0),
            (1, LeftDelay2, 2673, 1.0),
            (1, RightDelay1, 2111, -1.0),
            (1, RightDiffuser2, 335, -1.0),
            (1, RightDelay2, 121, -1.0),
        ];
        assert_eq!(
            TAPS.iter().flatten().count(),
            table_2.len(),
            "a tap Table 2 lacks"
        );
        let sr = SAMPLE_RATE;
        for (side, line, index, sign) in table_2 {
            let mut plate = Plate::new(sr);
            plate.impulse_into = Some(line);
            let at = scaled(f64::from(index), sr);
            let raw: Vec<[f64; 2]> = (0..=at).map(|_| plate.step(0.0)).collect();
            assert_eq!(
                raw[at][side],
                sign * TAP_GAIN,
                "side {side}: {line:?}[{index}] at frame {at}"
            );
        }
    }

    /// The time a band of `ir` takes to fall 60 dB, from its Schroeder
    /// integral's −5 to −25 dB, tripled.
    fn t60(ir: &[f64], sr: u32) -> f64 {
        let edc: Vec<f64> = (ir.iter().rev())
            .scan(0.0, |sum, x| {
                *sum += x * x;
                Some(*sum)
            })
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        let db = |n: usize| 10.0 * (edc[n] / edc[0]).log10();
        let at = |level: f64| {
            (0..edc.len())
                .find(|&n| db(n) <= level)
                .expect("the tail falls that far")
        };
        3.0 * (at(-25.0) - at(-5.0)) as f64 / f64::from(sr)
    }

    #[test]
    fn the_plate_rings_for_its_t60_in_the_band_neither_filter_shapes() {
        let sr = SAMPLE_RATE;
        let mut plate = Plate::new(sr);
        let [left, right] = returned(
            &mut plate,
            &[1.0],
            samples_at((4.0 * PLATE_T60_S) as f32, sr),
        );
        // the fold, above the return's high-pass, below where the tank's
        // damping acts
        let mut band: Vec<f64> = left
            .iter()
            .zip(&right)
            .map(|(l, r)| f64::midpoint(*l, *r))
            .collect();
        butter_highpass(600.0, 4, sr).process(&mut band);
        butter_lowpass(1500.0, 4, sr).process(&mut band);
        // the decay's relation assumes the loop's mean delay, while the
        // tank's lattices spread the modes' delay, so the late modes ring a
        // little longer (our inference); 10 % still catches a dropped
        // `decay` multiply, which about doubles T60
        const TOLERANCE: f64 = 0.1;
        let got = t60(&band, sr);
        assert!(
            (got / PLATE_T60_S - 1.0).abs() < TOLERANCE,
            "the plate rings {got:.3} s, not {PLATE_T60_S} s"
        );
    }

    #[test]
    fn the_plate_plays_the_same_however_it_is_chunked() {
        let mut out = [Vec::new(), Vec::new()];
        crate::stream::assert_chunk_invariant(
            rest_bound_secs().ceil() as u32,
            || Plate::new(SAMPLE_RATE),
            |plate, buf| {
                plate.process(buf, &mut out);
                for (x, (l, r)) in buf.iter_mut().zip(out[0].iter().zip(&out[1])) {
                    *x = l - 0.5 * r;
                }
            },
        );
    }

    /// The longest a plate may ring past its last send: the fastest day
    /// song's loop, which the next song's first loop must outlast.
    fn rest_bound_secs() -> f64 {
        f64::from(GEN_LOOP_BARS as f32 * BEATS_PER_BAR * 60.0 / DAY_BPM.1)
    }

    #[test]
    fn a_plate_rests_exactly_within_a_loop_of_its_last_send() {
        let sr = SAMPLE_RATE;
        let mut plate = Plate::new(sr);
        let burst = crate::stream::burst_then_silence(0);
        let mut out = [Vec::new(), Vec::new()];
        plate.process(&burst, &mut out);
        assert!(!plate.at_rest());
        let step = sr as usize / 10;
        let mut rang = 0;
        while !plate.at_rest() {
            plate.process(&vec![0.0; step], &mut out);
            rang += step;
            assert!(
                rang as f64 / f64::from(sr) <= rest_bound_secs(),
                "the plate still rings {rang} frames on"
            );
        }
        assert!(
            out.iter().flatten().all(|&x| x == 0.0),
            "a plate at rest still plays"
        );
    }

    #[test]
    fn the_returns_two_sides_correlate_little_for_noise_in() {
        let sr = SAMPLE_RATE;
        let mut rng = NoiseStream::new(3);
        let noise: Vec<f64> = (0..samples_at(8.0, sr))
            .map(|_| f64::from(rng.norm()))
            .collect();
        let [l, r] = returned(&mut Plate::new(sr), &noise, noise.len());
        // past the tank's fill
        let skip = samples_at(2.0, sr);
        let dot = |a: &[f64], b: &[f64]| -> f64 {
            a[skip..].iter().zip(&b[skip..]).map(|(x, y)| x * y).sum()
        };
        let rho = dot(&l, &r) / (dot(&l, &l) * dot(&r, &r)).sqrt();
        assert!(
            (-0.1..=0.5).contains(&rho),
            "the return's sides correlate at {rho:.3}"
        );
    }

    #[test]
    fn nothing_returns_before_a_send_and_its_pre_delay() {
        let sr = SAMPLE_RATE;
        let at = 1000;
        let mut send = vec![0.0; at];
        send.push(1.0);
        let out = returned(&mut Plate::new(sr), &send, at + samples_at(1.0, sr));
        let first = out
            .iter()
            .map(|side| side.iter().position(|&x| x != 0.0))
            .min()
            .flatten();
        let earliest = at + samples_at(PLATE_PREDELAY_S, sr);
        assert!(
            first.is_some_and(|f| f >= earliest),
            "the plate answered at {first:?}, before {earliest}"
        );
    }
}
