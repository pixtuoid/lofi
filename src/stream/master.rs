//! The master bus: each lane at its gain for how busy the room is, the mix's
//! level, and the ceiling it plays under.

use std::collections::VecDeque;

use crate::synth::Lane;

/// The intensities at which the mix plays pixtuoid's Empty, Moderate and Busy
/// offices.
pub const TIER_INTENSITIES: [f32; 3] = [0.0, 0.5, 1.0];

/// Each lane's gain at each of [`TIER_INTENSITIES`]: pixtuoid's `TierGain`s,
/// moved here from
/// <https://github.com/pixtuoid/pixtuoid/blob/a63f1782f477c357666e71a9c1147090b47fe17d/crates/pixtuoid-scene/src/audio/mod.rs#L98-L107>.
fn tier_gains(lane: Lane) -> [f32; 3] {
    match lane {
        Lane::Pad => [0.75, 0.70, 0.65],
        Lane::Sparkle => [0.70, 0.0, 0.0],
        Lane::Keys => [0.0, 0.60, 0.70],
        Lane::Drums => [0.0, 0.35, 0.60],
        // far past the Phase-0 ratification: at its ratified level the hiss
        // and crackle were inaudible
        Lane::Texture => [0.78, 0.84, 0.78],
        // never 0, and rising where the pad's falls: the lane must hold up
        // once the drums, the keys and pixtuoid's typing enter
        Lane::Bass => [0.60, 0.70, 0.75],
    }
}

/// `lane`'s gain at `intensity`, in `[0, 1]`: its tier's at each of
/// [`TIER_INTENSITIES`], linear between them, which is opinion.
pub fn lane_gain(lane: Lane, intensity: f32) -> f32 {
    let gains = tier_gains(lane);
    let k = usize::from(intensity >= TIER_INTENSITIES[1]);
    let (lo, hi) = (TIER_INTENSITIES[k], TIER_INTENSITIES[k + 1]);
    let t = (intensity - lo) / (hi - lo);
    // exact at both ends, so each anchor plays its tier's gains
    gains[k] * (1.0 - t) + gains[k + 1] * t
}

/// `lane`'s level at `intensity` where it plays no note: its gain at the
/// nearest of [`TIER_INTENSITIES`] at which it plays, so the tails it rings
/// out, and the entry it next makes, sound at a level it plays at.
pub(crate) fn resting_gain(lane: Lane, intensity: f32) -> f32 {
    let gain = lane_gain(lane, intensity);
    if gain > 0.0 {
        return gain;
    }
    (TIER_INTENSITIES.iter().zip(tier_gains(lane)))
        .filter(|&(_, gain)| gain > 0.0)
        .min_by(|a, b| (a.0 - intensity).abs().total_cmp(&(b.0 - intensity).abs()))
        .map_or(gain, |(_, gain)| gain)
}

/// The mix's gain into the ceiling: lofi's mastering level for a background
/// bed, the level pixtuoid's office has played the lanes at
/// (<https://github.com/pixtuoid/pixtuoid/blob/a63f1782f477c357666e71a9c1147090b47fe17d/crates/pixtuoid-scene/src/audio/mixer.rs#L63-L65>),
/// which leaves the ceiling a rare backstop. How a caller plays it:
/// [`render`](crate::Performance::render).
pub const MASTER_GAIN: f64 = 0.35;

/// The peak no sample of [`render`](crate::Performance::render) passes, after
/// EBU R 128 (<https://tech.ebu.ch/docs/r/r128.pdf>), which recommends "that
/// the True Peak Level of a programme shall not exceed −1 dBTP". Held on
/// sample peaks: a true peak needs oversampling, so a peak between two samples
/// can still pass it.
const CEILING_DBFS: f32 = -1.0;

/// How far ahead the limiter looks: how late the mix plays, and how long the
/// gain takes to fall to a peak's need. Opinion: [Luff]'s example.
///
/// [Luff]: https://signalsmith-audio.co.uk/writing/2022/limiter/
pub(crate) const LOOKAHEAD_S: f64 = 0.005;

/// How much longer than the lookahead a peak holds the gain down, which keeps
/// it from fluttering between peaks close together. Opinion: [Luff]'s
/// example.
///
/// [Luff]: https://signalsmith-audio.co.uk/writing/2022/limiter/
const HOLD_S: f64 = 0.015;

/// How long the gain takes to double once a peak's hold ends: an exponential
/// rise, "thus resulting in a linear hearing sensation"
/// (<https://users.iem.at/zmoelnig/publications/limiter>). Unlike a one-pole's
/// approach it lands on its target exactly, so the gain back at 1 keeps no
/// trace of the peak. Opinion.
const RELEASE_S: f64 = 0.1;

/// The limiter averages its gains as integer steps of 2^−`GAIN_BITS`, whose
/// running sum keeps no rounding from frames long gone; 40 bits keep the sum
/// over up to 2¹³ frames exact in an `f64`.
const GAIN_BITS: i32 = 40;

/// The master ceiling: a feedforward limiter with a lookahead, then a hard
/// clamp. Each frame's needed gain is held as a moving minimum over the
/// lookahead and the hold, released, then averaged over the lookahead, and the
/// mix played that late: every gain the average takes was held at or under
/// the need of the frame it scales, so the limiter alone holds the ceiling
/// ([Luff]). Each stage is per frame, so no chunking moves it.
///
/// [Luff]: https://signalsmith-audio.co.uk/writing/2022/limiter/
pub(crate) struct Ceiling {
    /// The ceiling's amplitude: in `f32`, the sample type, so a sample limited
    /// to it in `f64` never rounds past it.
    limit: f32,
    /// The frames the moving minimum spans: the lookahead's and the hold's.
    window: u64,
    /// The released gain's factor of rise per frame.
    rise: f64,
    /// The mix's last frames, as many as the lookahead.
    late: VecDeque<f64>,
    /// `(frame, need)` for each frame in the window whose need no later frame's
    /// undercuts: increasing in both, so the first is the window's minimum.
    needs: VecDeque<(u64, f64)>,
    /// Frames seen.
    frame: u64,
    /// The held gain, released.
    released: f64,
    /// The released gains over the lookahead and one frame, in steps, and
    /// their sum.
    steps: VecDeque<u64>,
    sum: u64,
}

impl Ceiling {
    pub(crate) fn new(sr: u32) -> Self {
        let rate = f64::from(sr);
        let lookahead = (LOOKAHEAD_S * rate) as usize;
        let window = lookahead + (HOLD_S * rate) as usize + 1;
        let unit = 1u64 << GAIN_BITS;
        // each with room for one more than it keeps: a frame pushes before it
        // pops
        let mut late = VecDeque::with_capacity(lookahead + 1);
        late.resize(lookahead, 0.0);
        let mut steps = VecDeque::with_capacity(lookahead + 2);
        steps.resize(lookahead + 1, unit);
        Self {
            limit: 10f32.powf(CEILING_DBFS / 20.0),
            window: window as u64,
            rise: 2f64.powf(1.0 / (RELEASE_S * rate)),
            late,
            needs: VecDeque::with_capacity(window + 1),
            frame: 0,
            released: 1.0,
            steps,
            sum: unit * (lookahead + 1) as u64,
        }
    }

    /// Limits `buf` in place, as late as the lookahead.
    pub(crate) fn limit(&mut self, buf: &mut [f64]) {
        let limit = f64::from(self.limit);
        let unit = (1u64 << GAIN_BITS) as f64;
        for x in buf {
            let need = if x.abs() > limit {
                limit / x.abs()
            } else {
                1.0
            };
            while self.needs.back().is_some_and(|&(_, n)| n >= need) {
                self.needs.pop_back();
            }
            self.needs.push_back((self.frame, need));
            while self
                .needs
                .front()
                .is_some_and(|&(at, _)| at + self.window <= self.frame)
            {
                self.needs.pop_front();
            }
            let held = self.needs.front().map_or(1.0, |&(_, n)| n);
            self.released = held.min(self.released * self.rise);
            // rounded down, so the average stays at or under every need
            let step = (self.released * unit) as u64;
            self.steps.push_back(step);
            self.sum += step;
            self.sum -= self.steps.pop_front().unwrap_or(0);
            self.late.push_back(*x);
            *x = self.late.pop_front().unwrap_or(0.0) * self.gain();
            self.frame += 1;
        }
    }

    /// The gain the last frame limited played at: the average of the steps.
    pub(crate) fn gain(&self) -> f64 {
        self.sum as f64 / (self.steps.len() as f64 * (1u64 << GAIN_BITS) as f64)
    }

    /// `x` as a sample, clamped to the ceiling.
    pub(crate) fn clamp(&self, x: f64) -> f32 {
        (x as f32).clamp(-self.limit, self.limit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::{NoiseStream, SAMPLE_RATE};
    use crate::stream::assert_chunk_invariant;

    /// `amplitude` cos at `hz` for `secs`, from its crest.
    fn tone(amplitude: f64, hz: f64, secs: f64) -> Vec<f64> {
        let sr = f64::from(SAMPLE_RATE);
        (0..(secs * sr) as usize)
            .map(|i| amplitude * (std::f64::consts::TAU * hz * i as f64 / sr).cos())
            .collect()
    }

    /// `buf` through a fresh ceiling's limiter, before its clamp.
    fn limited(buf: &[f64]) -> (Vec<f64>, Ceiling) {
        let mut ceiling = Ceiling::new(SAMPLE_RATE);
        let mut out = buf.to_vec();
        ceiling.limit(&mut out);
        (out, ceiling)
    }

    /// How many of `buf`'s samples the clamp changes.
    fn clamped(ceiling: &Ceiling, buf: &[f64]) -> usize {
        buf.iter()
            .filter(|&&x| ceiling.clamp(x) != x as f32)
            .count()
    }

    /// The frames the limiter plays its input late by: where a tone from
    /// silence first sounds.
    fn delay(out: &[f64]) -> usize {
        out.iter().position(|&v| v != 0.0).unwrap_or(out.len())
    }

    #[test]
    fn a_burst_from_silence_never_passes_the_ceiling() {
        let mut input = vec![0.0; SAMPLE_RATE as usize / 2];
        input.extend(tone(2.0, 440.0, 1.0));
        let (out, ceiling) = limited(&input);
        assert_eq!(
            clamped(&ceiling, &out),
            0,
            "the clamp caught the limiter's misses"
        );
        let peak = out.iter().fold(0.0f64, |p, v| p.max(v.abs()));
        assert!(
            peak > 0.99 * f64::from(ceiling.limit),
            "the burst should play up to the ceiling: {peak}"
        );
    }

    #[test]
    fn a_sustained_overload_comes_out_limited_not_clipped() {
        let input = tone(2.0, 110.0, 2.0);
        let (out, ceiling) = limited(&input);
        let late = delay(&out);
        let gain = f64::from(ceiling.limit) / 2.0;
        // a limited tone is the tone, quieter; a clipped one is flattened at
        // its crests
        let settled = SAMPLE_RATE as usize / 10;
        let worst = (out[settled..].iter().zip(&input[settled - late..]))
            .fold(0.0f64, |w, (y, x)| w.max((y - x * gain).abs()));
        assert!(
            worst < 1e-3 * f64::from(ceiling.limit),
            "the tone strays {worst} from itself at a steady gain"
        );
        assert_eq!(clamped(&ceiling, &out), 0);
    }

    #[test]
    fn a_quiet_signal_passes_unchanged_at_most_the_lookahead_late() {
        let input = tone(0.1, 440.0, 1.0);
        let (out, _) = limited(&input);
        let late = delay(&out);
        assert!(
            late > 0 && late as f64 <= LOOKAHEAD_S * f64::from(SAMPLE_RATE),
            "the limiter plays {late} frames late"
        );
        let first_diff =
            (out[late..].iter().zip(&input)).position(|(y, x)| y.to_bits() != x.to_bits());
        assert_eq!(first_diff, None, "a sample under the ceiling moved");
    }

    #[test]
    fn the_gain_comes_all_the_way_back_after_an_overload() {
        // noise, so the gains the limiter averages vary frame to frame
        let mut rng = NoiseStream::new(3);
        let mut input: Vec<f64> = (0..SAMPLE_RATE / 2)
            .map(|_| 2.0 * f64::from(rng.norm()))
            .collect();
        input.extend(tone(0.1, 440.0, 2.0));
        let (out, ceiling) = limited(&input);
        let late = delay(&out);
        assert_eq!(clamped(&ceiling, &out), 0, "the overload played unlimited");
        // the last second, long after the overload's release
        let from = out.len() - SAMPLE_RATE as usize;
        let first_diff = (out[from..].iter().zip(&input[from - late..]))
            .position(|(y, x)| y.to_bits() != x.to_bits());
        assert_eq!(
            first_diff, None,
            "the gain never got back to exactly 1, so it still remembers the overload"
        );
    }

    #[test]
    fn a_peak_holds_the_gain_down_then_releases_it_doubling_per_release_time() {
        // a second rate, so a span that counts `SAMPLE_RATE`'s frames misses it
        for sr in [SAMPLE_RATE, 96_000] {
            let frames = |secs: f64| (secs * f64::from(sr)) as usize;
            let (lookahead, hold, release) =
                (frames(LOOKAHEAD_S), frames(HOLD_S), frames(RELEASE_S));
            let mut ceiling = Ceiling::new(sr);
            // a lone peak that needs a quarter of the gain, then silence
            let mut input = vec![0.0; lookahead + hold + 2 * release + 2 * lookahead];
            input[0] = 4.0 * f64::from(ceiling.limit);
            let gains: Vec<f64> = (input.iter_mut())
                .map(|x| {
                    ceiling.limit(std::slice::from_mut(x));
                    ceiling.gain()
                })
                .collect();
            let held = lookahead + hold;
            assert!(
                gains[lookahead..=held].iter().all(|&g| g == 0.25),
                "the gain must hold at the peak's need through the hold"
            );
            // halfway back one release time on: the box averages the release
            // over the lookahead, so a little under
            let half = gains[held + release];
            let floor = 0.5 * 2f64.powf(-(lookahead as f64) / release as f64);
            assert!(
                (floor..0.5).contains(&half),
                "one release time on, the gain must have doubled: {half}"
            );
            assert!(gains[held + 2 * release - 1] < 1.0, "the release ran fast");
            assert!(
                gains[held + 2 * release + lookahead + 1..]
                    .iter()
                    .all(|&g| g == 1.0),
                "two release times and the lookahead on, the gain must be back at 1"
            );
        }
    }

    #[test]
    fn the_ceiling_plays_in_the_room_it_starts_with() {
        let mut ceiling = Ceiling::new(SAMPLE_RATE);
        let room = |c: &Ceiling| [c.late.capacity(), c.needs.capacity(), c.steps.capacity()];
        let before = room(&ceiling);
        // an overload quieter every frame for two windows, so the moving
        // minimum keeps every frame's need
        let frames = 2 * ceiling.window as usize;
        let limit = f64::from(ceiling.limit);
        let mut input: Vec<f64> = (0..frames)
            .map(|i| limit * (4.0 - 2.0 * i as f64 / frames as f64))
            .collect();
        ceiling.limit(&mut input);
        assert_eq!(room(&ceiling), before, "a buffer grew while playing");
    }

    #[test]
    fn the_ceiling_limits_the_same_however_it_is_chunked() {
        assert_chunk_invariant(|| Ceiling::new(SAMPLE_RATE), Ceiling::limit);
    }
}
