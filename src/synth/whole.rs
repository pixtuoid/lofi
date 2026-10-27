//! The whole-lane chain behind `render_lane`.

use super::recipe::{PadTone, Part, part};
use super::voices::{ChordTone, chord, drum_hit};
use super::{NoteVoice, SR, n_samples, place, samples_at};
use crate::compose::{GEN_LOOP_BARS, Hit, Lane, Mood, Note, Score};
use crate::dsp::{NoiseStream, SAMPLE_RATE, bandpass, lowpass, white_noise_gain};

/// Scales `buf` to peak at `peak`, and returns the gain it applied.
fn normalize(buf: &mut [f32], peak: f32) -> f32 {
    let m = buf.iter().fold(0.0f32, |a, &v| a.max(v.abs())).max(1e-9);
    let gain = peak / m;
    buf.iter_mut().for_each(|v| *v *= gain);
    gain
}

/// A whole-lane render and the gains its peak normalizations applied: the
/// first, which sets the level into the drive (the texture's, into the duck),
/// and the product of the rest. The stream's staging stands in for the two.
#[derive(Debug)]
pub(crate) struct Staged {
    pub(crate) buf: Vec<f32>,
    pub(crate) gains: [f32; 2],
}

/// Seamless-loop length for the texture bed. Must stay a power of two: its FFT
/// lowpass pads to one, and only an unpadded block filters circularly, so the
/// bed loops click-free.
const BED_LOOP_SAMPLES: usize = 1 << 19;
const _: () = assert!(BED_LOOP_SAMPLES.is_power_of_two());

/// Vinyl-crackle pop density shared by every room-tone bed — one knob, so the
/// day and night textures age identically.
pub(crate) const CRACKLE_POPS_PER_SEC: f32 = 0.18;

/// The texture's tape hiss: noise under this low-pass, at this level.
pub(crate) const HISS_HZ: f32 = 3800.0;
pub(crate) const HISS_LEVEL: f32 = 0.010;

/// The room hum's pitch, and the rate its level swells at.
pub(crate) const HUM_HZ: f32 = 90.0;
pub(crate) const HUM_SWELL_HZ: f32 = 0.4;

/// The faint warm room hum under the hiss, at the hum's and its swell's
/// phases in radians: the caller keeps the clock, since the whole-lane beds'
/// f32 clock is ratified and a song's outruns f32.
pub(crate) fn room_tone(hum: f32, swell: f32) -> f32 {
    hum.sin() * (1.0 + 0.2 * swell.sin()) * 0.006
}

/// [`room_tone`] `t` seconds into a whole-lane bed.
fn bed_room_tone(t: f32) -> f32 {
    let tau = std::f32::consts::TAU;
    room_tone(tau * HUM_HZ * t, tau * HUM_SWELL_HZ * t)
}

/// One vinyl-crackle pop at `sr`, and the gain it mixes in at.
pub(crate) fn crackle_pop(rng: &mut NoiseStream, sr: u32) -> (Vec<f32>, f32) {
    let pn = samples_at(0.003 + 0.004 * rng.unit(), sr);
    let white = white_noise_gain(sr);
    let pop = (0..pn)
        .map(|i| rng.norm() * white * (-(i as f32 / sr as f32) * 800.0).exp())
        .collect();
    (pop, 0.03 + 0.06 * rng.unit())
}

/// The vinyl/room texture bed: tape hiss + a faint warm room hum + sparse soft
/// crackle. Its crackle draws from the caller's stream ON PURPOSE — unpitched, so
/// variation is a feature; everything melodic comes from the [`Score`].
pub(crate) fn texture_bed(rng: &mut NoiseStream) -> Staged {
    let n = BED_LOOP_SAMPLES;
    let mut buf = room_bed(n, n, rng);
    let gain = normalize(&mut buf, TEXTURE_PEAK);
    Staged {
        buf,
        gains: [gain, 1.0],
    }
}

/// The texture stem's peak, day or night.
const TEXTURE_PEAK: f32 = 0.45;

/// The latest a crackle pop starts, before the bed's end.
const LAST_POP_BEFORE_END_S: f32 = 0.01;

/// The room's bed: `total` samples of hiss over the room tone, then the
/// crackle's pops over its first `n`, drawn from `rng` in that order.
fn room_bed(n: usize, total: usize, rng: &mut NoiseStream) -> Vec<f32> {
    let raw: Vec<f32> = (0..total).map(|_| rng.norm()).collect();
    let hiss = lowpass(&raw, HISS_HZ);
    let mut buf: Vec<f32> = hiss
        .iter()
        .enumerate()
        .map(|(i, hiss)| hiss * HISS_LEVEL + bed_room_tone(i as f32 / SR))
        .collect();
    let n_pops = (n as f32 / SR * CRACKLE_POPS_PER_SEC) as usize;
    for _ in 0..n_pops {
        let at = rng.unit() * (n as f32 / SR - LAST_POP_BEFORE_END_S);
        let (pop, gain) = crackle_pop(rng, SAMPLE_RATE);
        place(&mut buf[..n], &pop, at, gain);
    }
    buf
}

/// Tape wow and flutter, `(rate_hz, pitch deviation)` pairs as
/// [`warp_resample`](crate::dsp::warp_resample) takes them.
pub(crate) const WARPS: [(f32, f32); 2] = [WOW, (8.0, 0.0006)];

/// The slow, deep warp of [`WARPS`], named apart so what leans on it finds
/// it by name, not by place.
pub(crate) const WOW: (f32, f32) = (0.7, 0.0025);

/// The tape drive of every musical lane but the drums, which
/// [`DRUMS`](super::recipe::DRUMS) gives their own.
pub(crate) const DRIVE: f32 = 1.6;

/// The tape's head bump: the band it lifts, low enough to catch the bass
/// fundamentals (`.claude/skills/procedural-lofi/reference/LOFI-BIBLE.md`
/// §4), and how much of that band adds back.
pub(crate) const HEAD_BUMP_HZ: (f32, f32) = (60.0, 120.0);
pub(crate) const HEAD_BUMP_MIX: f32 = 0.35;

/// Where the tape's top rolls off.
pub(crate) const ROLLOFF_HZ: f32 = 6500.0;

/// The lofi master chain, applied per MUSICAL stem (texture skips it — it
/// IS the medium). No hiss here: per-stem hiss STACKS across the musical
/// stems, so the medium noise is `texture_bed`'s job, exactly once.
fn lofi_post(buf: &[f32], drive: f32) -> Vec<f32> {
    let warped = crate::dsp::warp_resample(buf, &WARPS);
    let t = drive.tanh();
    let sat: Vec<f32> = warped.iter().map(|&x| (x * drive).tanh() / t).collect();
    let bump = bandpass(&sat, HEAD_BUMP_HZ.0, HEAD_BUMP_HZ.1);
    let bumped: Vec<f32> = sat
        .iter()
        .zip(&bump)
        .map(|(&x, &b)| x + HEAD_BUMP_MIX * b)
        .collect();
    lowpass(&bumped, ROLLOFF_HZ)
}

/// ONE site, so a stem can't silently skip the ratified post: `buf` to
/// `peak`, which sets the level into the drive, then through the tape and
/// back to `peak`.
fn master(mut buf: Vec<f32>, drive: f32, peak: f32) -> Staged {
    let gain_in = normalize(&mut buf, peak);
    let mut out = lofi_post(&buf, drive);
    let gain = normalize(&mut out, peak);
    Staged {
        buf: out,
        gains: [gain_in, gain],
    }
}

/// The night texture's duck under each kick: down this far, then back up
/// over this long.
pub(crate) const DUCK_DB: f32 = -4.0;
pub(crate) const DUCK_RELEASE_S: f32 = 0.15;

/// Loop-seam splice length: the night texture's buffer is the night loop length,
/// NOT a power of two, so its FFT-shaped hiss is not circularly seamless.
const NIGHT_TEXTURE_SPLICE_S: f32 = 0.03;

/// The night room's bed over a loop `loop_secs` long, its seam spliced, at
/// the level [`room_bed`] makes it.
fn night_room_bed(loop_secs: f32, rng: &mut NoiseStream) -> Vec<f32> {
    let n = n_samples(loop_secs);
    let f = n_samples(NIGHT_TEXTURE_SPLICE_S);
    // synthesize f EXTRA samples past the loop end so the splice blends a genuine
    // CONTINUATION of the tail into the head; a head-only crossfade of unrelated
    // content leaves the discontinuity in place
    let mut buf = room_bed(n, n + f, rng);
    // CORRELATED content (the pre-roll continues the same stream), so a linear
    // blend suffices — no equal-power law needed
    for j in 0..f {
        let a = j as f32 / f as f32;
        buf[j] = buf[n + j] * (1.0 - a) + buf[j] * a;
    }
    buf.truncate(n);
    buf
}

fn night_texture_core(loop_secs: f32, kick_times: &[f32], rng: &mut NoiseStream) -> Staged {
    let mut buf = night_room_bed(loop_secs, rng);
    // normalize BEFORE the duck: duck-then-normalize re-inflates the level by up
    // to the duck depth
    let gain = normalize(&mut buf, TEXTURE_PEAK);
    let depth = 10f32.powf(DUCK_DB / 20.0);
    let rel = n_samples(DUCK_RELEASE_S);
    for &kt in kick_times {
        let i0 = n_samples(kt);
        for j in 0..rel {
            let Some(slot) = buf.get_mut(i0 + j) else {
                break;
            };
            let g = depth + (1.0 - depth) * (j as f32 / rel as f32);
            *slot *= g;
        }
    }
    Staged {
        buf,
        gains: [gain, 1.0],
    }
}

/// The bright master's one-pole tilt shelf: its corner, and how far above
/// it the top lifts.
pub(crate) const TILT_HZ: f32 = 1200.0;
pub(crate) const TILT_LIFT: f32 = 0.8;

/// The "bright" master for GENERATED takes: a one-pole tilt shelf, then
/// re-normalize to the bed's pre-tilt peak, returning that gain. FROZEN takes
/// keep their darker chain — brightening them would invalidate their
/// fingerprint pins.
fn brighten(buf: &mut [f32]) -> f32 {
    let pre_peak = buf.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    let a = (-std::f32::consts::TAU * TILT_HZ / SR).exp();
    let mut lp = 0.0f32;
    for s in buf.iter_mut() {
        lp = a * lp + (1.0 - a) * *s;
        *s += TILT_LIFT * (*s - lp);
    }
    normalize(buf, pre_peak.max(1e-6))
}

/// A `mood` pad over `loop_bars` bars `bar_s` long, `bar` giving each bar's
/// sound.
pub(super) fn pad_core(
    mood: Mood,
    bar_s: f32,
    loop_bars: usize,
    bar: impl Fn(usize) -> Vec<f32>,
) -> Staged {
    let mut buf = vec![0.0f32; n_samples(bar_s * loop_bars as f32)];
    for b in 0..loop_bars {
        place(&mut buf, &bar(b), b as f32 * bar_s, 1.0);
    }
    let tone = PadTone::of(mood);
    let mut buf = lowpass(&buf, tone.cutoff_hz);
    for (i, v) in buf.iter_mut().enumerate() {
        *v *= tone.bed_swell(i);
    }
    master(buf, DRIVE, tone.peak)
}

/// A note lane's stem over a loop `loop_secs` long: each of `events` through
/// `note_fn`, then the stem's low-pass, its peak and the tape.
pub(super) fn events_stem_voiced(
    loop_secs: f32,
    events: &[Note],
    dur_s: f32,
    cutoff_hz: f32,
    peak: f32,
    note_fn: NoteVoice,
) -> Staged {
    let mut buf = vec![0.0f32; n_samples(loop_secs)];
    for n in events {
        place(
            &mut buf,
            &note_fn(n.midi, dur_s, n.vel, SAMPLE_RATE),
            n.at,
            1.0,
        );
    }
    master(lowpass(&buf, cutoff_hz), DRIVE, peak)
}

pub(super) fn drums_core(
    loop_secs: f32,
    table: &[Hit],
    cutoff_hz: f32,
    drive: f32,
    peak: f32,
    rng: &mut NoiseStream,
) -> Staged {
    let mut buf = vec![0.0f32; n_samples(loop_secs)];
    for h in table {
        let hit = drum_hit(rng, h.kind, SAMPLE_RATE);
        place(&mut buf, &hit, h.at, h.gain);
    }
    master(lowpass(&buf, cutoff_hz), drive, peak)
}

/// Render one lane of `score`'s loop to mono f32 at [`SAMPLE_RATE`], peaking
/// at most 1.0, for lofi's tools to audition. Every lane but the day
/// [`Lane::Texture`] is a whole number of the score's loops long, so a caller
/// loops them in phase; the day texture runs free on its own length.
pub fn render_lane(score: &Score, lane: Lane, rng: &mut NoiseStream) -> Vec<f32> {
    staged_lane(score, lane, rng).buf
}

/// [`render_lane`], with the gains its peak normalizations applied.
pub(crate) fn staged_lane(score: &Score, lane: Lane, rng: &mut NoiseStream) -> Staged {
    let mut staged = render_part(&part(score, lane), score.loop_secs(), rng);
    staged.gains[1] *= brighten(&mut staged.buf);
    staged
}

/// `part` over a loop `loop_s` long through its whole-lane core: what
/// [`render_lane`] brightens, and the frozen takes' parts.
pub(super) fn render_part(part: &Part<'_>, loop_s: f32, rng: &mut NoiseStream) -> Staged {
    match *part {
        // the generated night pad renders NO baked sub — the bass lane owns that register
        Part::Chords {
            chords,
            bar_s,
            mood,
        } => pad_core(mood, bar_s, GEN_LOOP_BARS, |bar| {
            let tone = ChordTone::of(mood);
            chord(tone, &chords[bar % chords.len()], bar_s, SAMPLE_RATE)
        }),
        Part::Notes {
            events,
            voice,
            recipes,
            mood,
        } => {
            let r = recipes.at(mood);
            events_stem_voiced(loop_s, events, r.dur_s, r.cutoff_hz, r.peak, voice)
        }
        Part::Hits {
            hits,
            recipes,
            mood,
        } => {
            let r = recipes.at(mood);
            drums_core(loop_s, hits, r.cutoff_hz, r.drive, r.peak, rng)
        }
        Part::Bed {
            mood: Mood::Day, ..
        } => texture_bed(rng),
        Part::Bed {
            mood: Mood::Night, ..
        } => night_texture_core(loop_s, &part.ducks(), rng),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::{band_energy_share, centroid_hz};
    use crate::score;

    #[test]
    fn a_day_lead_sits_above_its_pad() {
        use crate::compose::compose;
        let score = compose(Mood::Day, 0);
        let mut rng = NoiseStream::new(0);
        let pad = render_lane(&score, Lane::Pad, &mut rng);
        let sparkle = render_lane(&score, Lane::Sparkle, &mut rng);
        assert!(
            centroid_hz(&sparkle) > centroid_hz(&pad) * 1.5,
            "the lead must sit clearly above the pad: {:.0} vs {:.0}",
            centroid_hz(&sparkle),
            centroid_hz(&pad)
        );
    }

    #[test]
    fn lofi_post_saturates_bumps_and_rolls_off() {
        let tau = std::f32::consts::TAU;
        let tone: Vec<f32> = (0..n_samples(2.0))
            .map(|i| (tau * 440.0 * i as f32 / SR).sin() * 0.8)
            .collect();
        let posted = lofi_post(&tone, 1.6);
        let third = band_energy_share(&posted, 1200.0, 1450.0);
        assert!(
            third > 0.005,
            "tanh drive must write a 3rd harmonic: {third:.4}"
        );
        let mut rng = NoiseStream::new(21);
        let noise: Vec<f32> = (0..n_samples(2.0)).map(|_| rng.norm() * 0.3).collect();
        let posted_n = lofi_post(&noise, 1.6);
        let bump_in = band_energy_share(&noise, 80.0, 120.0);
        let bump_out = band_energy_share(&posted_n, 80.0, 120.0);
        assert!(
            bump_out > bump_in * 1.3,
            "head bump must lift 80-120Hz: {bump_in:.4} -> {bump_out:.4}"
        );
        // only this reds on a revert to the old 80-120 floor
        let low_in = band_energy_share(&noise, 60.0, 80.0);
        let low_out = band_energy_share(&posted_n, 60.0, 80.0);
        assert!(
            low_out > low_in * 1.3,
            "head bump must reach down to 60Hz: {low_in:.4} -> {low_out:.4}"
        );
        let top = band_energy_share(&posted_n, 7000.0, 12000.0);
        assert!(top < 0.001, "HF must die past the 6.5k rolloff: {top:.5}");
    }

    /// The tape's two normalizations, as the staging stands in for them: the
    /// first sets the input's peak, and the second the taped buffer's.
    #[test]
    fn the_master_reports_its_two_normalizations() {
        const PEAK: f32 = 0.7;
        let mut rng = NoiseStream::new(5);
        let dry: Vec<f32> = (0..n_samples(1.0)).map(|_| rng.norm() * 0.3).collect();
        let peak = |b: &[f32]| b.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let staged = master(dry.clone(), DRIVE, PEAK);
        assert_eq!(staged.gains[0], PEAK / peak(&dry));
        let into_drive: Vec<f32> = dry.iter().map(|v| v * staged.gains[0]).collect();
        assert_eq!(staged.gains[1], PEAK / peak(&lofi_post(&into_drive, DRIVE)));
    }

    /// A bed, day or night, sets its level by its peak alone, and the tape
    /// doesn't touch it.
    #[test]
    fn a_texture_bed_reports_its_peak_normalization() {
        let peak = |b: &[f32]| b.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let day = room_bed(BED_LOOP_SAMPLES, BED_LOOP_SAMPLES, &mut NoiseStream::new(4));
        let staged = texture_bed(&mut NoiseStream::new(4));
        assert_eq!(staged.gains, [TEXTURE_PEAK / peak(&day), 1.0], "day");
        let loop_s = score::night_loop_secs();
        let night = night_room_bed(loop_s, &mut NoiseStream::new(4));
        let staged = night_texture_core(loop_s, &[], &mut NoiseStream::new(4));
        assert_eq!(staged.gains, [TEXTURE_PEAK / peak(&night), 1.0], "night");
    }

    /// A lane's staged render is its core's, brightened, with the brighten's
    /// gain folded into the second: a bed, a chord lane and a hit lane each.
    #[test]
    fn a_staged_lane_is_its_core_brightened() {
        let score = crate::compose::compose(Mood::Day, 0);
        for lane in [Lane::Texture, Lane::Pad, Lane::Drums] {
            let core = render_part(
                &part(&score, lane),
                score.loop_secs(),
                &mut NoiseStream::new(2),
            );
            let mut bright = core.buf.clone();
            let gain = brighten(&mut bright);
            let staged = staged_lane(&score, lane, &mut NoiseStream::new(2));
            assert!(staged.buf == bright, "{lane:?}'s render");
            assert_eq!(
                staged.gains,
                [core.gains[0], core.gains[1] * gain],
                "{lane:?}"
            );
            assert!(render_lane(&score, lane, &mut NoiseStream::new(2)) == staged.buf);
        }
    }

    #[test]
    fn brighten_lifts_the_top_and_preserves_the_peak() {
        let mut rng = NoiseStream::new(17);
        let noise: Vec<f32> = (0..n_samples(2.0)).map(|_| rng.norm() * 0.3).collect();
        let mut posted = noise.clone();
        brighten(&mut posted);
        // a hi/lo RATIO, not a share-of-total: the tilt lifts most of the
        // spectrum, so the denominator grows too
        let ratio =
            |b: &[f32]| band_energy_share(b, 4000.0, 8000.0) / band_energy_share(b, 100.0, 500.0);
        let (r_in, r_out) = (ratio(&noise), ratio(&posted));
        assert!(
            r_out > r_in * 2.0,
            "tilt must lift hi/lo by the shelf: {r_in:.3} -> {r_out:.3}"
        );
        let peak = |b: &[f32]| b.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        assert!(
            (peak(&posted) - peak(&noise)).abs() < 1e-3,
            "brighten must re-normalize to the pre-tilt peak"
        );
    }

    #[test]
    fn generated_beds_are_finite_phase_locked_and_in_the_sound_world() {
        // the musical invariants are the compose seed-sweep's job; a couple of
        // seeds per mood suffice as the synthesis smoke
        for mood in [Mood::Day, Mood::Night] {
            for seed in [0u64, 7] {
                let score = crate::compose::compose(mood, seed);
                let mut rng = NoiseStream::new(9);
                let beds = Lane::ALL.map(|l| render_lane(&score, l, &mut rng));
                let n = n_samples(score.loop_secs());
                let [pad, sparkle, keys, drums, texture, bass] = &beds;
                for (name, bed) in [
                    ("pad", pad),
                    ("sparkle", sparkle),
                    ("keys", keys),
                    ("drums", drums),
                    ("texture", texture),
                    ("bass", bass),
                ] {
                    assert!(
                        bed.iter().all(|v| v.is_finite() && v.abs() <= 1.0),
                        "{mood:?} seed {seed} {name}: NaN/over-peak"
                    );
                }
                // every musical lane phase-locks; only texture runs free by day
                for (name, bed) in [
                    ("pad", pad),
                    ("sparkle", sparkle),
                    ("keys", keys),
                    ("drums", drums),
                    ("bass", bass),
                ] {
                    assert_eq!(bed.len(), n, "{mood:?} seed {seed} {name} length");
                }
                match mood {
                    Mood::Day => assert_eq!(
                        texture.len(),
                        1 << 19,
                        "the ratified texture loop; a new length is a new sound"
                    ),
                    Mood::Night => assert_eq!(texture.len(), n),
                }
                let low = band_energy_share(drums, 62.5, 125.0);
                let mid = band_energy_share(drums, 125.0, 250.0);
                assert!(
                    low > 0.2 && low > mid * 0.8,
                    "{mood:?} seed {seed}: drums lost the kick floor ({low:.3}/{mid:.3})"
                );
                let sub = band_energy_share(bass, 31.0, 125.0);
                assert!(
                    sub > 0.35 && sub > band_energy_share(bass, 250.0, 1000.0),
                    "{mood:?} seed {seed}: the bass lane must own the sub ({sub:.3})"
                );
            }
        }
    }
}
