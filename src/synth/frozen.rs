//! The frozen takes and the fixtures that render them.

use super::recipe::{DRUMS, KEYS, Part, SPARKLE};
use super::voices::{
    ChordTone, chord, drums_core, ep_pluck_h2, ep_pluck_vel, events_stem_voiced, pad_core,
};
use super::whole::render_part;
use super::{SR, env_ar, midi_freq, n_samples, place};
use crate::compose::{BEATS_PER_BAR, DrumKind, GEN_LOOP_BARS, Hit, Mood, Note};
use crate::dsp::{NoiseStream, SAMPLE_RATE};
use crate::score;

// The `#[cfg(test)]` stems below are fingerprint ANCHORS, not dead code: their
// pins prove the voices and cores `render_lane` shares still render the ratified sound.

fn beat_s() -> f32 {
    score::beat_s()
}

fn bar_s() -> f32 {
    score::beat_s() * BEATS_PER_BAR
}

/// The day-1 anchor's EP, Phase 0's: a fixed 2nd harmonic, where the voice
/// that ships keys it to the velocity.
fn ep_pluck(midi: u8, dur_s: f32, vel: f32, sr: u32) -> Vec<f32> {
    ep_pluck_h2(midi, dur_s, vel, 0.35, sr)
}

/// How long the day-1 anchor's lead notes ring, where the shipped day lead's
/// ring [`SPARKLE`]'s.
const ANCHOR_LEAD_DUR_S: f32 = 1.6;

/// The day-1 anchor's `(beats, midi, vel)` rows as the notes its cores
/// place.
fn anchor_events(rows: &[(f32, u8, f32)]) -> Vec<Note> {
    (rows.iter())
        .map(|&(beats, midi, vel)| Note {
            at: beats * beat_s(),
            midi,
            vel,
        })
        .collect()
}

pub(crate) fn stem_pad() -> Vec<f32> {
    let chords: [[u8; 4]; GEN_LOOP_BARS] = std::array::from_fn(score::chord_at_bar);
    let pad = Part::Chords {
        chords: &chords,
        bar_s: bar_s(),
        mood: Mood::Day,
    };
    render_part(&pad, score::loop_secs(), &mut NoiseStream::new(0)).buf
}

pub(crate) fn stem_sparkle() -> Vec<f32> {
    let (r, events) = (SPARKLE.day, anchor_events(&score::SPARKLE_SCORE));
    let loop_s = score::loop_secs();
    events_stem_voiced(
        loop_s,
        &events,
        ANCHOR_LEAD_DUR_S,
        r.cutoff_hz,
        r.peak,
        ep_pluck,
    )
    .buf
}

pub(crate) fn stem_keys() -> Vec<f32> {
    let (r, events) = (KEYS.day, anchor_events(&score::KEYS_SCORE));
    events_stem_voiced(
        score::loop_secs(),
        &events,
        r.dur_s,
        r.cutoff_hz,
        r.peak,
        ep_pluck,
    )
    .buf
}

/// The day-1 anchor's kick/snare/swung-hat groove, its hat velocities the
/// frozen score's, through the shipped drums' core; the kick's and snare's
/// NOISE is fresh each hit, and a hat is the fixed metallic hash.
pub(crate) fn stem_drums(rng: &mut NoiseStream) -> Vec<f32> {
    let swing = 0.10 * beat_s();
    let hit = |at, kind, gain| Hit { at, kind, gain };
    let mut hits = Vec::new();
    for bar in 0..score::LOOP_BARS {
        let b0 = bar as f32 * bar_s();
        hits.push(hit(b0, DrumKind::Kick, 1.0));
        hits.push(hit(b0 + 2.5 * beat_s(), DrumKind::Kick, 0.8));
        hits.push(hit(b0 + 2.0 * beat_s(), DrumKind::Snare, 1.0));
        for eighth in 0..8 {
            let at =
                b0 + eighth as f32 * (beat_s() / 2.0) + if eighth % 2 == 1 { swing } else { 0.0 };
            let kind = if eighth == 7 && bar % 2 == 1 {
                DrumKind::OpenHat
            } else {
                DrumKind::Hat
            };
            hits.push(hit(at, kind, score::DRUM_HAT_VELS[bar * 8 + eighth]));
        }
    }
    let r = DRUMS.day;
    drums_core(score::loop_secs(), &hits, r.cutoff_hz, r.drive, r.peak, rng).buf
}

fn night_bar_s() -> f32 {
    score::night_beat_s() * BEATS_PER_BAR
}

/// The FROZEN night pad's baked sub level. The ratified v4 take renders the
/// chords and the sub-bass floor in ONE buffer.
const NIGHT_PAD_SUB_GAIN: f32 = 3.2;

/// Night pad: chords + the SUB-BASS floor in ONE buffer, so the harmonic floor
/// moves as one — the FROZEN anchor predates the bass stem and keeps its pins.
pub(crate) fn night_pad() -> Vec<f32> {
    let (chords, bar_s) = (&score::NIGHT_CHORDS, night_bar_s());
    pad_core(Mood::Night, bar_s, score::NIGHT_LOOP_BARS, |bar| {
        let tone = ChordTone::of(Mood::Night);
        let mut sig = chord(tone, &chords[bar % chords.len()], bar_s, SAMPLE_RATE);
        let roots = &score::NIGHT_BASS_ROOTS;
        bake_night_sub(&mut sig, roots[bar % roots.len()], bar_s);
        sig
    })
    .buf
}

/// The FROZEN v4 anchor's sub floor under one bar's chord in `sig`: the bar's
/// root twice, the second half softer.
fn bake_night_sub(sig: &mut [f32], root: u8, bar_s: f32) {
    let tau = std::f32::consts::TAU;
    let fb = midi_freq(f32::from(root));
    for half in 0..2 {
        let hdur = bar_s / 2.0 + 0.4;
        let hn = n_samples(hdur);
        let env = env_ar(hn, 0.06, 0.9, SAMPLE_RATE);
        let g = if half == 0 { 1.0 } else { 0.7 };
        let bass: Vec<f32> = (0..hn)
            .map(|i| {
                let t = i as f32 / SR;
                ((tau * fb * t).sin() + 0.15 * (tau * 2.0 * fb * t).sin())
                    * env(i)
                    * g
                    * NIGHT_PAD_SUB_GAIN
            })
            .collect();
        place(sig, &bass, half as f32 * bar_s / 2.0, 1.0);
    }
}

/// The frozen night take's `part` through the shipped whole-lane core.
fn night_part(part: &Part<'_>, rng: &mut NoiseStream) -> Vec<f32> {
    render_part(part, score::night_loop_secs(), rng).buf
}

pub(crate) fn night_keys() -> Vec<f32> {
    let keys = Part::Notes {
        events: &score::NIGHT_KEYS,
        voice: ep_pluck_vel,
        recipes: KEYS,
        mood: Mood::Night,
    };
    night_part(&keys, &mut NoiseStream::new(0))
}

pub(crate) fn night_sparkle() -> Vec<f32> {
    let sparkle = Part::Notes {
        events: &score::NIGHT_SPARKLE,
        voice: ep_pluck_vel,
        recipes: SPARKLE,
        mood: Mood::Night,
    };
    night_part(&sparkle, &mut NoiseStream::new(0))
}

pub(crate) fn night_drums(rng: &mut NoiseStream) -> Vec<f32> {
    let drums = Part::Hits {
        hits: &score::NIGHT_DRUMS,
        recipes: DRUMS,
        mood: Mood::Night,
    };
    night_part(&drums, rng)
}

/// The night texture: the day room-tone recipe at the NIGHT loop length with
/// the kick-duck BAKED IN, so the whole lane ducks in phase with its drums by
/// construction.
pub(crate) fn night_texture(rng: &mut NoiseStream) -> Vec<f32> {
    night_part(&NIGHT_BED, rng)
}

/// The frozen night take's bed, ducking under its drums.
const NIGHT_BED: Part<'static> = Part::Bed {
    mood: Mood::Night,
    hits: &score::NIGHT_DRUMS,
};

/// A frozen day take's `part` through the shipped whole-lane core.
fn day_take_part(take: &score::DayTake, part: &Part<'_>, rng: &mut NoiseStream) -> Vec<f32> {
    render_part(part, take.loop_secs(), rng).buf
}

pub(super) fn day_take_pad(take: &score::DayTake) -> Vec<f32> {
    let chords: [[u8; 4]; GEN_LOOP_BARS] = std::array::from_fn(|bar| take.chord_at_bar(bar));
    let pad = Part::Chords {
        chords: &chords,
        bar_s: take.bar_s(),
        mood: Mood::Day,
    };
    day_take_part(take, &pad, &mut NoiseStream::new(0))
}

pub(super) fn day_take_sparkle(take: &score::DayTake) -> Vec<f32> {
    let sparkle = Part::Notes {
        events: take.sparkle,
        voice: ep_pluck_vel,
        recipes: SPARKLE,
        mood: Mood::Day,
    };
    day_take_part(take, &sparkle, &mut NoiseStream::new(0))
}

pub(super) fn day_take_keys(take: &score::DayTake) -> Vec<f32> {
    let keys = Part::Notes {
        events: take.keys,
        voice: ep_pluck_vel,
        recipes: KEYS,
        mood: Mood::Day,
    };
    day_take_part(take, &keys, &mut NoiseStream::new(0))
}

pub(super) fn day_take_drums(take: &score::DayTake, rng: &mut NoiseStream) -> Vec<f32> {
    let drums = Part::Hits {
        hits: take.drums,
        recipes: DRUMS,
        mood: Mood::Day,
    };
    day_take_part(take, &drums, rng)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::{band_energy_share, centroid_hz};

    #[test]
    fn musical_stems_match_the_ratified_p3_fingerprints() {
        // reference = the ratified composition measured on the python FLOAT
        // chain, NOT the wav files — write_wav's stereo interleave + soft clip
        // once poisoned these numbers.
        type Case = (&'static str, Vec<f32>, f32, &'static [(f32, f32, f32)]);
        let cases: [Case; 4] = [
            (
                "pad",
                stem_pad(),
                291.1,
                &[(125.0, 250.0, 0.438), (250.0, 500.0, 0.516)],
            ),
            (
                "sparkle",
                stem_sparkle(),
                607.9,
                &[(250.0, 500.0, 0.313), (500.0, 1000.0, 0.641)],
            ),
            (
                "keys",
                stem_keys(),
                350.1,
                &[(125.0, 250.0, 0.312), (250.0, 500.0, 0.542)],
            ),
            (
                "drums",
                stem_drums(&mut NoiseStream::new(4)),
                214.6,
                &[(62.5, 125.0, 0.519), (125.0, 250.0, 0.408)],
            ),
        ];
        for (name, buf, ref_centroid, bands) in cases {
            let c = centroid_hz(&buf);
            assert!(
                (c - ref_centroid).abs() <= ref_centroid * 0.20,
                "{name}: centroid {c:.1} vs ratified {ref_centroid}"
            );
            // drums need a wider band tolerance: their noise content is fresh
            // per build
            let tol = if name == "drums" { 0.15 } else { 0.10 };
            for &(lo, hi, ref_share) in bands {
                let s = band_energy_share(&buf, lo, hi);
                assert!(
                    (s - ref_share).abs() <= tol,
                    "{name} band {lo}-{hi}: {s:.3} vs ratified {ref_share}"
                );
            }
        }
        let drums = stem_drums(&mut NoiseStream::new(4));
        let low = band_energy_share(&drums, 62.5, 125.0);
        let mid = band_energy_share(&drums, 125.0, 250.0);
        let high = band_energy_share(&drums, 250.0, 500.0);
        assert!(
            low > mid && mid > high,
            "drums must stay kick-dominant: {low:.3} > {mid:.3} > {high:.3}"
        );
        // the hi-hat layer is the ONLY content in 3.5-6.5k (kick/snare live below
        // 3.2k), so the coarse octave bands cannot see the groove
        let hats = band_energy_share(&drums, 3500.0, 6500.0);
        assert!(
            hats > 0.0015,
            "the swung-hat groove must be audible in 3.5-6.5k: {hats:.5}"
        );
    }

    #[test]
    fn musical_stems_share_one_loop_length() {
        let n = stem_pad().len();
        assert_eq!(n, n_samples(score::loop_secs()));
        assert_eq!(stem_sparkle().len(), n);
        assert_eq!(stem_keys().len(), n);
        assert_eq!(stem_drums(&mut NoiseStream::new(4)).len(), n);
    }

    #[test]
    fn night_stems_match_the_ratified_v4_fingerprints() {
        // reference = the owner-LISTEN-passed v4 float chain
        type Case = (&'static str, Vec<f32>, f32, f32, &'static [(f32, f32, f32)]);
        let cases: [Case; 4] = [
            (
                "night_pad",
                night_pad(),
                103.6,
                0.10,
                &[
                    (31.0, 62.0, 0.538),
                    // moved by lofi_post's bump widening to 60-120 (LOFI-BIBLE
                    // §4); re-LISTEN-gated, not curve-fitted
                    (62.0, 125.0, 0.299),
                    (125.0, 250.0, 0.172),
                ],
            ),
            (
                "night_sparkle",
                night_sparkle(),
                541.3,
                0.10,
                &[(250.0, 500.0, 0.527), (500.0, 1000.0, 0.449)],
            ),
            (
                "night_keys",
                night_keys(),
                209.2,
                0.10,
                &[(125.0, 250.0, 0.729), (250.0, 500.0, 0.253)],
            ),
            (
                "night_drums",
                night_drums(&mut NoiseStream::new(4)),
                118.8,
                0.15,
                &[(62.0, 125.0, 0.561), (125.0, 250.0, 0.412)],
            ),
        ];
        for (name, buf, ref_centroid, tol, bands) in cases {
            let c = centroid_hz(&buf);
            assert!(
                (c - ref_centroid).abs() <= ref_centroid * 0.20,
                "{name}: centroid {c:.1} vs ratified {ref_centroid}"
            );
            for &(lo, hi, ref_share) in bands {
                let s = band_energy_share(&buf, lo, hi);
                assert!(
                    (s - ref_share).abs() <= tol,
                    "{name} band {lo}-{hi}: {s:.3} vs ratified {ref_share}"
                );
            }
        }
    }

    #[test]
    fn day_take_stems_match_the_ratified_fingerprints() {
        // reference = the owner-LISTEN-passed candidates, measured on the python
        // float chain with the RUST formulas
        type Case = (&'static str, Vec<f32>, f32, f32, &'static [(f32, f32, f32)]);
        let cases: [Case; 8] = [
            (
                "day2_pad",
                day_take_pad(&score::DAY2),
                277.7,
                0.10,
                &[(125.0, 250.0, 0.497), (250.0, 500.0, 0.462)],
            ),
            (
                "day2_sparkle",
                day_take_sparkle(&score::DAY2),
                580.4,
                0.10,
                &[(250.0, 500.0, 0.312), (500.0, 1000.0, 0.659)],
            ),
            (
                "day2_keys",
                day_take_keys(&score::DAY2),
                355.6,
                0.10,
                &[(125.0, 250.0, 0.323), (250.0, 500.0, 0.556)],
            ),
            (
                "day2_drums",
                day_take_drums(&score::DAY2, &mut NoiseStream::new(4)),
                235.5,
                0.15,
                &[(62.5, 125.0, 0.492), (125.0, 250.0, 0.420)],
            ),
            (
                "day3_pad",
                day_take_pad(&score::DAY3),
                217.5,
                0.10,
                &[(125.0, 250.0, 0.746), (250.0, 500.0, 0.235)],
            ),
            (
                "day3_sparkle",
                day_take_sparkle(&score::DAY3),
                611.3,
                0.10,
                &[(250.0, 500.0, 0.218), (500.0, 1000.0, 0.751)],
            ),
            (
                "day3_keys",
                day_take_keys(&score::DAY3),
                303.7,
                0.10,
                &[(125.0, 250.0, 0.404), (250.0, 500.0, 0.558)],
            ),
            (
                "day3_drums",
                day_take_drums(&score::DAY3, &mut NoiseStream::new(4)),
                192.2,
                0.15,
                &[(62.5, 125.0, 0.521), (125.0, 250.0, 0.414)],
            ),
        ];
        for (name, buf, ref_centroid, tol, bands) in cases {
            assert!(
                buf.iter().all(|v| v.is_finite() && v.abs() <= 1.0),
                "{name}: NaN/over-peak"
            );
            let c = centroid_hz(&buf);
            assert!(
                (c - ref_centroid).abs() <= ref_centroid * 0.20,
                "{name}: centroid {c:.1} vs ratified {ref_centroid}"
            );
            for &(lo, hi, ref_share) in bands {
                let s = band_energy_share(&buf, lo, hi);
                assert!(
                    (s - ref_share).abs() <= tol,
                    "{name} band {lo}-{hi}: {s:.3} vs ratified {ref_share}"
                );
            }
            // the full-kit signatures night deliberately lacks: the snare body
            // and the open-hat ring both put audible energy in 3.5-6.5k
            if name.ends_with("drums") {
                let hats = band_energy_share(&buf, 3500.0, 6500.0);
                assert!(
                    hats > 0.0015,
                    "{name} hats/snare must be audible in 3.5-6.5k: {hats:.5}"
                );
            }
        }
    }

    #[test]
    fn day_take_stems_share_one_loop_length_per_take() {
        for take in [&score::DAY2, &score::DAY3] {
            let n = day_take_pad(take).len();
            assert_eq!(n, n_samples(take.loop_secs()));
            assert_eq!(day_take_sparkle(take).len(), n);
            assert_eq!(day_take_keys(take).len(), n);
            assert_eq!(day_take_drums(take, &mut NoiseStream::new(4)).len(), n);
            assert_ne!(n, n_samples(score::loop_secs()));
            assert_ne!(n, n_samples(score::night_loop_secs()));
        }
        assert_ne!(
            n_samples(score::DAY2.loop_secs()),
            n_samples(score::DAY3.loop_secs())
        );
    }

    #[test]
    fn night_stems_share_one_loop_length_including_texture() {
        // night phase-lock covers TEXTURE too: its kick-duck is baked at frozen
        // kick times, so it must tile in lockstep with the drums
        let n = night_pad().len();
        assert_eq!(n, n_samples(score::night_loop_secs()));
        assert_eq!(night_sparkle().len(), n);
        assert_eq!(night_keys().len(), n);
        assert_eq!(night_drums(&mut NoiseStream::new(4)).len(), n);
        assert_eq!(night_texture(&mut NoiseStream::new(4)).len(), n);
        assert_ne!(n, n_samples(score::loop_secs()));
    }

    #[test]
    fn night_texture_seam_is_spliced_and_duck_dips_after_kicks() {
        let tex = night_texture(&mut NoiseStream::new(4));
        let n = tex.len();
        // the wrap must read as a CONTINUATION, not merely "no huge jump": a
        // loose body-max check passed while the discontinuity survived
        let body_median = {
            let mut d: Vec<f32> = tex.windows(2).map(|w| (w[1] - w[0]).abs()).collect();
            d.sort_by(f32::total_cmp);
            d[d.len() / 2]
        };
        let seam = (tex[0] - tex[n - 1]).abs();
        assert!(
            seam <= body_median * 8.0,
            "wrap must read as adjacent samples: seam {seam} vs median {body_median}"
        );
        let rms = |s: &[f32]| (s.iter().map(|v| v * v).sum::<f32>() / s.len() as f32).sqrt();
        let w = n_samples(0.01);
        let head = rms(&tex[..w]);
        let tail = rms(&tex[n - w..]);
        assert!(
            (head - tail).abs() <= 0.10 * head.max(tail),
            "no level step across the wrap: head {head:.5} vs tail {tail:.5}"
        );
        let kt = NIGHT_BED.ducks()[2];
        let i = n_samples(kt);
        let w = n_samples(0.05);
        let rms = |s: &[f32]| (s.iter().map(|v| v * v).sum::<f32>() / s.len() as f32).sqrt();
        let before = rms(&tex[i - w..i]);
        let after = rms(&tex[i..i + w]);
        assert!(
            after < before * 0.85,
            "texture must duck after the kick: {before:.5} -> {after:.5}"
        );
    }
}
