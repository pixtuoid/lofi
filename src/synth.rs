//! The procedural sound recipes. Every constant is an owner-ratified value:
//! change it through an A/B batch (`examples/lofi_audition.rs`) and
//! `ab_judge.py`'s verdict, which the commit records. The voices render into
//! mono f32 buffers, FFT-backed, never inside an audio callback, since
//! callers render ahead: a voice renders one note at the rate it is given as
//! the engine reaches it, and [`render_lane`] a whole lane at
//! [`SAMPLE_RATE`] for lofi's own tools. The engine plays the same recipe
//! through `stream::lanes`' causal chains.

use super::dsp::SAMPLE_RATE;

const SR: f32 = SAMPLE_RATE as f32;

/// `dur_s` in samples at `sr`.
pub fn samples_at(dur_s: f32, sr: u32) -> usize {
    (dur_s * sr as f32).round() as usize
}

fn n_samples(dur_s: f32) -> usize {
    samples_at(dur_s, SAMPLE_RATE)
}

/// The sample a note at `at_s` starts on, at `sr`.
fn onset(at_s: f32, sr: u32) -> usize {
    (at_s * sr as f32) as usize
}

/// The sample bar `bar` of a loop starts on, at `sr`: where its chord strikes.
pub(crate) fn bar_onset(bar: usize, bar_s: f32, sr: u32) -> usize {
    onset(bar as f32 * bar_s, sr)
}

fn env_ar(n: usize, attack_s: f32, release_s: f32, sr: u32) -> impl Fn(usize) -> f32 {
    let na = samples_at(attack_s, sr).min(n).max(1);
    let nr = samples_at(release_s, sr).min(n).max(1);
    move |i| {
        let a = (i as f32 / na as f32).min(1.0);
        let r = ((n - i) as f32 / nr as f32).min(1.0);
        a.min(r)
    }
}

fn place(buf: &mut [f32], snippet: &[f32], at_s: f32, gain: f32) {
    let start = onset(at_s, SAMPLE_RATE);
    for (i, &s) in snippet.iter().enumerate() {
        if let Some(slot) = buf.get_mut(start + i) {
            *slot += s * gain;
        }
    }
}

/// A voice that plays one `(midi, dur_s, vel)` note at a sample rate.
type NoteVoice = fn(u8, f32, f32, u32) -> Vec<f32>;

const MIDI_A4: f32 = 440.0;
fn midi_freq(m: f32) -> f32 {
    MIDI_A4 * 2f32.powf((m - 69.0) / 12.0)
}

#[cfg(test)]
mod frozen;
mod recipe;
mod voices;
mod whole;

#[cfg(test)]
pub(crate) use recipe::{BASS, DRUMS, HitRecipe, KEYS, NoteRecipe, SPARKLE};
pub(crate) use recipe::{PadTone, Part, part};
pub use whole::render_lane;
pub(crate) use whole::{
    CRACKLE_POPS_PER_SEC, DRIVE, DUCK_DB, DUCK_RELEASE_S, HEAD_BUMP_HZ, HEAD_BUMP_MIX, HISS_HZ,
    HISS_LEVEL, HUM_HZ, HUM_SWELL_HZ, ROLLOFF_HZ, TILT_HZ, TILT_LIFT, WARPS, crackle_pop,
    room_tone, staged_lane,
};

#[cfg(test)]
pub(crate) use whole::WOW;

#[cfg(test)]
mod tests {
    use super::frozen::{
        night_drums, night_keys, night_pad, night_sparkle, night_texture, stem_drums, stem_keys,
        stem_pad, stem_sparkle,
    };
    use super::whole::texture_bed;
    use crate::dsp::NoiseStream;

    #[test]
    fn every_buffer_is_finite_and_peak_bounded() {
        let mut rng = NoiseStream::new(9);
        for (name, buf) in [
            ("texture", texture_bed(&mut rng).buf),
            ("pad", stem_pad()),
            ("sparkle", stem_sparkle()),
            ("keys", stem_keys()),
            ("drums", stem_drums(&mut rng)),
            ("night_pad", night_pad()),
            ("night_sparkle", night_sparkle()),
            ("night_keys", night_keys()),
            ("night_drums", night_drums(&mut rng)),
            ("night_texture", night_texture(&mut rng)),
            // The 8 day2/day3 take-stems are checked in
            // `day_take_stems_match_the_ratified_fingerprints`; rebuilding them
            // here would be a duplicate synthesis pass.
        ] {
            assert!(!buf.is_empty(), "{name} empty");
            assert!(
                buf.iter().all(|v| v.is_finite() && v.abs() <= 1.0),
                "{name} has NaN/over-peak samples"
            );
        }
    }
}
