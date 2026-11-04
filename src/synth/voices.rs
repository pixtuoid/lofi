//! The note and hit voices.

use super::{NoteVoice, env_ar, midi_freq, samples_at};
use crate::compose::{DrumKind, LeadVoice};
use crate::dsp::{NoiseStream, bandpass_at, highpass, white_noise_gain};

/// An electric piano's three partials: each one's fall rate; the 3rd's weight
/// and its ratio to the note, slightly sharp of a harmonic. The 2nd's weight is
/// the caller's `h2`.
const EP_RATE: f32 = 5.5;
const EP_H2_RATE: f32 = 9.0;
const EP_H3_LEVEL: f32 = 0.12;
const EP_H3_RATIO: f32 = 3.01;
const EP_H3_RATE: f32 = 14.0;
/// The velocity-keyed 2nd harmonic: its weight at rest and what velocity
/// squared adds.
const EP_VEL_H2_BASE: f32 = 0.18;
const EP_VEL_H2_GAIN: f32 = 0.45;

/// One soft electric-piano note; `h2` is the 2nd-harmonic weight — the one axis
/// the hiphop and sleep voices differ on.
pub(super) fn ep_pluck_h2(midi: u8, dur_s: f32, vel: f32, h2: f32, sr: u32) -> Vec<f32> {
    let n = samples_at(dur_s, sr);
    let f = midi_freq(f32::from(midi));
    let tau = std::f32::consts::TAU;
    (0..n)
        .map(|i| {
            let t = i as f32 / sr as f32;
            let sig = (tau * f * t).sin() * (-t * EP_RATE).exp()
                + h2 * (tau * 2.0 * f * t).sin() * (-t * EP_H2_RATE).exp()
                + EP_H3_LEVEL * (tau * EP_H3_RATIO * f * t).sin() * (-t * EP_H3_RATE).exp();
            sig * vel
        })
        .collect()
}

/// A kick's length, the Hz its pitch settles on, how far over that it starts
/// and how fast it falls there, and how fast its level falls: set so the
/// rendered kick reads as LOFI-BIBLE.md §3's does, which the commit that set
/// them measures.
const KICK_S: f32 = 0.2;
const KICK_FLOOR_HZ: f32 = 44.0;
const KICK_SWEEP_HZ: f32 = 50.0;
const KICK_SWEEP_RATE: f32 = 40.0;
const KICK_DECAY_RATE: f32 = 27.0;
/// The noise click on a kick's front: its length, its level over the body and
/// how fast it falls.
const KICK_CLICK_S: f32 = 0.006;
const KICK_CLICK_LEVEL: f32 = 0.25;
const KICK_CLICK_RATE: f32 = 300.0;

fn kick(rng: &mut NoiseStream, sr: u32) -> Vec<f32> {
    let n = samples_at(KICK_S, sr);
    let tau = std::f32::consts::TAU;
    let mut phase = 0.0f32;
    let mut buf: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32 / sr as f32;
            let f = KICK_SWEEP_HZ * (-t * KICK_SWEEP_RATE).exp() + KICK_FLOOR_HZ;
            phase += tau * f / sr as f32;
            phase.sin() * (-t * KICK_DECAY_RATE).exp()
        })
        .collect();
    let white = white_noise_gain(sr);
    for (i, slot) in buf
        .iter_mut()
        .enumerate()
        .take(samples_at(KICK_CLICK_S, sr))
    {
        let t = i as f32 / sr as f32;
        *slot += rng.norm() * white * KICK_CLICK_LEVEL * (-t * KICK_CLICK_RATE).exp();
    }
    buf
}

/// A snare's length, its noise band, how fast its noise falls, and its tone,
/// set toward LOFI-BIBLE.md §3's.
const SNARE_S: f32 = 0.16;
const SNARE_BAND_HZ: (f32, f32) = (450.0, 2400.0);
const SNARE_DECAY_RATE: f32 = 40.0;
const SNARE_TONE_HZ: f32 = 185.0;
const SNARE_TONE_RATE: f32 = 40.0;
const SNARE_TONE_LEVEL: f32 = 0.3;
const SNARE_LEVEL: f32 = 0.8;

fn snare(rng: &mut NoiseStream, sr: u32) -> Vec<f32> {
    let n = samples_at(SNARE_S, sr);
    let white = white_noise_gain(sr);
    let raw: Vec<f32> = (0..n).map(|_| rng.norm() * white).collect();
    let noise = bandpass_at(&raw, SNARE_BAND_HZ.0, SNARE_BAND_HZ.1, sr);
    let tau = std::f32::consts::TAU;
    noise
        .iter()
        .enumerate()
        .map(|(i, noise)| {
            let t = i as f32 / sr as f32;
            let tone =
                (tau * SNARE_TONE_HZ * t).sin() * (-t * SNARE_TONE_RATE).exp() * SNARE_TONE_LEVEL;
            (noise * (-t * SNARE_DECAY_RATE).exp() + tone) * SNARE_LEVEL
        })
        .collect()
}

/// Where a hat's hash starts: the bottom of the 3–6 kHz band LOFI-BIBLE.md
/// §3 low-passes the drums into, so the hat sounds under the drums' cutoff
/// and the tape's [`ROLLOFF_HZ`](super::ROLLOFF_HZ).
const HAT_HZ: f32 = 3000.0;
/// A hat's level under the kick and snare, sized by the mix's 2-4 kHz in the
/// commit that set it (opinion, judged).
const HAT_LEVEL: f32 = 0.15;

/// The metallic hash an analogue hat strikes: six square waves "tuned to
/// random intervals" across "around 200-800 Hz, though it's not that
/// critical" (<https://docs.cherryaudio.com/misfit-audio/transistor-808-hats>),
/// the TR-808's "six square-wave oscillators tuned enharmonically"
/// (<https://www.soundonsound.com/techniques/practical-cymbal-synthesis>).
/// These tunings are opinion: no two share a simple ratio.
const HAT_SQUARES_HZ: [f32; 6] = [207.0, 293.0, 371.0, 513.0, 593.0, 769.0];
/// The hash's gain to the level white noise had in the hat's band up to the
/// drums' cutoff, its RMS there over 32 seeds' noise.
const HAT_HASH_GAIN: f32 = 5.29;

/// The residual PolyBLEP adds near a unit step at phase `t` in `[0, 1)`, for
/// a phase step of `dt` a sample: a polynomial over the sample each side of
/// the step, which cancels most of a naive waveform's aliasing
/// (<https://metafunction.co.uk/post/all-about-digital-oscillators-part-2-blits-bleps>,
/// after Välimäki et al. 2010).
fn poly_blep(t: f32, dt: f32) -> f32 {
    if t < dt {
        let x = t / dt;
        2.0 * x - x * x - 1.0
    } else if t > 1.0 - dt {
        let x = (t - 1.0) / dt;
        x * x + 2.0 * x + 1.0
    } else {
        0.0
    }
}

/// Sample `i` of a unit square wave at `hz` at `sr`, rising at phase 0 and
/// falling at a half, each step [`poly_blep`]'d.
fn square(hz: f32, i: usize, sr: u32) -> f32 {
    let dt = hz / sr as f32;
    let t = (dt * i as f32).fract();
    let naive = if t < 0.5 { 1.0 } else { -1.0 };
    naive + poly_blep(t, dt) - poly_blep((t + 0.5).fract(), dt)
}

/// A hat's length and how fast it falls, open and closed.
const HAT_OPEN: (f32, f32) = (0.35, 9.0);
const HAT_CLOSED: (f32, f32) = (0.06, 60.0);

fn hat(open: bool, sr: u32) -> Vec<f32> {
    let (dur, decay) = if open { HAT_OPEN } else { HAT_CLOSED };
    let n = samples_at(dur, sr);
    let raw: Vec<f32> = (0..n)
        .map(|i| {
            let sum: f32 = HAT_SQUARES_HZ.iter().map(|&hz| square(hz, i, sr)).sum();
            sum * HAT_HASH_GAIN / HAT_SQUARES_HZ.len() as f32
        })
        .collect();
    let hp = highpass(&raw, HAT_HZ, sr);
    hp.iter()
        .enumerate()
        .map(|(i, hp)| hp * (-(i as f32 / sr as f32) * decay).exp() * HAT_LEVEL)
        .collect()
}

/// The velocity-keyed 2nd-harmonic EP voice (the Rhodes "bell vs bark"), shared
/// by the night events AND the day takes' sparkle/keys.
pub(super) fn ep_pluck_vel(midi: u8, dur_s: f32, vel: f32, sr: u32) -> Vec<f32> {
    ep_pluck_h2(
        midi,
        dur_s,
        vel,
        EP_VEL_H2_BASE + EP_VEL_H2_GAIN * vel * vel,
        sr,
    )
}

/// The plucked string's 2nd harmonic: its weight at rest and what velocity adds.
const PLUCK_H2_BASE: f32 = 0.30;
const PLUCK_H2_VEL: f32 = 0.30;
/// The MIDI note whose decay is the string's base decay, and how much a
/// semitone off it scales that decay.
const PLUCK_REF_MIDI: f32 = 72.0;
const PLUCK_PITCH_K: f32 = 0.02;
/// The per-note decay jitter's floor, its span and the velocity multiplier
/// that spreads velocities across it.
const PLUCK_BREATH_BASE: f32 = 0.85;
const PLUCK_BREATH_SPAN: f32 = 0.30;
const PLUCK_BREATH_MIX: f32 = 137.5;
/// The string's base decay rate, before pitch and breath.
const PLUCK_DECAY: f32 = 3.4;
/// The attack's length.
const PLUCK_ATTACK_S: f32 = 0.004;
/// The 2nd partial's fall relative to the fundamental's.
const PLUCK_H2_DECAY_K: f32 = 1.76;
/// The 3rd partial: its weight, its ratio to the note (slightly sharp of a
/// harmonic) and its fall relative to the fundamental's.
const PLUCK_H3_LEVEL: f32 = 0.18;
const PLUCK_H3_RATIO: f32 = 3.004;
const PLUCK_H3_DECAY_K: f32 = 2.94;
/// The 4th partial: its weight, its inharmonic ratio and its fixed fall rate.
const PLUCK_H4_LEVEL: f32 = 0.07;
const PLUCK_H4_RATIO: f32 = 4.21;
const PLUCK_H4_RATE: f32 = 22.0;
/// The pick's noise click.
const PLUCK_PICK: Strike = Strike {
    salt: 0,
    len_s: 0.006,
    band: (2000.0, 5000.0),
    decay: 700.0,
    level: 0.10,
};

/// The plucked-string lead voice ("nylon" family).
fn pluck_note(midi: u8, dur_s: f32, vel: f32, sr: u32) -> Vec<f32> {
    let n = samples_at(dur_s, sr);
    let f = midi_freq(f32::from(midi));
    let tau = std::f32::consts::TAU;
    let h2 = PLUCK_H2_BASE + PLUCK_H2_VEL * vel;
    // decay follows pitch (higher strings die faster) and breathes per note; the
    // velocity BITS are the jitter source, not an rng, so identical inputs still
    // render identical buffers (the purity contract)
    let pitch_k = 1.0 + (f32::from(midi) - PLUCK_REF_MIDI) * PLUCK_PITCH_K;
    let breath = PLUCK_BREATH_BASE + PLUCK_BREATH_SPAN * (vel * PLUCK_BREATH_MIX).fract();
    let d = PLUCK_DECAY * pitch_k * breath;
    let mut buf: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32 / sr as f32;
            let attack = (t / PLUCK_ATTACK_S).min(1.0);
            let sig = (tau * f * t).sin() * (-t * d).exp()
                + h2 * (tau * 2.0 * f * t).sin() * (-t * d * PLUCK_H2_DECAY_K).exp()
                + PLUCK_H3_LEVEL
                    * (tau * PLUCK_H3_RATIO * f * t).sin()
                    * (-t * d * PLUCK_H3_DECAY_K).exp()
                + PLUCK_H4_LEVEL
                    * (tau * PLUCK_H4_RATIO * f * t).sin()
                    * (-t * PLUCK_H4_RATE).exp();
            sig * attack * vel
        })
        .collect();
    strike(&mut buf, PLUCK_PICK, midi, vel, sr);
    buf
}

/// A struck note's noise: `len_s` of white noise in `band`, falling at
/// `decay` per second from `level`, seeded by the note XOR `salt`, so
/// identical notes strike identical noise and two voices' strikes differ.
#[derive(Clone, Copy)]
struct Strike {
    salt: u64,
    len_s: f32,
    band: (f32, f32),
    decay: f32,
    level: f32,
}

/// Adds `strike`'s noise for the note `(midi, vel)` to the head of `buf`.
fn strike(buf: &mut [f32], strike: Strike, midi: u8, vel: f32, sr: u32) {
    let pn = samples_at(strike.len_s, sr);
    let mut prng =
        NoiseStream::new(strike.salt ^ (u64::from(midi) << 32) ^ u64::from(vel.to_bits()));
    let white = white_noise_gain(sr);
    let raw: Vec<f32> = (0..pn).map(|_| prng.norm() * white).collect();
    let noise = bandpass_at(&raw, strike.band.0, strike.band.1, sr);
    for (i, &v) in noise.iter().enumerate() {
        if let Some(slot) = buf.get_mut(i) {
            *slot += v * (-(i as f32 / sr as f32) * strike.decay).exp() * strike.level * vel;
        }
    }
}

/// The bass's attack length, and the share of the note its release takes.
const BASS_ATTACK_S: f32 = 0.03;
const BASS_RELEASE_SHARE: f32 = 0.5;
/// The bass's drive into the tanh: at rest and what velocity adds.
const BASS_DRIVE_BASE: f32 = 1.1;
const BASS_DRIVE_VEL: f32 = 1.6;
/// The level a note settles to after its onset, how far above that it starts
/// and how fast it falls there.
const BASS_BODY_SETTLED: f32 = 0.75;
const BASS_BODY_ONSET: f32 = 0.25;
const BASS_BODY_RATE: f32 = 2.0;

/// The bass voice: sine + vel-keyed tanh harmonics, soft attack so the kick
/// keeps the transient, held-note plateau envelope.
pub(super) fn bass_note(midi: u8, dur_s: f32, vel: f32, sr: u32) -> Vec<f32> {
    let n = samples_at(dur_s, sr);
    let f = midi_freq(f32::from(midi));
    let tau = std::f32::consts::TAU;
    // half-note release: a sub tail crossing into the next root reads as mud
    let env = env_ar(n, BASS_ATTACK_S, dur_s * BASS_RELEASE_SHARE, sr);
    let drive = BASS_DRIVE_BASE + BASS_DRIVE_VEL * vel;
    let norm = drive.tanh();
    (0..n)
        .map(|i| {
            let t = i as f32 / sr as f32;
            // the note settles slightly after the onset, like a damped string
            let body = BASS_BODY_SETTLED + BASS_BODY_ONSET * (-t * BASS_BODY_RATE).exp();
            let sig = ((tau * f * t).sin() * drive).tanh() / norm;
            sig * env(i) * body * vel
        })
        .collect()
}

/// The kalimba's decay: the MIDI note whose decay is the base, how much a
/// semitone off it scales that, the floor of that scale, the per-note jitter's
/// floor, span and velocity multiplier, and the base decay rate.
const KALIMBA_REF_MIDI: f32 = 72.0;
const KALIMBA_PITCH_K: f32 = 0.025;
const KALIMBA_PITCH_K_MIN: f32 = 0.2;
const KALIMBA_BREATH_BASE: f32 = 0.9;
const KALIMBA_BREATH_SPAN: f32 = 0.2;
const KALIMBA_BREATH_MIX: f32 = 91.7;
const KALIMBA_DECAY: f32 = 2.6;
/// The attack's length.
const KALIMBA_ATTACK_S: f32 = 0.002;
/// The 2nd partial: its weight, its ratio to the note, and how its fall
/// scales with, and is offset from, the fundamental's.
const KALIMBA_H2_LEVEL: f32 = 0.20;
const KALIMBA_H2_RATIO: f32 = 5.9;
const KALIMBA_H2_DECAY_K: f32 = 5.0;
const KALIMBA_H2_DECAY_ADD: f32 = 14.0;
/// The 3rd partial: its weight, its ratio and its fixed fall rate.
const KALIMBA_H3_LEVEL: f32 = 0.08;
const KALIMBA_H3_RATIO: f32 = 16.5;
const KALIMBA_H3_RATE: f32 = 40.0;
/// The resonator body's thump.
const KALIMBA_THUMP: Strike = Strike {
    salt: 0x4B41_4C49,
    len_s: 0.010,
    band: (120.0, 700.0),
    decay: 350.0,
    level: 0.12,
};

/// The kalimba lead voice: a plucked tine — inharmonic cantilever modes near
/// 1 : 5.9 : 16.5, fast-dying overtones, a resonator-body thump.
fn kalimba_note(midi: u8, dur_s: f32, vel: f32, sr: u32) -> Vec<f32> {
    let n = samples_at(dur_s, sr);
    let f = midi_freq(f32::from(midi));
    let tau = std::f32::consts::TAU;
    // higher tines die faster; the vel BITS jitter decay per note, not an rng,
    // so identical inputs still render identical buffers (the purity contract)
    // floored, so a low note still dies away
    let pitch_k =
        (1.0 + (f32::from(midi) - KALIMBA_REF_MIDI) * KALIMBA_PITCH_K).max(KALIMBA_PITCH_K_MIN);
    let breath = KALIMBA_BREATH_BASE + KALIMBA_BREATH_SPAN * (vel * KALIMBA_BREATH_MIX).fract();
    let d = KALIMBA_DECAY * pitch_k * breath;
    let mut buf: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32 / sr as f32;
            let attack = (t / KALIMBA_ATTACK_S).min(1.0);
            let sig = (tau * f * t).sin() * (-t * d).exp()
                + KALIMBA_H2_LEVEL
                    * (tau * KALIMBA_H2_RATIO * f * t).sin()
                    * (-t * (d * KALIMBA_H2_DECAY_K + KALIMBA_H2_DECAY_ADD)).exp()
                + KALIMBA_H3_LEVEL
                    * (tau * KALIMBA_H3_RATIO * f * t).sin()
                    * (-t * KALIMBA_H3_RATE).exp();
            sig * attack * vel
        })
        .collect();
    strike(&mut buf, KALIMBA_THUMP, midi, vel, sr);
    buf
}

/// The vibraphone's decay: the MIDI note whose decay is the base, how much a
/// semitone off it scales that, and the base decay rate.
const VIBE_REF_MIDI: f32 = 69.0;
const VIBE_PITCH_K: f32 = 0.015;
const VIBE_DECAY: f32 = 1.3;
/// The fan's rate: its floor and the span the velocity bits spread it over,
/// and their multiplier.
const VIBE_TREM_BASE_HZ: f32 = 4.2;
const VIBE_TREM_SPAN_HZ: f32 = 0.8;
const VIBE_TREM_MIX: f32 = 53.1;
/// The fan's depth.
const VIBE_TREM_DEPTH: f32 = 0.175;
/// The attack's length.
const VIBE_ATTACK_S: f32 = 0.008;
/// The 2nd partial: its ratio to the note, its weight at rest and what
/// velocity adds, and how its fall scales with, and is offset from, the
/// fundamental's.
const VIBE_H2_RATIO: f32 = 4.0;
const VIBE_H2_BASE: f32 = 0.15;
const VIBE_H2_VEL: f32 = 0.20;
const VIBE_H2_DECAY_K: f32 = 3.0;
const VIBE_H2_DECAY_ADD: f32 = 2.0;
/// The 3rd partial: its weight, its ratio and its fixed fall rate.
const VIBE_H3_LEVEL: f32 = 0.06;
const VIBE_H3_RATIO: f32 = 9.8;
const VIBE_H3_RATE: f32 = 18.0;
/// The mallet's strike.
const VIBE_MALLET: Strike = Strike {
    salt: 0x5649_4245,
    len_s: 0.008,
    band: (600.0, 1800.0),
    decay: 500.0,
    level: 0.06,
};

/// The vibraphone lead voice: a struck bar tuned to 1 : 4 : ~9.8 partials,
/// long ring, the rotating-fan tremolo baked in.
fn vibe_note(midi: u8, dur_s: f32, vel: f32, sr: u32) -> Vec<f32> {
    let n = samples_at(dur_s, sr);
    let f = midi_freq(f32::from(midi));
    let tau = std::f32::consts::TAU;
    let pitch_k = 1.0 + (f32::from(midi) - VIBE_REF_MIDI) * VIBE_PITCH_K;
    let d = VIBE_DECAY * pitch_k;
    // the vel BITS detune the fan rate per note (purity contract, as above)
    let trem_hz = VIBE_TREM_BASE_HZ + VIBE_TREM_SPAN_HZ * (vel * VIBE_TREM_MIX).fract();
    let mut buf: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32 / sr as f32;
            let attack = (t / VIBE_ATTACK_S).min(1.0);
            let trem = 1.0 - VIBE_TREM_DEPTH * (1.0 - (tau * trem_hz * t).cos());
            let sig = (tau * f * t).sin() * (-t * d).exp()
                + (VIBE_H2_BASE + VIBE_H2_VEL * vel)
                    * (tau * VIBE_H2_RATIO * f * t).sin()
                    * (-t * (d * VIBE_H2_DECAY_K + VIBE_H2_DECAY_ADD)).exp()
                + VIBE_H3_LEVEL * (tau * VIBE_H3_RATIO * f * t).sin() * (-t * VIBE_H3_RATE).exp();
            sig * attack * trem * vel
        })
        .collect();
    strike(&mut buf, VIBE_MALLET, midi, vel, sr);
    buf
}

/// A pad's chord voice: how long a chord rings past its bar, its notes'
/// attack, each note's later by a step, their release, the 2nd and 3rd
/// harmonics' weights, and the voice's gain.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ChordTone {
    pub(crate) ring_s: f32,
    pub(crate) attack_s: f32,
    pub(crate) attack_step_s: f32,
    pub(crate) release_s: f32,
    pub(crate) h2: f32,
    pub(crate) h3: f32,
    pub(crate) gain: f32,
}

/// One pad chord in `tone`, held through the bar and its ring.
pub(super) fn chord(tone: ChordTone, chord: &[u8; 4], bar_s: f32, sr: u32) -> Vec<f32> {
    let tau = std::f32::consts::TAU;
    let nd = samples_at(bar_s + tone.ring_s, sr);
    let mut sig = vec![0.0f32; nd];
    for (i, &m) in chord.iter().enumerate() {
        let f = midi_freq(f32::from(m));
        let env = env_ar(
            nd,
            tone.attack_s + tone.attack_step_s * i as f32,
            tone.release_s,
            sr,
        );
        for (j, slot) in sig.iter_mut().enumerate() {
            let t = j as f32 / sr as f32;
            let partials = (tau * f * t).sin()
                + tone.h2 * (tau * 2.0 * f * t).sin()
                + tone.h3 * (tau * 3.0 * f * t).sin();
            *slot += partials * env(j) * tone.gain;
        }
    }
    sig
}

/// A kick's or snare's NOISE is fresh each call, drawn from `rng`; a hat
/// draws nothing, its hash fixed.
pub(super) fn drum_hit(rng: &mut NoiseStream, kind: DrumKind, sr: u32) -> Vec<f32> {
    match kind {
        DrumKind::Kick => kick(rng, sr),
        DrumKind::Snare => snare(rng, sr),
        DrumKind::Hat => hat(false, sr),
        DrumKind::OpenHat => hat(true, sr),
    }
}

impl LeadVoice {
    /// The voice that sings the lead.
    pub(super) fn voice(self) -> NoteVoice {
        match self {
            Self::EpVel => ep_pluck_vel,
            Self::Pluck => pluck_note,
            Self::Kalimba => kalimba_note,
            Self::Vibraphone => vibe_note,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::{SAMPLE_RATE, band_energy_share};
    use crate::synth::{ROLLOFF_HZ, n_samples};

    #[test]
    fn new_lead_voices_are_pure_finite_and_keep_their_decay_character() {
        let rms = |s: &[f32]| (s.iter().map(|v| v * v).sum::<f32>() / s.len() as f32).sqrt();
        let tail_ratio = |b: &[f32]| {
            let q = b.len() / 4;
            rms(&b[b.len() - q..]) / rms(&b[..q]).max(1e-9)
        };
        let (kal, vib) = (
            kalimba_note(72, 2.0, 0.5, SAMPLE_RATE),
            vibe_note(72, 2.0, 0.5, SAMPLE_RATE),
        );
        for (name, buf) in [("kalimba", &kal), ("vibe", &vib)] {
            assert_eq!(buf.len(), n_samples(2.0), "{name}: wrong note length");
            assert!(buf.iter().all(|v| v.is_finite()), "{name}: non-finite");
            assert!(rms(buf) > 1e-4, "{name}: silent note");
        }
        // purity contract: identical inputs render identical buffers
        assert_eq!(kal, kalimba_note(72, 2.0, 0.5, SAMPLE_RATE));
        assert_eq!(vib, vibe_note(72, 2.0, 0.5, SAMPLE_RATE));
        // the tine dies fast, the bar rings — that ordering IS the two timbres
        assert!(
            tail_ratio(&vib) > 2.0 * tail_ratio(&kal),
            "vibe tail {} must ring well past the kalimba tail {}",
            tail_ratio(&vib),
            tail_ratio(&kal)
        );
    }

    #[test]
    fn a_voice_at_48_khz_peaks_where_it_does_at_the_ratified_rate() {
        const RATE: u32 = 48_000;
        // Goertzel's power at `hz`
        // (<https://en.wikipedia.org/wiki/Goertzel_algorithm#Power-spectrum_terms>)
        let power = |buf: &[f32], sr: u32, hz: f64| {
            let c = 2.0 * (std::f64::consts::TAU * hz / f64::from(sr)).cos();
            let (s1, s2) = buf
                .iter()
                .fold((0.0, 0.0), |(s1, s2), &x| (f64::from(x) + c * s1 - s2, s1));
            s1 * s1 + s2 * s2 - c * s1 * s2
        };
        let peak_hz = |buf: &[f32], sr: u32, (lo, hi): (f64, f64)| {
            let bins = ((hi - lo) * 4.0) as usize;
            (0..=bins)
                .map(|k| lo + k as f64 / 4.0)
                .max_by(|&a, &b| power(buf, sr, a).total_cmp(&power(buf, sr, b)))
                .unwrap()
        };
        // each wide enough to hold the pitch a voice that kept counting
        // `SAMPLE_RATE` would sound at `RATE`
        let a4 = (400.0, 520.0);
        let notes: [(&str, NoteVoice); 5] = [
            ("ep", ep_pluck_vel),
            ("pluck", pluck_note),
            ("kalimba", kalimba_note),
            ("vibe", vibe_note),
            ("bass", bass_note),
        ];
        type Render = Box<dyn Fn(u32) -> Vec<f32>>;
        let mut voices: Vec<(String, Render, (f64, f64))> = Vec::new();
        for (name, voice) in notes {
            voices.push((name.into(), Box::new(move |sr| voice(69, 1.0, 0.8, sr)), a4));
        }
        for style in crate::style::STYLES {
            let tone = style.sound.chord;
            let render = move |sr| chord(tone, &[69; 4], 1.0, sr);
            voices.push((format!("{style:?} chord"), Box::new(render), a4));
        }
        for (name, kind, band) in [
            ("kick", DrumKind::Kick, (30.0, 200.0)),
            ("snare", DrumKind::Snare, (150.0, 230.0)),
        ] {
            let hit = move |sr| drum_hit(&mut NoiseStream::new(3), kind, sr);
            voices.push((name.into(), Box::new(hit), band));
        }
        for (name, render, band) in voices {
            let want = peak_hz(&render(SAMPLE_RATE), SAMPLE_RATE, band);
            let got = peak_hz(&render(RATE), RATE, band);
            assert!(
                (got - want).abs() <= 1.0,
                "{name}: {got} Hz at {RATE} vs {want} Hz"
            );
        }
    }

    /// A square at 2.5 kHz folds its harmonics past Nyquist onto 1.6, 6.6 and
    /// 11.6 kHz, each 0.9 kHz under one of its own, further than
    /// `HALF_WIDTH_HZ`: PolyBLEP keeps them 20 dB under the naive square's
    /// there (opinion).
    #[test]
    fn a_squares_aliases_stay_under_a_naive_ones() {
        const HZ: f32 = 2500.0;
        const ALIASES_HZ: [f32; 3] = [1600.0, 6600.0, 11600.0];
        const HALF_WIDTH_HZ: f32 = 300.0;
        const MARGIN_DB: f32 = 20.0;
        let sr = SAMPLE_RATE;
        let n = 1 << 16;
        let blep: Vec<f32> = (0..n).map(|i| square(HZ, i, sr)).collect();
        let naive: Vec<f32> = (0..n)
            .map(|i| {
                if (HZ / sr as f32 * i as f32).fract() < 0.5 {
                    1.0
                } else {
                    -1.0
                }
            })
            .collect();
        let aliased = |buf: &[f32]| -> f32 {
            (ALIASES_HZ.iter())
                .map(|&hz| band_energy_share(buf, hz - HALF_WIDTH_HZ, hz + HALF_WIDTH_HZ))
                .sum()
        };
        let under_db = 10.0 * (aliased(&naive) / aliased(&blep)).log10();
        assert!(
            under_db > MARGIN_DB,
            "aliases {under_db:.1} dB under the naive square's"
        );
    }

    /// The hat's hash spans an octave under the lowest corner the drums'
    /// chain low-passes at: a hat that starts at the corner, as one did,
    /// plays nothing the chain lets through.
    #[test]
    fn a_hat_sounds_under_the_drums_chain() {
        let corner = (crate::style::STYLES.iter())
            .map(|style| style.sound.drums.cutoff_hz)
            .fold(ROLLOFF_HZ, f32::min);
        assert!(2.0 * HAT_HZ <= corner, "{HAT_HZ} Hz under {corner} Hz");
    }
}
