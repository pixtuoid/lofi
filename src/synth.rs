//! The procedural sound recipes. Every constant is an owner-ratified value:
//! change it through an A/B batch (`examples/lofi_audition.rs`) and
//! `ab_judge.py`'s verdict, which the commit records. The voices render into
//! mono f32 buffers, FFT-backed, never inside an audio callback, since
//! callers render ahead: a voice renders one note at the rate it is given as
//! the engine reaches it, and [`render_lane`] a whole lane at
//! [`SAMPLE_RATE`] for lofi's own tools. The engine plays the same recipe
//! through `stream::lanes`' causal chains.

use super::dsp::{
    NoiseStream, SAMPLE_RATE, bandpass, bandpass_at, highpass, lowpass, white_noise_gain,
};

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

fn normalize(buf: &mut [f32], peak: f32) {
    let m = buf.iter().fold(0.0f32, |a, &v| a.max(v.abs())).max(1e-9);
    buf.iter_mut().for_each(|v| *v *= peak / m);
}

/// A voice that plays one `(midi, dur_s, vel)` note at a sample rate.
type NoteVoice = fn(u8, f32, f32, u32) -> Vec<f32>;

const MIDI_A4: f32 = 440.0;
fn midi_freq(m: f32) -> f32 {
    MIDI_A4 * 2f32.powf((m - 69.0) / 12.0)
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
pub(crate) fn texture_bed(rng: &mut NoiseStream) -> Vec<f32> {
    let n = BED_LOOP_SAMPLES;
    let mut buf = room_bed(n, n, rng);
    normalize(&mut buf, TEXTURE_PEAK);
    buf
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

// The `#[cfg(test)]` stems below are fingerprint ANCHORS, not dead code: their
// pins prove the voices and cores `render_lane` shares still render the ratified sound.

use super::compose::{DrumKind, Hit, Note};
#[cfg(test)]
use super::{compose::BEATS_PER_BAR, score};

#[cfg(test)]
fn beat_s() -> f32 {
    score::beat_s()
}

#[cfg(test)]
fn bar_s() -> f32 {
    score::beat_s() * BEATS_PER_BAR
}

/// Tape wow and flutter, `(rate_hz, pitch deviation)` pairs as
/// [`warp_resample`](crate::dsp::warp_resample) takes them.
pub(crate) const WARPS: [(f32, f32); 2] = [WOW, (8.0, 0.0006)];

/// The slow, deep warp of [`WARPS`], named apart so what leans on it finds
/// it by name, not by place.
pub(crate) const WOW: (f32, f32) = (0.7, 0.0025);

/// The tape drive of every musical lane but the drums, which [`DRUMS`]
/// gives their own.
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

/// ONE site, so a stem can't silently skip the ratified post.
fn master(buf: &[f32], drive: f32, peak: f32) -> Vec<f32> {
    let mut out = lofi_post(buf, drive);
    normalize(&mut out, peak);
    out
}

/// One soft electric-piano note; `h2` is the 2nd-harmonic weight — the one axis
/// the day and night voices differ on.
fn ep_pluck_h2(midi: u8, dur_s: f32, vel: f32, h2: f32, sr: u32) -> Vec<f32> {
    let n = samples_at(dur_s, sr);
    let f = midi_freq(f32::from(midi));
    let tau = std::f32::consts::TAU;
    (0..n)
        .map(|i| {
            let t = i as f32 / sr as f32;
            let sig = (tau * f * t).sin() * (-t * 5.5).exp()
                + h2 * (tau * 2.0 * f * t).sin() * (-t * 9.0).exp()
                + 0.12 * (tau * 3.01 * f * t).sin() * (-t * 14.0).exp();
            sig * vel
        })
        .collect()
}

/// The day-1 anchor's EP, Phase 0's: a fixed 2nd harmonic, where the voice
/// that ships keys it to the velocity.
#[cfg(test)]
fn ep_pluck(midi: u8, dur_s: f32, vel: f32, sr: u32) -> Vec<f32> {
    ep_pluck_h2(midi, dur_s, vel, 0.35, sr)
}

/// How long the day-1 anchor's lead notes ring, where the shipped day lead's
/// ring [`SPARKLE`]'s.
#[cfg(test)]
const ANCHOR_LEAD_DUR_S: f32 = 1.6;

/// The day-1 anchor's `(beats, midi, vel)` rows as the notes its cores
/// place.
#[cfg(test)]
fn anchor_events(rows: &[(f32, u8, f32)]) -> Vec<Note> {
    (rows.iter())
        .map(|&(beats, midi, vel)| Note {
            at: beats * beat_s(),
            midi,
            vel,
        })
        .collect()
}

fn kick(rng: &mut NoiseStream, sr: u32) -> Vec<f32> {
    let n = samples_at(0.32, sr);
    let tau = std::f32::consts::TAU;
    let mut phase = 0.0f32;
    let mut buf: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32 / sr as f32;
            let f = 110.0 * (-t * 9.0).exp() + 42.0;
            phase += tau * f / sr as f32;
            phase.sin() * (-t * 11.0).exp()
        })
        .collect();
    let white = white_noise_gain(sr);
    for (i, slot) in buf.iter_mut().enumerate().take(samples_at(0.006, sr)) {
        let t = i as f32 / sr as f32;
        *slot += rng.norm() * white * 0.25 * (-t * 300.0).exp();
    }
    buf
}

fn snare(rng: &mut NoiseStream, sr: u32) -> Vec<f32> {
    let n = samples_at(0.22, sr);
    let white = white_noise_gain(sr);
    let raw: Vec<f32> = (0..n).map(|_| rng.norm() * white).collect();
    let noise = bandpass_at(&raw, 400.0, 3200.0, sr);
    let tau = std::f32::consts::TAU;
    noise
        .iter()
        .enumerate()
        .map(|(i, noise)| {
            let t = i as f32 / sr as f32;
            let tone = (tau * 185.0 * t).sin() * (-t * 25.0).exp() * 0.5;
            (noise * (-t * 22.0).exp() + tone) * 0.8
        })
        .collect()
}

/// Where a hat's hash starts: the bottom of the 3–6 kHz band LOFI-BIBLE.md
/// §3 low-passes the drums into, so the hat sounds under the drums' cutoff
/// and the tape's [`ROLLOFF_HZ`].
const HAT_HZ: f32 = 3000.0;
/// A hat's level under the kick and snare: 6 dB under the one that played
/// above the chain, now that its band sounds (opinion, judged).
const HAT_LEVEL: f32 = 0.25;

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

fn hat(open: bool, sr: u32) -> Vec<f32> {
    let (dur, decay) = if open { (0.35, 9.0) } else { (0.06, 60.0) };
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

#[cfg(test)]
pub(crate) fn stem_pad() -> Vec<f32> {
    let chords: [[u8; 4]; GEN_LOOP_BARS] = std::array::from_fn(score::chord_at_bar);
    let pad = Part::Chords {
        chords: &chords,
        bar_s: bar_s(),
        mood: Mood::Day,
    };
    render_part(&pad, score::loop_secs(), &mut NoiseStream::new(0))
}

#[cfg(test)]
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
}

#[cfg(test)]
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
}

/// The day-1 anchor's kick/snare/swung-hat groove, its hat velocities the
/// frozen score's, through the shipped drums' core; the kick's and snare's
/// NOISE is fresh each hit, and a hat is the fixed metallic hash.
#[cfg(test)]
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
    drums_core(score::loop_secs(), &hits, r.cutoff_hz, r.drive, r.peak, rng)
}

/// The velocity-keyed 2nd-harmonic EP voice (the Rhodes "bell vs bark"), shared
/// by the night events AND the day takes' sparkle/keys.
fn ep_pluck_vel(midi: u8, dur_s: f32, vel: f32, sr: u32) -> Vec<f32> {
    ep_pluck_h2(midi, dur_s, vel, 0.18 + 0.45 * vel * vel, sr)
}

/// The plucked-string lead voice ("nylon" family).
fn pluck_note(midi: u8, dur_s: f32, vel: f32, sr: u32) -> Vec<f32> {
    let n = samples_at(dur_s, sr);
    let f = midi_freq(f32::from(midi));
    let tau = std::f32::consts::TAU;
    let h2 = 0.30 + 0.30 * vel;
    // decay follows pitch (higher strings die faster) and breathes per note; the
    // velocity BITS are the jitter source, not an rng, so identical inputs still
    // render identical buffers (the purity contract)
    let pitch_k = 1.0 + (f32::from(midi) - 72.0) * 0.02;
    let breath = 0.85 + 0.30 * (vel * 137.5).fract();
    let d = 3.4 * pitch_k * breath;
    let mut buf: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32 / sr as f32;
            let attack = (t / 0.004).min(1.0);
            let sig = (tau * f * t).sin() * (-t * d).exp()
                + h2 * (tau * 2.0 * f * t).sin() * (-t * d * 1.76).exp()
                + 0.18 * (tau * 3.004 * f * t).sin() * (-t * d * 2.94).exp()
                + 0.07 * (tau * 4.21 * f * t).sin() * (-t * 22.0).exp();
            sig * attack * vel
        })
        .collect();
    let pick = Strike {
        salt: 0,
        len_s: 0.006,
        band: (2000.0, 5000.0),
        decay: 700.0,
        level: 0.10,
    };
    strike(&mut buf, pick, midi, vel, sr);
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

/// The bass voice: sine + vel-keyed tanh harmonics, soft attack so the kick
/// keeps the transient, held-note plateau envelope.
fn bass_note(midi: u8, dur_s: f32, vel: f32, sr: u32) -> Vec<f32> {
    let n = samples_at(dur_s, sr);
    let f = midi_freq(f32::from(midi));
    let tau = std::f32::consts::TAU;
    // half-note release: a sub tail crossing into the next root reads as mud
    let env = env_ar(n, 0.03, dur_s * 0.5, sr);
    let drive = 1.1 + 1.6 * vel;
    let norm = drive.tanh();
    (0..n)
        .map(|i| {
            let t = i as f32 / sr as f32;
            // the note settles slightly after the onset, like a damped string
            let body = 0.75 + 0.25 * (-t * 2.0).exp();
            let sig = ((tau * f * t).sin() * drive).tanh() / norm;
            sig * env(i) * body * vel
        })
        .collect()
}

/// The kalimba lead voice: a plucked tine — inharmonic cantilever modes near
/// 1 : 5.9 : 16.5, fast-dying overtones, a resonator-body thump.
fn kalimba_note(midi: u8, dur_s: f32, vel: f32, sr: u32) -> Vec<f32> {
    let n = samples_at(dur_s, sr);
    let f = midi_freq(f32::from(midi));
    let tau = std::f32::consts::TAU;
    // higher tines die faster; the vel BITS jitter decay per note, not an rng,
    // so identical inputs still render identical buffers (the purity contract)
    // floored, so a low note still dies away
    let pitch_k = (1.0 + (f32::from(midi) - 72.0) * 0.025).max(0.2);
    let breath = 0.9 + 0.2 * (vel * 91.7).fract();
    let d = 2.6 * pitch_k * breath;
    let mut buf: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32 / sr as f32;
            let attack = (t / 0.002).min(1.0);
            let sig = (tau * f * t).sin() * (-t * d).exp()
                + 0.20 * (tau * 5.9 * f * t).sin() * (-t * (d * 5.0 + 14.0)).exp()
                + 0.08 * (tau * 16.5 * f * t).sin() * (-t * 40.0).exp();
            sig * attack * vel
        })
        .collect();
    let thump = Strike {
        salt: 0x4B41_4C49,
        len_s: 0.010,
        band: (120.0, 700.0),
        decay: 350.0,
        level: 0.12,
    };
    strike(&mut buf, thump, midi, vel, sr);
    buf
}

/// The vibraphone lead voice: a struck bar tuned to 1 : 4 : ~9.8 partials,
/// long ring, the rotating-fan tremolo baked in.
fn vibe_note(midi: u8, dur_s: f32, vel: f32, sr: u32) -> Vec<f32> {
    let n = samples_at(dur_s, sr);
    let f = midi_freq(f32::from(midi));
    let tau = std::f32::consts::TAU;
    let pitch_k = 1.0 + (f32::from(midi) - 69.0) * 0.015;
    let d = 1.3 * pitch_k;
    // the vel BITS detune the fan rate per note (purity contract, as above)
    let trem_hz = 4.2 + 0.8 * (vel * 53.1).fract();
    let mut buf: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32 / sr as f32;
            let attack = (t / 0.008).min(1.0);
            let trem = 1.0 - 0.175 * (1.0 - (tau * trem_hz * t).cos());
            let sig = (tau * f * t).sin() * (-t * d).exp()
                + (0.15 + 0.20 * vel) * (tau * 4.0 * f * t).sin() * (-t * (d * 3.0 + 2.0)).exp()
                + 0.06 * (tau * 9.8 * f * t).sin() * (-t * 18.0).exp();
            sig * attack * trem * vel
        })
        .collect();
    let mallet = Strike {
        salt: 0x5649_4245,
        len_s: 0.008,
        band: (600.0, 1800.0),
        decay: 500.0,
        level: 0.06,
    };
    strike(&mut buf, mallet, midi, vel, sr);
    buf
}

#[cfg(test)]
fn night_bar_s() -> f32 {
    score::night_beat_s() * BEATS_PER_BAR
}

/// The FROZEN night pad's baked sub level. The ratified v4 take renders the
/// chords and the sub-bass floor in ONE buffer.
#[cfg(test)]
const NIGHT_PAD_SUB_GAIN: f32 = 3.2;

/// Night pad: chords + the SUB-BASS floor in ONE buffer, so the harmonic floor
/// moves as one — the FROZEN anchor predates the bass stem and keeps its pins.
#[cfg(test)]
pub(crate) fn night_pad() -> Vec<f32> {
    let (chords, bar_s) = (&score::NIGHT_CHORDS, night_bar_s());
    pad_core(Mood::Night, bar_s, score::NIGHT_LOOP_BARS, |bar| {
        let tone = ChordTone::of(Mood::Night);
        let mut sig = chord(tone, &chords[bar % chords.len()], bar_s, SAMPLE_RATE);
        let roots = &score::NIGHT_BASS_ROOTS;
        bake_night_sub(&mut sig, roots[bar % roots.len()], bar_s);
        sig
    })
}

/// The FROZEN v4 anchor's sub floor under one bar's chord in `sig`: the bar's
/// root twice, the second half softer.
#[cfg(test)]
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

/// A pad's chord voice: how long a chord rings past its bar, its notes'
/// attack, each note's later by a step, their release, the 2nd and 3rd
/// harmonics' weights, and the voice's gain.
#[derive(Clone, Copy, Debug)]
struct ChordTone {
    ring_s: f32,
    attack_s: f32,
    attack_step_s: f32,
    release_s: f32,
    h2: f32,
    h3: f32,
    gain: f32,
}

impl ChordTone {
    const fn of(mood: Mood) -> Self {
        match mood {
            Mood::Day => Self {
                ring_s: 0.9,
                attack_s: 0.25,
                attack_step_s: 0.08,
                release_s: 1.2,
                h2: 0.30,
                h3: 0.08,
                gain: 1.0,
            },
            Mood::Night => Self {
                ring_s: 1.2,
                attack_s: 0.5,
                attack_step_s: 0.1,
                release_s: 1.8,
                h2: 0.22,
                h3: 0.05,
                gain: 0.8,
            },
        }
    }
}

/// One pad chord in `tone`, held through the bar and its ring.
fn chord(tone: ChordTone, chord: &[u8; 4], bar_s: f32, sr: u32) -> Vec<f32> {
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

/// A `mood` pad over `loop_bars` bars `bar_s` long, `bar` giving each bar's
/// sound.
fn pad_core(mood: Mood, bar_s: f32, loop_bars: usize, bar: impl Fn(usize) -> Vec<f32>) -> Vec<f32> {
    let mut buf = vec![0.0f32; n_samples(bar_s * loop_bars as f32)];
    for b in 0..loop_bars {
        place(&mut buf, &bar(b), b as f32 * bar_s, 1.0);
    }
    let tone = PadTone::of(mood);
    let mut buf = lowpass(&buf, tone.cutoff_hz);
    for (i, v) in buf.iter_mut().enumerate() {
        *v *= tone.bed_swell(i);
    }
    normalize(&mut buf, tone.peak);
    master(&buf, DRIVE, tone.peak)
}

/// A note lane's stem over a loop `loop_secs` long: each of `events` through
/// `note_fn`, then the stem's low-pass, its peak and the tape.
fn events_stem_voiced(
    loop_secs: f32,
    events: &[Note],
    dur_s: f32,
    cutoff_hz: f32,
    peak: f32,
    note_fn: NoteVoice,
) -> Vec<f32> {
    let mut buf = vec![0.0f32; n_samples(loop_secs)];
    for n in events {
        place(
            &mut buf,
            &note_fn(n.midi, dur_s, n.vel, SAMPLE_RATE),
            n.at,
            1.0,
        );
    }
    let mut buf = lowpass(&buf, cutoff_hz);
    normalize(&mut buf, peak);
    master(&buf, DRIVE, peak)
}

/// The frozen night take's `part` through the shipped whole-lane core.
#[cfg(test)]
fn night_part(part: &Part<'_>, rng: &mut NoiseStream) -> Vec<f32> {
    render_part(part, score::night_loop_secs(), rng)
}

#[cfg(test)]
pub(crate) fn night_keys() -> Vec<f32> {
    let keys = Part::Notes {
        events: &score::NIGHT_KEYS,
        voice: ep_pluck_vel,
        recipes: KEYS,
        mood: Mood::Night,
    };
    night_part(&keys, &mut NoiseStream::new(0))
}

#[cfg(test)]
pub(crate) fn night_sparkle() -> Vec<f32> {
    let sparkle = Part::Notes {
        events: &score::NIGHT_SPARKLE,
        voice: ep_pluck_vel,
        recipes: SPARKLE,
        mood: Mood::Night,
    };
    night_part(&sparkle, &mut NoiseStream::new(0))
}

/// A kick's or snare's NOISE is fresh each call, drawn from `rng`; a hat
/// draws nothing, its hash fixed.
fn drum_hit(rng: &mut NoiseStream, kind: DrumKind, sr: u32) -> Vec<f32> {
    match kind {
        DrumKind::Kick => kick(rng, sr),
        DrumKind::Snare => snare(rng, sr),
        DrumKind::Hat => hat(false, sr),
        DrumKind::OpenHat => hat(true, sr),
    }
}

#[cfg(test)]
pub(crate) fn night_drums(rng: &mut NoiseStream) -> Vec<f32> {
    let drums = Part::Hits {
        hits: &score::NIGHT_DRUMS,
        recipes: DRUMS,
        mood: Mood::Night,
    };
    night_part(&drums, rng)
}

fn drums_core(
    loop_secs: f32,
    table: &[Hit],
    cutoff_hz: f32,
    drive: f32,
    peak: f32,
    rng: &mut NoiseStream,
) -> Vec<f32> {
    let mut buf = vec![0.0f32; n_samples(loop_secs)];
    for h in table {
        let hit = drum_hit(rng, h.kind, SAMPLE_RATE);
        place(&mut buf, &hit, h.at, h.gain);
    }
    let mut buf = lowpass(&buf, cutoff_hz);
    normalize(&mut buf, peak);
    master(&buf, drive, peak)
}

/// The night texture's duck under each kick: down this far, then back up
/// over this long.
pub(crate) const DUCK_DB: f32 = -4.0;
pub(crate) const DUCK_RELEASE_S: f32 = 0.15;

/// Loop-seam splice length: the night texture's buffer is the night loop length,
/// NOT a power of two, so its FFT-shaped hiss is not circularly seamless.
const NIGHT_TEXTURE_SPLICE_S: f32 = 0.03;

/// The night texture: the day room-tone recipe at the NIGHT loop length with
/// the kick-duck BAKED IN, so the whole lane ducks in phase with its drums by
/// construction.
#[cfg(test)]
pub(crate) fn night_texture(rng: &mut NoiseStream) -> Vec<f32> {
    night_part(&NIGHT_BED, rng)
}

/// The frozen night take's bed, ducking under its drums.
#[cfg(test)]
const NIGHT_BED: Part<'static> = Part::Bed {
    mood: Mood::Night,
    hits: &score::NIGHT_DRUMS,
};

fn night_texture_core(loop_secs: f32, kick_times: &[f32], rng: &mut NoiseStream) -> Vec<f32> {
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
    // normalize BEFORE the duck: duck-then-normalize re-inflates the level by up
    // to the duck depth
    normalize(&mut buf, TEXTURE_PEAK);
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
    buf
}

/// A frozen day take's `part` through the shipped whole-lane core.
#[cfg(test)]
fn day_take_part(take: &score::DayTake, part: &Part<'_>, rng: &mut NoiseStream) -> Vec<f32> {
    render_part(part, take.loop_secs(), rng)
}

#[cfg(test)]
pub(super) fn day_take_pad(take: &score::DayTake) -> Vec<f32> {
    let chords: [[u8; 4]; GEN_LOOP_BARS] = std::array::from_fn(|bar| take.chord_at_bar(bar));
    let pad = Part::Chords {
        chords: &chords,
        bar_s: take.bar_s(),
        mood: Mood::Day,
    };
    day_take_part(take, &pad, &mut NoiseStream::new(0))
}

/// A generated pad's tone before the tape: a low-pass, then a slow swell.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PadTone {
    pub(crate) cutoff_hz: f32,
    pub(crate) swell_hz: f32,
    swell_depth: f32,
    /// Its whole-lane stem's peak.
    peak: f32,
}

impl PadTone {
    pub(crate) fn of(mood: Mood) -> Self {
        match mood {
            Mood::Day => Self {
                cutoff_hz: 2600.0,
                swell_hz: 0.22,
                swell_depth: 0.05,
                peak: 0.7,
            },
            Mood::Night => Self {
                cutoff_hz: 2200.0,
                swell_hz: 0.15,
                swell_depth: 0.04,
                peak: 0.75,
            },
        }
    }

    /// The swell's gain at its phase in radians, which the caller's clock
    /// keeps as [`room_tone`]'s does.
    pub(crate) fn swell(self, phase: f32) -> f32 {
        1.0 + self.swell_depth * phase.sin()
    }

    /// The swell's gain on sample `i` of a whole-lane pad.
    fn bed_swell(self, i: usize) -> f32 {
        self.swell(std::f32::consts::TAU * self.swell_hz * i as f32 / SR)
    }
}

#[cfg(test)]
pub(super) fn day_take_sparkle(take: &score::DayTake) -> Vec<f32> {
    let sparkle = Part::Notes {
        events: take.sparkle,
        voice: ep_pluck_vel,
        recipes: SPARKLE,
        mood: Mood::Day,
    };
    day_take_part(take, &sparkle, &mut NoiseStream::new(0))
}

#[cfg(test)]
pub(super) fn day_take_keys(take: &score::DayTake) -> Vec<f32> {
    let keys = Part::Notes {
        events: take.keys,
        voice: ep_pluck_vel,
        recipes: KEYS,
        mood: Mood::Day,
    };
    day_take_part(take, &keys, &mut NoiseStream::new(0))
}

#[cfg(test)]
pub(super) fn day_take_drums(take: &score::DayTake, rng: &mut NoiseStream) -> Vec<f32> {
    let drums = Part::Hits {
        hits: take.drums,
        recipes: DRUMS,
        mood: Mood::Day,
    };
    day_take_part(take, &drums, rng)
}

use super::compose::{GEN_LOOP_BARS, Lane, LeadVoice, Mood, Score};

/// The bright master's one-pole tilt shelf: its corner, and how far above
/// it the top lifts.
pub(crate) const TILT_HZ: f32 = 1200.0;
pub(crate) const TILT_LIFT: f32 = 0.8;

/// The "bright" master for GENERATED takes: a one-pole tilt shelf, then
/// re-normalize to the bed's pre-tilt peak. FROZEN takes keep their darker
/// chain — brightening them would invalidate their fingerprint pins.
fn brighten(buf: &mut [f32]) {
    let pre_peak = buf.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    let a = (-std::f32::consts::TAU * TILT_HZ / SR).exp();
    let mut lp = 0.0f32;
    for s in buf.iter_mut() {
        lp = a * lp + (1.0 - a) * *s;
        *s += TILT_LIFT * (*s - lp);
    }
    normalize(buf, pre_peak.max(1e-6));
}

impl LeadVoice {
    /// The voice that sings the lead.
    fn voice(self) -> NoteVoice {
        match self {
            Self::EpVel => ep_pluck_vel,
            Self::Pluck => pluck_note,
            Self::Kalimba => kalimba_note,
            Self::Vibraphone => vibe_note,
        }
    }
}

/// A lane's recipe at night and by day.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ByMood<T> {
    pub(crate) night: T,
    pub(crate) day: T,
}

impl<T: Copy> ByMood<T> {
    /// The recipe a `mood` song plays.
    pub(crate) const fn at(&self, mood: Mood) -> T {
        match mood {
            Mood::Night => self.night,
            Mood::Day => self.day,
        }
    }
}

/// A note lane's recipe: how long each note rings, its stem's low-pass corner
/// and its stem's peak.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct NoteRecipe {
    pub(crate) dur_s: f32,
    pub(crate) cutoff_hz: f32,
    pub(crate) peak: f32,
}

/// The drums' recipe: their stem's low-pass corner, the tape's drive and the
/// stem's peak.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct HitRecipe {
    pub(crate) cutoff_hz: f32,
    pub(crate) drive: f32,
    pub(crate) peak: f32,
}

/// Each lane's recipe: the ONE authority [`render_lane`] and the lane chains
/// read.
pub(crate) const SPARKLE: ByMood<NoteRecipe> = ByMood {
    night: NoteRecipe {
        dur_s: 2.2,
        cutoff_hz: 2800.0,
        peak: 0.5,
    },
    day: NoteRecipe {
        dur_s: 2.0,
        cutoff_hz: 3200.0,
        peak: 0.6,
    },
};
pub(crate) const KEYS: ByMood<NoteRecipe> = ByMood {
    night: NoteRecipe {
        dur_s: 1.1,
        cutoff_hz: 2000.0,
        peak: 0.7,
    },
    day: NoteRecipe {
        dur_s: 0.9,
        cutoff_hz: 2400.0,
        peak: 0.8,
    },
};
pub(crate) const BASS: ByMood<NoteRecipe> = ByMood {
    night: NoteRecipe {
        dur_s: 1.7,
        cutoff_hz: 700.0,
        peak: 0.8,
    },
    day: NoteRecipe {
        dur_s: 1.5,
        cutoff_hz: 900.0,
        peak: 0.8,
    },
};
pub(crate) const DRUMS: ByMood<HitRecipe> = ByMood {
    night: HitRecipe {
        cutoff_hz: 6000.0,
        drive: 2.0,
        peak: 0.8,
    },
    day: HitRecipe {
        cutoff_hz: 7500.0,
        drive: 2.2,
        peak: 0.85,
    },
};

/// What one lane of a score plays: the ONE mapping from a lane to its events
/// and voice, read by [`render_lane`] and by the stream alike.
pub(crate) enum Part<'a> {
    /// A chord a bar, each held through the bar and its ring.
    Chords {
        chords: &'a [[u8; 4]; GEN_LOOP_BARS],
        bar_s: f32,
        mood: Mood,
    },
    /// Notes, played through `voice` as `recipes` gives a `mood` song.
    Notes {
        events: &'a [Note],
        voice: NoteVoice,
        recipes: ByMood<NoteRecipe>,
        mood: Mood,
    },
    /// Drum hits, played as `recipes` gives a `mood` song.
    Hits {
        hits: &'a [Hit],
        recipes: ByMood<HitRecipe>,
        mood: Mood,
    },
    /// The texture: a bed with no notes, which [`Part::ducks`] under the
    /// kicks of `hits`.
    Bed { mood: Mood, hits: &'a [Hit] },
}

pub(crate) fn part(score: &Score, lane: Lane) -> Part<'_> {
    let mood = score.mood;
    match lane {
        Lane::Pad => Part::Chords {
            chords: &score.bar_chords,
            bar_s: score.bar_s(),
            mood,
        },
        Lane::Sparkle => Part::Notes {
            events: &score.sparkle,
            voice: score.lead_voice.voice(),
            recipes: SPARKLE,
            mood,
        },
        Lane::Keys => Part::Notes {
            events: &score.keys,
            voice: ep_pluck_vel,
            recipes: KEYS,
            mood,
        },
        Lane::Bass => Part::Notes {
            events: &score.bass,
            voice: bass_note,
            recipes: BASS,
            mood,
        },
        Lane::Drums => Part::Hits {
            hits: &score.drums,
            recipes: DRUMS,
            mood,
        },
        Lane::Texture => Part::Bed {
            mood,
            hits: &score.drums,
        },
    }
}

impl Part<'_> {
    /// Each note's first sample in the loop at `sr`, in event order: where
    /// [`render_lane`] places it at [`SAMPLE_RATE`].
    pub(crate) fn onsets(&self, sr: u32) -> Vec<usize> {
        match *self {
            Part::Chords { chords, bar_s, .. } => (0..chords.len())
                .map(|bar| bar_onset(bar, bar_s, sr))
                .collect(),
            Part::Notes { events, .. } => events.iter().map(|n| onset(n.at, sr)).collect(),
            Part::Hits { hits, .. } => hits.iter().map(|h| onset(h.at, sr)).collect(),
            Part::Bed { .. } => Vec::new(),
        }
    }

    /// The times in the loop a bed ducks under, in order: a night bed's
    /// kicks, so the night room breathes with its groove, while the day room
    /// runs free. Nothing else ducks.
    pub(crate) fn ducks(&self) -> Vec<f32> {
        match *self {
            Part::Bed {
                mood: Mood::Night,
                hits,
            } => {
                let mut kicks: Vec<f32> = (hits.iter())
                    .filter(|h| h.kind == DrumKind::Kick)
                    .map(|h| h.at)
                    .collect();
                kicks.sort_by(f32::total_cmp);
                kicks
            }
            _ => Vec::new(),
        }
    }

    /// Note `i` of [`onsets`](Self::onsets) at `sr`, and the gain it mixes in
    /// at; a hit draws its noise from `rng`.
    pub(crate) fn note(&self, i: usize, rng: &mut NoiseStream, sr: u32) -> (Vec<f32>, f32) {
        match *self {
            Part::Chords {
                chords,
                bar_s,
                mood,
            } => (chord(ChordTone::of(mood), &chords[i], bar_s, sr), 1.0),
            Part::Notes {
                events,
                voice,
                recipes,
                mood,
            } => {
                let n = events[i];
                (voice(n.midi, recipes.at(mood).dur_s, n.vel, sr), 1.0)
            }
            Part::Hits { hits, .. } => {
                let h = hits[i];
                (drum_hit(rng, h.kind, sr), h.gain)
            }
            Part::Bed { .. } => (Vec::new(), 1.0),
        }
    }
}

/// Render one lane of `score`'s loop to mono f32 at [`SAMPLE_RATE`], peaking
/// at most 1.0, for lofi's tools to audition. Every lane but the day
/// [`Lane::Texture`] is a whole number of the score's loops long, so a caller
/// loops them in phase; the day texture runs free on its own length.
pub fn render_lane(score: &Score, lane: Lane, rng: &mut NoiseStream) -> Vec<f32> {
    let mut bed = render_part(&part(score, lane), score.loop_secs(), rng);
    brighten(&mut bed);
    bed
}

/// `part` over a loop `loop_s` long through its whole-lane core: what
/// [`render_lane`] brightens, and the frozen takes' parts.
fn render_part(part: &Part<'_>, loop_s: f32, rng: &mut NoiseStream) -> Vec<f32> {
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
        let chords = [("day chord", Mood::Day), ("night chord", Mood::Night)];
        type Render = Box<dyn Fn(u32) -> Vec<f32>>;
        let mut voices: Vec<(&str, Render, (f64, f64))> = Vec::new();
        for (name, voice) in notes {
            voices.push((name, Box::new(move |sr| voice(69, 1.0, 0.8, sr)), a4));
        }
        for (name, mood) in chords {
            let tone = ChordTone::of(mood);
            voices.push((name, Box::new(move |sr| chord(tone, &[69; 4], 1.0, sr)), a4));
        }
        for (name, kind, band) in [
            ("kick", DrumKind::Kick, (30.0, 200.0)),
            ("snare", DrumKind::Snare, (150.0, 230.0)),
        ] {
            let hit = move |sr| drum_hit(&mut NoiseStream::new(3), kind, sr);
            voices.push((name, Box::new(hit), band));
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
        let corner = DRUMS
            .night
            .cutoff_hz
            .min(DRUMS.day.cutoff_hz)
            .min(ROLLOFF_HZ);
        assert!(2.0 * HAT_HZ <= corner, "{HAT_HZ} Hz under {corner} Hz");
    }

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

    /// The day bed loops on its own length, out of phase with the groove, so
    /// only the night bed, a loop long, ducks: under every kick it lays.
    #[test]
    fn only_a_night_bed_ducks_and_under_every_kick() {
        for seed in 0..8 {
            let day = crate::compose::compose(Mood::Day, seed);
            assert!(part(&day, Lane::Texture).ducks().is_empty());
            let night = crate::compose::compose(Mood::Night, seed);
            let mut kicks: Vec<f32> = night.kicks().collect();
            kicks.sort_by(f32::total_cmp);
            assert!(!kicks.is_empty());
            assert_eq!(part(&night, Lane::Texture).ducks(), kicks, "seed {seed}");
        }
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

    #[test]
    fn every_buffer_is_finite_and_peak_bounded() {
        let mut rng = NoiseStream::new(9);
        for (name, buf) in [
            ("texture", texture_bed(&mut rng)),
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
