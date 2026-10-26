//! The theory-constrained COMPOSER — generative lofi as constrained sampling:
//! a pure function `seed -> Score`. Harmony walks a vetted progression
//! grammar, and the lead follows melody rules: chord tones on strong beats,
//! diatonic passing tones, leaps bounded to a fifth and resolved by a step,
//! and two phrases, the answer quoting the statement, peaking, and closing
//! the loop when its last bar sings. A quoted, peaking or closing note may
//! leap further, and a resolving step may land off the chord; the tests pin
//! each rule with its exceptions.
//!
//! Quality contract: a generator with unbounded output can't pass a per-take
//! gate, so its gate is STATISTICAL — `ab_judge.py` judges a batch of seeds
//! (`examples/lofi_audition`), and the generator as a whole passes or regresses.
//! The frozen takes are `#[cfg(test)]` fingerprint anchors, not a fallback.
//!
//! The seed fully determines the song, so generation is DETERMINISTIC — a
//! caller that wants a new song picks a new seed; the same seed composes the same
//! score everywhere.

use std::ops::RangeInclusive;

use super::dsp::{NoiseStream, splitmix_draw, splitmix64};

mod form;

#[cfg(test)]
pub(crate) use form::Plays;
pub(crate) use form::{Edge, Layer, arrange};
pub use form::{Form, spans_a_form};

/// The listening context a song is composed for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Mood {
    /// The daytime style.
    Day,
    /// The night-time style.
    Night,
}

impl Mood {
    /// Every knob the mood's songs are composed with.
    const fn style(self) -> &'static Style {
        match self {
            Mood::Day => &DAY,
            Mood::Night => &NIGHT,
        }
    }
}

/// One lane of a generated take. A take's lanes render in [`Lane::ALL`] order
/// from one rng stream, so that order is part of the sound; a song's form
/// names which of their layers strike in each bar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Lane {
    /// Sustained chord bed.
    Pad,
    /// The lead voice's melody.
    Sparkle,
    /// The keys part under the melody.
    Keys,
    /// The drum kit.
    Drums,
    /// The ambient texture bed.
    Texture,
    /// Bass line.
    Bass,
}

impl Lane {
    /// Every lane, in render order, which is also each lane's index.
    pub const ALL: [Lane; 6] = [
        Lane::Pad,
        Lane::Sparkle,
        Lane::Keys,
        Lane::Drums,
        Lane::Texture,
        Lane::Bass,
    ];

    /// Each lane's name in [`Lane::ALL`] order: the ONE spelling lofi prints
    /// and its tools' `--solo` accepts.
    pub const NAMES: [&'static str; Lane::ALL.len()] =
        ["pad", "sparkle", "keys", "drums", "texture", "bass"];

    /// The lane's name.
    pub const fn name(self) -> &'static str {
        Self::NAMES[self as usize]
    }
}

/// The add-an-instrument seam: a new timbre = one synth voice fn, one variant
/// here with its arms in [`LeadVoice::name`] and `LeadVoice::voice`, a weight
/// in a mood's `voices`, a listen batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LeadVoice {
    /// The velocity-keyed EP (the ratified night/day-take Rhodes).
    EpVel,
    /// Plucked-string lead ("nylon" family).
    Pluck,
    /// Plucked-tine lead (day pool) — inharmonic cantilever overtones + a
    /// resonator-body thump.
    Kalimba,
    /// Struck-bar lead (night pool) — double-octave tuned partials with the
    /// fan tremolo baked in.
    Vibraphone,
}

impl LeadVoice {
    /// The voice's display name.
    const fn name(self) -> &'static str {
        match self {
            LeadVoice::EpVel => "ep",
            LeadVoice::Pluck => "pluck",
            LeadVoice::Kalimba => "kalimba",
            LeadVoice::Vibraphone => "vibe",
        }
    }
}

/// Beats in one bar (4/4).
pub(super) const BEATS_PER_BAR: f32 = 4.0;

/// Eighth notes in one bar: the comp's and the hats' slots.
const EIGHTHS_PER_BAR: usize = 8;

/// Drum event kinds. The night groove deliberately uses only Kick + Hat (the
/// sleepy register); the day grooves carry the full kit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DrumKind {
    Kick,
    Snare,
    Hat,
    OpenHat,
}

/// One note a lane plays: when it sounds, in seconds into the loop, its
/// pitch (MIDI) and its velocity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Note {
    pub(super) at: f32,
    pub(super) midi: u8,
    pub(super) vel: f32,
}

/// One drum hit: when it lands, in seconds into the loop, its drum and its
/// gain.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Hit {
    pub(super) at: f32,
    pub(super) kind: DrumKind,
    pub(super) gain: f32,
}

/// One composed song — tempo, lead voice, chords and every lane's events —
/// that `synth`'s voices play.
#[derive(Debug, Clone, PartialEq)]
pub struct Score {
    pub(super) mood: Mood,
    pub(super) bpm: f32,
    /// Each bar's chord, which the pad holds.
    pub(super) bar_chords: [[u8; 4]; GEN_LOOP_BARS],
    /// The lead melody (the sparkle lane).
    pub(super) sparkle: Vec<Note>,
    pub(super) keys: Vec<Note>,
    pub(super) drums: Vec<Hit>,
    /// The bass lane: each bar's root, an answering root or fifth, and the
    /// day's pickup a semitone under the next root.
    pub(super) bass: Vec<Note>,
    /// Which instrument sings the lead.
    pub(super) lead_voice: LeadVoice,
}

pub(super) const GEN_LOOP_BARS: usize = 8;

/// Bars in a phrase: the lead's statement and answer each span one, and the
/// day kit's ghost kick and open hat mark each one's end.
const PHRASE_BARS: usize = GEN_LOOP_BARS / 2;

/// How far a hat may land from its slot, end to end: each lands within half of
/// it either side, so a downbeat's hat may sound that much before its bar.
pub(crate) const HAT_JITTER_S: f32 = 0.012;

impl Score {
    /// Tempo in beats per minute.
    pub fn bpm(&self) -> f32 {
        self.bpm
    }

    pub(super) fn beat_s(&self) -> f32 {
        60.0 / self.bpm
    }

    pub(super) fn bar_s(&self) -> f32 {
        self.beat_s() * BEATS_PER_BAR
    }

    /// The loop's length in seconds: the material every bar of the song cuts
    /// from.
    pub fn loop_secs(&self) -> f32 {
        self.bar_s() * GEN_LOOP_BARS as f32
    }

    /// Display name of the lead instrument this score drew.
    pub fn lead_voice_name(&self) -> &'static str {
        self.lead_voice.name()
    }

    /// Every kick's time in the loop, in the order the drums lay them.
    #[cfg(test)]
    pub(super) fn kicks(&self) -> impl Iterator<Item = f32> + '_ {
        (self.drums.iter())
            .filter(|hit| hit.kind == DrumKind::Kick)
            .map(|hit| hit.at)
    }
}

/// Every knob a mood's songs are composed with, one part per stage, so each
/// stage reads only its own.
struct Style {
    harmony: HarmonyStyle,
    /// The tempo window, in beats per minute, drawn once a song and rounded.
    bpm: Spread,
    lead: LeadStyle,
    keys: KeysStyle,
    drums: Grooves,
    bass: BassStyle,
    /// The lead voices the mood draws from, each with its weight.
    voices: &'static [(LeadVoice, u32)],
}

struct HarmonyStyle {
    /// The grammar a song draws its template from.
    progressions: &'static [Progression],
    /// Whether the loop's last bar turns around to the dominant of its
    /// first root.
    turnaround: bool,
}

struct LeadStyle {
    /// The lowest and highest notes the lead sings (MIDI): the span of the
    /// mood's frozen take's lead.
    lo: u8,
    hi: u8,
    /// The chance a statement bar rests, where the mood rests at all.
    rest_p: Option<f32>,
    /// The most notes a statement bar draws: the most the mood's frozen
    /// takes sing in a bar.
    max_per_bar: usize,
    velocity: Spread,
}

struct KeysStyle {
    /// The chance each eighth strikes.
    density: Density,
    velocity: Spread,
    comp: Comp,
}

/// How a mood sets the keys' density.
enum Density {
    /// Drawn once a song.
    Drawn(Spread),
    /// The same every song, drawing nothing.
    Fixed(f32),
}

/// What the keys strike on an eighth.
enum Comp {
    /// A rolled rootless shell, the chord's 3rd and 7th, the 7th sometimes
    /// lifted to the 9th: comping means chords, and the pad and bass own the
    /// low end, so two voices keep 250-500 Hz controlled.
    Shell,
    /// One chord tone: the sleepy register must not thicken.
    Single,
}

/// How a mood picks its groove.
enum Grooves {
    /// One of these, drawn once a song.
    Drawn(&'static [GrooveTemplate]),
    /// This one every song, drawing nothing.
    Fixed(&'static GrooveTemplate),
}

struct BassStyle {
    /// The chance a bar answers its root.
    answer_p: f32,
    /// The beat the answer lands on.
    answer_beat: f32,
    /// The chance the answer is the fifth, not the root again.
    fifth_p: f32,
    /// The chance a bar picks up into the next a semitone under its root,
    /// where the mood picks up at all.
    pickup_p: Option<f32>,
    /// The root's velocity; the answer's is [`ANSWER_SOFTER`] under it.
    velocity: Spread,
}

const DAY: Style = Style {
    harmony: HarmonyStyle {
        progressions: &DAY_PROGRESSIONS,
        turnaround: true,
    },
    bpm: Spread::between(70.0, 82.0),
    lead: LeadStyle {
        lo: 65,
        hi: 81,
        rest_p: None,
        max_per_bar: 3,
        velocity: Spread {
            base: 0.34,
            span: 0.10,
        },
    },
    keys: KeysStyle {
        density: Density::Drawn(Spread::between(0.28, 0.38)),
        velocity: Spread {
            base: 0.42,
            span: 0.30,
        },
        comp: Comp::Shell,
    },
    drums: Grooves::Drawn(&DAY_GROOVES),
    bass: BassStyle {
        answer_p: 0.55,
        answer_beat: 2.5,
        fifth_p: 0.4,
        pickup_p: Some(0.25),
        velocity: Spread {
            base: 0.62,
            span: 0.08,
        },
    },
    voices: &[
        (LeadVoice::Pluck, 35),
        (LeadVoice::Kalimba, 25),
        (LeadVoice::EpVel, 40),
    ],
};

const NIGHT: Style = Style {
    harmony: HarmonyStyle {
        progressions: &NIGHT_PROGRESSIONS,
        turnaround: false,
    },
    bpm: Spread::between(64.0, 72.0),
    lead: LeadStyle {
        lo: 60,
        hi: 79,
        rest_p: Some(0.5),
        max_per_bar: 1,
        velocity: Spread {
            base: 0.25,
            span: 0.12,
        },
    },
    keys: KeysStyle {
        density: Density::Fixed(0.15),
        velocity: Spread {
            base: 0.35,
            span: 0.25,
        },
        comp: Comp::Single,
    },
    drums: Grooves::Fixed(&NIGHT_GROOVE),
    bass: BassStyle {
        answer_p: 0.7,
        // the old baked sub's pulse
        answer_beat: 2.0,
        fifth_p: 0.2,
        pickup_p: None,
        velocity: Spread {
            base: 0.55,
            span: 0.08,
        },
    },
    voices: &[(LeadVoice::Vibraphone, 35), (LeadVoice::EpVel, 65)],
};

/// The fastest any song plays, in beats per minute.
#[cfg(test)]
pub(crate) fn fastest_bpm() -> f32 {
    [Mood::Day, Mood::Night]
        .map(|mood| mood.style().bpm.hi())
        .into_iter()
        .fold(0.0, f32::max)
}

/// A uniform draw from `base` up to `base + span`.
#[derive(Clone, Copy, Debug)]
struct Spread {
    base: f32,
    span: f32,
}

impl Spread {
    /// The draw from `lo` up to `hi`.
    const fn between(lo: f32, hi: f32) -> Self {
        Self {
            base: lo,
            span: hi - lo,
        }
    }

    /// The draw's top end.
    #[cfg(test)]
    const fn hi(self) -> f32 {
        self.base + self.span
    }

    fn draw(self, rng: &mut NoiseStream) -> f32 {
        self.base + self.span * rng.unit()
    }

    /// `t`, delayed by a draw.
    fn after(self, t: f32, rng: &mut NoiseStream) -> f32 {
        t + self.base + self.span * rng.unit()
    }
}

/// A timing jitter `width` wide, `early` of it landing before the grid.
#[derive(Clone, Copy)]
struct Jitter {
    early: f32,
    width: f32,
}

impl Jitter {
    fn draw(self, rng: &mut NoiseStream) -> f32 {
        (rng.unit() - self.early) * self.width
    }
}

/// Intervals above a root, in semitones.
const MINOR_THIRD: u8 = 3;
const MAJOR_THIRD: u8 = 4;
const FIFTH: u8 = 7;
const MINOR_SEVENTH: u8 = 10;
const MAJOR_SEVENTH: u8 = 11;
/// A ninth's pitch class above the root's: a whole tone.
const NINTH: u8 = 2;
const OCTAVE: u8 = 12;

/// The stages a song is composed in. Each draws from its own stream, keyed
/// by its fixed id, so a change to one stage's draws moves no other stage's,
/// and a stage can join, leave or move without re-rolling the rest.
#[derive(Clone, Copy)]
pub(crate) enum Stage {
    Form = 1,
    Harmony = 2,
    Tempo = 3,
    Lead = 4,
    Keys = 5,
    Drums = 6,
    Bass = 7,
    LeadVoice = 8,
}

impl Stage {
    /// The stage's stream for song seed `seed`: the id folded into the hashed
    /// seed, never added to the raw one, whose neighbours' streams would
    /// overlap.
    pub(crate) fn stream(self, seed: u64) -> NoiseStream {
        NoiseStream::new(splitmix_draw(splitmix64(seed), self as u64))
    }
}

/// One chord: its voicing, and its root's pitch class, which an inversion's
/// lowest note is not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Chord {
    notes: [u8; 4],
    root_pc: u8,
}

const fn chord(notes: [u8; 4], root_pc: u8) -> Chord {
    Chord { notes, root_pc }
}

impl Chord {
    /// The chord `t` semitones up.
    fn transposed(self, t: i8) -> Self {
        Self {
            notes: self.notes.map(|n| (i16::from(n) + i16::from(t)) as u8),
            root_pc: ((i16::from(self.root_pc) + i16::from(t)).rem_euclid(OCTAVE.into())) as u8,
        }
    }
}

/// One vetted progression template, pre-voiced in C.
#[derive(Debug)]
struct Progression {
    chords: [Chord; 4],
    scale_pcs: [u8; 7],
    /// Carries ONE deliberate out-of-scale color move — exempt from the
    /// diatonic pin, which instead asserts the color tone EXISTS.
    chromatic: bool,
}

impl Progression {
    /// The template `t` semitones up, its scale's classes in order.
    fn transposed(&self, t: i8) -> Self {
        let mut scale_pcs = self
            .scale_pcs
            .map(|s| ((i16::from(s) + i16::from(t)).rem_euclid(OCTAVE.into())) as u8);
        scale_pcs.sort_unstable();
        Self {
            chords: self.chords.map(|c| c.transposed(t)),
            scale_pcs,
            chromatic: self.chromatic,
        }
    }
}

/// C major pitch classes.
const C_MAJOR: [u8; 7] = [0, 2, 4, 5, 7, 9, 11];
/// F major pitch classes.
const F_MAJOR: [u8; 7] = [0, 2, 4, 5, 7, 9, 10];

/// The day grammar — every chord tone is in the template's scale unless the
/// row sets `chromatic`.
const DAY_PROGRESSIONS: [Progression; 10] = [
    // royal road: Fmaj7 G7 Em7 Am7
    Progression {
        chords: [
            chord([53, 57, 60, 64], 5),
            chord([55, 59, 62, 65], 7),
            chord([52, 55, 59, 62], 4),
            chord([57, 60, 64, 67], 9),
        ],
        scale_pcs: C_MAJOR,
        chromatic: false,
    },
    // I-vi-ii-V turnaround with voice-led inversions; bar 2 is Am7/C as a TRUE
    // 7th-chord pc set — a voicing with no 7th breaks the shell derivation
    Progression {
        chords: [
            chord([48, 52, 55, 59], 0),
            chord([48, 52, 55, 57], 9),
            chord([50, 53, 57, 60], 2),
            chord([53, 55, 59, 62], 7),
        ],
        scale_pcs: C_MAJOR,
        chromatic: false,
    },
    // modal set: Dm7 Gm7 B♭maj7 Am7
    Progression {
        chords: [
            chord([50, 53, 57, 60], 2),
            chord([55, 58, 62, 65], 7),
            chord([58, 62, 65, 69], 10),
            chord([57, 60, 64, 67], 9),
        ],
        scale_pcs: F_MAJOR,
        chromatic: false,
    },
    // stepwise descent: Fmaj7 Em7 Dm7 Cmaj7
    Progression {
        chords: [
            chord([53, 57, 60, 64], 5),
            chord([52, 55, 59, 62], 4),
            chord([50, 53, 57, 60], 2),
            chord([48, 52, 55, 59], 0),
        ],
        scale_pcs: C_MAJOR,
        chromatic: false,
    },
    // pop cadence: Am7 Fmaj7 Cmaj7 G7
    Progression {
        chords: [
            chord([57, 60, 64, 67], 9),
            chord([53, 57, 60, 64], 5),
            chord([48, 52, 55, 59], 0),
            chord([55, 59, 62, 65], 7),
        ],
        scale_pcs: C_MAJOR,
        chromatic: false,
    },
    // gentle lift: Cmaj7 Fmaj7 Em7 Am7
    Progression {
        chords: [
            chord([48, 52, 55, 59], 0),
            chord([53, 57, 60, 64], 5),
            chord([52, 55, 59, 62], 4),
            chord([57, 60, 64, 67], 9),
        ],
        scale_pcs: C_MAJOR,
        chromatic: false,
    },
    // V7/vi: Fmaj7 E7 Am7 G7 — the E7's G# pulls into Am
    Progression {
        chords: [
            chord([53, 57, 60, 64], 5),
            chord([52, 56, 59, 62], 4),
            chord([57, 60, 64, 67], 9),
            chord([55, 59, 62, 65], 7),
        ],
        scale_pcs: C_MAJOR,
        chromatic: true,
    },
    // borrowed iv: Cmaj7 Fmaj7 Fm7 Cmaj7 — the minor-plagal fade
    Progression {
        chords: [
            chord([48, 52, 55, 59], 0),
            chord([53, 57, 60, 64], 5),
            chord([53, 56, 60, 63], 5),
            chord([48, 52, 55, 59], 0),
        ],
        scale_pcs: C_MAJOR,
        chromatic: true,
    },
    // rhythm-changes turnaround: Cmaj7 A7 Dm7 G7 — the A7's C# leans into Dm
    Progression {
        chords: [
            chord([48, 52, 55, 59], 0),
            chord([49, 55, 57, 64], 9),
            chord([50, 53, 57, 60], 2),
            chord([55, 59, 62, 65], 7),
        ],
        scale_pcs: C_MAJOR,
        chromatic: true,
    },
    // full jazz turnaround: Em7 A7 Dm7 G7 — iii-VI7-ii-V, the long way home
    Progression {
        chords: [
            chord([52, 55, 59, 62], 4),
            chord([55, 57, 61, 64], 9),
            chord([50, 53, 57, 60], 2),
            chord([55, 59, 62, 65], 7),
        ],
        scale_pcs: C_MAJOR,
        chromatic: true,
    },
];

/// The night grammar — minor-leaning, root-position, the sleepy register.
const NIGHT_PROGRESSIONS: [Progression; 8] = [
    // Am7 F Cmaj7 Em7 — the frozen night take's loop, its F a triad
    Progression {
        chords: [
            chord([57, 60, 64, 67], 9),
            chord([53, 57, 60, 65], 5),
            chord([48, 52, 55, 59], 0),
            chord([52, 55, 59, 62], 4),
        ],
        scale_pcs: C_MAJOR,
        chromatic: false,
    },
    // minor descent: Am7 G7 Fmaj7 Em7
    Progression {
        chords: [
            chord([57, 60, 64, 67], 9),
            chord([55, 59, 62, 65], 7),
            chord([53, 57, 60, 64], 5),
            chord([52, 55, 59, 62], 4),
        ],
        scale_pcs: C_MAJOR,
        chromatic: false,
    },
    // suspended drift: Dm7 Am7 B♭maj7 Fmaj7
    Progression {
        chords: [
            chord([50, 53, 57, 60], 2),
            chord([57, 60, 64, 67], 9),
            chord([58, 62, 65, 69], 10),
            chord([53, 57, 60, 64], 5),
        ],
        scale_pcs: F_MAJOR,
        chromatic: false,
    },
    // deep water: Am7 Em7 Fmaj7 Cmaj7
    Progression {
        chords: [
            chord([57, 60, 64, 67], 9),
            chord([52, 55, 59, 62], 4),
            chord([53, 57, 60, 64], 5),
            chord([48, 52, 55, 59], 0),
        ],
        scale_pcs: C_MAJOR,
        chromatic: false,
    },
    // late round: Gm7 C7 Fmaj7 Dm7 — a ii-V that lands and settles on the vi
    Progression {
        chords: [
            chord([55, 58, 62, 65], 7),
            chord([48, 52, 55, 58], 0),
            chord([53, 57, 60, 64], 5),
            chord([50, 53, 57, 60], 2),
        ],
        scale_pcs: F_MAJOR,
        chromatic: false,
    },
    // half-lit stairs: Dm7 Em7♭5 Fmaj7 B♭maj7 — a stepwise climb that never cadences
    Progression {
        chords: [
            chord([50, 53, 57, 60], 2),
            chord([52, 55, 58, 62], 4),
            chord([53, 57, 60, 64], 5),
            chord([58, 62, 65, 69], 10),
        ],
        scale_pcs: F_MAJOR,
        chromatic: false,
    },
    // minor turnaround: Dm7 B♭maj7 Gm7 A7 — harmonic-minor color, the C# pulls home
    Progression {
        chords: [
            chord([50, 53, 57, 60], 2),
            chord([58, 62, 65, 69], 10),
            chord([55, 58, 62, 65], 7),
            chord([57, 61, 64, 67], 9),
        ],
        scale_pcs: F_MAJOR,
        chromatic: true,
    },
    // slow orbit: Am7 Dm7 Fmaj7 G7 — vi-ii-IV-V, drifting without landing
    Progression {
        chords: [
            chord([57, 60, 64, 67], 9),
            chord([50, 53, 57, 60], 2),
            chord([53, 57, 60, 64], 5),
            chord([55, 59, 62, 65], 7),
        ],
        scale_pcs: C_MAJOR,
        chromatic: false,
    },
];

/// The semitones a template may move, so a song's key is drawn.
const TRANSPOSE: RangeInclusive<i8> = -3..=2;

/// Where the turnaround dominant's root sits: the octave of the templates'
/// lowest notes.
const DOMINANT_ROOTS_FROM: u8 = 46;

fn pick(rng: &mut NoiseStream, n: usize) -> usize {
    ((rng.unit() * n as f32) as usize).min(n.saturating_sub(1))
}

fn chance(rng: &mut NoiseStream, p: f32) -> bool {
    rng.unit() < p
}

/// A fair coin.
fn coin(rng: &mut NoiseStream) -> bool {
    chance(rng, 0.5)
}

/// The band one unit draw `u` falls in, each band as wide as its weight.
fn band(u: f32, weights: &[u32]) -> usize {
    let total: u32 = weights.iter().sum();
    let at = ((u * total as f32) as u32).min(total - 1);
    let mut below = 0;
    weights
        .iter()
        .position(|&w| {
            below += w;
            at < below
        })
        .unwrap_or(weights.len() - 1)
}

/// The row of `table` one unit draw falls in.
fn kind<T: Copy>(u: f32, table: &[(T, u32)]) -> T {
    let weights: Vec<u32> = table.iter().map(|&(_, w)| w).collect();
    table[band(u, &weights)].0
}

/// The pitch with class `pc` in the octave from `lo` up.
fn in_octave_from(lo: u8, pc: u8) -> u8 {
    lo + (pc + OCTAVE - lo % OCTAVE) % OCTAVE
}

/// A song's harmony: its template in the song's key, and the timeline every
/// lane reads.
#[derive(Debug)]
struct Harmony {
    template: Progression,
    /// The 8-bar timeline: the template played twice, its last bar turned
    /// around to the dominant where the style says so.
    bars: [Chord; GEN_LOOP_BARS],
}

/// The harmony stage: a template from the mood's grammar, in a drawn key.
fn harmony(rng: &mut NoiseStream, style: &HarmonyStyle) -> Harmony {
    let progs = style.progressions;
    let prog = &progs[pick(rng, progs.len())];
    let t = *TRANSPOSE.start() + pick(rng, TRANSPOSE.count()) as i8;
    let template = prog.transposed(t);
    let bars = timeline(&template, style.turnaround);
    Harmony { template, bars }
}

/// The 8-bar timeline: `template` twice, its last bar swapped for the
/// dominant of the returning first root when it turns around.
fn timeline(template: &Progression, turnaround: bool) -> [Chord; GEN_LOOP_BARS] {
    let chords = &template.chords;
    let mut bars = std::array::from_fn(|bar| chords[bar % chords.len()]);
    if turnaround {
        bars[GEN_LOOP_BARS - 1] = dominant7_of(chords[0].root_pc);
    }
    bars
}

/// The V7 of `target_root_pc`: a major triad and its minor seventh.
fn dominant7_of(target_root_pc: u8) -> Chord {
    let root_pc = (target_root_pc + FIFTH) % OCTAVE;
    let r = in_octave_from(DOMINANT_ROOTS_FROM, root_pc);
    chord([r, r + MAJOR_THIRD, r + FIFTH, r + MINOR_SEVENTH], root_pc)
}

/// The rootless comp shell: the chord's 3rd + 7th, found degree-aware from the
/// harmonic root. Every day template chord is a 7th chord; the fallbacks only
/// guard a future non-7th row.
fn shell_of(chord: &Chord) -> (u8, u8) {
    let find = |offsets: [u8; 2]| {
        (chord.notes.iter().copied()).find(|&n| {
            offsets
                .iter()
                .any(|&o| n % OCTAVE == (chord.root_pc + o) % OCTAVE)
        })
    };
    let third = find([MINOR_THIRD, MAJOR_THIRD]).unwrap_or(chord.notes[1]);
    let seventh = find([MINOR_SEVENTH, MAJOR_SEVENTH])
        .or_else(|| (chord.notes.iter().copied()).find(|&n| n % OCTAVE != third % OCTAVE))
        .unwrap_or(third.saturating_add(FIFTH));
    (third, seventh)
}

/// The pitch with class `pc` nearest to `around`.
fn nearest_with_pc(around: u8, pc: u8) -> u8 {
    let a = i16::from(around);
    for d in 0..=i16::from(OCTAVE / 2) {
        for cand in [a - d, a + d] {
            if cand > 0 && cand.rem_euclid(OCTAVE.into()) as u8 == pc {
                return cand as u8;
            }
        }
    }
    around
}

fn chord_tones_in(chord: &[u8; 4], lo: u8, hi: u8) -> Vec<u8> {
    let mut out = Vec::new();
    for &c in chord {
        let mut n = c;
        while n < lo {
            n += OCTAVE;
        }
        while n <= hi {
            out.push(n);
            n += OCTAVE;
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// Nearest member of `pool` to `target`; ties resolve low.
fn nearest(pool: &[u8], target: u8) -> u8 {
    *pool
        .iter()
        .min_by_key(|&&n| ((i16::from(n) - i16::from(target)).abs(), n))
        .expect("pool is never empty within the register windows")
}

/// Nearest in-scale note to `n` inside `[lo, hi]` — a diatonic scale has no gap
/// wider than a whole tone, so the outward search always lands within ±2.
fn snap_to_scale(n: i16, scale: &[u8; 7], lo: u8, hi: u8) -> u8 {
    for d in 0..=i16::from(OCTAVE / 2) {
        for cand in [n - d, n + d] {
            if cand >= i16::from(lo)
                && cand <= i16::from(hi)
                && scale.contains(&(cand.rem_euclid(OCTAVE.into()) as u8))
            {
                return cand as u8;
            }
        }
    }
    lo // unreachable for any 7-pc scale with hi-lo >= 12; safe fallback
}

/// Step `k` scale degrees from `note` (k may be negative), clamped to the
/// register window. Stays in-scale by construction.
fn scale_step(note: u8, k: i8, scale: &[u8; 7], lo: u8, hi: u8) -> u8 {
    let mut n = i16::from(note);
    let mut remaining = k.abs();
    let dir = if k >= 0 { 1 } else { -1 };
    while remaining > 0 {
        n += dir;
        if n < i16::from(lo) || n > i16::from(hi) {
            n -= dir;
            break;
        }
        if scale.contains(&(n.rem_euclid(OCTAVE.into()) as u8)) {
            remaining -= 1;
        }
    }
    snap_to_scale(n, scale, lo, hi)
}

/// The friendly beat grid a lead note may land on (0.0 is the kick's).
const LEAD_GRID: [f32; 6] = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0];

/// The chances a statement bar draws a second note, then a third, where its
/// mood's density allows.
const SECOND_NOTE_P: f32 = 0.6;
const THIRD_NOTE_P: f32 = 0.25;

/// The lofi push: the chance an on-beat note lands a 16th ahead.
const PUSH_P: f32 = 0.15;
const SIXTEENTH_BEATS: f32 = 0.25;

/// Two of a bar's notes closer than this, in beats, merge into the first.
const MIN_GAP_BEATS: f32 = 0.75;

/// The answer's variations on its statement bar: the chance a note shifts an
/// eighth either way, and the chance one is dropped.
const SHIFT_P: f32 = 0.2;
const EIGHTH_BEATS: f32 = 0.5;
const DROP_P: f32 = 0.10;

/// The statement a song sings when its draws leave it fewer notes: a bar and
/// a beat each, in the statement's first and third bars.
const FALLBACK_STATEMENT: [(usize, f32); 2] = [(0, 1.0), (2, 2.0)];

/// The statement's first notes, which the answer opens by quoting.
const QUOTE_NOTES: usize = 2;
/// The answer's peak, in the bar after the quote.
const PEAK_BAR: usize = PHRASE_BARS + 1;

/// The lead starts on the first chord's tone in its register's top octave
/// nearest this far over the register's middle.
const START_LIFT: u8 = 2;
/// The peak's lift over the last note, in semitones, onto a chord tone.
const PEAK_LIFT: RangeInclusive<i16> = 3..=5;
/// A move wider than this, in semitones, is a leap the next note resolves.
const LEAP: i16 = MAJOR_THIRD as i16;
/// How far a chord-tone note wanders from the last, in semitones.
const DRIFT: RangeInclusive<i16> = -4..=4;
/// The chance an off-beat note is a chord tone, not a passing one.
const CHORD_TONE_P: f32 = 0.55;
/// The chance a passing tone skips a degree.
const SKIP_P: f32 = 0.3;
/// The peak note's velocity over its draw.
const PEAK_ACCENT: f32 = 0.06;

/// How far behind the grid a lead note sounds, in seconds, and how its
/// velocity swells or softens as played.
const LEAD_LAG: Spread = Spread {
    base: 0.008,
    span: 0.014,
};
const LEAD_ACCENT: Spread = Spread {
    base: 0.92,
    span: 0.16,
};

/// What a note does in the melody, which picks its pitch: the first of these,
/// in order, that applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Role {
    /// The answer opens on the statement's opening notes.
    Quote,
    /// The loop's last note returns near the opening, onto a chord tone.
    Close,
    /// The answer's peak: a lift onto a chord tone.
    Peak,
    /// After a leap, a step against it.
    Resolve,
    /// A chord tone near the last note, the leap bounded to a fifth.
    ChordTone,
    /// A scale step or two from the last note.
    Passing,
}

/// One melody note before it is played: its bar, beat in the bar, pitch and
/// velocity.
struct LeadNote {
    bar: usize,
    beat: f32,
    note: u8,
    vel: f32,
}

/// The lead stage: a melody over `harmony`, as played behind the grid.
fn lead(rng: &mut NoiseStream, harmony: &Harmony, style: &LeadStyle, beat_s: f32) -> Vec<Note> {
    let statement = statement(rng, style);
    let answer = answer(rng, &statement, style);
    let line = melody(rng, &[statement, answer].concat(), harmony, style);
    let bar_s = beat_s * BEATS_PER_BAR;
    (line.into_iter())
        .map(|n| {
            let at = LEAD_LAG.after(n.bar as f32 * bar_s + n.beat * beat_s, rng);
            Note {
                at,
                midi: n.note,
                vel: n.vel * LEAD_ACCENT.draw(rng),
            }
        })
        .collect()
}

/// The statement's rhythm: each of its bars' beats.
fn statement(rng: &mut NoiseStream, style: &LeadStyle) -> Vec<Vec<f32>> {
    let max = style.max_per_bar;
    let mut bars: Vec<Vec<f32>> = (0..PHRASE_BARS)
        .map(|_| {
            if style.rest_p.is_some_and(|p| chance(rng, p)) {
                return Vec::new();
            }
            let n = 1
                + usize::from(max > 1 && chance(rng, SECOND_NOTE_P))
                + usize::from(max > 2 && chance(rng, THIRD_NOTE_P));
            let mut beats: Vec<f32> = (0..n)
                .map(|_| LEAD_GRID[pick(rng, LEAD_GRID.len())])
                .collect();
            for b in &mut beats {
                push(rng, b, style);
            }
            settle(&mut beats);
            beats
        })
        .collect();
    // a mood that rests can rest through the whole phrase, and a take with no
    // lead has lost its singable identity
    if bars.iter().map(Vec::len).sum::<usize>() < FALLBACK_STATEMENT.len() {
        for (bar, beat) in FALLBACK_STATEMENT {
            bars[bar] = vec![beat];
        }
    }
    bars
}

/// The answer's rhythm: each statement bar's, varied.
fn answer(rng: &mut NoiseStream, statement: &[Vec<f32>], style: &LeadStyle) -> Vec<Vec<f32>> {
    let grid = LEAD_GRID[0]..=LEAD_GRID[LEAD_GRID.len() - 1];
    (statement.iter())
        .map(|bar| {
            let mut beats = bar.clone();
            for b in &mut beats {
                if chance(rng, SHIFT_P) {
                    let shift = if coin(rng) {
                        EIGHTH_BEATS
                    } else {
                        -EIGHTH_BEATS
                    };
                    *b = (*b + shift).clamp(*grid.start(), *grid.end());
                }
                push(rng, b, style);
            }
            if !beats.is_empty() && chance(rng, DROP_P) {
                beats.remove(pick(rng, beats.len()));
            }
            settle(&mut beats);
            beats
        })
        .collect()
}

/// The lofi push, where the mood sings more than a note a bar.
fn push(rng: &mut NoiseStream, beat: &mut f32, style: &LeadStyle) {
    if style.max_per_bar > 1 && beat.fract() == 0.0 && chance(rng, PUSH_P) {
        *beat -= SIXTEENTH_BEATS;
    }
}

/// A bar's beats in order, each a gap from the last.
fn settle(beats: &mut Vec<f32>) {
    beats.sort_by(f32::total_cmp);
    beats.dedup_by(|a, b| (*a - *b).abs() < MIN_GAP_BEATS);
}

/// Each note of `rhythm`'s pitch and velocity, by its [`Role`].
fn melody(
    rng: &mut NoiseStream,
    rhythm: &[Vec<f32>],
    harmony: &Harmony,
    style: &LeadStyle,
) -> Vec<LeadNote> {
    let (lo, hi) = (style.lo, style.hi);
    let scale = &harmony.template.scale_pcs;
    let start_pool = chord_tones_in(
        &harmony.bars[0].notes,
        lo.max(hi.saturating_sub(OCTAVE)),
        hi,
    );
    let mut prev = nearest(&start_pool, u8::midpoint(lo, hi) + START_LIFT);
    let mut last_leap: i16 = 0;
    let mut head: Vec<u8> = Vec::new();
    let mut out: Vec<LeadNote> = Vec::new();
    let last_bar = rhythm.len() - 1;
    for (bar, beats) in rhythm.iter().enumerate() {
        let tones = chord_tones_in(&harmony.bars[bar].notes, lo, hi);
        for (i, &beat) in beats.iter().enumerate() {
            let last = bar == last_bar && i == beats.len() - 1;
            // an anticipation is a pushed strong beat, not a passing tone
            let strong = beat.fract() == 0.0 || beat.fract() == 1.0 - SIXTEENTH_BEATS;
            let role = if bar == PHRASE_BARS && i < head.len() && !last {
                Role::Quote
            } else if last {
                Role::Close
            } else if bar == PEAK_BAR && i == 0 {
                Role::Peak
            } else if last_leap.abs() > LEAP {
                Role::Resolve
            } else if strong || chance(rng, CHORD_TONE_P) {
                Role::ChordTone
            } else {
                Role::Passing
            };
            let note = match role {
                Role::Quote => head[i],
                Role::Close => nearest(&tones, out.first().map_or(prev, |n| n.note)),
                Role::Peak => {
                    let lift = *PEAK_LIFT.start() + pick(rng, PEAK_LIFT.count()) as i16;
                    nearest(&tones, (i16::from(prev) + lift).min(i16::from(hi)) as u8)
                }
                Role::Resolve => scale_step(prev, -last_leap.signum() as i8, scale, lo, hi),
                Role::ChordTone => {
                    let drift = *DRIFT.start() + pick(rng, DRIFT.count()) as i16;
                    let target = (i16::from(prev) + drift).clamp(i16::from(lo), i16::from(hi));
                    let n = nearest(&tones, target as u8);
                    if (i16::from(n) - i16::from(prev)).abs() > i16::from(FIFTH) {
                        nearest(&tones, prev)
                    } else {
                        n
                    }
                }
                Role::Passing => {
                    let k = if coin(rng) { 1 } else { -1 };
                    scale_step(prev, k * (1 + i8::from(chance(rng, SKIP_P))), scale, lo, hi)
                }
            };
            last_leap = i16::from(note) - i16::from(prev);
            prev = note;
            if bar == 0 && head.len() < QUOTE_NOTES {
                head.push(note);
            }
            let base = style.velocity.draw(rng);
            let vel = if role == Role::Peak {
                base + PEAK_ACCENT
            } else {
                base
            };
            out.push(LeadNote {
                bar,
                beat,
                note,
                vel,
            });
        }
    }
    out
}

/// One groove: where its kicks, snares and hats land, and how they swing and
/// drag.
struct GrooveTemplate {
    /// The bar's downbeat kick, then its second.
    kicks: [Kick; 2],
    /// Whether a ghost kick picks up into each phrase's next bar.
    ghost_pickup: bool,
    /// How far behind the grid the kicks land, in seconds.
    drag_s: f32,
    /// The beats the snare lands on.
    snares: &'static [f32],
    hats: Hats,
    /// Whether each phrase's last hat opens.
    open_hat: bool,
}

/// A kick: the beat it lands on in even bars and in odd ones, and its gain.
struct Kick {
    beats: [f32; 2],
    gain: f32,
}

struct Hats {
    /// Whether only the off-eighths may strike, skipping the rest unseen.
    off_eighths_only: bool,
    /// The chance a slot that may strike stays silent.
    skip_p: f32,
    /// The off-eighths' swing, as [`swing_delay`] reads it.
    swing: f32,
    velocity: Spread,
}

const DOWNBEAT: f32 = 0.0;

const DAY_GROOVES: [GrooveTemplate; 3] = [
    // boom-bap: backbeats on 2&4
    GrooveTemplate {
        kicks: [
            Kick {
                beats: [DOWNBEAT; 2],
                gain: 1.0,
            },
            Kick {
                beats: [2.25, 2.5],
                gain: 0.75,
            },
        ],
        ghost_pickup: true,
        drag_s: DRAG_KICK_S,
        snares: &[1.0, 3.0],
        hats: DAY_HATS.skipping(0.15),
        open_hat: true,
    },
    // half-time: the lazy backbeat
    GrooveTemplate {
        kicks: [
            Kick {
                beats: [DOWNBEAT; 2],
                gain: 1.0,
            },
            Kick {
                beats: [2.75; 2],
                gain: 0.7,
            },
        ],
        ghost_pickup: false,
        drag_s: DRAG_KICK_S,
        snares: &[2.0],
        hats: DAY_HATS.skipping(0.30),
        open_hat: true,
    },
    // sparse: quiet-focus
    GrooveTemplate {
        kicks: [
            Kick {
                beats: [DOWNBEAT; 2],
                gain: 0.9,
            },
            Kick {
                beats: [2.5; 2],
                gain: 0.6,
            },
        ],
        ghost_pickup: false,
        drag_s: DRAG_KICK_S,
        snares: &[2.0],
        hats: DAY_HATS.skipping(0.45),
        open_hat: false,
    },
];

/// The day kit's hats, but for how often they skip.
const DAY_HATS: Hats = Hats {
    off_eighths_only: false,
    skip_p: 0.0,
    swing: 0.56,
    velocity: Spread {
        base: 0.4,
        span: 0.2,
    },
};

impl Hats {
    const fn skipping(self, skip_p: f32) -> Self {
        Self { skip_p, ..self }
    }
}

/// The night groove: kick + soft closed hats only (the sleepy register).
const NIGHT_GROOVE: GrooveTemplate = GrooveTemplate {
    kicks: [
        Kick {
            beats: [DOWNBEAT; 2],
            gain: 0.6,
        },
        Kick {
            beats: [2.5; 2],
            gain: 0.4,
        },
    ],
    ghost_pickup: false,
    // a separate lag on purpose, not a drifted copy of DRAG_KICK_S
    drag_s: 0.018,
    snares: &[],
    hats: Hats {
        off_eighths_only: true,
        skip_p: 0.45,
        swing: 0.58,
        velocity: Spread {
            base: 0.40,
            span: 0.24,
        },
    },
    open_hat: false,
};

const DRAG_KICK_S: f32 = 0.015;
const SWING_KICK: f32 = 0.53;
/// The keys' swing: their own knob, set where the kick's is.
const SWING_KEYS: f32 = 0.53;

const GHOST_KICK: Kick = Kick {
    beats: [3.75; 2],
    gain: 0.35,
};

/// How far behind the grid a snare lands, in seconds.
const SNARE_DRAG_S: f32 = 0.008;
const SNARE_GAIN: f32 = 0.85;

/// Each bar's velocity wobble, which all its hits share, and each hit's own.
const WOBBLE: Spread = Spread {
    base: 0.9,
    span: 0.2,
};
const HIT_ACCENT: Spread = Spread {
    base: 0.95,
    span: 0.1,
};

/// How far each hit lands from its slot, the kick and snare mostly late.
const KICK_JITTER: Jitter = Jitter {
    early: 0.3,
    width: 0.010,
};
const SNARE_JITTER: Jitter = Jitter {
    early: 0.3,
    width: 0.008,
};
const HAT_JITTER: Jitter = Jitter {
    early: 0.5,
    width: HAT_JITTER_S,
};

/// An off-eighth's delay at swing `s`: the BIBLE's `(s − 0.5) × 8th`, which
/// it gives for a 16th pair. An eighth pair spans a quarter, so an off-eighth
/// delayed by this sits at `0.5 + (s − 0.5) / 2` of its pair.
fn swing_delay(s: f32, eighth_s: f32) -> f32 {
    (s - 0.5) * eighth_s
}

/// One bar of a groove: where it starts, how long its beats are, and the
/// velocity wobble every hit in it shares.
#[derive(Clone, Copy)]
struct Bar {
    start: f32,
    beat_s: f32,
    wobble: f32,
}

impl Bar {
    /// The time of bar-relative beat `beat`.
    fn at(self, beat: f32) -> f32 {
        self.start + beat * self.beat_s
    }

    /// An eighth note's length: the swing's unit.
    fn eighth(self) -> f32 {
        self.beat_s / 2.0
    }
}

/// Place ONE humanized kick at bar-relative beat `at_beat`. The RNG DRAW
/// ORDER is LOAD-BEARING (frozen-seed fidelity): the jitter draw FIRST, then
/// the gain-wobble draw.
fn push_kick(
    out: &mut Vec<Hit>,
    bar: Bar,
    drag: f32,
    at_beat: f32,
    gain: f32,
    rng: &mut NoiseStream,
) {
    let mut at = bar.at(at_beat) + drag;
    if at_beat.fract() != 0.0 {
        at += swing_delay(SWING_KICK, bar.eighth());
    }
    let at = (at + KICK_JITTER.draw(rng)).max(0.0);
    out.push(Hit {
        at,
        kind: DrumKind::Kick,
        gain: gain * bar.wobble * HIT_ACCENT.draw(rng),
    });
}

/// The drums stage: a groove of the mood's, laid.
fn drums(rng: &mut NoiseStream, grooves: &Grooves, beat_s: f32) -> Vec<Hit> {
    let g = match *grooves {
        Grooves::Drawn(pool) => &pool[pick(rng, pool.len())],
        Grooves::Fixed(g) => g,
    };
    groove(rng, g, beat_s)
}

/// The kit `g` lays, clamped non-negative so a bar-0 jitter can't leave the
/// loop.
fn groove(rng: &mut NoiseStream, g: &GrooveTemplate, beat_s: f32) -> Vec<Hit> {
    let bar_s = beat_s * BEATS_PER_BAR;
    let mut out = Vec::new();
    for bar in 0..GEN_LOOP_BARS {
        let at = Bar {
            start: bar as f32 * bar_s,
            beat_s,
            wobble: WOBBLE.draw(rng),
        };
        let phrase_end = bar % PHRASE_BARS == PHRASE_BARS - 1;
        let ghost = (g.ghost_pickup && phrase_end).then_some(&GHOST_KICK);
        for kick in g.kicks.iter().chain(ghost) {
            push_kick(&mut out, at, g.drag_s, kick.beats[bar % 2], kick.gain, rng);
        }
        for &sb in g.snares {
            let t = at.at(sb) + SNARE_DRAG_S + SNARE_JITTER.draw(rng);
            out.push(Hit {
                at: t.max(0.0),
                kind: DrumKind::Snare,
                gain: SNARE_GAIN * at.wobble * HIT_ACCENT.draw(rng),
            });
        }
        let hats = &g.hats;
        for e in 0..EIGHTHS_PER_BAR {
            if (hats.off_eighths_only && e % 2 == 0) || chance(rng, hats.skip_p) {
                continue;
            }
            let mut t = at.start + e as f32 * at.eighth();
            if e % 2 == 1 {
                t += swing_delay(hats.swing, at.eighth());
            }
            t += HAT_JITTER.draw(rng);
            let kind = if g.open_hat && phrase_end && e == EIGHTHS_PER_BAR - 1 {
                DrumKind::OpenHat
            } else {
                DrumKind::Hat
            };
            out.push(Hit {
                at: t.max(0.0),
                kind,
                gain: hats.velocity.draw(rng) * at.wobble,
            });
        }
    }
    out
}

/// The bass window's lowest note, G1: a root folds into the octave from here.
/// Opinion, from the sweep the commit that set it records: the smallest move
/// that puts the mix's power under 62 Hz, as a share of its power under
/// 125 Hz, inside the reference corpus's interquartile range with room on
/// both sides.
pub const BASS_LOW: u8 = 31;

/// The note under which no two voices sound closer than a fifth: LOFI-BIBLE.md
/// R7's low-interval limit.
pub const LOW_INTERVAL_LIMIT: u8 = 48;

/// Whether a `bass` note sounds within a fifth of `voice` under
/// [`LOW_INTERVAL_LIMIT`].
fn clashes(bass: u8, voice: u8) -> bool {
    voice < LOW_INTERVAL_LIMIT && voice.abs_diff(bass) < FIFTH
}

/// A chord's lowest voice.
fn lowest(chord: &Chord) -> u8 {
    chord.notes.iter().copied().fold(u8::MAX, u8::min)
}

/// Fold a pitch class into the bass window, an octave under it where it would
/// clash with `chord`'s lowest voice — the ONE bass-register authority.
fn bass_pitch(pc: u8, chord: &Chord) -> u8 {
    let note = in_octave_from(BASS_LOW, pc);
    if clashes(note, lowest(chord)) {
        note - OCTAVE
    } else {
        note
    }
}

/// How far behind the beat each bass note lands, in seconds, so the kick keeps
/// the transient.
const BASS_LAG: Spread = Spread {
    base: 0.014,
    span: 0.010,
};
/// The answer's velocity under the root's.
const ANSWER_SOFTER: f32 = 0.14;
/// The pickup's beat, lag and velocity.
const PICKUP_BEAT: f32 = 3.5;
const PICKUP_LAG: Spread = Spread {
    base: 0.012,
    span: 0.008,
};
const PICKUP_VELOCITY: Spread = Spread {
    base: 0.34,
    span: 0.06,
};

/// The bass stage: the ONE on every bar, an optional answer, an occasional
/// pickup — root/5th/leading-tone only, a floor, not a melody.
fn bass(
    rng: &mut NoiseStream,
    bars: &[Chord; GEN_LOOP_BARS],
    style: &BassStyle,
    beat_s: f32,
) -> Vec<Note> {
    let bar_s = beat_s * BEATS_PER_BAR;
    let answer_velocity = Spread {
        base: style.velocity.base - ANSWER_SOFTER,
        span: style.velocity.span,
    };
    let mut out = Vec::new();
    for (bar, chord) in bars.iter().enumerate() {
        let b0 = bar as f32 * bar_s;
        let root = bass_pitch(chord.root_pc, chord);
        let at = BASS_LAG.after(b0, rng);
        out.push(Note {
            at,
            midi: root,
            vel: style.velocity.draw(rng),
        });
        if chance(rng, style.answer_p) {
            let note = if chance(rng, style.fifth_p) {
                bass_pitch(chord.root_pc + FIFTH, chord)
            } else {
                root
            };
            let at = BASS_LAG.after(b0 + style.answer_beat * beat_s, rng);
            out.push(Note {
                at,
                midi: note,
                vel: answer_velocity.draw(rng),
            });
        }
        if style.pickup_p.is_some_and(|p| chance(rng, p))
            && let Some(next) = bars.get(bar + 1)
        {
            // literal semitone below the folded next root (pc-folding rendered an
            // 11-semitone drop); no last-bar pickup — place() clips at the seam
            let at = PICKUP_LAG.after(b0 + PICKUP_BEAT * beat_s, rng);
            let midi = bass_pitch(next.root_pc, next) - 1;
            let vel = PICKUP_VELOCITY.draw(rng);
            // one that would clash under this bar's chord is left out, its
            // draws kept
            if !clashes(midi, lowest(chord)) {
                out.push(Note { at, midi, vel });
            }
        }
    }
    out
}

/// The comp's lag behind the grid, its roll between the shell's two voices,
/// both in seconds, and the rolled voice's softer velocity.
const KEYS_LAG: Spread = Spread {
    base: 0.010,
    span: 0.020,
};
const ROLL: Spread = Spread {
    base: 0.012,
    span: 0.008,
};
const ROLLED_SOFTER: f32 = 0.9;
/// The chance a shell's 7th lifts to the 9th.
const NINTH_P: f32 = 0.2;

/// The keys stage: the comp over each bar's chord.
fn keys(
    rng: &mut NoiseStream,
    bars: &[Chord; GEN_LOOP_BARS],
    style: &KeysStyle,
    beat_s: f32,
) -> Vec<Note> {
    let bar_s = beat_s * BEATS_PER_BAR;
    let eighth = beat_s / 2.0;
    let density = match style.density {
        Density::Drawn(spread) => spread.draw(rng),
        Density::Fixed(p) => p,
    };
    let mut out = Vec::new();
    for (bar, chord) in bars.iter().enumerate() {
        for e in 0..EIGHTHS_PER_BAR {
            if !chance(rng, density) {
                continue;
            }
            let mut at = bar as f32 * bar_s + e as f32 * eighth;
            if e % 2 == 1 {
                at += swing_delay(SWING_KEYS, eighth);
            }
            at += KEYS_LAG.draw(rng);
            match style.comp {
                Comp::Shell => {
                    let (third, seventh) = shell_of(chord);
                    let upper = if chance(rng, NINTH_P) {
                        nearest_with_pc(seventh, (chord.root_pc + NINTH) % OCTAVE)
                    } else {
                        seventh
                    };
                    let vel = style.velocity.draw(rng);
                    out.push(Note {
                        at,
                        midi: third,
                        vel,
                    });
                    // a hand rolls the dyad; a sequencer stamps it
                    out.push(Note {
                        at: ROLL.after(at, rng),
                        midi: upper,
                        vel: vel * ROLLED_SOFTER,
                    });
                }
                Comp::Single => {
                    let note = chord.notes[pick(rng, chord.notes.len())];
                    out.push(Note {
                        at,
                        midi: note,
                        vel: style.velocity.draw(rng),
                    });
                }
            }
        }
    }
    out
}

/// Compose one full take from a seed — deterministic, no clock and no I/O.
/// A [`Performance`](crate::Performance) plays it;
/// [`render_lane`](super::synth::render_lane) renders one lane whole.
pub(crate) fn compose(mood: Mood, seed: u64) -> Score {
    composed(mood, seed).0
}

/// The score `seed` composes in `mood`, and the harmony it is written over,
/// whose roots and key the score's voices don't read.
fn composed(mood: Mood, seed: u64) -> (Score, Harmony) {
    let style = mood.style();
    let stream = |stage: Stage| stage.stream(seed);
    let harmony = harmony(&mut stream(Stage::Harmony), &style.harmony);
    let bpm = style.bpm.draw(&mut stream(Stage::Tempo)).round();
    let beat_s = 60.0 / bpm;
    let sparkle = lead(&mut stream(Stage::Lead), &harmony, &style.lead, beat_s);
    let keys = keys(&mut stream(Stage::Keys), &harmony.bars, &style.keys, beat_s);
    let drums = drums(&mut stream(Stage::Drums), &style.drums, beat_s);
    let bass = bass(&mut stream(Stage::Bass), &harmony.bars, &style.bass, beat_s);
    let lead_voice = kind(stream(Stage::LeadVoice).unit(), style.voices);
    let score = Score {
        mood,
        bpm,
        bar_chords: harmony.bars.map(|c| c.notes),
        sparkle,
        keys,
        drums,
        bass,
        lead_voice,
    };
    (score, harmony)
}

/// FNV-1a over 64-bit words, the hash the goldens are recorded with.
#[cfg(test)]
pub(crate) fn fnv1a(words: impl IntoIterator<Item = u64>) -> u64 {
    words.into_iter().fold(0xcbf2_9ce4_8422_2325, |h, w| {
        (h ^ w).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

#[cfg(test)]
mod tests;
