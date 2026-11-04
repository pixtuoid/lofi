//! The styles lofi plays, one record each, so a style is defined in one place,
//! as jacbz/Lofi's `ProducerPreset` holds a preset's bass, harmony, arpeggio
//! and melody instruments
//! (<https://github.com/jacbz/Lofi/blob/main/client/src/producer_presets.ts>).

use std::fmt;

use crate::compose::{
    BassStyle, Comp, Density, Grooves, HIPHOP_GROOVES, HarmonyStyle, KeysStyle, Lane, LeadStyle,
    LeadVoice, SLEEP_GROOVE, Spread, Writing,
};
use crate::synth::{Bed, ChordTone, HitRecipe, NoteRecipe, PadTone, Sound};

/// Every knob a style's songs are written, sounded and staged with. Its
/// records are `static`s, so a style is its record: "All references and raw
/// pointers to the static refer to the same allocation"
/// (<https://doc.rust-lang.org/reference/items/static-items.html>).
pub(crate) struct Style {
    /// The playlist it learns from, which names it everywhere: its `Debug`
    /// prints this in place of every knob, and the judge reads a take's
    /// corpus from it.
    pub(crate) name: &'static str,
    pub(crate) writing: Writing,
    pub(crate) sound: Sound,
    /// Each lane's fixed gains `[in, out]` in the stream, in [`Lane::ALL`]
    /// order, standing in for the whole-lane render's peak normalizations:
    /// `in` for its first `normalize`, which sets the level into the drive
    /// (the texture's, into the duck), and `out` for the product of the rest.
    /// A gain sets its lane's level in the stream, so it moves as a sound
    /// change the judge scores; `lofi_measure --pins` prints each beside the
    /// median of its normalizations over `audition::STAGING_SEEDS`, and how
    /// far apart they sit.
    pub(crate) staging: [[f64; 2]; Lane::ALL.len()],
}

impl PartialEq for Style {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self, other)
    }
}

impl fmt::Debug for Style {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name)
    }
}

/// Every style, which the tests sweep.
#[cfg(test)]
pub(crate) const STYLES: [&Style; 2] = [&HIPHOP, &SLEEP];

pub(crate) static HIPHOP: Style = Style {
    name: "hiphop",
    writing: Writing {
        harmony: HarmonyStyle { major_share: 0.8 },
        bpm: Spread::between(70.0, 82.0),
        lead: LeadStyle {
            lo: 65,
            hi: 81,
            rest_p: None,
            note_p: 0.45,
            velocity: Spread {
                base: 0.34,
                span: 0.10,
            },
            dur: 2.0,
        },
        keys: KeysStyle {
            density: Density::Drawn(Spread::between(0.28, 0.38)),
            velocity: Spread {
                base: 0.42,
                span: 0.30,
            },
            comp: Comp::Shell,
            dur: 0.9,
        },
        drums: Grooves::Drawn(&HIPHOP_GROOVES),
        bass: BassStyle {
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
    },
    sound: Sound {
        pad: PadTone {
            cutoff_hz: 2600.0,
            swell_hz: 0.22,
            swell_depth: 0.05,
            peak: 0.7,
        },
        chord: ChordTone {
            ring_s: 0.9,
            attack_s: 0.25,
            attack_step_s: 0.08,
            release_s: 1.2,
            h2: 0.30,
            h3: 0.08,
            gain: 1.0,
        },
        sparkle: NoteRecipe {
            cutoff_hz: 3200.0,
            peak: 0.6,
        },
        keys: NoteRecipe {
            cutoff_hz: 2400.0,
            peak: 0.8,
        },
        bass: NoteRecipe {
            cutoff_hz: 900.0,
            peak: 0.704,
        },
        drums: HitRecipe {
            cutoff_hz: 6500.0,
            drive: 2.2,
            peak: 0.85,
        },
        bed: Bed::Free,
    },
    staging: [
        [0.1376, 0.7424],
        [1.0414, 0.5369],
        [0.5419, 0.7848],
        [0.9134, 0.6438],
        [3.1570, 0.5869],
        [1.0323, 0.5610],
    ],
};

pub(crate) static SLEEP: Style = Style {
    name: "sleep",
    writing: Writing {
        harmony: HarmonyStyle { major_share: 0.42 },
        // the slower half of the one card's `tempo bpm` quartiles before #93
        // split it by style; #93's step 3 fits SLEEP to its own playlist
        bpm: Spread::between(73.0, 77.0),
        lead: LeadStyle {
            lo: 60,
            hi: 79,
            rest_p: Some(0.25),
            // under R10, the sleepy register, but over the keys' single tones,
            // so the lead stays the top line
            note_p: 0.35,
            velocity: Spread {
                base: 0.25,
                span: 0.12,
            },
            dur: 2.2,
        },
        keys: KeysStyle {
            density: Density::Fixed(0.15),
            velocity: Spread {
                base: 0.35,
                span: 0.25,
            },
            comp: Comp::Single,
            dur: 1.1,
        },
        drums: Grooves::Fixed(&SLEEP_GROOVE),
        bass: BassStyle {
            fifth_p: 0.2,
            pickup_p: None,
            velocity: Spread {
                base: 0.55,
                span: 0.08,
            },
        },
        voices: &[(LeadVoice::Vibraphone, 35), (LeadVoice::EpVel, 65)],
    },
    sound: Sound {
        pad: PadTone {
            cutoff_hz: 2200.0,
            swell_hz: 0.15,
            swell_depth: 0.04,
            peak: 0.75,
        },
        chord: ChordTone {
            ring_s: 1.2,
            attack_s: 0.5,
            attack_step_s: 0.1,
            release_s: 1.8,
            h2: 0.22,
            h3: 0.05,
            gain: 0.8,
        },
        sparkle: NoteRecipe {
            cutoff_hz: 2800.0,
            peak: 0.5,
        },
        keys: NoteRecipe {
            cutoff_hz: 2000.0,
            peak: 0.7,
        },
        bass: NoteRecipe {
            cutoff_hz: 700.0,
            peak: 0.82,
        },
        drums: HitRecipe {
            cutoff_hz: 6000.0,
            drive: 2.0,
            peak: 0.8,
        },
        bed: Bed::Breathing,
    },
    staging: [
        [0.2049, 0.7833],
        [1.1510, 0.5543],
        [1.0453, 0.6983],
        [1.3995, 0.6367],
        [3.0763, 0.6012],
        [1.3370, 0.6117],
    ],
};
