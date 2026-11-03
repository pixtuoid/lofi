//! The lane recipe both chains read.

use super::voices::{ChordTone, bass_note, chord, drum_hit, ep_pluck_vel};
use super::{NoteVoice, SR, bar_onset, onset};
use crate::compose::{GEN_LOOP_BARS, Hit, Lane, Note, Score, kicks};
use crate::dsp::NoiseStream;

/// Every recipe a style's lanes sound with: the ONE authority
/// [`render_lane`](super::render_lane) and the lane chains read.
pub(crate) struct Sound {
    pub(crate) pad: PadTone,
    pub(crate) chord: ChordTone,
    pub(crate) sparkle: NoteRecipe,
    pub(crate) keys: NoteRecipe,
    pub(crate) bass: NoteRecipe,
    pub(crate) drums: HitRecipe,
    pub(crate) bed: Bed,
}

/// A generated pad's tone before the tape: a low-pass, then a slow swell.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PadTone {
    pub(crate) cutoff_hz: f32,
    pub(crate) swell_hz: f32,
    pub(crate) swell_depth: f32,
    /// Its whole-lane stem's peak.
    pub(crate) peak: f32,
}

impl PadTone {
    /// The swell's gain at its phase in radians, which the caller's clock
    /// keeps as [`room_tone`](super::room_tone)'s does.
    pub(crate) fn swell(self, phase: f32) -> f32 {
        1.0 + self.swell_depth * phase.sin()
    }

    /// The swell's gain on sample `i` of a whole-lane pad.
    pub(super) fn bed_swell(self, i: usize) -> f32 {
        self.swell(std::f32::consts::TAU * self.swell_hz * i as f32 / SR)
    }
}

/// A note lane's recipe: its stem's low-pass corner and its stem's peak.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct NoteRecipe {
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

/// How a style's texture bed runs against its groove.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Bed {
    /// On a length of its own, out of phase with the groove.
    Free,
    /// A loop long, ducking under every kick its groove lays, so the room
    /// breathes with it.
    Breathing,
}

/// What one lane of a score plays: the ONE mapping from a lane to its events
/// and voice, read by [`render_lane`](super::render_lane) and by the stream
/// alike, each lane in the `sound` of its song's style.
pub(crate) enum Part<'a> {
    /// A chord a bar, each held through the bar and its ring.
    Chords {
        chords: &'a [[u8; 4]; GEN_LOOP_BARS],
        bar_s: f32,
        sound: &'static Sound,
    },
    /// Notes, played through `voice` as `recipe` reads a style's sound,
    /// which the stream grades between styles.
    Notes {
        events: &'a [Note],
        voice: NoteVoice,
        recipe: fn(&Sound) -> NoteRecipe,
        sound: &'static Sound,
    },
    /// Drum hits, played as `recipe` reads a style's sound.
    Hits {
        hits: &'a [Hit],
        recipe: fn(&Sound) -> HitRecipe,
        sound: &'static Sound,
    },
    /// The texture: a bed with no notes, which [`Part::ducks`] under the
    /// kicks of `hits`.
    Bed {
        sound: &'static Sound,
        hits: &'a [Hit],
    },
}

pub(crate) fn part(score: &Score, lane: Lane) -> Part<'_> {
    let sound = &score.style.sound;
    match lane {
        Lane::Pad => Part::Chords {
            chords: &score.bar_chords,
            bar_s: score.bar_s(),
            sound,
        },
        Lane::Sparkle => Part::Notes {
            events: &score.sparkle,
            voice: score.lead_voice.voice(),
            recipe: |sound| sound.sparkle,
            sound,
        },
        Lane::Keys => Part::Notes {
            events: &score.keys,
            voice: ep_pluck_vel,
            recipe: |sound| sound.keys,
            sound,
        },
        Lane::Bass => Part::Notes {
            events: &score.bass,
            voice: bass_note,
            recipe: |sound| sound.bass,
            sound,
        },
        Lane::Drums => Part::Hits {
            hits: &score.drums,
            recipe: |sound| sound.drums,
            sound,
        },
        Lane::Texture => Part::Bed {
            sound,
            hits: &score.drums,
        },
    }
}

impl Part<'_> {
    /// Each note's first sample in the loop at `sr`, in event order: where
    /// [`render_lane`](super::render_lane) places it at
    /// [`SAMPLE_RATE`](crate::dsp::SAMPLE_RATE).
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

    /// The times in the loop a bed ducks under, in order: a
    /// [`Bed::Breathing`] bed's kicks. The other lanes' sidechain is
    /// `whole::sidechained`'s.
    pub(crate) fn ducks(&self) -> Vec<f32> {
        match *self {
            Part::Bed { sound, hits } if sound.bed == Bed::Breathing => kicks(hits),
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
                sound,
            } => (chord(sound.chord, &chords[i], bar_s, sr), 1.0),
            Part::Notes { events, voice, .. } => {
                let n = events[i];
                (voice(n.midi, n.dur, n.vel, sr), 1.0)
            }
            Part::Hits { hits, .. } => {
                let h = hits[i];
                (drum_hit(rng, h.kind, sr), h.gain)
            }
            Part::Bed { .. } => (Vec::new(), 1.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A free bed loops on its own length, out of phase with the groove, so
    /// only a breathing bed, a loop long, ducks: under every kick it lays.
    #[test]
    fn only_a_breathing_bed_ducks_and_under_every_kick() {
        for style in crate::style::STYLES {
            for seed in 0..8 {
                let score = crate::compose::compose(style, seed);
                let ducks = part(&score, Lane::Texture).ducks();
                match style.sound.bed {
                    Bed::Free => assert!(ducks.is_empty(), "{style:?} seed {seed}"),
                    Bed::Breathing => {
                        let mut kicks: Vec<f32> = score.kicks().collect();
                        kicks.sort_by(f32::total_cmp);
                        assert!(!kicks.is_empty());
                        assert_eq!(ducks, kicks, "{style:?} seed {seed}");
                    }
                }
            }
        }
    }
}
