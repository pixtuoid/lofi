//! The lane recipe both chains read.

use super::voices::{ChordTone, bass_note, chord, drum_hit, ep_pluck_vel};
use super::{NoteVoice, SR, bar_onset, onset};
use crate::compose::{DrumKind, GEN_LOOP_BARS, Hit, Lane, Mood, Note, Score};
use crate::dsp::NoiseStream;

/// A generated pad's tone before the tape: a low-pass, then a slow swell.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PadTone {
    pub(crate) cutoff_hz: f32,
    pub(crate) swell_hz: f32,
    swell_depth: f32,
    /// Its whole-lane stem's peak.
    pub(super) peak: f32,
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
    /// keeps as [`room_tone`](super::room_tone)'s does.
    pub(crate) fn swell(self, phase: f32) -> f32 {
        1.0 + self.swell_depth * phase.sin()
    }

    /// The swell's gain on sample `i` of a whole-lane pad.
    pub(super) fn bed_swell(self, i: usize) -> f32 {
        self.swell(std::f32::consts::TAU * self.swell_hz * i as f32 / SR)
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

/// Each lane's recipe: the ONE authority
/// [`render_lane`](super::render_lane) and the lane chains read.
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
/// and voice, read by [`render_lane`](super::render_lane) and by the stream
/// alike.
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
