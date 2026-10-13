//! What both listening tools select and play: a batch of takes, each the first
//! song of a seed's run, heard at one of pixtuoid's office tiers through either
//! engine.

use std::io;
use std::ops::Range;
use std::time::{Duration, Instant};

use pixtuoid_lofi::audition::{
    self, LOOPS_PER_SONG, Lane, MASTER_GAIN, Score, TIER_INTENSITIES, lane_gain, render_lane,
};
use pixtuoid_lofi::dsp::NoiseStream;
use pixtuoid_lofi::{Controls, Frame};

/// The `--mood` vocabulary paired with the daylight it plays at, so the parse
/// and the error message can't list different spellings.
const MOODS: [(&str, f32); 2] = [("day", 1.0), ("night", 0.0)];

/// The lanes in [`Lane::ALL`] order: the ONE spelling the tools print and
/// `--solo` accepts.
pub(crate) const LANE_NAMES: [&str; Lane::ALL.len()] =
    ["pad", "sparkle", "keys", "drums", "texture", "bass"];

/// How busy pixtuoid's office is, which picks the intensity a take plays at.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) enum Tier {
    Empty,
    /// The taste anchor, `demo_2_moderate` in `docs/phase0-ratification.md`.
    #[default]
    Moderate,
    Busy,
}

impl Tier {
    const ALL: [Tier; 3] = [Tier::Empty, Tier::Moderate, Tier::Busy];

    /// The `--tier` spelling.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Tier::Empty => "empty",
            Tier::Moderate => "moderate",
            Tier::Busy => "busy",
        }
    }

    /// lofi's intensity anchor for the tier.
    pub(crate) fn intensity(self) -> f32 {
        let [empty, moderate, busy] = TIER_INTENSITIES;
        match self {
            Tier::Empty => empty,
            Tier::Moderate => moderate,
            Tier::Busy => busy,
        }
    }

    /// Each lane's gain in the tier's mix, in [`Lane::ALL`] order: lofi's, at
    /// the tier's intensity.
    pub(crate) fn lane_gains(self) -> [f32; Lane::ALL.len()] {
        Lane::ALL.map(|lane| lane_gain(lane, self.intensity()))
    }
}

/// Which engine renders a take. Both play the same score, so a batch from each
/// is a blind A/B of the engines.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) enum Engine {
    /// The lanes rendered whole as one loop each, then looped and mixed: how
    /// pixtuoid's office played lofi before the stream.
    V1,
    /// A `Performance`, which is what lofi ships, so a sound change is heard
    /// as its listeners will hear it.
    #[default]
    V2,
}

impl Engine {
    const ALL: [Engine; 2] = [Engine::V1, Engine::V2];

    /// The `--engine` spelling.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Engine::V1 => "v1",
            Engine::V2 => "v2",
        }
    }
}

/// The lanes' noise seed: any fixed value, so a take renders alike on every run
/// and a base and a candidate render differ only by their change.
const LANE_SEED: u64 = 9;

/// A usage error. These tools ARE the listen gate, so an unusable argument
/// fails instead of rendering or measuring a DIFFERENT take than the one asked
/// for.
pub(crate) fn usage_error(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, msg)
}

fn parse_count(v: Option<String>, flag: &str) -> io::Result<u64> {
    let raw = v.ok_or_else(|| usage_error(format!("{flag} needs a non-negative integer")))?;
    raw.parse()
        .map_err(|_| usage_error(format!("{flag} needs a non-negative integer, got {raw:?}")))
}

/// `v`'s index in `names`, the values `flag` accepts.
pub(crate) fn parse_choice(v: Option<String>, flag: &str, names: &[&str]) -> io::Result<usize> {
    let v = v.unwrap_or_default();
    names
        .iter()
        .position(|&name| name == v)
        .ok_or_else(|| usage_error(format!("unknown {flag} {v:?}; valid: {}", names.join("|"))))
}

/// The takes a run selects: `--seeds` seeds of one `--mood` from `--start`,
/// heard at `--tier` through `--engine`.
pub(crate) struct Batch {
    mood: usize,
    seeds: u64,
    start: u64,
    /// The `--tier` given.
    pub(crate) tier: Option<Tier>,
    pub(crate) engine: Engine,
}

impl Default for Batch {
    fn default() -> Self {
        Self {
            mood: 0,
            seeds: 12,
            start: 0,
            tier: None,
            engine: Engine::default(),
        }
    }
}

impl Batch {
    /// Applies `flag` when it is one of the batch's, taking its value from
    /// `args`; `Ok(false)` leaves `flag` to the caller.
    pub(crate) fn parse_flag(
        &mut self,
        flag: &str,
        args: &mut impl Iterator<Item = String>,
    ) -> io::Result<bool> {
        match flag {
            "--mood" => self.mood = parse_choice(args.next(), flag, &MOODS.map(|(name, _)| name))?,
            "--seeds" => {
                self.seeds = parse_count(args.next(), flag)?;
                if self.seeds == 0 {
                    return Err(usage_error("--seeds must be at least 1".into()));
                }
            }
            "--start" => self.start = parse_count(args.next(), flag)?,
            "--tier" => {
                self.tier =
                    Some(Tier::ALL[parse_choice(args.next(), flag, &Tier::ALL.map(Tier::name))?]);
            }
            "--engine" => {
                self.engine =
                    Engine::ALL[parse_choice(args.next(), flag, &Engine::ALL.map(Engine::name))?];
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    pub(crate) fn mood_name(&self) -> &'static str {
        MOODS[self.mood].0
    }

    pub(crate) fn seeds(&self) -> Range<u64> {
        self.start..self.start + self.seeds
    }

    /// The tier the batch is heard at.
    pub(crate) fn tier(&self) -> Tier {
        self.tier.unwrap_or_default()
    }

    /// What every take plays under: the mood's daylight, the tier's intensity.
    pub(crate) fn controls(&self) -> Controls {
        Controls::default()
            .with_daylight(MOODS[self.mood].1)
            .with_intensity(self.tier().intensity())
    }

    /// The score `seed`'s take plays, whichever engine renders it.
    pub(crate) fn score(&self, seed: u64) -> Score {
        audition::song_score(seed, 0, self.controls())
    }

    /// `seed`'s take as a `Performance` plays it: one channel of it, since it
    /// plays both alike.
    pub(crate) fn perform(&self, seed: u64) -> io::Result<Vec<f32>> {
        audition::song(seed, 0, self.controls())
            .into_iter()
            .map(|[left, right]: Frame| {
                (left.to_bits() == right.to_bits())
                    .then_some(left)
                    .ok_or_else(|| {
                        io::Error::other("v2 plays its channels apart, so one is not the take")
                    })
            })
            .collect()
    }
}

/// Every lane of `score` in [`Lane::ALL`] order, drawn from one rng stream as a
/// take renders, and how long each took.
pub(crate) fn render_take(
    score: &Score,
) -> ([Vec<f32>; Lane::ALL.len()], [Duration; Lane::ALL.len()]) {
    let mut rng = NoiseStream::new(LANE_SEED);
    let mut took = [Duration::ZERO; Lane::ALL.len()];
    let beds = std::array::from_fn(|i| {
        let started = Instant::now();
        let bed = render_lane(score, Lane::ALL[i], &mut rng);
        took[i] = started.elapsed();
        bed
    });
    (beds, took)
}

/// `beds` at `gains`, each looping on its own length, summed over as many
/// loops of the first bed as a `Performance` plays a song's, and mastered at
/// its [`MASTER_GAIN`]: the whole-lane mix, as long and at the stage where
/// the engine's own mix meets its ceiling.
pub(crate) fn soak_mix(beds: &[Vec<f32>], gains: &[f32]) -> Vec<f32> {
    let mut mix = vec![0.0f32; beds[0].len() * LOOPS_PER_SONG];
    for (bed, &gain) in beds.iter().zip(gains) {
        let gain = gain * MASTER_GAIN as f32;
        for (i, slot) in mix.iter_mut().enumerate() {
            *slot += bed[i % bed.len()] * gain;
        }
    }
    mix
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A default batch after `flag value`.
    fn parsed(flag: &str, value: &str) -> io::Result<Batch> {
        let mut batch = Batch::default();
        let applied = batch.parse_flag(flag, &mut std::iter::once(value.to_owned()))?;
        assert!(applied, "{flag} is the batch's");
        Ok(batch)
    }

    #[test]
    fn every_tier_name_parses_to_its_tier() {
        for (name, tier) in [
            ("empty", Tier::Empty),
            ("moderate", Tier::Moderate),
            ("busy", Tier::Busy),
        ] {
            let batch = parsed("--tier", name);
            assert_eq!(batch.map(|b| b.tier()).ok(), Some(tier), "{name}");
        }
        assert!(parsed("--tier", "loud").is_err());
    }

    #[test]
    fn every_engine_name_parses_to_its_engine() {
        for (name, engine) in [("v1", Engine::V1), ("v2", Engine::V2)] {
            let batch = parsed("--engine", name);
            assert_eq!(batch.map(|b| b.engine).ok(), Some(engine), "{name}");
        }
        assert!(parsed("--engine", "v3").is_err());
    }
}
