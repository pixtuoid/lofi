//! What both listening tools select and play: a batch of takes, rendered as
//! pixtuoid renders a take and heard at one of its office tiers.

use std::io;
use std::ops::Range;
use std::time::{Duration, Instant};

use pixtuoid_lofi::dsp::{NoiseStream, SAMPLE_RATE};
use pixtuoid_lofi::{Lane, Mood, Score, render_lane};

/// The `--mood` vocabulary paired with its [`Mood`], so the parse and the error
/// message can't list different spellings.
const MOODS: [(&str, Mood); 2] = [("day", Mood::Day), ("night", Mood::Night)];

/// The lanes in [`Lane::ALL`] order: the ONE spelling the tools print and
/// `--solo` accepts.
pub(crate) const LANE_NAMES: [&str; Lane::ALL.len()] =
    ["pad", "sparkle", "keys", "drums", "texture", "bass"];

/// How busy pixtuoid's office is, which picks each lane's [`tier_gain`].
#[derive(Clone, Copy, Default)]
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
}

/// `lane`'s gain at `tier`: pixtuoid's `TierGain`s
/// (<https://github.com/pixtuoid/pixtuoid/blob/9338cc3841f6d50f99d478a3b2290e2ac96feeec/crates/pixtuoid-scene/src/audio/mod.rs#L98-L107>),
/// copied across repos where no test can pin them, so a LISTEN first diffs
/// pixtuoid's `main` against these with the `procedural-lofi` skill's GENERATOR
/// step 5, which reads each row as `Lane::X => [empty, moderate, busy],`.
pub(crate) fn tier_gain(lane: Lane, tier: Tier) -> f32 {
    let [empty, moderate, busy] = match lane {
        Lane::Pad => [0.75, 0.70, 0.65],
        Lane::Sparkle => [0.70, 0.0, 0.0],
        Lane::Keys => [0.0, 0.60, 0.70],
        Lane::Drums => [0.0, 0.35, 0.60],
        Lane::Texture => [0.78, 0.84, 0.78],
        Lane::Bass => [0.60, 0.70, 0.75],
    };
    match tier {
        Tier::Empty => empty,
        Tier::Moderate => moderate,
        Tier::Busy => busy,
    }
}

/// Soak length target per take — long enough to hear the loop breathe.
const SOAK_SECS: f32 = 90.0;

/// A soak plays at least this many loops, so a take is judged on repeat, the
/// way the office plays it.
const MIN_SOAK_LOOPS: usize = 4;

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
/// heard at `--tier`.
pub(crate) struct Batch {
    mood: usize,
    seeds: u64,
    start: u64,
    /// The `--tier` given.
    pub(crate) tier: Option<Tier>,
}

impl Default for Batch {
    fn default() -> Self {
        Self {
            mood: 0,
            seeds: 12,
            start: 0,
            tier: None,
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
            _ => return Ok(false),
        }
        Ok(true)
    }

    pub(crate) fn mood(&self) -> Mood {
        MOODS[self.mood].1
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
}

/// Every lane of `score` in [`Lane::ALL`] order, drawn from one rng stream as a
/// take renders, each with how long it took.
pub(crate) fn render_take(score: &Score) -> [(Vec<f32>, Duration); Lane::ALL.len()] {
    let mut rng = NoiseStream::new(LANE_SEED);
    Lane::ALL.map(|lane| {
        let started = Instant::now();
        let bed = render_lane(score, lane, &mut rng);
        (bed, started.elapsed())
    })
}

/// `beds` at `gains`, each looping on its own length, summed over a soak of
/// whole take loops (the first bed's length) at least [`SOAK_SECS`] and
/// [`MIN_SOAK_LOOPS`] long.
pub(crate) fn soak_mix(beds: &[Vec<f32>], gains: &[f32]) -> Vec<f32> {
    let take_len = beds[0].len();
    let loops =
        ((SOAK_SECS * SAMPLE_RATE as f32 / take_len as f32).ceil() as usize).max(MIN_SOAK_LOOPS);
    let mut mix = vec![0.0f32; take_len * loops];
    for (bed, &gain) in beds.iter().zip(gains) {
        for (i, slot) in mix.iter_mut().enumerate() {
            *slot += bed[i % bed.len()] * gain;
        }
    }
    mix
}
