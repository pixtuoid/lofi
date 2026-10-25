//! What both listening tools select and play: a batch of takes, each the first
//! song of a seed's run, heard at one of pixtuoid's office tiers through either
//! engine.

use std::io;
use std::ops::Range;
use std::time::{Duration, Instant};

use pixtuoid_lofi::audition::{self, Lane, Score, TIER_INTENSITIES, lane_gain, render_lane};
use pixtuoid_lofi::dsp::NoiseStream;
use pixtuoid_lofi::{Controls, Frame, Performance};

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
}

/// Which engine renders a take. Both play the same score for as long, the
/// whole-lane engine its loop unarranged, so a batch from each is a blind A/B
/// of the stream and its song's form against the loop as the office played it
/// before.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) enum Engine {
    /// The lanes rendered whole as one loop each, then looped and mixed as long
    /// as the song: how pixtuoid's office played lofi before the stream, and
    /// before songs had a form.
    WholeLane,
    /// A `Performance`, which is what lofi ships, so a sound change is heard
    /// as its listeners will hear it.
    #[default]
    Stream,
}

impl Engine {
    const ALL: [Engine; 2] = [Engine::WholeLane, Engine::Stream];

    /// The `--engine` spelling.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Engine::WholeLane => "whole",
            Engine::Stream => "stream",
        }
    }
}

/// A batch `--batch` names, which picks its own moods, tiers and seeds.
#[derive(Clone, Copy)]
pub(crate) enum Named {
    /// What `ab_judge.py` scores a sound change on.
    Judge,
}

impl Named {
    const ALL: [Named; 1] = [Named::Judge];

    /// The `--batch` spelling.
    fn name(self) -> &'static str {
        match self {
            Named::Judge => "judge",
        }
    }
}

/// A sweep `--sweep` asks for: the controls each of its takes starts under,
/// and the calls it makes after, each heard when [`Performance::set_controls`]
/// latches it. A call sits in the middle of a bar: a latch falls at a bar
/// line, which a call on it lands on either side of by the latch's timing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Sweep {
    /// Asks for Empty, Busy over the take's middle half, then Empty again,
    /// at full day.
    Intensity,
    /// Asks for daylight from day to night across the first song, a step
    /// every bar, and night for the songs after it, [`DUSK_SONGS`] in all.
    Dusk,
    /// Daylight held at each of [`TWILIGHT`]'s, between night and day.
    Twilight,
    /// Asks for rain over the take's middle half, dry around it, at full day
    /// and a moderate room.
    Rain,
}

/// The songs the dusk sweep spans: a night song follows its dusk, since a
/// song's family latches at its first bar.
const DUSK_SONGS: u64 = 2;

/// The twilight sweep's daylights, named for their takes: one in each
/// family, between its ends.
const TWILIGHT: [(&str, f32); 2] = [("quarter", 0.25), ("threequarters", 0.75)];

impl Sweep {
    pub(crate) const ALL: [Sweep; 4] =
        [Sweep::Intensity, Sweep::Dusk, Sweep::Twilight, Sweep::Rain];

    /// The `--sweep` spelling.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Sweep::Intensity => "intensity",
            Sweep::Dusk => "dusk",
            Sweep::Twilight => "twilight",
            Sweep::Rain => "rain",
        }
    }

    /// Each take's controls to start under, named for what the sweep holds
    /// still in it.
    fn starts(self) -> Vec<(&'static str, Controls)> {
        let (day, moderate) = (MOODS[0], Tier::Moderate);
        let at_day = Controls::default().with_daylight(day.1);
        match self {
            Sweep::Intensity => vec![(day.0, at_day.with_intensity(Tier::Empty.intensity()))],
            Sweep::Dusk | Sweep::Rain => {
                vec![(moderate.name(), at_day.with_intensity(moderate.intensity()))]
            }
            Sweep::Twilight => (TWILIGHT.iter())
                .map(|&(name, daylight)| (name, at_day.with_daylight(daylight)))
                .collect(),
        }
    }

    /// The songs a take spans unless `--songs` says.
    fn songs(self) -> u64 {
        match self {
            Sweep::Dusk => DUSK_SONGS,
            Sweep::Intensity | Sweep::Twilight | Sweep::Rain => 1,
        }
    }

    /// What the sweep asks for in the middle of bar `bar` of a take `bars`
    /// long, of songs `song_bars` long, `now` being what it asked for last.
    fn call(self, bar: usize, song_bars: usize, bars: usize, now: Controls) -> Option<Controls> {
        match self {
            Sweep::Intensity => middle_half(bar, bars).map(|on| {
                let tier = if on { Tier::Busy } else { Tier::Empty };
                now.with_intensity(tier.intensity())
            }),
            Sweep::Rain => middle_half(bar, bars).map(|on| now.with_rain(f32::from(u8::from(on)))),
            Sweep::Dusk => (bar < song_bars).then(|| {
                let [(_, day), (_, night)] = MOODS;
                let done = (bar + 1) as f32 / song_bars as f32;
                now.with_daylight(day + (night - day) * done)
            }),
            Sweep::Twilight => None,
        }
    }
}

/// Whether bar `bar` of a take `bars` long starts its middle half (`true`) or
/// ends it (`false`), if either.
fn middle_half(bar: usize, bars: usize) -> Option<bool> {
    let quarter = bars / 4;
    match bar {
        b if b == quarter => Some(true),
        b if b == bars - quarter => Some(false),
        _ => None,
    }
}

/// What a take plays: each song's controls, which its first bar latches, and
/// its length; then each call its sweep makes, at its frame in the take.
#[derive(Default)]
pub(crate) struct Timeline {
    pub(crate) songs: Vec<(Controls, usize)>,
    pub(crate) calls: Vec<(usize, Controls)>,
}

/// `verdict.py`'s `MIN_UNTIED`, which `test_verdict.py` reads here: the
/// untied pairs each mood of the judge batch is built to reach.
pub(crate) const JUDGE_MIN_UNTIED: usize = 8;

/// The judge batch's seeds at `empty`, per mood in [`MOODS`] order: the fewest
/// from 0 that give every lead voice the mood draws [`JUDGE_MIN_UNTIED`] takes,
/// since `empty` is the tier where the lead sings alone over no beat or keys.
const JUDGE_EMPTY_SEEDS: [u64; MOODS.len()] = [29, 28];

/// The judge batch's seeds at each louder tier: [`JUDGE_MIN_UNTIED`] or more
/// between them, so a change to a lane `empty` mutes still reaches the floor.
const JUDGE_TIER_SEEDS: u64 = JUDGE_MIN_UNTIED.div_ceil(2) as u64;

/// The sweeps the judge batch plays, each a group of its own: those only a
/// listener can judge, which no closed-form test pins.
const JUDGE_SWEEPS: [Sweep; 2] = [Sweep::Twilight, Sweep::Rain];

/// The seeds each judged sweep spreads over its takes: room above
/// [`JUDGE_MIN_UNTIED`] for the pairs a change leaves tied (opinion).
const JUDGE_SWEEP_SEEDS: u64 = 12;

/// The lanes' noise seed: any fixed value, so a take renders alike on every run
/// and a base and a candidate render differ only by their change.
const LANE_SEED: u64 = 9;

/// A usage error. These tools render what the A/B gate judges, so an unusable
/// argument fails instead of rendering or measuring a DIFFERENT take than the
/// one asked for.
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

/// The takes a run selects: `--seeds` seeds from `--start`, each `--songs`
/// songs from the run's first, of one `--mood` heard at `--tier` or under a
/// `--sweep`, through `--engine`; or the `--batch` named.
#[derive(Clone, Copy)]
pub(crate) struct Batch {
    mood: usize,
    seeds: u64,
    start: u64,
    /// The `--tier` given.
    pub(crate) tier: Option<Tier>,
    pub(crate) engine: Engine,
    /// The `--batch` given.
    pub(crate) named: Option<Named>,
    /// The `--sweep` given, and which of its takes' starts this batch plays.
    pub(crate) sweep: Option<Sweep>,
    sweep_start: usize,
    /// The `--songs` given.
    pub(crate) songs: Option<u64>,
    /// Whether `--mood`, `--seeds`, `--start` or `--tier` was given, which a
    /// named batch picks itself.
    picked: bool,
    /// Whether `--mood` or `--tier` was given, which a sweep sets itself.
    controlled: bool,
}

impl Default for Batch {
    fn default() -> Self {
        Self {
            mood: 0,
            seeds: 12,
            start: 0,
            tier: None,
            engine: Engine::default(),
            named: None,
            sweep: None,
            sweep_start: 0,
            songs: None,
            picked: false,
            controlled: false,
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
        if matches!(flag, "--mood" | "--seeds" | "--start" | "--tier") {
            self.picked = true;
        }
        if matches!(flag, "--mood" | "--tier") {
            self.controlled = true;
        }
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
            "--batch" => {
                self.named = Some(
                    Named::ALL[parse_choice(args.next(), flag, &Named::ALL.map(Named::name))?],
                );
            }
            "--sweep" => {
                self.sweep = Some(
                    Sweep::ALL[parse_choice(args.next(), flag, &Sweep::ALL.map(Sweep::name))?],
                );
            }
            "--songs" => {
                let songs = parse_count(args.next(), flag)?;
                if songs == 0 {
                    return Err(usage_error("--songs must be at least 1".into()));
                }
                self.songs = Some(songs);
            }
            _ => return Ok(false),
        }
        if self.picked && self.named.is_some() {
            return Err(usage_error(
                "--batch picks its own moods, tiers and seeds, so it takes no --mood, --seeds, --start or --tier"
                    .into(),
            ));
        }
        if self.named.is_some() && (self.sweep.is_some() || self.songs.is_some()) {
            return Err(usage_error(
                "--batch plays each take's first song under fixed controls, so it takes no --sweep or --songs"
                    .into(),
            ));
        }
        if self.controlled && self.sweep.is_some() {
            return Err(usage_error(
                "--sweep sets its own controls, so it takes no --mood or --tier".into(),
            ));
        }
        Ok(true)
    }

    /// The batches the run plays: one per start of its sweep, each of its
    /// named batch's, or this one, through its engine.
    pub(crate) fn runs(&self) -> Vec<Batch> {
        if let Some(sweep) = self.sweep {
            return (0..sweep.starts().len())
                .map(|sweep_start| Batch {
                    sweep_start,
                    ..*self
                })
                .collect();
        }
        match self.named {
            None => vec![*self],
            Some(Named::Judge) => {
                let at = Batch {
                    named: None,
                    picked: true,
                    ..*self
                };
                let tiers = (0..MOODS.len()).flat_map(|mood| {
                    [
                        (Tier::Empty, JUDGE_EMPTY_SEEDS[mood]),
                        (Tier::Moderate, JUDGE_TIER_SEEDS),
                        (Tier::Busy, JUDGE_TIER_SEEDS),
                    ]
                    .map(|(tier, seeds)| Batch {
                        mood,
                        seeds,
                        start: 0,
                        tier: Some(tier),
                        ..at
                    })
                });
                // a sweep's takes split its seeds, so its group holds each once
                let sweeps = JUDGE_SWEEPS.into_iter().flat_map(move |sweep| {
                    let starts = sweep.starts().len();
                    let seeds = JUDGE_SWEEP_SEEDS / starts as u64;
                    (0..starts).map(move |sweep_start| Batch {
                        sweep: Some(sweep),
                        sweep_start,
                        seeds,
                        start: sweep_start as u64 * seeds,
                        ..at
                    })
                });
                tiers.chain(sweeps).collect()
            }
        }
    }

    /// The take's first name: its sweep's, or else its mood's.
    pub(crate) fn mood_name(&self) -> &'static str {
        self.sweep.map_or(MOODS[self.mood].0, Sweep::name)
    }

    /// The take's second name: what its sweep holds still, or else its tier.
    pub(crate) fn mix_name(&self) -> &'static str {
        self.sweep.map_or(self.tier().name(), |sweep| {
            sweep.starts()[self.sweep_start].0
        })
    }

    pub(crate) fn seeds(&self) -> Range<u64> {
        self.start..self.start + self.seeds
    }

    /// The tier the batch is heard at.
    pub(crate) fn tier(&self) -> Tier {
        self.tier.unwrap_or_default()
    }

    /// What every take starts under: its sweep's start, or else the mood's
    /// daylight and the tier's intensity.
    pub(crate) fn controls(&self) -> Controls {
        match self.sweep {
            Some(sweep) => sweep.starts()[self.sweep_start].1,
            None => Controls::default()
                .with_daylight(MOODS[self.mood].1)
                .with_intensity(self.tier().intensity()),
        }
    }

    /// The songs a take plays.
    pub(crate) fn songs(&self) -> u64 {
        (self.songs.or(self.sweep.map(Sweep::songs))).unwrap_or(1)
    }

    /// The score of `seed`'s take's first song, the run's first, whichever
    /// engine renders it.
    pub(crate) fn score(&self, seed: u64) -> Score {
        audition::song_score(seed, 0, self.controls())
    }

    /// What `seed`'s take plays: each song under the controls asked for last
    /// before its first bar, and each call at its bar's middle frame.
    pub(crate) fn timeline(&self, seed: u64) -> Timeline {
        let songs = self.songs();
        let mut asked = self.controls();
        let mut timeline = Timeline::default();
        let mut song_start = 0;
        for k in 0..songs {
            let bars = audition::bar_frames(seed, k, asked);
            let len = audition::song_len(seed, k, asked);
            timeline.songs.push((asked, len));
            // every song has as many bars, whatever its tempo
            let take_bars = bars.len() * songs as usize;
            for (j, &onset) in bars.iter().enumerate() {
                let end = bars.get(j + 1).copied().unwrap_or(len);
                let bar = k as usize * bars.len() + j;
                if let Some(next) =
                    (self.sweep).and_then(|sweep| sweep.call(bar, bars.len(), take_bars, asked))
                {
                    timeline
                        .calls
                        .push((song_start + onset.midpoint(end), next));
                    asked = next;
                }
            }
            song_start += len;
        }
        timeline
    }

    /// `seed`'s take as a `Performance` plays it, each call made at its
    /// frame.
    pub(crate) fn perform(&self, seed: u64) -> Vec<Frame> {
        self.perform_with(seed, |performance, out| performance.render(out))
    }

    /// [`perform`](Self::perform), each stretch between its calls played by
    /// `render`.
    pub(crate) fn perform_with(
        &self,
        seed: u64,
        mut render: impl FnMut(&mut Performance, &mut [Frame]),
    ) -> Vec<Frame> {
        let timeline = self.timeline(seed);
        let mut frames = vec![Frame::default(); timeline.songs.iter().map(|&(_, len)| len).sum()];
        let mut performance = Performance::new(seed, self.controls());
        let mut done = 0;
        for &(at, controls) in &timeline.calls {
            render(&mut performance, &mut frames[done..at]);
            performance.set_controls(controls);
            done = at;
        }
        render(&mut performance, &mut frames[done..]);
        frames
    }
}

/// Each lane's gain under `controls`, in [`Lane::ALL`] order: lofi's, at the
/// intensity a `Performance` mixes it at.
pub(crate) fn lane_gains(controls: Controls) -> [f32; Lane::ALL.len()] {
    Lane::ALL.map(|lane| lane_gain(lane, controls.intensity))
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

/// `beds` at `gains`, each looping on its own length, summed over `frames`
/// frames, and mastered at
/// its [`Performance::MASTER_GAIN`]: the whole-lane mix, as long and at the stage where
/// the engine's own mix meets its ceiling.
pub(crate) fn soak_mix(beds: &[Vec<f32>], gains: &[f32], frames: usize) -> Vec<f32> {
    let mut mix = vec![0.0f32; frames];
    for (bed, &gain) in beds.iter().zip(gains) {
        let gain = gain * Performance::MASTER_GAIN as f32;
        for (i, slot) in mix.iter_mut().enumerate() {
            *slot += bed[i % bed.len()] * gain;
        }
    }
    mix
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    /// A default batch after `flag value`.
    fn parsed(flag: &str, value: &str) -> io::Result<Batch> {
        let mut batch = Batch::default();
        let applied = batch.parse_flag(flag, &mut std::iter::once(value.to_owned()))?;
        assert!(applied, "{flag} is the batch's");
        Ok(batch)
    }

    #[test]
    fn every_tier_and_mood_plays_under_the_controls_it_names() {
        let [empty, moderate, busy] = TIER_INTENSITIES;
        for (name, intensity) in [("empty", empty), ("moderate", moderate), ("busy", busy)] {
            let batch = parsed("--tier", name).expect("the tier parses");
            assert_eq!(batch.controls().intensity, intensity, "{name}");
            let gains = Lane::ALL.map(|lane| lane_gain(lane, intensity));
            assert_eq!(
                lane_gains(batch.controls()),
                gains,
                "{name}'s whole-lane mix"
            );
        }
        assert!(parsed("--tier", "loud").is_err());
        let [day, night] =
            ["day", "night"].map(|mood| parsed("--mood", mood).map(|b| b.score(0)).ok());
        assert!(day != night, "a night take plays a day song");
    }

    #[test]
    fn the_judge_batch_gives_every_lead_voice_the_floor_at_empty_and_no_more() {
        let judge = parsed("--batch", "judge").expect("the batch parses").runs();
        for (mood, (name, _)) in MOODS.iter().enumerate() {
            let empty = (judge.iter())
                .find(|run| run.mood == mood && run.tier == Some(Tier::Empty))
                .expect("the judge batch plays each mood at empty");
            let voices = |seeds: Range<u64>| {
                let mut counts = BTreeMap::new();
                for seed in seeds {
                    *counts
                        .entry(empty.score(seed).lead_voice_name())
                        .or_insert(0) += 1;
                }
                counts
            };
            // every voice the mood draws over far more seeds than the batch has
            let pool = voices(0..256);
            let short = |counts: &BTreeMap<&str, usize>| {
                (pool.keys())
                    .find(|v| counts.get(*v).is_none_or(|&n| n < JUDGE_MIN_UNTIED))
                    .copied()
            };
            let seeds = empty.seeds();
            assert_eq!(
                short(&voices(seeds.clone())),
                None,
                "{name}'s lead voices at empty"
            );
            // one seed fewer would leave a voice short, so no pair is spare
            assert!(
                short(&voices(seeds.start..seeds.end - 1)).is_some(),
                "{name} has spare seeds at empty"
            );
        }
    }

    #[test]
    fn the_judge_batch_plays_each_of_its_sweeps_over_its_own_seeds() {
        let judge = parsed("--batch", "judge").expect("the batch parses").runs();
        for sweep in JUDGE_SWEEPS {
            let mut seeds: Vec<u64> = (judge.iter())
                .filter(|run| run.sweep == Some(sweep))
                .flat_map(Batch::seeds)
                .collect();
            seeds.sort_unstable();
            seeds.dedup();
            assert_eq!(
                seeds.len() as u64,
                JUDGE_SWEEP_SEEDS,
                "{sweep:?}'s distinct seeds"
            );
            let takes = judge.iter().filter(|run| run.sweep == Some(sweep));
            assert_eq!(
                takes.map(|run| run.seeds().count() as u64).sum::<u64>(),
                JUDGE_SWEEP_SEEDS,
                "{sweep:?}'s takes"
            );
        }
        assert!(
            judge
                .iter()
                .all(|run| run.sweep.is_none_or(|s| JUDGE_SWEEPS.contains(&s)))
        );
    }

    #[test]
    fn a_named_batch_takes_no_pick_of_its_own() {
        for pick in [
            ["--tier", "busy"],
            ["--mood", "night"],
            ["--seeds", "3"],
            ["--start", "2"],
        ] {
            for order in [[["--batch", "judge"], pick], [pick, ["--batch", "judge"]]] {
                let mut batch = Batch::default();
                let mut args = order.concat().into_iter().map(str::to_owned);
                let mut outcome = Ok(true);
                while let Some(flag) = args.next() {
                    outcome = outcome.and_then(|_| batch.parse_flag(&flag, &mut args));
                }
                assert!(outcome.is_err(), "{order:?} parsed");
            }
        }
    }

    /// Every take each sweep renders, with its seed.
    fn sweep_takes() -> Vec<(Sweep, Batch, u64)> {
        (Sweep::ALL.iter())
            .flat_map(|&sweep| {
                let runs = parsed("--sweep", sweep.name())
                    .expect("the sweep parses")
                    .runs();
                runs.into_iter()
                    .flat_map(move |run| run.seeds().map(move |seed| (sweep, run, seed)))
            })
            .collect()
    }

    #[test]
    fn every_sweep_calls_in_the_middle_of_a_bar_for_every_seed() {
        for (sweep, run, seed) in sweep_takes() {
            let timeline = run.timeline(seed);
            let mut song_start = 0;
            let mut calls = timeline.calls.iter().peekable();
            for (k, &(controls, len)) in timeline.songs.iter().enumerate() {
                let bars = audition::bar_frames(seed, k as u64, controls);
                while let Some(&&(at, _)) = calls.peek()
                    && at < song_start + len
                {
                    let at = at - song_start;
                    let bar = bars
                        .iter()
                        .rposition(|&start| start <= at)
                        .expect("a call before the song's first bar");
                    let end = bars.get(bar + 1).copied().unwrap_or(len);
                    assert!(
                        bars[bar] < at,
                        "{sweep:?} seed {seed}: a call on bar {bar}'s line"
                    );
                    assert!(
                        (at - bars[bar]).abs_diff(end - at) <= 1,
                        "{sweep:?} seed {seed}: a call off bar {bar}'s middle"
                    );
                    calls.next();
                }
                song_start += len;
            }
            assert_eq!(
                calls.next(),
                None,
                "{sweep:?} seed {seed}: a call past the take"
            );
        }
    }

    #[test]
    fn every_sweep_asks_for_the_controls_it_names() {
        let [(_, day), (_, night)] = MOODS;
        let [empty, _, busy] = TIER_INTENSITIES;
        for (sweep, run, seed) in sweep_takes() {
            let timeline = run.timeline(seed);
            let asked: Vec<Controls> = timeline.calls.iter().map(|&(_, c)| c).collect();
            match sweep {
                Sweep::Intensity => {
                    assert_eq!(
                        run.controls(),
                        Controls::default().with_daylight(day).with_intensity(empty)
                    );
                    let tiers: Vec<f32> = asked.iter().map(|c| c.intensity).collect();
                    assert_eq!(tiers, [busy, empty], "seed {seed}");
                    assert!(asked.iter().all(|c| c.daylight == day), "seed {seed}");
                }
                Sweep::Dusk => {
                    assert_eq!(run.controls().daylight, day);
                    let families: Vec<f32> =
                        timeline.songs.iter().map(|&(c, _)| c.daylight).collect();
                    assert_eq!(
                        families,
                        [day, night],
                        "seed {seed}: a day song, then a night one"
                    );
                    let daylights: Vec<f32> = asked.iter().map(|c| c.daylight).collect();
                    assert!(
                        daylights.windows(2).all(|w| w[1] < w[0]),
                        "seed {seed}: dusk turns back"
                    );
                    assert_eq!(daylights.last(), Some(&night), "seed {seed}");
                }
                Sweep::Rain => {
                    assert_eq!(run.controls().rain, 0.0, "the rain sweep starts dry");
                    let rains: Vec<f32> = asked.iter().map(|c| c.rain).collect();
                    assert_eq!(rains, [1.0, 0.0], "seed {seed}");
                    assert!(
                        asked.iter().all(|c| c.with_rain(0.0) == run.controls()),
                        "seed {seed}: rain moved another control"
                    );
                }
                Sweep::Twilight => {
                    assert!(asked.is_empty(), "twilight holds its daylight");
                    let daylight = run.controls().daylight;
                    assert!(
                        night < daylight && daylight < day,
                        "{daylight} is no twilight"
                    );
                }
            }
        }
    }

    #[test]
    fn the_twilight_sweep_plays_one_take_in_each_family() {
        let runs = parsed("--sweep", "twilight")
            .expect("the sweep parses")
            .runs();
        let [quarter, threequarters] =
            [0, 1].map(|i| audition::song_score(0, 0, runs[i].controls()));
        assert!(
            quarter != threequarters,
            "both twilight takes compose one family's song"
        );
    }

    #[test]
    fn songs_plays_that_many_songs_of_the_run() {
        let mut batch = parsed("--songs", "3").expect("--songs parses");
        batch.engine = Engine::Stream;
        let c = batch.controls();
        let lens: Vec<usize> = batch
            .timeline(1)
            .songs
            .iter()
            .map(|&(_, len)| len)
            .collect();
        assert_eq!(
            lens,
            (0..3)
                .map(|k| audition::song_len(1, k, c))
                .collect::<Vec<_>>()
        );
        assert!(parsed("--songs", "0").is_err());
    }

    #[test]
    fn a_sweep_sets_its_own_controls_and_a_named_batch_its_own_songs() {
        for order in [
            [["--sweep", "dusk"], ["--mood", "night"]],
            [["--tier", "busy"], ["--sweep", "dusk"]],
            [["--batch", "judge"], ["--sweep", "dusk"]],
            [["--songs", "2"], ["--batch", "judge"]],
        ] {
            let mut batch = Batch::default();
            let mut args = order.concat().into_iter().map(str::to_owned);
            let mut outcome = Ok(true);
            while let Some(flag) = args.next() {
                outcome = outcome.and_then(|_| batch.parse_flag(&flag, &mut args));
            }
            assert!(outcome.is_err(), "{order:?} parsed");
        }
        let mut batch = Batch::default();
        let mut args = ["--sweep", "dusk", "--seeds", "3", "--songs", "4"]
            .into_iter()
            .map(str::to_owned);
        while let Some(flag) = args.next() {
            assert!(batch.parse_flag(&flag, &mut args).is_ok(), "{flag} parsed");
        }
    }

    #[test]
    fn every_engine_name_parses_to_its_engine() {
        for (name, engine) in [("whole", Engine::WholeLane), ("stream", Engine::Stream)] {
            let batch = parsed("--engine", name);
            assert_eq!(batch.map(|b| b.engine).ok(), Some(engine), "{name}");
        }
        assert!(parsed("--engine", "both").is_err());
    }
}
