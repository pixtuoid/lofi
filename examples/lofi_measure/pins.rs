//! The constants the source pins to a run's measurement, measured again: the
//! loudness spreads, the solo's safe level, the plate's sends and the
//! staging, each printed beside the value the source holds, and the
//! whole-lane rows as the level test should hold them. A sound change moves
//! them, and no test reads them off a run.

use std::num::NonZero;
use std::ops::Range;
use std::thread;

use pixtuoid_lofi::Frame;
use pixtuoid_lofi::audition::{
    self, LUFS_SEEDS, Lane, SONG_LUFS_SD, Stems, TIER_INTENSITIES, WHOLE_LANE_LUFS_SD,
};
use pixtuoid_lofi::dsp::SAMPLE_RATE;

use crate::batch::{Batch, Engine, MOODS, SOLO_SAFE_LUFS, Tier, lane_gains};
use crate::staging::median;
use crate::{frames_lufs, mono_lufs, parse, staging};

/// The seeds the plate's sends were measured over, both moods.
const PLATE_SEEDS: Range<u64> = 0..6;

/// The tier each wet lane's return is read at, the commit that set the
/// sends': the pad and keys at `moderate`, and the lead at `empty`, where it
/// plays alone.
const PLATE_TIERS: [(Lane, Tier); 3] = [
    (Lane::Pad, Tier::Moderate),
    (Lane::Keys, Tier::Moderate),
    (Lane::Sparkle, Tier::Empty),
];

/// Prints each pinned constant as the source holds it, from a run.
pub(crate) fn print() {
    spreads("WHOLE_LANE_LUFS_SD", Engine::WholeLane, &WHOLE_LANE_LUFS_SD);
    spreads("SONG_LUFS_SD", Engine::Stream, &SONG_LUFS_SD);
    solo();
    plate();
    staging::print();
}

/// The batch `--engine`, `--tier` and `--mood` select over `seeds`, as the
/// flags select it.
fn batch(engine: Engine, tier: Tier, mood: &str, seeds: Range<u64>) -> Batch {
    let args = [
        "--engine".into(),
        engine.name().into(),
        "--tier".into(),
        tier.name().into(),
        "--mood".into(),
        mood.into(),
        "--start".into(),
        seeds.start.to_string(),
        "--seeds".into(),
        (seeds.end - seeds.start).to_string(),
    ];
    parse(args.into_iter()).expect("a pinned row's flags parse")
}

/// `f` of each seed in order, as many on their own threads at once as the
/// machine runs in parallel: a take's render holds a few hundred MB.
fn over<T: Send>(seeds: Range<u64>, f: impl Fn(u64) -> T + Sync) -> Vec<T> {
    let f = &f;
    let width = thread::available_parallelism().map_or(1, NonZero::get);
    let seeds: Vec<u64> = seeds.collect();
    (seeds.chunks(width))
        .flat_map(|chunk| {
            thread::scope(|scope| {
                let runs: Vec<_> = chunk
                    .iter()
                    .map(|&seed| scope.spawn(move || f(seed)))
                    .collect();
                (runs.into_iter())
                    .map(|run| run.join().expect("a render panicked"))
                    .collect::<Vec<_>>()
            })
        })
        .collect()
}

/// A `(intensity, daylight, LU)` table's rows, each measured over
/// [`LUFS_SEEDS`] through `engine` at its tier and mood.
fn spreads(name: &str, engine: Engine, rows: &[(f32, f32, f64)]) {
    println!(
        "// {name}: each row's loudness sd over LUFS_SEEDS through the {} engine",
        engine.name()
    );
    for &(intensity, daylight, held) in rows {
        let index = (TIER_INTENSITIES.iter())
            .position(|&i| i == intensity)
            .expect("a pinned row's intensity is a tier's");
        let (mood, _) = (MOODS.into_iter())
            .find(|&(_, d)| d == daylight)
            .expect("a pinned row's daylight is a mood's");
        let batch = batch(engine, Tier::ALL[index], mood, LUFS_SEEDS);
        let lufs = over(LUFS_SEEDS, |seed| match engine {
            Engine::WholeLane => mono_lufs(&batch.whole(seed, lane_gains).remove(0).mix()),
            Engine::Stream => frames_lufs(&batch.perform(seed)),
        });
        println!(
            "(TIER_INTENSITIES[{index}], {daylight:.1}, {:.4}), // held {held:.4}",
            audition::lufs_sd(&lufs)
        );
    }
}

/// [`SOLO_SAFE_LUFS`]: the quietest of every solo take's safe level, both
/// moods and [`LUFS_SEEDS`], as `lofi_audition --solo` renders them.
fn solo() {
    let mut safest: Option<(f64, String)> = None;
    for (mood, _) in MOODS {
        let batch = batch(Engine::WholeLane, Tier::default(), mood, LUFS_SEEDS);
        for lane in Lane::ALL {
            let safe = over(LUFS_SEEDS, |seed| safe_lufs(&batch.take(seed, Some(lane))));
            for (seed, safe) in LUFS_SEEDS.zip(safe) {
                if safest.as_ref().is_none_or(|&(s, _)| safe < s) {
                    safest = Some((safe, format!("{mood} {}, seed {seed}", lane.name())));
                }
            }
        }
    }
    let (safe, take) = safest.expect("a solo take");
    println!(
        "SOLO_SAFE_LUFS: {:.1}, // held {SOLO_SAFE_LUFS:.1}; the loudest safe solo reads {safe:.2} ({take})",
        (safe * 10.0).floor() / 10.0
    );
}

/// The gated level at which `take` peaks at full scale.
fn safe_lufs(take: &[Frame]) -> f64 {
    let peak = take.iter().flatten().fold(0.0f32, |m, v| m.max(v.abs()));
    frames_lufs(take) - 20.0 * f64::from(peak).log10()
}

/// The plate's sends: the send that puts each wet lane's return as far
/// under its dry lane on the fold as `plate.rs` asks, from its return at
/// unity, the median over [`PLATE_SEEDS`] in both moods.
fn plate() {
    println!("// the plate's sends, each from its return at unity over PLATE_SEEDS");
    for (lane, tier) in PLATE_TIERS {
        let (held, under) = audition::plate_wet(lane).expect("a wet lane sends");
        let mut gaps = Vec::new();
        for (mood, _) in MOODS {
            let batch = batch(Engine::Stream, tier, mood, PLATE_SEEDS);
            gaps.extend(over(PLATE_SEEDS, |seed| unity_gap(&batch, seed, lane)));
        }
        let gap = median(gaps);
        println!(
            "{}_SEND_DB: {:.1}, // held {held:.1}; at unity its return reads {gap:.2} LU under its dry lane, for {under}",
            lane.name().to_uppercase(),
            gap - under,
        );
    }
}

/// How far under `lane`'s dry stem its plate return at unity reads on the
/// fold, over `seed`'s first song's second loop, the first one its own tails
/// ring into.
fn unity_gap(batch: &Batch, seed: u64, lane: Lane) -> f64 {
    let mut stems = Stems::default();
    batch.perform_with(seed, |performance, out| stems.render(performance, out));
    let dry = &stems.lanes()[lane as usize];
    let [left, right] = audition::plate_return(dry);
    let wet: Vec<f32> = left
        .iter()
        .zip(&right)
        .map(|(l, r)| (l + r) / 2.0)
        .collect();
    let loop_len = audition::samples_at(batch.score(seed).loop_secs(), SAMPLE_RATE);
    let second = loop_len..2 * loop_len;
    mono_lufs(&dry[second.clone()]) - mono_lufs(&wet[second])
}
