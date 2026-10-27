//! Every constant the source pins to a measurement, measured again and
//! printed beside it: a sound change moves them, and no test reads them off a
//! run.

use std::ops::Range;
use std::thread;

use pixtuoid_lofi::Frame;
use pixtuoid_lofi::audition::{
    self, LUFS_SEEDS, Lane, SONG_LUFS_SD, TIER_INTENSITIES, WHOLE_LANE_LUFS_SD,
};

use crate::batch::{Batch, Engine, MOODS, SOLO_SAFE_LUFS, Tier, lane_gains};
use crate::{parse, staging, stream_lufs, whole_lufs};

/// Prints each pinned constant as the source holds it, from a run.
pub(crate) fn print() {
    spreads("WHOLE_LANE_LUFS_SD", Engine::WholeLane, &WHOLE_LANE_LUFS_SD);
    spreads("SONG_LUFS_SD", Engine::Stream, &SONG_LUFS_SD);
    solo();
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

/// `f` of each seed, each on its own thread.
fn over<T: Send>(seeds: Range<u64>, f: impl Fn(u64) -> T + Sync) -> Vec<T> {
    let f = &f;
    thread::scope(|scope| {
        let runs: Vec<_> = seeds.map(|seed| scope.spawn(move || f(seed))).collect();
        (runs.into_iter())
            .map(|run| run.join().expect("a render panicked"))
            .collect()
    })
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
            Engine::WholeLane => whole_lufs(&batch.whole(seed, lane_gains).remove(0).mix()),
            Engine::Stream => stream_lufs(&batch.perform(seed)),
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
    stream_lufs(take) - 20.0 * f64::from(peak).log10()
}
