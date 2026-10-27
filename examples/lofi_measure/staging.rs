//! The stream's staging beside what it stands in for: each lane's whole-lane
//! render over [`STAGING_SEEDS`], read for the gains its peak normalizations
//! applied and for its gated loudness. `lanes.rs`'s whole-lane level test holds
//! the loudness's medians.

use pixtuoid_lofi::Controls;
use pixtuoid_lofi::audition::{self, Lane, STAGING_SEEDS, gated_lufs};
use pixtuoid_lofi::dsp::SAMPLE_RATE;

use crate::corpus::spread;

/// The moods the stream stages, by a daylight each plays at.
const MOODS: [(&str, f32); 2] = [("Day", 1.0), ("Night", 0.0)];

pub(crate) fn median(values: Vec<f64>) -> f64 {
    let [_, _, median, _, _] = spread(values);
    median
}

/// Prints the stream's staging, each lane's beside the medians of its
/// whole-lane normalizations and how far the stream's level sits from theirs,
/// and the whole-lane loudness rows, as the stream's source holds them. Each
/// seed's first song renders each lane on its own fork of that seed's noise,
/// as a whole-lane take does.
pub(crate) fn print() {
    let mut rows = Vec::new();
    println!(
        "// staging, then medians over STAGING_SEEDS of each lane's whole-lane normalizations"
    );
    for (mood, daylight) in MOODS {
        let controls = Controls::default().with_daylight(daylight);
        let mut lufs_row = Vec::new();
        let first = audition::song_score(STAGING_SEEDS.start, 0, controls);
        for lane in Lane::ALL {
            let (mut ins, mut outs, mut lufs) = (Vec::new(), Vec::new(), Vec::new());
            for seed in STAGING_SEEDS {
                let score = audition::song_score(seed, 0, controls);
                let mut noise = audition::lane_noise(seed);
                let (buf, [gain_in, gain_out]) =
                    audition::staged_lane(&score, lane, &mut noise[lane as usize]);
                ins.push(f64::from(gain_in));
                outs.push(f64::from(gain_out));
                lufs.push(gated_lufs(&[&buf], SAMPLE_RATE));
            }
            let [gain_in, gain_out] = audition::staging(&first, lane);
            let (median_in, median_out) = (median(ins), median(outs));
            let off_db = 20.0 * (gain_in * gain_out / (median_in * median_out)).log10();
            println!(
                "(Mood::{mood}, Lane::{lane:?}) => [{gain_in:.4}, {gain_out:.4}], \
                 // medians [{median_in:.4}, {median_out:.4}], {off_db:+.2} dB from them"
            );
            lufs_row.push(format!("{:.2}", median(lufs)));
            eprintln!("staged {mood} {}", lane.name());
        }
        rows.push(format!("(Mood::{mood}, [{}]),", lufs_row.join(", ")));
    }
    println!("// whole-lane rows: medians over STAGING_SEEDS of each lane's gated loudness");
    for row in rows {
        println!("{row}");
    }
}
