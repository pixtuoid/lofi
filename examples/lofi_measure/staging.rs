//! The stream's staging, measured: each lane's whole-lane render over
//! [`STAGING_SEEDS`], read for the gains its peak normalizations applied and for
//! its gated loudness. `lanes.rs`'s `staging` holds the gains' medians, and
//! its whole-lane level test the loudness's.

use pixtuoid_lofi::Controls;
use pixtuoid_lofi::audition::{self, Lane, STAGING_SEEDS, gated_lufs};
use pixtuoid_lofi::dsp::SAMPLE_RATE;

use crate::corpus::spread;

/// The moods the stream stages, by a daylight each plays at.
const MOODS: [(&str, f32); 2] = [("Day", 1.0), ("Night", 0.0)];

fn median(values: Vec<f64>) -> f64 {
    let [_, _, median, _, _] = spread(values);
    median
}

/// Prints the staging and the whole-lane loudness rows, as the stream's
/// source holds them. Each seed's first song renders each lane on its own
/// fork of that seed's noise, as a whole-lane take does.
pub(crate) fn print() {
    let mut rows = Vec::new();
    println!("// staging: medians over STAGING_SEEDS of each lane's whole-lane normalizations");
    for (mood, daylight) in MOODS {
        let controls = Controls::default().with_daylight(daylight);
        let mut lufs_row = Vec::new();
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
            println!(
                "(Mood::{mood}, Lane::{lane:?}) => [{:.4}, {:.4}],",
                median(ins),
                median(outs)
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
