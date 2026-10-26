//! How rain recolours the playing song: darker, its texture over its melody,
//! its drums softer. Each map is exact at rain 0, so a dry room plays as it
//! did before rain existed.

use crate::compose::Lane;
use crate::synth::TILT_LIFT;

/// The tape tilt's lift on the sparkle, keys and drums at rain 1, from
/// [`TILT_LIFT`] dry: a darker top from the shelf every tape already has.
/// Opinion: dull enough to read as rain under a background bed.
pub(crate) const RAIN_TILT_LIFT: f32 = -0.098;

/// The sparkle's and the pad's level change at rain 1, in dB. Opinion: the
/// melody steps back and the pad's bed swells under it.
pub(crate) const RAIN_SPARKLE_DB: f64 = -3.0;
pub(crate) const RAIN_PAD_DB: f64 = 1.0;

/// Each drum hit's gain change at rain 1, in dB, read at its onset. Opinion:
/// a softer touch, short of thinning the groove.
pub(crate) const RAIN_DRUM_DB: f64 = -2.0;

/// How long, in bars, a latch's rain takes to arrive. Opinion: a phrase, so
/// it settles in as weather does rather than cutting in.
pub(crate) const RAIN_RAMP_BARS: f32 = 4.0;

/// The lanes rain darkens, spelled apart from [`darkens`] so a test checks
/// the set instead of restating it.
#[cfg(test)]
pub(crate) const DARKENED: [Lane; 3] = [Lane::Sparkle, Lane::Keys, Lane::Drums];

/// The tilt's lift at `rain`.
pub(crate) fn lift(rain: f64) -> f64 {
    let dry = f64::from(TILT_LIFT);
    dry + (f64::from(RAIN_TILT_LIFT) - dry) * rain
}

/// Whether rain tilts `lane`'s top.
pub(crate) fn darkens(lane: Lane) -> bool {
    matches!(lane, Lane::Sparkle | Lane::Keys | Lane::Drums)
}

/// `lane`'s level at `rain`, as a factor on its intensity's.
pub(crate) fn level(lane: Lane, rain: f64) -> f64 {
    let db = match lane {
        Lane::Sparkle => RAIN_SPARKLE_DB,
        Lane::Pad => RAIN_PAD_DB,
        _ => return 1.0,
    };
    super::gain_of_db(db * rain)
}

/// A drum hit's gain at `rain`.
pub(crate) fn hit(rain: f64) -> f32 {
    super::gain_of_db(RAIN_DRUM_DB * rain) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `f` at rain 0, 1 and in sixteenths between.
    fn swept<T>(f: impl Fn(f64) -> T) -> Vec<T> {
        (0..=16).map(|i| f(f64::from(i) / 16.0)).collect()
    }

    #[test]
    fn rain_darkens_the_tilt_from_its_dry_lift_to_its_wet_one() {
        let lifts = swept(lift);
        assert_eq!(lifts[0], f64::from(TILT_LIFT), "dry");
        assert_eq!(lifts[16], f64::from(RAIN_TILT_LIFT), "wet");
        assert!(lifts.windows(2).all(|w| w[1] < w[0]), "{lifts:?}");
    }

    #[test]
    fn rain_turns_the_sparkle_down_and_the_pad_up_and_no_other_lane() {
        for lane in Lane::ALL {
            let gains = swept(|rain| level(lane, rain));
            let wet = 20.0 * gains[16].log10();
            assert_eq!(gains[0], 1.0, "{lane:?} dry");
            let (want, rising) = match lane {
                Lane::Sparkle => (RAIN_SPARKLE_DB, false),
                Lane::Pad => (RAIN_PAD_DB, true),
                _ => (0.0, true),
            };
            assert!((wet - want).abs() < 1e-9, "{lane:?} wet: {wet} dB");
            let ordered = gains
                .windows(2)
                .all(|w| if rising { w[1] >= w[0] } else { w[1] < w[0] });
            assert!(ordered, "{lane:?}: {gains:?}");
        }
    }

    #[test]
    fn rain_softens_each_drum_hit() {
        let gains = swept(hit);
        assert_eq!(gains[0], 1.0, "dry");
        assert!((20.0 * f64::from(gains[16]).log10() - RAIN_DRUM_DB).abs() < 1e-5);
        assert!(gains.windows(2).all(|w| w[1] < w[0]), "{gains:?}");
    }
}
