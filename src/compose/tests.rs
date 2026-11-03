//! The seed-sweep property suite: frozen takes pin ONE realization
//! (checksums); these pin the RULES — any seed must be musically well-formed.

use std::collections::BTreeSet;

use super::*;
use crate::score;

const SWEEP: u64 = 64;

const MOODS: [Mood; 2] = [Mood::Day, Mood::Night];

/// Every loop a song can draw.
fn loops() -> impl Iterator<Item = &'static [Degree]> {
    MAJOR_LOOPS.iter().chain(&MINOR_LOOPS).copied()
}

/// The mood and seed of every song the sweep composes.
fn songs() -> impl Iterator<Item = (Mood, u64)> {
    MOODS
        .into_iter()
        .flat_map(|mood| (0..SWEEP).map(move |seed| (mood, seed)))
}

/// Whether two voices sound muddy together: the lower one under its
/// interval's floor in [`LOW_INTERVAL_LIMITS`].
fn muddy(a: u8, b: u8) -> bool {
    let (lo, hi) = (a.min(b), a.max(b));
    (LOW_INTERVAL_LIMITS.get(usize::from(hi - lo))).is_some_and(|&floor| lo < floor)
}

/// The pitch classes of `chord`'s notes.
fn pcs(chord: &[u8; 4]) -> [u8; 4] {
    chord.map(|n| n % 12)
}

#[test]
fn a_whole_number_of_every_loop_fills_the_song_loop() {
    for degrees in loops() {
        assert!(degrees.len() > 1, "{degrees:?} never moves");
        assert_eq!(GEN_LOOP_BARS % degrees.len(), 0, "{degrees:?}");
    }
}

/// A song's bars are the loop it drew, cycled, in its key: one seam wrong and
/// a 2-bar loop would change on the wrong bar.
#[test]
fn a_songs_bars_cycle_the_loop_it_drew() {
    for (mood, seed) in songs() {
        let style = &mood.style().harmony;
        let degrees = drawn_loop(&mut Stage::Harmony.stream(seed), style);
        let bars = composed(mood, seed).1.bars;
        let above = |root_pc: u8, first: u8| (root_pc + OCTAVE - first) % OCTAVE;
        for (bar, chord) in bars.iter().enumerate() {
            let at = bar % degrees.len();
            assert_eq!(*chord, bars[at], "{mood:?} seed {seed}");
            assert_eq!(
                above(chord.root_pc, bars[0].root_pc),
                above(degrees[at].root_pc, degrees[0].root_pc),
                "{mood:?} seed {seed} bar {bar}"
            );
        }
    }
}

/// A song keeps to its key as the tracks do (LOFI-BIBLE R3): every note in
/// it, its root, third and 7th sounding (R1).
#[test]
fn every_bar_keeps_to_its_songs_key() {
    for (mood, seed) in songs() {
        let harmony = composed(mood, seed).1;
        for chord in &harmony.bars {
            let sounding = pcs(&chord.notes);
            assert!(
                (sounding.iter()).all(|pc| harmony.scale_pcs.contains(pc)),
                "{mood:?} seed {seed}: {:?} leaves {:?}",
                chord.notes,
                harmony.scale_pcs
            );
            for offsets in [[MINOR_THIRD, MAJOR_THIRD], [MINOR_SEVENTH, MAJOR_SEVENTH]] {
                let tone = offsets.map(|i| (chord.root_pc + i) % OCTAVE);
                assert!(
                    sounding.contains(&chord.root_pc) && tone.iter().any(|t| sounding.contains(t)),
                    "{mood:?} seed {seed}: {:?} over root {}",
                    chord.notes,
                    chord.root_pc
                );
            }
        }
    }
}

/// A key a mood never reaches would narrow what hours of songs can sound
/// like: the tracks sit in all twelve.
#[test]
fn every_mood_reaches_every_key() {
    for mood in MOODS {
        let keys: BTreeSet<[u8; 7]> = (0..SWEEP)
            .map(|seed| composed(mood, seed).1.scale_pcs)
            .collect();
        assert_eq!(keys.len(), usize::from(OCTAVE), "{mood:?}");
    }
}

/// The chart's own pairs: an interval at its floor sounds clear, a semitone
/// under muddy, and a unison or an octave at any height clear.
#[test]
fn an_interval_under_its_floor_sounds_muddy() {
    for (lo, hi) in [(52, 53), (48, 51), (46, 50), (34, 41), (41, 49), (34, 50)] {
        assert!(!muddy(lo, hi) && muddy(lo - 1, hi - 1), "{lo}-{hi}");
    }
    assert!(!muddy(24, 24) && !muddy(24, 36));
}

/// Every chord a song voices sounds clear, whatever its key.
#[test]
fn every_chord_a_song_voices_sounds_clear() {
    for (mood, seed) in songs() {
        for notes in compose(mood, seed).bar_chords {
            assert!(
                (notes.iter()).all(|&a| notes.iter().all(|&b| !muddy(a, b))),
                "{mood:?} seed {seed}: {notes:?}"
            );
        }
    }
}

/// A key moves no chord's register: its root stays in one octave, lowest.
#[test]
fn every_chord_sits_in_root_position_in_one_octave() {
    for (mood, seed) in songs() {
        for chord in composed(mood, seed).1.bars {
            let [root, ..] = chord.notes;
            assert_eq!(root % OCTAVE, chord.root_pc, "{mood:?} seed {seed}");
            assert!(chord.notes.is_sorted(), "{mood:?} seed {seed}");
            assert!(
                (VOICING_FLOOR..VOICING_FLOOR + OCTAVE).contains(&root),
                "{mood:?} seed {seed}: {:?}",
                chord.notes
            );
        }
    }
}

/// A mood's loops are drawn from both grammars, at its share.
#[test]
fn a_mood_draws_its_major_share_of_major_loops() {
    let draws = 4000;
    for mood in MOODS {
        let style = &mood.style().harmony;
        let mut rng = NoiseStream::new(1);
        let major = (0..draws)
            .filter(|_| MAJOR_LOOPS.contains(&drawn_loop(&mut rng, style)))
            .count();
        let share = major as f32 / draws as f32;
        assert!(
            (share - style.major_share).abs() < 0.03,
            "{mood:?}: {share}"
        );
    }
}

/// The lead's register is the span its mood's frozen take sings.
#[test]
fn each_moods_lead_register_is_its_frozen_takes() {
    let day = score::SPARKLE_SCORE.map(|(_, midi, _)| midi);
    let night = score::NIGHT_SPARKLE.map(|n| n.midi);
    for (mood, take) in [(Mood::Day, &day[..]), (Mood::Night, &night[..])] {
        let span = (take.iter().min().copied(), take.iter().max().copied());
        let lead = &mood.style().lead;
        assert_eq!(span, (Some(lead.lo), Some(lead.hi)), "{mood:?}");
    }
}

/// A lead sparser than its comp lets the keys take the top line, which then
/// hops between them: the day sings LOFI-BIBLE R10's density, the night under
/// it but over its keys' single tones.
#[test]
fn a_lead_sings_over_its_comp_and_less_by_night() {
    let per_bar = |mood, part: fn(&Score) -> usize| {
        let notes: usize = (0..SWEEP).map(|seed| part(&compose(mood, seed))).sum();
        notes as f32 / (SWEEP as usize * GEN_LOOP_BARS) as f32
    };
    let lead = |s: &Score| s.sparkle.len();
    let (day, night) = (per_bar(Mood::Day, lead), per_bar(Mood::Night, lead));
    let night_keys = per_bar(Mood::Night, |s| s.keys.len());
    assert!((2.0..=4.0).contains(&day), "{day}");
    assert!(
        (night_keys..day).contains(&night),
        "{night} against {night_keys} and {day}"
    );
}

/// Two of a bar's lead notes closer than [`MIN_GAP_BEATS`] would smear into one
/// struck twice.
#[test]
fn a_bars_lead_notes_keep_their_gap() {
    for mood in MOODS {
        let style = &mood.style().lead;
        for seed in 0..SWEEP {
            let mut rng = NoiseStream::new(seed);
            let statement = statement(&mut rng, style);
            for bar in [statement.clone(), answer(&mut rng, &statement)].concat() {
                for pair in bar.windows(2) {
                    assert!(
                        pair[1] - pair[0] >= MIN_GAP_BEATS,
                        "{mood:?} seed {seed}: {bar:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn lead_voice_pools_are_mood_curated() {
    let mut day_saw = (false, false, false);
    let mut night_saw = (false, false);
    for seed in 0..SWEEP {
        match compose(Mood::Day, seed).lead_voice {
            LeadVoice::EpVel => day_saw.0 = true,
            LeadVoice::Pluck => day_saw.1 = true,
            LeadVoice::Kalimba => day_saw.2 = true,
            LeadVoice::Vibraphone => panic!("seed {seed}: vibraphone is night-pool only"),
        }
        match compose(Mood::Night, seed).lead_voice {
            LeadVoice::EpVel => night_saw.0 = true,
            LeadVoice::Vibraphone => night_saw.1 = true,
            v => panic!("seed {seed}: night drew the day-pool {v:?}"),
        }
    }
    assert!(
        day_saw.0 && day_saw.1 && day_saw.2,
        "all three day lead voices must appear: {day_saw:?}"
    );
    assert!(
        night_saw.0 && night_saw.1,
        "both night lead voices must appear: {night_saw:?}"
    );
}

#[test]
fn lead_voice_distribution_tracks_the_draw_weights() {
    // day p(Pluck)=0.35 / p(Kalimba)=0.25, night p(Vibraphone)=0.35; each band
    // is ±3.5σ over 400 seeds, so a biased or misplaced draw fails but an
    // unlucky sample does not
    let (mut plucks, mut kalimbas, mut vibes) = (0usize, 0usize, 0usize);
    for s in 0..400 {
        match compose(Mood::Day, s).lead_voice {
            LeadVoice::Pluck => plucks += 1,
            LeadVoice::Kalimba => kalimbas += 1,
            _ => {}
        }
        if compose(Mood::Night, s).lead_voice == LeadVoice::Vibraphone {
            vibes += 1;
        }
    }
    assert!(
        (107..=173).contains(&plucks),
        "pluck drew {plucks}/400 vs p=0.35"
    );
    assert!(
        (70..=130).contains(&kalimbas),
        "kalimba drew {kalimbas}/400 vs p=0.25"
    );
    assert!(
        (107..=173).contains(&vibes),
        "vibe drew {vibes}/400 vs p=0.35"
    );
}

#[test]
fn the_bass_starts_its_notes_as_bass_step_p_weighs_each_16th() {
    // a bar's root in its first beat by each 16th's share of that beat's
    // weights, then each later 16th by its own chance; each band is ±3.5σ
    // over every bar of `SEEDS` seeds, so a misplaced or misweighted draw fails
    // but an unlucky sample does not
    const SEEDS: u64 = 400;
    let mut onsets = [0usize; BASS_STEP_P.len()];
    for seed in 0..SEEDS {
        let s = compose(Mood::Day, seed);
        let (bar_s, step_s) = (s.bar_s(), s.beat_s() * SIXTEENTH_BEATS);
        let mut first_beat = [0usize; GEN_LOOP_BARS];
        for &Note { at, .. } in &s.bass {
            let bar = (at / bar_s) as usize;
            // a note lags its 16th by less than a 16th
            let step = ((at - bar as f32 * bar_s) / step_s) as usize;
            onsets[step] += 1;
            first_beat[bar] += usize::from(step < STEPS_PER_BEAT);
        }
        assert_eq!(
            first_beat, [1; GEN_LOOP_BARS],
            "seed {seed}: a bar without one root in its first beat"
        );
    }
    let bars = (SEEDS as usize * GEN_LOOP_BARS) as f64;
    let first: f32 = BASS_STEP_P[..STEPS_PER_BEAT].iter().sum();
    for (step, (&n, &p)) in onsets.iter().zip(&BASS_STEP_P).enumerate() {
        let p = f64::from(if step < STEPS_PER_BEAT { p / first } else { p });
        let (mean, sd) = (bars * p, (bars * p * (1.0 - p)).sqrt());
        assert!(
            (n as f64 - mean).abs() <= 3.5 * sd,
            "16th {step}: {n} onsets over {bars} bars vs p={p:.3}"
        );
    }
}

#[test]
fn each_bass_bar_plays_its_own_rhythm_each_note_held_toward_the_next() {
    for mood in MOODS {
        for seed in 0..SWEEP {
            let (s, h) = composed(mood, seed);
            let step_s = s.beat_s() * SIXTEENTH_BEATS;
            let bar_steps = BASS_STEP_P.len();
            // each note's 16th from the loop's start: it lags that 16th by
            // less than a 16th
            let onsets: Vec<usize> = (s.bass.iter())
                .map(|&Note { at, .. }| (at / step_s) as usize)
                .collect();
            let bars: BTreeSet<Vec<usize>> = (0..GEN_LOOP_BARS)
                .map(|bar| {
                    (onsets.iter())
                        .filter(|&&o| o / bar_steps == bar)
                        .map(|&o| o % bar_steps)
                        .collect()
                })
                .collect();
            assert!(bars.len() > 1, "{mood:?} seed {seed}: every bar one rhythm");
            for (i, (&Note { at, midi, dur, .. }, &onset)) in s.bass.iter().zip(&onsets).enumerate()
            {
                // the last note's next is the loop's first, a loop later
                let next =
                    (onsets.get(i + 1).copied()).unwrap_or(onsets[0] + GEN_LOOP_BARS * bar_steps);
                let held = ((next - onset) as f32 * step_s * BASS_GATE).min(s.loop_secs() - at);
                assert!(
                    (dur - held).abs() < 1e-4,
                    "{mood:?} seed {seed}: bass note at {at}s held {dur}s, not {held}s"
                );
                let bar = onset / bar_steps;
                let chord = &h.bars[bar];
                let tone = [bass_pitch(chord.root_pc), bass_pitch(chord.root_pc + FIFTH)];
                if !tone.contains(&midi) {
                    let ends_bar = onsets.get(i + 1).is_none_or(|&n| n / bar_steps != bar);
                    assert!(
                        ends_bar && onset % bar_steps >= bar_steps - STEPS_PER_BEAT,
                        "{mood:?} seed {seed}: pickup at {at}s not its bar's last-beat last note"
                    );
                }
            }
        }
    }
}

#[test]
fn every_seed_is_well_formed() {
    for mood in MOODS {
        let style = mood.style();
        for seed in 0..SWEEP {
            let (s, h) = composed(mood, seed);
            assert_well_formed(&s, &h, seed);
            assert!(
                (style.bpm.base..=style.bpm.hi()).contains(&s.bpm),
                "{mood:?} seed {seed}: bpm {}",
                s.bpm
            );
            for &Note { midi: note, .. } in &s.sparkle {
                assert!(
                    (style.lead.lo..=style.lead.hi).contains(&note),
                    "{mood:?} seed {seed}: lead note {note} out of its register"
                );
            }
            let kinds = |k: DrumKind| s.drums.iter().any(|&Hit { kind, .. }| kind == k);
            assert!(kinds(DrumKind::Snare), "seed {seed}: no backbeat");
            if mood == Mood::Night {
                assert!(
                    !kinds(DrumKind::OpenHat),
                    "seed {seed}: night grew an open hat"
                );
            }
        }
    }
}

fn assert_well_formed(s: &Score, h: &Harmony, seed: u64) {
    let loop_s = s.loop_secs();
    let bar_s = s.bar_s();
    let bar_of = |at: f32| ((at / bar_s) as usize).min(GEN_LOOP_BARS - 1);
    for &Note { at, vel, .. } in s.sparkle.iter().chain(s.keys.iter()).chain(s.bass.iter()) {
        assert!(
            at >= 0.0 && at < loop_s,
            "seed {seed}: event at {at} outside loop {loop_s}"
        );
        assert!(vel > 0.0 && vel <= 1.0, "seed {seed}: velocity {vel}");
    }
    for (bar, chord) in h.bars.iter().enumerate() {
        // the floor: the root, sounding in the bar's first beat
        let one = (bar as f32 * bar_s)..(bar as f32 * bar_s + s.beat_s());
        assert!(
            (s.bass.iter()).any(
                |&Note { at, midi, .. }| one.contains(&at) && midi == bass_pitch(chord.root_pc)
            ),
            "seed {seed}: the bass must sound bar {bar}'s root on its ONE"
        );
    }
    for &Note {
        at,
        midi: note,
        dur,
        ..
    } in &s.bass
    {
        // the window G1-F#2, a root an octave under it, and the leading-tone
        // pickup a step under that
        assert!(
            (26..=42).contains(&note),
            "seed {seed}: bass note {note} at {at}s out of the bass window"
        );
        // closed vocabulary: root/5th/pickup — anything else is a wrong join
        let bar = bar_of(at);
        let chord = &h.bars[bar];
        let mut allowed = vec![bass_pitch(chord.root_pc), bass_pitch(chord.root_pc + 7)];
        if let Some(next) = h.bars.get(bar + 1) {
            // a semitone under the next root as its bar folds it, not that
            // pitch class folded in
            allowed.push(bass_pitch(next.root_pc) - 1);
        }
        assert!(
            allowed.contains(&note),
            "seed {seed}: bass note {note} in bar {bar} is not root/5th/pickup {allowed:?}"
        );
        // place() clips at the loop end — no note may ring into the seam
        assert!(
            dur > 0.0 && at + dur <= loop_s,
            "seed {seed}: bass note at {at}s held {dur}s rings into the loop seam"
        );
    }
    for &Hit { at, gain, .. } in &s.drums {
        assert!(
            at >= 0.0 && at < loop_s,
            "seed {seed}: drum at {at} outside loop"
        );
        assert!(gain > 0.0 && gain <= 1.5, "seed {seed}: drum gain {gain}");
    }
    for &Note { at, midi: note, .. } in &s.keys {
        let bar = bar_of(at);
        let chord = s.bar_chords[bar];
        let ninth_pc = (h.bars[bar].root_pc + 2) % 12;
        assert!(
            chord
                .iter()
                .any(|&c| note == c || note == c + 12 || note == c + 24)
                || note % 12 == ninth_pc,
            "seed {seed}: keys note {note} at {at}s not a tone/ninth of {chord:?}"
        );
    }
    for &Note { at, midi: note, .. } in &s.sparkle {
        assert!(
            h.scale_pcs.contains(&(note % 12)),
            "seed {seed}: lead note {note} at {at}s outside its key"
        );
    }
    assert!(
        s.sparkle.len() >= 2,
        "seed {seed}: the lead lost its identity"
    );
}

/// One played lead note read back off the grid: its bar, its index in the
/// bar, its beat in the bar and its pitch. The lag is under a 16th, so the
/// nearest 16th is the note's.
struct Sung {
    bar: usize,
    i: usize,
    beat: f32,
    note: u8,
}

fn sung(s: &Score) -> Vec<Sung> {
    let mut out: Vec<Sung> = Vec::new();
    for &Note { at, midi: note, .. } in &s.sparkle {
        let sixteenths = (at / s.beat_s() * 4.0).round();
        let bar = (sixteenths / 16.0) as usize;
        let beat = (sixteenths - bar as f32 * 16.0) / 4.0;
        let i = out.iter().filter(|n| n.bar == bar).count();
        out.push(Sung { bar, i, beat, note });
    }
    out
}

/// The rules that outrank the walk: the answer's quote of the statement's
/// first two notes, the close on the loop's last bar, and the peak opening
/// the bar after the quote.
fn outranks_the_walk(line: &[Sung], j: usize) -> bool {
    let n = &line[j];
    let head = line.iter().filter(|m| m.bar == 0).count().min(2);
    let close = j == line.len() - 1 && n.bar == GEN_LOOP_BARS - 1;
    let quote = n.bar == 4 && n.i < head && !close;
    let peak = n.bar == 5 && n.i == 0;
    quote || close || peak
}

/// Whether note `j` follows a leap past a major third, which it resolves; the
/// lead's first leap starts from a note it never sings, so note 1's is unknown.
fn resolves(line: &[Sung], j: usize) -> Option<bool> {
    (j >= 2).then(|| (i16::from(line[j - 1].note) - i16::from(line[j - 2].note)).abs() > 4)
}

#[test]
fn strong_beats_sing_chord_tones_unless_a_rule_outranks_them() {
    let mut checked = 0;
    for mood in MOODS {
        for seed in 0..SWEEP {
            let s = compose(mood, seed);
            let line = sung(&s);
            for (j, n) in line.iter().enumerate() {
                let strong = n.beat.fract() == 0.0 || n.beat.fract() == 0.75;
                let may_resolve = j == 1 || resolves(&line, j) == Some(true);
                if !strong || may_resolve || outranks_the_walk(&line, j) {
                    continue;
                }
                checked += 1;
                assert!(
                    pcs(&s.bar_chords[n.bar]).contains(&(n.note % 12)),
                    "{mood:?} seed {seed}: strong note {} in bar {} off its chord",
                    n.note,
                    n.bar
                );
            }
        }
    }
    assert!(checked > 0);
}

#[test]
fn leaps_stay_within_a_fifth_unless_a_rule_outranks_them() {
    for mood in MOODS {
        for seed in 0..SWEEP {
            let line = sung(&compose(mood, seed));
            for j in 1..line.len() {
                let leap = (i16::from(line[j].note) - i16::from(line[j - 1].note)).abs();
                assert!(
                    leap <= 7 || outranks_the_walk(&line, j),
                    "{mood:?} seed {seed}: note {j} leaps {leap}"
                );
            }
        }
    }
}

#[test]
fn a_leap_resolves_by_a_step_against_it() {
    let mut checked = 0;
    for mood in MOODS {
        for seed in 0..SWEEP {
            let line = sung(&compose(mood, seed));
            for j in 2..line.len() {
                if resolves(&line, j) != Some(true) || outranks_the_walk(&line, j) {
                    continue;
                }
                checked += 1;
                let leap = i16::from(line[j - 1].note) - i16::from(line[j - 2].note);
                let step = i16::from(line[j].note) - i16::from(line[j - 1].note);
                // a step held at the register's edge stays put
                assert!(
                    step.signum() != leap.signum() && step.abs() <= 2,
                    "{mood:?} seed {seed}: a leap of {leap} met by {step}"
                );
            }
        }
    }
    assert!(checked > 0);
}

#[test]
fn the_peak_and_the_close_land_on_chord_tones_and_the_close_comes_home() {
    let (mut peaks, mut closes) = (0, 0);
    for mood in MOODS {
        let lead = &mood.style().lead;
        for seed in 0..SWEEP {
            let s = compose(mood, seed);
            let line = sung(&s);
            if let Some(peak) = line.iter().find(|n| n.bar == 5 && n.i == 0) {
                peaks += 1;
                assert!(
                    pcs(&s.bar_chords[5]).contains(&(peak.note % 12)),
                    "{mood:?} seed {seed}: the peak {} is off its chord",
                    peak.note
                );
            }
            let (first, last) = (&line[0], &line[line.len() - 1]);
            if last.bar != GEN_LOOP_BARS - 1 {
                continue;
            }
            closes += 1;
            let chord = pcs(&s.bar_chords[last.bar]);
            let distance = |n: u8| (i16::from(n) - i16::from(first.note)).abs();
            let nearer = (lead.lo..=lead.hi)
                .filter(|n| chord.contains(&(n % 12)))
                .find(|&n| distance(n) < distance(last.note));
            assert!(
                chord.contains(&(last.note % 12)) && nearer.is_none(),
                "{mood:?} seed {seed}: the close {} isn't the chord tone nearest the opening {}",
                last.note,
                first.note
            );
        }
    }
    assert!(peaks > 0 && closes > 0);
}

#[test]
fn a_bass_note_folds_into_its_window() {
    for pc in 0..2 * OCTAVE {
        let note = bass_pitch(pc);
        assert_eq!(note % OCTAVE, pc % OCTAVE, "pc {pc}");
        assert!((BASS_LOW..BASS_LOW + OCTAVE).contains(&note), "pc {pc}");
    }
}

/// A comp without a chord's 3rd and 7th would stop saying which chord it is:
/// the other tests pass any two of its tones.
#[test]
fn a_shell_is_a_chords_third_and_seventh() {
    for (degree, third, seventh) in [
        (C, MAJOR_THIRD, MAJOR_SEVENTH),
        (A_MIN, MINOR_THIRD, MINOR_SEVENTH),
        (G, MAJOR_THIRD, MINOR_SEVENTH),
    ] {
        let chord = degree.voiced(0);
        let root = chord.notes[0];
        assert_eq!(
            shell_of(&chord),
            (root + third, root + seventh),
            "{degree:?}"
        );
    }
}

#[test]
fn day_keys_comp_in_rolled_shell_pairs() {
    for seed in 0..SWEEP {
        let s = compose(Mood::Day, seed);
        assert_eq!(s.keys.len() % 2, 0, "seed {seed}: unpaired shell note");
        for pair in s.keys.chunks(2) {
            let (a, b) = (pair[0], pair[1]);
            let roll = b.at - a.at;
            assert!(
                (0.010..=0.022).contains(&roll),
                "seed {seed}: shell roll {roll}s outside the hand's range"
            );
            assert!(
                b.vel < a.vel,
                "seed {seed}: the rolled upper voice rides softer"
            );
            assert_ne!(
                a.midi % 12,
                b.midi % 12,
                "seed {seed}: shell voices collapsed"
            );
        }
    }
}

#[test]
fn day_lead_uses_sixteenth_anticipations_somewhere_in_the_sweep() {
    let mut seen_push = false;
    for seed in 0..SWEEP {
        let s = compose(Mood::Day, seed);
        let beat_s = s.beat_s();
        for &Note { at, .. } in &s.sparkle {
            // strip the played-not-sequenced lag before reading the grid
            let beats = (at - 0.015) / beat_s;
            let frac = beats - beats.floor();
            let on_grid = [0.0f32, 0.25, 0.5, 0.75]
                .iter()
                .any(|g| (frac - g).abs() < 0.07 || (frac - g - 1.0).abs() < 0.07);
            assert!(on_grid, "seed {seed}: lead event off-grid (frac {frac})");
            let bar_beat = beats % 4.0;
            if [0.75f32, 1.75, 2.75]
                .iter()
                .any(|g| (bar_beat - g).abs() < 0.07)
            {
                seen_push = true;
            }
        }
    }
    assert!(seen_push, "no 16th anticipation surfaced across the sweep");
}

#[test]
fn the_answer_quotes_the_statements_opening() {
    for mood in MOODS {
        for seed in 0..SWEEP {
            let line = sung(&compose(mood, seed));
            let bar = |b: usize| line.iter().filter(move |n| n.bar == b).map(|n| n.note);
            if let (Some(statement), Some(answer)) = (bar(0).next(), bar(4).next()) {
                assert_eq!(
                    answer, statement,
                    "{mood:?} seed {seed}: the answer must open on the statement's pitch"
                );
            }
        }
    }
}

/// Every hit a seeded groove lays, bit for bit: the drums the generator
/// writes, which no frozen take pins. A change to a hit's time, gain or the
/// order its noise is drawn in moves this.
#[test]
fn a_seeded_groove_lays_every_hit_where_it_did() {
    let mut words = Vec::new();
    let beat_s = 60.0 / 74.0;
    for g in DAY_GROOVES.iter().chain([&NIGHT_GROOVE]) {
        let hits = groove(&mut NoiseStream::new(7), g, beat_s);
        for &Hit { at, kind, gain } in &hits {
            words.extend([at.to_bits(), kind as u32, gain.to_bits()]);
        }
        // the kicks again, as the hash was recorded while the drums kept a
        // second list of them
        let kicks = hits.iter().filter(|hit| hit.kind == DrumKind::Kick);
        words.extend(kicks.map(|hit| hit.at.to_bits()));
    }
    let h = fnv1a(words.into_iter().map(u64::from));
    assert_eq!(h, GOLDEN, "{h:#018x}");
}

const GOLDEN: u64 = 0xf07e_5f74_f00f_fbfc;

/// Both moods' first eight seeds, every event and the harmonic timeline as f32
/// bits. `compose` runs no transcendental function, so these bits hold on every
/// platform (Rust RFC 3514), unlike the synth's.
#[test]
fn every_seed_composes_the_score_it_did() {
    let mut words: Vec<u64> = Vec::new();
    for mood in MOODS {
        for seed in 0..8 {
            let s = compose(mood, seed);
            words.push(u64::from(s.bpm.to_bits()));
            for lane in [&s.sparkle, &s.keys, &s.bass] {
                for &Note {
                    at,
                    midi: note,
                    vel,
                    dur,
                } in lane
                {
                    words.extend([
                        u64::from(at.to_bits()),
                        u64::from(note),
                        u64::from(vel.to_bits()),
                        u64::from(dur.to_bits()),
                    ]);
                }
            }
            for &Hit { at, kind, gain } in &s.drums {
                words.extend([
                    u64::from(at.to_bits()),
                    kind as u64,
                    u64::from(gain.to_bits()),
                ]);
            }
            // the night kicks again, as the hash was recorded while a night
            // score kept a second list of them
            if mood == Mood::Night {
                words.extend(s.kicks().map(|t| u64::from(t.to_bits())));
            }
            words.extend(s.bar_chords.iter().flatten().map(|&n| u64::from(n)));
            words.push(s.lead_voice as u64);
        }
    }
    let h = fnv1a(words);
    assert_eq!(h, EVERY_SEED_GOLDEN, "{h:#018x}");
}

const EVERY_SEED_GOLDEN: u64 = 0xb46e_9b09_90b2_957b;
