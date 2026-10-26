//! The seed-sweep property suite: frozen takes pin ONE realization
//! (checksums); these pin the RULES — any seed must be musically well-formed.

use super::*;
use crate::score;

const SWEEP: u64 = 64;

const MOODS: [Mood; 2] = [Mood::Day, Mood::Night];

/// Every template of every mood's grammar, with its mood and index.
fn templates() -> impl Iterator<Item = (Mood, usize, &'static Progression)> {
    MOODS.into_iter().flat_map(|mood| {
        let progs = mood.style().harmony.progressions;
        progs.iter().enumerate().map(move |(i, p)| (mood, i, p))
    })
}

/// The pitch classes of `chord`'s notes.
fn pcs(chord: &[u8; 4]) -> [u8; 4] {
    chord.map(|n| n % 12)
}

#[test]
fn template_chords_and_roots_are_diatonic_or_deliberately_chromatic() {
    for (mood, i, p) in templates() {
        let color_tones = (p.chords.iter())
            .flat_map(|c| c.notes)
            .filter(|&n| !p.scale_pcs.contains(&(n % 12)))
            .count();
        if p.chromatic {
            // at most one chord's worth of color, so it stays in the key
            assert!(
                (1..=4).contains(&color_tones),
                "{mood:?}[{i}]: chromatic template carries {color_tones} color tones"
            );
        } else {
            assert_eq!(
                color_tones, 0,
                "{mood:?}[{i}]: diatonic template leaked a color tone"
            );
        }
        for c in &p.chords {
            assert!(
                p.scale_pcs.contains(&c.root_pc),
                "{mood:?}[{i}]: root pc {} outside its scale",
                c.root_pc
            );
            assert!(
                pcs(&c.notes).contains(&c.root_pc),
                "{mood:?}[{i}]: {:?} doesn't sound its root {}",
                c.notes,
                c.root_pc
            );
        }
    }
}

#[test]
fn every_shift_keeps_a_template_in_its_own_key() {
    for (mood, i, p) in templates() {
        let colors = |p: &Progression| {
            (p.chords.iter())
                .flat_map(|c| c.notes)
                .filter(|&n| !p.scale_pcs.contains(&(n % 12)))
                .count()
        };
        for t in TRANSPOSE {
            let q = p.transposed(t);
            assert_eq!(colors(&q), colors(p), "{mood:?}[{i}] shifted {t}");
            for (a, b) in p.chords.iter().zip(&q.chords) {
                let moved = |from: u8, to: u8| i16::from(to) - i16::from(from) == i16::from(t);
                assert!(a.notes.iter().zip(&b.notes).all(|(&x, &y)| moved(x, y)));
                assert_eq!(
                    (i16::from(a.root_pc) + i16::from(t)).rem_euclid(12),
                    i16::from(b.root_pc)
                );
            }
            assert!(q.scale_pcs.is_sorted(), "{mood:?}[{i}] shifted {t}");
        }
    }
}

#[test]
fn a_timeline_plays_its_template_twice_and_a_turnaround_ends_on_the_dominant() {
    for (mood, i, p) in templates() {
        for turnaround in [false, true] {
            let bars = timeline(p, turnaround);
            for (bar, chord) in bars.iter().enumerate().take(GEN_LOOP_BARS - 1) {
                assert_eq!(*chord, p.chords[bar % 4], "{mood:?}[{i}] bar {bar}");
            }
            let last = bars[GEN_LOOP_BARS - 1];
            if turnaround {
                // the hinge: a dominant 7th of the returning bar-1 root
                let dom = (p.chords[0].root_pc + 7) % 12;
                let want = [0u8, 4, 7, 10].map(|o| (dom + o) % 12);
                assert_eq!(pcs(&last.notes), want, "{mood:?}[{i}]: not V7 of the root");
                assert_eq!(last.root_pc, dom);
            } else {
                assert_eq!(last, p.chords[3], "{mood:?}[{i}]: the plain cycle");
            }
        }
    }
    assert!(Mood::Day.style().harmony.turnaround);
    // the sleepy loop keeps the plain cycle
    assert!(!Mood::Night.style().harmony.turnaround);
}

#[test]
fn every_shell_comping_template_chord_carries_a_third_and_seventh() {
    // the shell voicings are only well-defined over true 7th chords
    for (mood, i, p) in templates() {
        if !matches!(mood.style().keys.comp, Comp::Shell) {
            continue;
        }
        for (bar, c) in p.chords.iter().enumerate() {
            let has = |offs: [u8; 2]| {
                (c.notes.iter()).any(|&n| offs.iter().any(|&o| n % 12 == (c.root_pc + o) % 12))
            };
            assert!(has([3, 4]), "{mood:?}[{i}] bar {bar}: no third");
            assert!(has([10, 11]), "{mood:?}[{i}] bar {bar}: no seventh");
        }
    }
}

/// The lead's register is the span its mood's frozen take sings, and its
/// density the most notes any of its mood's frozen takes sings in a bar: the
/// authorities its bounds were read from.
#[test]
fn each_moods_lead_register_and_density_are_its_frozen_takes() {
    let day = score::SPARKLE_SCORE.map(|(_, midi, _)| midi);
    let night = score::NIGHT_SPARKLE.map(|n| n.midi);
    for (mood, take) in [(Mood::Day, &day[..]), (Mood::Night, &night[..])] {
        let span = (take.iter().min().copied(), take.iter().max().copied());
        let lead = &mood.style().lead;
        assert_eq!(span, (Some(lead.lo), Some(lead.hi)), "{mood:?}");
    }
    let most = |bars: Vec<usize>| {
        (bars.iter())
            .map(|b| bars.iter().filter(|&c| c == b).count())
            .max()
    };
    let in_bars =
        |notes: &[Note], bar_s: f32| notes.iter().map(|n| (n.at / bar_s) as usize).collect();
    let day1 = score::SPARKLE_SCORE.map(|(beats, ..)| (beats / BEATS_PER_BAR) as usize);
    let day = [
        most(day1.to_vec()),
        most(in_bars(score::DAY2.sparkle, score::DAY2.bar_s())),
        most(in_bars(score::DAY3.sparkle, score::DAY3.bar_s())),
    ];
    let night_bar_s = score::night_loop_secs() / score::NIGHT_LOOP_BARS as f32;
    let night = most(in_bars(&score::NIGHT_SPARKLE, night_bar_s));
    assert_eq!(
        day.into_iter().max().flatten(),
        Some(Mood::Day.style().lead.max_per_bar)
    );
    assert_eq!(night, Some(Mood::Night.style().lead.max_per_bar));
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
            match mood {
                Mood::Day => assert!(kinds(DrumKind::Snare), "seed {seed}: no backbeat"),
                Mood::Night => assert!(
                    !kinds(DrumKind::Snare) && !kinds(DrumKind::OpenHat),
                    "seed {seed}: night grew a snare/open hat"
                ),
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
            (s.bass.iter()).any(|&Note { at, midi, .. }| one.contains(&at)
                && midi == bass_pitch(chord.root_pc, chord)),
            "seed {seed}: the bass must sound bar {bar}'s root on its ONE"
        );
    }
    for &Note { at, midi: note, .. } in &s.bass {
        // the window G1-F#2, a root an octave under it, and the leading-tone
        // pickup a step under that
        assert!(
            (26..=42).contains(&note),
            "seed {seed}: bass note {note} at {at}s out of the bass window"
        );
        // closed vocabulary: root/5th/pickup — anything else is a wrong join
        let bar = bar_of(at);
        let chord = &h.bars[bar];
        let mut allowed = vec![
            bass_pitch(chord.root_pc, chord),
            bass_pitch(chord.root_pc + 7, chord),
        ];
        if let Some(next) = h.bars.get(bar + 1) {
            // a semitone under the next root as its bar folds it, not that
            // pitch class folded in
            allowed.push(bass_pitch(next.root_pc, next) - 1);
        }
        assert!(
            allowed.contains(&note),
            "seed {seed}: bass note {note} in bar {bar} is not root/5th/pickup {allowed:?}"
        );
        // place() clips at the loop end — no onset may ring into the seam
        assert!(
            at < loop_s - 1.4 * (bar_s / 4.0),
            "seed {seed}: bass onset {at}s rings into the loop seam"
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
        let bar = bar_of(at);
        let in_turnaround = pcs(&s.bar_chords[bar]).contains(&(note % 12));
        assert!(
            h.template.scale_pcs.contains(&(note % 12)) || in_turnaround,
            "seed {seed}: lead note {note} at {at}s outside key AND bar chord"
        );
    }
    assert!(
        s.sparkle.len() >= 2,
        "seed {seed}: the lead lost its identity"
    );
    let max_per_bar = s.mood.style().lead.max_per_bar;
    for bar in 0..GEN_LOOP_BARS {
        let n = (s.sparkle.iter())
            .filter(|&&Note { at, .. }| (at / bar_s) as usize == bar)
            .count();
        // the humanization lag can push a bar-boundary event into the
        // next bar's count, so allow one over
        assert!(
            n <= max_per_bar + 1,
            "seed {seed}: bar {bar} lead density {n} > {max_per_bar}"
        );
    }
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

/// R7's low-interval limit, checked against every voice the pad and keys
/// sound in the bass's bar.
#[test]
fn no_bass_note_sounds_within_a_fifth_of_a_chord_voice_below_48() {
    for mood in MOODS {
        for seed in 0..200 {
            let s = compose(mood, seed);
            let bar_s = s.bar_s();
            let bar_of = |at: f32| ((at / bar_s) as usize).min(GEN_LOOP_BARS - 1);
            for bass in &s.bass {
                let bar = bar_of(bass.at);
                let keys = (s.keys.iter())
                    .filter(|k| bar_of(k.at) == bar)
                    .map(|k| k.midi);
                for voice in s.bar_chords[bar].into_iter().chain(keys) {
                    assert!(
                        voice >= 48 || voice.abs_diff(bass.midi) >= 7,
                        "{mood:?} seed {seed} bar {bar}: bass {} under {voice}",
                        bass.midi
                    );
                }
            }
        }
    }
}

#[test]
fn a_bass_note_drops_an_octave_only_where_it_would_clash() {
    let over = |low: u8| chord([low, 52, 55, 59], 0);
    // (pitch class, the bar's lowest chord voice, the bass note)
    for (pc, low, want) in [
        (7, 45, 31),
        (2, 45, 38),
        (3, 45, 27),
        (4, 45, 28),
        (6, 47, 30),
        (6, 48, 42),
        (6 + 12, 48, 42),
    ] {
        assert_eq!(bass_pitch(pc, &over(low)), want, "pc {pc} over {low}");
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

#[test]
fn adopted_color_templates_actually_rotate_in() {
    for mood in MOODS {
        let seen_color = (0..SWEEP).any(|seed| composed(mood, seed).1.template.chromatic);
        assert!(
            seen_color,
            "{mood:?}: no chromatic template across the sweep"
        );
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

const GOLDEN: u64 = 0x71ca_6f22_f488_2daf;

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
                } in lane
                {
                    words.extend([
                        u64::from(at.to_bits()),
                        u64::from(note),
                        u64::from(vel.to_bits()),
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

const EVERY_SEED_GOLDEN: u64 = 0xbfde_8336_275f_100a;
