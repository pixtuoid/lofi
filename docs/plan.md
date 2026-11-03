# The plan

What the scorecard can't say: the rows it can't work top-down yet, each with
the reason, and the work no row holds. A PR that finishes, adds or reorders
an item edits this file in the same commit ([AGENTS.md](../AGENTS.md)).
Where a figure backs an item, the commit cited holds it.

## Order

The card is worked top-down. These rows wait on others:

- Layer 1, rows `triads %` and `maj7 %`: wait on layer 4. Triads written by
  doubling the root lost on FAD as the take brightened (LOFI-BIBLE R1), and
  no voicing of four tried holds the mix's centroid (the commit that wrote
  this file, with the tracks' triad shares by degree and mode).
- Layer 2, row `onsets /s`: waits on layer 3's row `top-line onsets /bar`.
  The drums' and the bass's rows hold, so the gap is the top line's (#77).
- Layer 3, rows `pentatonic share`, `step share` and `polyphony`: wait on the
  comp's voicing, layer 1's triads. The pad and keys sing about half the top
  line (#77).
- Layer 3, rows `repeat at 8 bars` and `repeat at 16 bars`: wait on the long
  loop below (#78).
- Next: layer 4, rows `drums FAD`, `bass FAD`, `hat decay ms` and
  `bass 62-90 Hz %`, then layer 5. Layers 4 to 6 also take back what #76 cost,
  rows `bass FAD`, `bass %` and `low punch dB`.

## Work no row holds

- **The long loop.** The tracks' top line repeats as little at 16 bars as at
  8 (#78), so the lockstep loop must run longer than a song before it
  repeats. A 32-bar loop, with the form's phrase and the lead's period kept,
  ran the library tests about 3.3 times as long, on the whole-lane renders
  and each song's texture loop (the commit that wrote this file). Two
  refactors come first, each a PR of its own: `GEN_LOOP_BARS`, which names the
  loop, the form's phrase and the lead's period at once, splits in three,
  byte-identical; and those renders stop scaling with the loop.
- **`lofi_bench` times by the wall clock**, so load from elsewhere fails its
  gate (#72). `std::time`'s clocks are `Instant` and `SystemTime`, none a
  thread's CPU time (<https://doc.rust-lang.org/1.99.1/std/time/index.html>),
  and the crate takes no dependency (`ci.yml`'s count, REVIEW.md's Design 2).
- **The card has no form or variation row.**
- **Styles**: targets per style, and a judge batch split by style, whose
  floor test then covers the style axis.
- **A seed's day and night songs draw one bass rhythm** (#76).
- **Release 0.4**, which the owner dispatches, then pixtuoid's follow-up.
