# Review rules

What the automatic review ([`claude-review.yml`](.github/workflows/claude-review.yml)) checks in this repo, on top of [`AGENTS.md`](AGENTS.md) and the [review contract](https://github.com/pixtuoid/.github/blob/main/.github/workflows/claude-review.yml), which owns severity, format and evidence. Each rule names the failure it catches.

**Published contracts** here: the public API, the songs a seed composes, and a shipped sound.

## Traps: the obvious reading is wrong here

- **The rng draw order is the sound.** A draw added anywhere but the tail of `compose`'s stream or of `form::arrange`'s, or a re-pinned `every_seed_composes_the_score_it_did`, `every_seed_arranges_the_form_it_did`, `a_seeded_groove_lays_every_hit_where_it_did` or `day_take_stems_match_the_ratified_fingerprints`, is a sound change, never a test fix: CI goes green on a recomposed office.
- **The score is bit-identical; samples are not.** A transcendental function on `compose`'s path splits a seed's score across platforms, and a test pinning rendered sample values passes on one OS or toolchain and fails on another.
- **Semver rides the title the squash merge takes** (the PR's, or a lone commit's), which is the changelog line. A change [`AGENTS.md`](AGENTS.md#workflow) says takes `!`, without `!` there, ships a break as a patch.

## Always check

- New behavior without a test; scope beyond what the PR body states; each claim the PR body makes.

## Do not flag

- Anything a [`ci.yml`](.github/workflows/ci.yml) job gates, compiling included; pure style.
- Behavior documented where it is constrained: read the item's doc comment and the comments on the lines it governs first.
- Speculative defense in depth where a primary defense holds; risks that need unreachable preconditions.
- Runtime cost off `Performance::render`'s path, unless measurable. `render` runs for as long as the listener plays, so a cost on its path is flagged against `examples/lofi_bench.rs`'s per-chunk and mean-CPU gates.

## Correctness

- **CI-only gates**: name each `ci.yml` job the diff can turn red that a local `cargo test` does not run.
- **Siblings**: a guard added to some of a sibling set (the `Day` and `Night` pools, the `Lane` arms, the `LeadVoice` seam, the frozen, whole-lane and streaming chains) or to one caller of a shared fn: check the full set. A new lane joins `generated_beds_are_finite_phase_locked_and_in_the_sound_world`, `every_lane_stays_finite_for_ten_minutes` and `every_lane_plays_at_its_whole_lane_level`.
- **Unwired additions**: every new field, parameter, constant or gate has a live consumer in the same diff.
- **Population**: a list that selects a set (`Cargo.toml`'s `include`, the CI matrix, a test's seed sweep): name the set before and after; a silent shrink is a defect.
- **Test teeth**: every new or changed test fails when the code it names is wrong, and reads no ambient state.
- **Security**: the trust boundary is CI. On a `.github/` change, report a secret in a command or log, untrusted code in a secret-bearing job, an argument spliced into a shell as code, or a gate the diff loosens, a fork, bot or cancelled run that no longer fails closed included. When the diff touches none, the summary says so.

## Design

1. **Consumers**: trace each changed `pub` item to its consumers here (`examples/`, the README's doctest) and to pixtuoid, whose office plays a `Performance` and whose foley calls `dsp`.
2. **Layering**: `compose` makes a `Score`, `synth`'s voices render its notes from `dsp`'s primitives, `stream` plays them through each lane's chain and mixes them under the ceiling, and `score` is test-only.
3. **Proportion**: a branch, helper, parameter, type or test machinery that serves no reachable case; an invariant held by prose where a type could hold it; a hand-rolled mechanism `std` already provides ("reuse X", never a new dependency). The zero dependencies, the chunk-invariant and history-free render, and the frozen chains are load-bearing.
4. **Naming**: a new or renamed name says what the item is at the head, reads right without this PR's context, uses the domain's existing word, and is the shortest that stays clear ([Rust API Guidelines](https://rust-lang.github.io/api-guidelines/naming.html)).
5. **Sourced practice**: an external claim in the diff or the PR body that cites no source and is not labeled opinion breaches AGENTS.md's conventions.
