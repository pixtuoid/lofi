# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0](https://github.com/pixtuoid/lofi/compare/v0.2.0...v0.3.0) - 2026-10-27

### Added

- *(stream)* the plate rings wetter, as deep as the judge took it ([#50](https://github.com/pixtuoid/lofi/pull/50))
- *(tools)* lofi_measure --pins measures every pinned constant again, beside the source's value ([#46](https://github.com/pixtuoid/lofi/pull/46))
- *(tools)* lofi_measure --staging prints each staging gain beside the whole-lane medians it stands in for ([#45](https://github.com/pixtuoid/lofi/pull/45))
- *(synth)* hats sound, a metallic hash under the drums' chain ([#42](https://github.com/pixtuoid/lofi/pull/42))
- the night ducks only under kicks that sound, its tempo sits inside MEASURED, and chords sound clear ([#40](https://github.com/pixtuoid/lofi/pull/40))
- the bass window rises to G1-F#2, an octave under a chord voice it would crowd ([#37](https://github.com/pixtuoid/lofi/pull/37))
- *(compose)* each stage draws from its own stream ([#36](https://github.com/pixtuoid/lofi/pull/36))
- a song opens, breathes and closes, and its lead sings at every tier ([#30](https://github.com/pixtuoid/lofi/pull/30))
- *(tools)* lofi_measure reads the stream's lanes and the corpus's percentiles ([#26](https://github.com/pixtuoid/lofi/pull/26))
- a stereo plate rings under the pad, keys and lead ([#23](https://github.com/pixtuoid/lofi/pull/23))

### Other

- main runs never cancel, every failing test reports, and the examples' and wasm's items are linted and documented ([#53](https://github.com/pixtuoid/lofi/pull/53))
- *(synth)* voices, the lane recipe and the whole-lane chain each get a file ([#48](https://github.com/pixtuoid/lofi/pull/48))
- *(dsp)* [**breaking**] a noise envelope is `Band`s in fractions, smoothed in Hz at any loop length ([#44](https://github.com/pixtuoid/lofi/pull/44))
- *(tools)* loudness reads both channels, the corpus reads again from the repo, and one FFT ([#39](https://github.com/pixtuoid/lofi/pull/39))
- pin actions by commit, declare and check rust-version 1.88, and drop the foley tools ([#35](https://github.com/pixtuoid/lofi/pull/35))
- one style per mood, stages that read only their own, and named events ([#34](https://github.com/pixtuoid/lofi/pull/34))
- typed lane recipes, and frozen takes that play the shipped voices ([#33](https://github.com/pixtuoid/lofi/pull/33))
- a song owns its automation, and the schedule its steps ([#32](https://github.com/pixtuoid/lofi/pull/32))
- one exact newest toolchain from rust-toolchain.toml ([#28](https://github.com/pixtuoid/lofi/pull/28))

## [0.2.0](https://github.com/pixtuoid/lofi/compare/v0.1.0...v0.2.0) - 2026-10-19

### Added

- Performance::MASTER_GAIN names the stage a host's sounds share ([#21](https://github.com/pixtuoid/lofi/pull/21))
- Controls::DUSK names where daylight picks the night song ([#20](https://github.com/pixtuoid/lofi/pull/20))
- rain recolours the playing song ([#19](https://github.com/pixtuoid/lofi/pull/19))
- intensity and daylight latch at bar lines ([#18](https://github.com/pixtuoid/lofi/pull/18))
- *(tools)* the judge batch plays the twilight sweep; one-sided takes are listed, not failed ([#16](https://github.com/pixtuoid/lofi/pull/16))
- *(tools)* lofi_audition renders named control sweeps and --songs N ([#15](https://github.com/pixtuoid/lofi/pull/15))
- *(tools)* the A/B judge runs on Modal and scores only the pairs a change touched ([#13](https://github.com/pixtuoid/lofi/pull/13))
- [**breaking**] lofi plays as an endless stream: Performance, Controls and stereo frames ([#10](https://github.com/pixtuoid/lofi/pull/10))
- *(tools)* measure lanes on gated loudness; tier renders and a blind A/B for every sound change ([#8](https://github.com/pixtuoid/lofi/pull/8))

### Other

- a changed sound is gated by ab_judge.py's verdict; the owner's listen overrides it ([#11](https://github.com/pixtuoid/lofi/pull/11))
- point links at pixtuoid/pixtuoid; CI holds zero dependencies ([#3](https://github.com/pixtuoid/lofi/pull/3))
