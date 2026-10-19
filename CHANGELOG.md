# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
