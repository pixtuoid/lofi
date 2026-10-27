# The commands the git hooks and ci.yml run, so each changes in one place;
# ci.yml keeps inline only what runs nowhere else.

# One dialect on every platform: GitHub's Windows runners ship Git with Bash
# (https://github.com/actions/runner-images/blob/main/images/windows/Windows2025-Readme.md),
# so the test job's recipe runs there as it does here.
set shell := ["bash", "-euo", "pipefail", "-c"]

# mise.toml's tools, found without a shell that activates mise
_mise_paths := shell('command -v mise >/dev/null || exit 0; MISE_OFFLINE=1 mise env --paths')
export PATH := (if _mise_paths == '' { '' } else { replace_regex(_mise_paths, '\r?\n', PATH_VAR_SEP) + PATH_VAR_SEP }) + env('PATH')

# The nightly api/pixtuoid-lofi.txt is built on: rustdoc JSON is nightly-only,
# and cargo-public-api 0.52 reads nightly-2025-11-22 or later
# (https://github.com/cargo-public-api/cargo-public-api/blob/v0.52.0/README.md)
API_NIGHTLY := "nightly-2026-10-22"
# the listing: `-s` is "--omit blanket-impls", in cargo-public-api's "shorthand
# for omitting noisy items" (`cargo public-api --help`)
API_LIST := "RUSTUP_TOOLCHAIN=" + API_NIGHTLY + " cargo public-api -s"

# Turn on the hooks and install mise.toml's tools, once per clone
setup:
    git config core.hooksPath .githooks
    mise install

# What pre-push runs: quick over a warm build, so the render-heavy tests stay CI's
preflight: fmt-check clippy doc api-check

fmt-check:
    cargo fmt --all --check

clippy:
    cargo clippy --all-targets -- -D warnings

# The public docs, then every item's and every example's: rustdoc resolves a
# link only on an item it renders
[doc('The public docs, then every item and example, warnings as errors')]
doc:
    RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
    RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --document-private-items --lib --examples

# The public API, `#[doc(hidden)]` items aside (AGENTS.md), as a listing a PR's
# diff shows at review time; release-plz's semver check reports "in the release
# Pull Request and in the output of the `release-plz update` command"
# (https://release-plz.dev/docs/semver-check)
[doc('Rewrite api/pixtuoid-lofi.txt, the public API cargo-public-api lists')]
api: _api-nightly
    {{ API_LIST }} > api/pixtuoid-lofi.txt

# Fail when the public API moved off api/pixtuoid-lofi.txt
api-check: _api-nightly
    #!/usr/bin/env bash
    set -euo pipefail
    listed=$(mktemp)
    trap 'rm -f "$listed"' EXIT
    {{ API_LIST }} > "$listed"
    diff -u api/pixtuoid-lofi.txt "$listed" || {
        # a workflow command, as `workflows` prints its own
        echo "::error::the public API moved off api/pixtuoid-lofi.txt: run 'just api' and commit it"
        exit 1
    }

_api-nightly:
    rustup toolchain install {{ API_NIGHTLY }} --profile minimal --no-self-update

# Every failing test binary, not only the first
test:
    cargo test --all-targets --no-fail-fast
    cargo test --doc

# Cargo.toml's rust-version, checked as the Cargo book's CI guide checks it
# (https://doc.rust-lang.org/cargo/guide/continuous-integration.html#verifying-rust-version)
[doc("Check the build at Cargo.toml's rust-version")]
msrv:
    cargo hack check --rust-version --all-targets --locked

# The wasm32 build, and the lints and docs of the items the host build compiles out
wasm:
    cargo build --lib --target wasm32-unknown-unknown
    cargo build --example lofi_bench --target wasm32-unknown-unknown
    cargo clippy --lib --example lofi_bench --target wasm32-unknown-unknown -- -D warnings
    RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --document-private-items --lib --example lofi_bench --target wasm32-unknown-unknown

# The repo root, not just workflows/, so dependabot.yml is audited too; then
# whether ci-gate needs every job, since a job blocks a merge only through its
# needs (`coverage` is advisory)
[doc('Audit the workflows, and check ci-gate needs every job')]
workflows:
    #!/usr/bin/env bash
    set -euo pipefail
    zizmor --config .github/zizmor.yml .
    actionlint
    jobs=$(yq '.jobs | keys | .[] | select(. != "ci-gate" and . != "coverage")' .github/workflows/ci.yml | sort)
    needs=$(yq '.jobs["ci-gate"].needs[]' .github/workflows/ci.yml | sort)
    if [ "$jobs" != "$needs" ]; then
        # a workflow command, which runs "sent to the runner over `stdout`" and which
        # "will create an annotation" (https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-commands);
        # run locally, it prints as plain text
        echo "::error::ci-gate's needs differ from the gating jobs (< jobs, > needs)"
        diff <(echo "$jobs") <(echo "$needs") >&2 || true
        exit 1
    fi
