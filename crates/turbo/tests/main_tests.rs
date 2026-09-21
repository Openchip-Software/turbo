//! The `turbo` integration test suite.
//!
//! One test binary, not one per topic: the shared [`harness`] used to be
//! recompiled into every `tests/*.rs` target, which is why it needed a blanket
//! `allow(dead_code)` -- each copy saw only the slice of the harness its own
//! file happened to use. With a single target every helper has a real user, so
//! unused ones are now caught.
//!
//! Test names gain their module as a prefix, e.g.
//! `cargo nextest run -p turbo trace::flash_trace`.

/// Declare a table of tests that each put one guest binary through `$runner`.
///
/// Every such test was previously five near-identical lines whose only
/// distinguishing content was a binary path, a name repeated verbatim as a
/// string, and sometimes guest arguments. Here the name is the test's own
/// identifier via `stringify!`, so the test and its `target/test_output/`
/// directory cannot drift apart.
///
/// `$runner` is `fn(name: &str, bin: &str, args: &[&str])`.
macro_rules! turbo_tests {
    (@args) => { &[] };
    (@args $args:expr) => { $args };

    ($runner:path { $(
        $(#[$attr:meta])*
        $name:ident => $bin:expr $(, $args:expr)? ;
    )* }) => { $(
        $(#[$attr])*
        #[test]
        fn $name() {
            $runner(stringify!($name), $bin, turbo_tests!(@args $($args)?));
        }
    )* };
}

/// The same idea as [`turbo_tests`] for runners that need nothing but the
/// name -- they derive the binary path and the expected-results fixture key
/// from it.
///
/// `$runner` is `fn(name: &str) -> _`.
macro_rules! turbo_tests_named {
    ($runner:path { $($name:ident),* $(,)? }) => { $(
        #[test]
        fn $name() {
            $runner(stringify!($name));
        }
    )* };
}

mod harness;

mod annotate;
mod asm_validation;
mod cache_sim;
mod cli;
mod config_cmd;
mod intensity;
mod ld_library_path;
mod mem_replay;
mod pc_equivalence;
mod roi;
mod threading;
mod trace;
mod user_performance;
mod vdso;
mod vlen_sweep;
