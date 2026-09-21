//! The `turbo` command-line entry point.
//!
//! # Exit codes
//!
//! The distinction is a contract, not an accident, and validation may assert on
//! it:
//!
//! - **0** -- the command did what it was asked.
//! - **1** -- a returned `anyhow::Error`, printed as one `Error: ...` line.
//!   This is the exit code for every *user-input* fault: a path that is not
//!   there, a directory that holds no recording, a `--vcpu` set matching no
//!   recorded hart, a config that contradicts the trace. Such a fault is
//!   reported by the layer that still knows what the user typed, and nothing
//!   is written to disk before that check.
//! - **2** -- clap rejected the argv itself (unknown flag, missing positional).
//! - **101** -- a Rust panic. Panics are reserved for violated internal
//!   invariants, so a 101 out of `turbo` is a *turbo bug* regardless of what
//!   the message says, and should be reported as one.

use clap::Parser;
use turbo::main_interface::{self, *};

#[cfg(feature = "profile")]
#[global_allocator]
static HP_ALLOC: hotpath::CountingAllocator = hotpath::CountingAllocator::new();

fn main() -> anyhow::Result<()> {
    // Top-level "--version" / "-v": print the version stamp on one line and
    // exit before any other initialization. Checked positionally (the first
    // argument) so the flag is strictly top-level and can never shadow a
    // subcommand option. The commit is stamped at build time by build.rs.
    let version_requested = std::env::args()
        .nth(1)
        .is_some_and(|arg| arg == "--version" || arg == "-v");
    if version_requested {
        println!(
            "turbo {} (git commit {})",
            env!("CARGO_PKG_VERSION"),
            option_env!("GIT_COMMIT").unwrap_or("unknown")
        );
        return Ok(());
    }

    // Initialize logger
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // hotpath profiler guard - prints timing/alloc report on drop at process exit
    #[cfg(feature = "profile")]
    let _hp_guard = hotpath::HotpathGuardBuilder::new("turbo").build();

    ctrlc::set_handler(|| {
        log::info!("Ctrl-C received, exiting.");
        std::process::exit(0);
    })
    .expect("Failed to set ctrl-C handler");

    // proxy the CLI args into the library "main_interface" module
    // This design makes it easier to "unit-test" at "command/integration" level.
    let args = Cli::parse();
    main_interface::main_interface(args)
}
