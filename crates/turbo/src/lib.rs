//! # Turbo: RISC-V Instruction Trace Analysis and Performance Profiling
//!
//! Turbo is a Rust crate for analyzing RISC-V instruction execution traces captured from QEMU
//! and hardware simulators. It reconstructs the dynamic instruction stream from the
//! E-Trace format, enriches trace events with symbol and region information, computes detailed
//! performance metrics (including vector operations), and presents results via a web server.
//!
//! ## Core Workflow
//!
//! The main entry point is [`workflow::run_perf_workflow_opt`], which:
//! 1. Loads ELF binary metadata (symbols, functions, shared libraries)
//! 2. Runs the whole pipeline on the calling thread: an [`source::TraceSource`]
//!    is polled for trace chunks (live QEMU spool or a finished trace file),
//!    each chunk is decoded into PC batches, and each batch is enriched inline
//!    (instruction stream reconstruction + statistics accumulation) as it is
//!    produced. The only other thread is the optional **HTTP server thread**
//!    that streams live performance data.
//! 3. Merges results from all hardware threads (harts) and outputs JSON performance reports
//!
//! ## Key Modules
//!
//! - [`source`]: Input adapters for QEMU (live execution or plugin output), trace files, pipetrace traces
//! - [`processing`]: E-Trace decoding pipeline, instruction analysis, metadata extraction
//! - [`processing::enricher`]: Multi-stage instruction enrichment (VL tracking, region markers, statistics)
//! - [`server`]: HTTP server for live progress monitoring
//! - [`common`]: Core data structures (RoiInfo for functions/regions, PerformanceData aggregates)
//! - [`config`]: `EnvConfig` for tool paths (QEMU) and `CpuConfig` for the
//!   modelled CPU (VLEN, cache geometry)
//! - [`workflow`]: Top-level pipeline entry points (`run_perf_workflow_opt` and its live-QEMU/file/load wrappers)
//! - [`report`]: Console summary reporting
//! - [`error`]: `RerfError`, the library's error type
//!
//! ## Data Flow
//!
//! All of this runs on a single thread -- each stage is a direct call into the
//! next, with no channels and no producer/consumer handoff:
//!
//! ```text
//! [QEMU child process] --writes--> [turbo_trace_vcpu<N>.bin spool files]
//!                                            |
//!                                            v  poll (tail)
//!                                     [TraceSource]
//!                                            |
//!                                            v  per chunk
//!                                     [DataProcessor]
//!                                        (decode)
//!                                            |
//!                                            v  per 100K-PC batch
//!                                   [Enricher per hart]
//!                                  (stats accumulation)
//!                                            |
//!                                            v
//!            [PerformanceData] -> [HTTP server] & [JSON output]
//! ```
//!
//! ## Trace Formats
//!
//! The trace encoding scheme is carried by [`processing::TraceFormat`]:
//! - **Bbt**: the QEMU turbo tracer plugin's only software-tracing format (one block ID
//!   per block entry plus a shared block dictionary); what `run`/`process` use.
//!
//! ## Multi-Hart Support
//!
//! Each hardware thread (hart) maintains independent decoding state and statistics accumulation.
//! Per-hart results are merged into a global PerformanceData structure for reporting.

pub mod assets;
pub mod common;
pub mod config;
pub mod config_cmd;
pub mod error;
pub mod inspect;
pub mod main_interface;
pub mod processing;
pub mod progress;
pub mod report;
pub mod server;
pub mod source;
pub mod util;
pub mod workflow;

pub use error::{RerfError, RerfErrorKind};
pub use report::{
    print_function_summary, print_program_summary, print_regions_summary, print_summary,
};
pub use util::{extract_panic, get_build_artifact_dir, workspace_root};
pub use workflow::{
    process_trace_input_file, record_with_qemu, run_from_input_rois, run_perf_workflow_opt,
    run_with_live_qemu,
};
