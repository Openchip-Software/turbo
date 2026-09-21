//! Core data structures for performance analysis results.
//!
//! This module defines the primary output types that hold aggregated instruction statistics:
//! - [`PerformanceData`]: Top-level results (ROIs, functions, and trace metadata)
//! - [`RoiInfo`]: Statistics for a single region/function (instruction counts, vector metrics, memory operations)
//! - [`RoiExecution`]: A single dynamic execution of an ROI-marked region (with begin/end marker ordinals)
//! - [`Lmul`]: RISC-V vector register grouping mode
//! - [`SystemVlen`]: Global vector length configuration
//!
//! These structures support both scalar and vector instruction accounting, including:
//! - Element counts (weighted by vector length for vector instructions)
//! - FMA operations (fused multiply-accumulate)
//! - Vector load/store histograms per SEW and LMUL
//!
//! The types live in focused submodules; this module re-exports all of them so
//! `crate::common::*` remains a single flat import surface.

pub mod insn_class;
pub mod perf_data;
pub mod roi;
pub mod vector;

pub use insn_class::*;
pub use perf_data::*;
pub use roi::*;
pub use vector::*;
