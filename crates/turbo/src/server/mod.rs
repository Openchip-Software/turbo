//! HTTP server for real-time performance data visualization.
//!
//! This module implements a simple HTTP server that streams live PerformanceData
//! as the enrichment pipeline processes trace events. Clients can connect to view
//! progress and intermediate results while trace analysis is ongoing.

pub mod http;
pub mod sse;
pub use http::*;
pub use sse::*;
