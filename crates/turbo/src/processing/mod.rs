//! Instruction trace processing pipeline.
//!
//! This module orchestrates the multi-stage pipeline for converting raw E-Trace bytes
//! into enriched instruction events with performance statistics:
//!
//! 1. **Binary Metadata** ([`elf_metadata`]): Load ELF symbols, function names, and library info
//! 2. **Trace Decoding** ([`data_processor`], [`instruction_decoder`]): Parse E-Trace format streams
//! 3. **Instruction Analysis** ([`instruction_decoder`]): Decode RISC-V instructions, detect ROI markers
//! 4. **Enrichment** ([`enricher`]): Track vector state, accumulate statistics per region/function
//! 5. **Perfetto Export** ([`enricher::trace_emitter`]): Generate time-stamped trace events
//!
//! ## Key Components
//!
//! - [`DataProcessor`]: Main decoder that orchestrates instruction pipeline per hart
//! - [`ElfMetadata`]: Reads ELF binary and shared library symbols, manages address-to-function mapping
//! - [`ElfResolver`]: Converts PC to function name via binary metadata
//! - [`Enricher`]: Per-hart statistics accumulator and trace emitter
//! - [`InstructionDecoder`]: RISC-V instruction decoding and ROI marker detection
//! - [`TraceFormat`]: Wire format of an incoming trace stream
//!
//! ## Data Flow
//!
//! ```text
//! [TraceData bytes]
//!     |
//!     v
//! [DataProcessor.decode_into]
//! (format-aware decoding)
//!     |
//! +---+---+
//! |       |
//! v       v
//! [PC batch] [VL values]
//!     |         |
//!     +----+----+
//!          |
//!          v
//!   [Enricher.transform_optimized]
//!   (stats accumulation)
//!          |
//!          v
//!   [PerformanceData] -> JSON + HTTP

pub mod cache_model;
pub mod cache_sim;
pub mod data_processor;
pub mod decode_pool;
mod elf_metadata;
pub mod enricher;
mod function_lookup;
pub mod histogram;
mod instruction_decoder;
pub mod mem_replay;
pub mod roi_flamegraph;

pub use cache_model::{AccessKind, Caches, Config as CacheConfig, Counts, Footprint};
pub use cache_sim::{simulate_trace_file, CacheSim, SimSummary};
pub use data_processor::BbtBatch;
pub use data_processor::BbtBlock;
pub use data_processor::BbtDecoder;
pub use data_processor::DataProcessor;
pub use data_processor::HartDecoder;
pub use data_processor::TraceBatch;
pub use data_processor::TraceFormat;
pub use elf_metadata::ElfMetadata;
pub use elf_metadata::SharedLibInfo;
pub use enricher::annotate::read_page_gz;
pub use enricher::BatchSideChannels;
pub use enricher::Enricher;
pub use enricher::PcProfile;
pub use function_lookup::ElfResolver;
pub use function_lookup::MultiResolver;
pub use instruction_decoder::InstructionDecoder;
pub use instruction_decoder::MultiDecoder;
pub use instruction_decoder::RoiMarker;
pub use mem_replay::{replay_trace_file, MemAccess, MemReplay};

// Re-export common types from crate root
pub use crate::common::RoiInfo;
