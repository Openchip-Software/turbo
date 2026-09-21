//! Driving the cache model from a trace's captured memory footprint.
//!
//! The bridge between [`MemReplay`](crate::processing::MemReplay), which turns
//! captured address groups back into footprints, and
//! [`Caches`](crate::processing::cache_model::Caches), which turns footprints
//! into hit/miss counts. It is deliberately thin: the interesting decisions are
//! in those two, and the wiring between them is the part that has to be
//! obviously right.
//!
//! Per **instruction**, not per block. `Caches::access` is called once per
//! instruction whatever its addressing mode, so `(pc, counts)` is what the model
//! naturally hands back; summing it here would discard the per-PC attribution
//! for nothing. A caller wanting only a block total folds it in the closure.
//!
//! # Ordering
//!
//! A cache model is order-sensitive in a way none of the pipeline's other
//! counters are. Blocks must arrive in execution order and the model must not be
//! reset between batches, both of which hold inside one hart's processing and
//! would break if this were ever moved to a per-batch parallel stage. One
//! instance per hart, never shared.

use std::path::Path;
use std::sync::Arc;

use anyhow::{bail, Context, Result};

use crate::processing::cache_model::{Caches, Config, Counts};
use crate::processing::enricher::Modules;
use crate::processing::mem_replay::MemReplay;
use crate::processing::{BbtBlock, DataProcessor, ElfMetadata, TraceBatch, TraceFormat};

/// One hart's cache modelling state.
pub struct CacheSim {
    caches: Caches,
    replay: MemReplay,
}

impl CacheSim {
    pub fn new(modules: Arc<Modules>, vlenb: u64, config: Config) -> Result<Self> {
        Ok(Self {
            caches: Caches::new(config).map_err(|e| anyhow::anyhow!("cache config: {e}"))?,
            replay: MemReplay::new(modules, vlenb),
        })
    }

    /// Feed every vector memory instruction of `block` to the model, in program
    /// order, reporting each one's own counts to `on_insn`.
    pub fn block(
        &mut self,
        block: &BbtBlock<'_>,
        mut on_insn: impl FnMut(u64, Counts),
    ) -> Result<()> {
        let caches = &mut self.caches;
        self.replay.block_footprints(block, |pc, kind, fp| {
            let counts = caches.access(kind, fp);

            // hardcore debug
            if false {
                log::trace!(
                    "cache: pc={pc:#x} {kind:?} {fp:?} -> l1 {}h/{}m l2 {}h/{}m",
                    counts.l1_hits,
                    counts.l1_misses,
                    counts.l2_hits,
                    counts.l2_misses
                );
            }
            on_insn(pc, counts);
        })
    }

    /// Cumulative counts over every access so far.
    pub fn counts(&self) -> Counts {
        self.caches.counts()
    }

    /// How many times the model was called, i.e. how many vector memory
    /// instructions were replayed. The wiring check: this must equal the number
    /// of such instructions executed, not the number of lines or elements they
    /// touched.
    pub fn calls(&self) -> u64 {
        self.caches.calls()
    }
}

/// What [`simulate_trace_file`] found.
pub struct SimSummary {
    pub counts: Counts,
    /// Model calls, i.e. vector memory instructions replayed.
    pub calls: u64,
    /// Per-PC counts, for attribution and for tests that want to name an
    /// instruction rather than a total.
    pub per_pc: std::collections::BTreeMap<u64, Counts>,
}

/// Replay one recorded per-hart trace through a cache model, whole.
///
/// The offline entry point: the same walk the enricher does, without any of the
/// enricher. Used by the integration tests, and by anyone who wants cache
/// numbers out of a trace that has already been recorded.
pub fn simulate_trace_file(
    trace_path: &Path,
    elf_path: &Path,
    vlenb: u64,
    config: Config,
) -> Result<SimSummary> {
    use turbo_tracer_shared::{
        check_record_log_header, decode_next_record, StreamRecord, RECORD_LOG_HEADER_BYTES,
    };

    let buf = std::fs::read(trace_path)
        .with_context(|| format!("reading trace file {}", trace_path.display()))?;
    check_record_log_header(&buf)
        .map_err(anyhow::Error::msg)
        .with_context(|| format!("{}", trace_path.display()))?
        .with_context(|| {
            format!(
                "{}: too short for a record-log header",
                trace_path.display()
            )
        })?;
    let mut cursor = RECORD_LOG_HEADER_BYTES;

    let metadata = ElfMetadata::new(elf_path.to_path_buf());
    let modules = Arc::new(Modules::from_metadata(&metadata)?);
    let dp = DataProcessor::with_modules(modules.clone(), TraceFormat::Bbt)?;
    let mut decoder = dp.make_decoder();
    let mut sim = CacheSim::new(modules, vlenb, config)?;
    let mut per_pc: std::collections::BTreeMap<u64, Counts> = Default::default();

    let mut record_decoder = turbo_tracer_shared::RecordLogDecoder::new();
    loop {
        match decode_next_record(&mut record_decoder, &buf, &mut cursor) {
            // A torn tail is what a run still being written looks like; what was
            // decoded before it is still valid.
            None | Some(Ok(StreamRecord::Done)) => break,
            Some(Err(e)) => bail!("{}: {e}", trace_path.display()),
            Some(Ok(StreamRecord::Batch { trace_data, .. })) => {
                let mut err = None;
                dp.decode_into(&mut decoder, &trace_data, |batch| {
                    let TraceBatch::Blocks(blocks) = batch else {
                        err.get_or_insert_with(|| {
                            anyhow::anyhow!("BBT decode produced a non-block batch")
                        });
                        return;
                    };
                    for block in blocks {
                        if err.is_some() {
                            return;
                        }
                        match block.map_err(anyhow::Error::from) {
                            Ok(b) => {
                                let r = sim.block(&b, |pc, counts| {
                                    *per_pc.entry(pc).or_default() += counts;
                                });
                                if let Err(e) = r {
                                    err = Some(e);
                                }
                            }
                            Err(e) => err = Some(e),
                        }
                    }
                })?;
                if let Some(e) = err {
                    return Err(e);
                }
            }
        }
    }

    Ok(SimSummary {
        counts: sim.counts(),
        calls: sim.calls(),
        per_pc,
    })
}
