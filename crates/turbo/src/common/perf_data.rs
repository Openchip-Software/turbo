//! Top-level analysis results: [`PerformanceData`] and [`TraceMetadata`].

use super::roi::RoiInfo;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// PerformanceInfo
#[derive(Serialize, Deserialize, Clone, Debug, Default, JsonSchema)]
pub struct PerformanceData {
    pub rois: Vec<RoiInfo>,
    pub functions: Vec<RoiInfo>,
    pub trace_info: Option<TraceMetadata>,
    /// VLEN in bits the run modelled, so the UI can draw each VL histogram over
    /// its full `0..=VLMAX` range (`VLEN * LMUL / SEW`) instead of scaling to
    /// the data. `None` in a `perf_data*.json` written before this existed.
    #[serde(default)]
    pub vlen_bits: Option<u32>,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default, JsonSchema)]
pub struct TraceMetadata {
    // calculated metrics: trace size as percent of workload memory usage
    pub trace_bytes_ratio: f32,
    pub instructions_per_trace_byte: f32,
    /// One field to branch on: whether these numbers describe the WHOLE
    /// recording. See [`TraceIntegrity::is_complete`].
    ///
    /// `serde(default)` on both integrity fields, so a `perf_data*.json`
    /// written before they existed still loads (`turbo load` is routinely
    /// pointed at an older run's output) -- as not-complete, which is the
    /// honest reading of a file that cannot say.
    #[serde(default)]
    pub trace_complete: bool,
    /// The evidence behind `trace_complete`, per hart.
    #[serde(default)]
    pub trace_integrity: TraceIntegrity,
}

/// What the decoded trace was, as opposed to what it reported.
///
/// Exists because "the trace was damaged" and "the program did less work" look
/// identical in a profile: both are simply smaller numbers. A run that ends
/// without its `Done` sentinel is legitimately worth reporting -- a killed or
/// crashed guest is the usual cause and the prefix is real data -- but a caller
/// has to be able to TELL, and log lines are not something a caller can assert
/// on. Hence a machine-readable block in `perf_data_final.json`.
#[derive(Serialize, Deserialize, Clone, Debug, Default, JsonSchema)]
pub struct TraceIntegrity {
    /// One entry per hart whose record log was read, in hart order.
    pub harts: Vec<HartTraceStatus>,
    /// Harts present in the recording but deliberately NOT processed, because
    /// `--vcpu` selected against them or a single log was named explicitly.
    /// A run with any of these is partial by construction.
    pub harts_excluded: Vec<u32>,
}

impl TraceIntegrity {
    /// Whether these numbers describe the whole recording: at least one hart
    /// was read, every hart read reached its `Done` sentinel, and no hart was
    /// left out.
    pub fn is_complete(&self) -> bool {
        !self.harts.is_empty()
            && self.harts_excluded.is_empty()
            && self.harts.iter().all(|h| h.complete)
    }

    /// The harts that were read but did not finish, for the report.
    pub fn incomplete_harts(&self) -> Vec<u32> {
        self.harts
            .iter()
            .filter(|h| !h.complete)
            .map(|h| h.hart_id)
            .collect()
    }
}

/// How completely one hart's record log was read.
#[derive(Serialize, Deserialize, Clone, Debug, Default, JsonSchema)]
pub struct HartTraceStatus {
    pub hart_id: u32,
    /// Bytes read from this hart's `turbo_trace_vcpu<N>.bin`.
    pub bytes_read: u64,
    /// Batches (flush windows) decoded from it.
    pub batches: u64,
    /// The producer's terminal `Done` record was reached, so this log is whole.
    /// False means the recording was cut short: QEMU was killed, the machine
    /// died, or the file was copied while it was still being written.
    pub complete: bool,
    /// Bytes of an incomplete trailing record, when the log ends mid-write.
    /// Zero for a `complete` log, and also zero for one that ends cleanly on a
    /// record boundary without a `Done`.
    pub partial_tail_bytes: u64,
}

impl PartialEq for PerformanceData {
    fn eq(&self, other: &Self) -> bool {
        let mut self_rois = self.rois.clone();
        let mut other_rois = other.rois.clone();
        let mut self_functions = self.functions.clone();
        let mut other_functions = other.functions.clone();

        self_rois.sort_by(|a, b| a.name.cmp(&b.name));
        other_rois.sort_by(|a, b| a.name.cmp(&b.name));
        self_functions.sort_by(|a, b| a.name.cmp(&b.name));
        other_functions.sort_by(|a, b| a.name.cmp(&b.name));

        self_rois == other_rois && self_functions == other_functions
    }
}

impl PerformanceData {
    /// Read a `perf_data*.json` written by [`save`](Self::save).
    ///
    /// Fallible rather than panicking: the file is an ordinary artifact on
    /// disk, so it can be truncated by a killed run, hand-edited, or simply
    /// from an older schema. Every one of those is a bad file the caller can
    /// name in an error message, not a bug in this process.
    pub fn load(file: PathBuf) -> anyhow::Result<PerformanceData> {
        use anyhow::Context as _;
        let json = std::fs::read_to_string(&file)
            .with_context(|| format!("reading performance data from {}", file.display()))?;
        // nosemgrep: deserialization-from-untrusted-source -- JSON is data-only; a bad file surfaces as an Err naming the path
        serde_json::from_str(&json)
            .with_context(|| format!("parsing performance data from {}", file.display()))
    }

    pub fn save(&self, file: PathBuf) {
        log::debug!("saving PerformanceData to {}", file.display());
        use serde_json;

        // Here we serialize to "pretty" to help LLMs understand the data. Without it
        // all data is on a single line, resulting in "context window munching" without
        // value.
        let json = serde_json::to_string_pretty(self).expect("Failed to serialize");
        let json_len = json.len();
        match std::fs::write(&file, &json) {
            Ok(_) => {
                log::info!(
                    "saved PerformanceData as {} JSON bytes to {}",
                    json_len,
                    file.display()
                )
            }
            Err(e) => {
                log::error!(
                    "failed to write PerformanceData as JSON to {}, {e:?}",
                    file.display()
                );
                log::info!("performance data as JSON is:\n\n{}\n\n", json);
            }
        }
    }

    pub fn with_rois_and_funcs(rois: Vec<RoiInfo>, funcs: Vec<RoiInfo>) -> Self {
        Self {
            rois,
            functions: funcs,
            ..Self::default()
        }
    }

    pub fn add_trace_metadata(
        &mut self,
        insn_trace_bytes: u64,
        vl_trace_bytes: u64,
        instruction_count: u64,
        trace_integrity: TraceIntegrity,
    ) {
        // collect all functions' bytes_moved,
        let workload_bytes = self
            .functions
            .iter()
            .map(|roi| roi.load_bytes + roi.store_bytes)
            .sum::<u64>();

        let trace_bytes = insn_trace_bytes + vl_trace_bytes;

        log::debug!(
            "trace bytes {}, workload bytes {}",
            trace_bytes,
            workload_bytes
        );
        log::debug!(
            "instruction count {}, trace bytes {}",
            trace_bytes,
            workload_bytes
        );

        let trace_bytes_ratio = if workload_bytes > 0 {
            trace_bytes as f32 / workload_bytes as f32
        } else {
            trace_bytes as f32
        };

        let trace_metadata = TraceMetadata {
            trace_bytes_ratio,
            instructions_per_trace_byte: if trace_bytes > 0 {
                instruction_count as f32 / trace_bytes as f32
            } else {
                0.0_f32
            },
            trace_complete: trace_integrity.is_complete(),
            trace_integrity,
        };

        // At info, because "these numbers are only part of the run" is not a
        // detail: a reader who does not know it draws conclusions about a
        // program from a fragment of its execution.
        let integrity = &trace_metadata.trace_integrity;
        if trace_metadata.trace_complete {
            log::info!(
                "trace integrity: complete, {} hart(s)",
                integrity.harts.len()
            );
        } else {
            log::warn!(
                "trace integrity: INCOMPLETE -- {} hart(s) read, cut short: {:?}, \
                 not processed: {:?}. These numbers cover only part of the run; \
                 see trace_info.trace_integrity in perf_data_final.json",
                integrity.harts.len(),
                integrity.incomplete_harts(),
                integrity.harts_excluded,
            );
        }
        for hart in &integrity.harts {
            log::debug!("trace integrity: {hart:?}");
        }

        // Calculate trace metrics (in context of Perfdata?)
        log::info!(
                "trace done: insn trace {} vl {} instructions {}. instructions / (insn trace + vl bytes) = {:0.2} => instructions / insn trace (only!) = {:0.2}. trace bytes ratio {:0.4}",
                insn_trace_bytes,
                vl_trace_bytes,
                instruction_count,
                instruction_count as f32 / (insn_trace_bytes + vl_trace_bytes) as f32,
                instruction_count as f32 / insn_trace_bytes as f32,
                trace_metadata.trace_bytes_ratio
            );

        self.trace_info = Some(trace_metadata);
    }

    /// Assert that this PerformanceData equals another, with detailed error messages for testing
    ///
    /// # Arguments
    /// * `expected` - The expected PerformanceData to compare against
    /// * `test_name` - Name of the test (for error messages)
    ///
    /// # Panics
    /// Panics with a detailed message if any metric doesn't match
    pub fn assert_equals(&self, expected: &PerformanceData, test_name: &str) {
        let mut self_rois = self.rois.clone();
        let mut expected_rois = expected.rois.clone();
        let mut self_functions = self.functions.clone();
        let mut expected_functions = expected.functions.clone();

        self_rois.sort_by(|a, b| a.name.cmp(&b.name));
        expected_rois.sort_by(|a, b| a.name.cmp(&b.name));
        self_functions.sort_by(|a, b| a.name.cmp(&b.name));
        expected_functions.sort_by(|a, b| a.name.cmp(&b.name));

        assert_eq!(
            self_rois.len(),
            expected_rois.len(),
            "ROI count mismatch for {}: expected {}, got {}",
            test_name,
            expected_rois.len(),
            self_rois.len()
        );

        assert_eq!(
            self_functions.len(),
            expected_functions.len(),
            "Function count mismatch for {}: expected {}, got {}",
            test_name,
            expected_functions.len(),
            self_functions.len()
        );

        for (i, (actual, expected)) in self_rois.iter().zip(expected_rois.iter()).enumerate() {
            actual.assert_equals(
                expected,
                &format!("{} ROI[{}] '{}'", test_name, i, actual.name),
            );
        }

        for (i, (actual, expected)) in self_functions
            .iter()
            .zip(expected_functions.iter())
            .enumerate()
        {
            actual.assert_equals(
                expected,
                &format!("{} Function[{}] '{}'", test_name, i, actual.name),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `turbo load` is routinely pointed at an older run's output directory, so
    /// a `perf_data*.json` written before the trace-integrity block existed
    /// must still load -- as not-complete, which is the honest reading of a
    /// file that cannot say either way.
    #[test]
    fn a_report_without_the_integrity_block_still_loads() {
        let dir = std::env::temp_dir().join("turbo_perf_data_compat");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("perf_data_final.json");
        std::fs::write(
            &path,
            r#"{"rois":[],"functions":[],
                "trace_info":{"trace_bytes_ratio":0.5,"instructions_per_trace_byte":1.5}}"#,
        )
        .unwrap();

        let pd = PerformanceData::load(path).expect("an older report must still load");
        let info = pd.trace_info.expect("trace_info survives");
        assert_eq!(info.instructions_per_trace_byte, 1.5);
        assert!(
            !info.trace_complete,
            "a file that cannot say is not complete"
        );
        assert!(info.trace_integrity.harts.is_empty());
    }

    /// The three ways a run can fail to cover the whole recording.
    #[test]
    fn completeness_needs_every_hart_read_and_finished() {
        let whole = |hart_id| HartTraceStatus {
            hart_id,
            bytes_read: 100,
            batches: 2,
            complete: true,
            partial_tail_bytes: 0,
        };

        assert!(TraceIntegrity {
            harts: vec![whole(0), whole(1)],
            harts_excluded: vec![],
        }
        .is_complete());

        // Nothing was read at all.
        assert!(!TraceIntegrity::default().is_complete());

        // A hart was read but never reached its Done sentinel.
        let mut cut_short = whole(1);
        cut_short.complete = false;
        let integrity = TraceIntegrity {
            harts: vec![whole(0), cut_short],
            harts_excluded: vec![],
        };
        assert!(!integrity.is_complete());
        assert_eq!(integrity.incomplete_harts(), vec![1]);

        // A hart was left unread (`--vcpu`).
        assert!(!TraceIntegrity {
            harts: vec![whole(0)],
            harts_excluded: vec![1],
        }
        .is_complete());
    }
}
