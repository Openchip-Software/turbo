//! Server-sent-event payload wrappers for the live web UI.

use crate::common::RoiInfo;
use serde::Serialize;

/// Data types that can be sent to clients via SSE. `Machine` carries run-wide
/// facts about the modelled CPU and goes first, so the detail pane knows VLEN
/// before it sizes the VL histograms.
#[derive(Serialize, Debug)]
#[serde(tag = "type", content = "data", rename_all = "lowercase")]
pub enum DataType {
    Machine { vlen_bits: Option<u32> },
    Rois { rois: Vec<RoiInfo> },
    Functions { functions: Vec<RoiInfo> },
    Roofline { measurements: Vec<u64> },
}

/// SSE data wrapper
#[derive(Serialize, Debug)]
pub struct SseData {
    pub data: Vec<DataType>,
}

impl SseData {
    pub fn new(data: Vec<DataType>) -> Self {
        SseData { data }
    }
}
