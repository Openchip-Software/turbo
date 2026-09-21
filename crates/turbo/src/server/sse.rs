//! Server-sent-event payload wrappers for the live web UI.

use crate::common::RoiInfo;
use serde::Serialize;

/// Data types that can be sent to clients via SSE
#[derive(Serialize, Debug)]
#[serde(tag = "type", content = "data", rename_all = "lowercase")]
pub enum DataType {
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
