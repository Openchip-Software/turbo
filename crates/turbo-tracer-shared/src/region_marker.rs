/// Metadata about the trace, in JSON format
use serde::{Deserialize, Serialize};

/// Type of ROI marker detected in the instruction stream.
/// Only R-type markers (those requiring runtime register values) are captured here.
/// Immediate-only markers (li x0, -N) are statically decodable from the binary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoiRtMarkerKind {
    /// and x0, rs1, rs2 -- NameEvent (V1/V2) or NameValue preamble
    And,
    /// or x0, rs1, rs2 -- EventAndValue
    Or,
    /// sll x0, rs1, rs2 -- V2 EventString (name ptr + len)
    Sll,
    /// srl x0, rs1, rs2 -- V2 ValueString (name ptr + len)
    Srl,
    /// add x0, rs1, rs2 -- V2 BeginRegion (name ptr + len)
    Add,
    /// sub x0, rs1, rs2 -- V2 EndRegion (name ptr + len)
    Sub,
    /// xor x0, rs1, rs2 -- reserved. Consumed plugin-internally by whichever
    /// build defines a meaning for it, and never forwarded as a RoiRtMarker.
    Xor,
}

/// A ROI marker captured at runtime with the actual register values.
/// These are only generated for R-type instructions writing to x0,
/// which require runtime register values to decode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoiRtMarker {
    /// PC of the instruction that generated this marker
    pub pc: u64,
    /// The kind of R-type operation
    pub kind: RoiRtMarkerKind,
    /// Runtime value of rs1
    pub rs1_val: u64,
    /// Runtime value of rs2
    pub rs2_val: u64,
    /// For pointer-carrying V2 markers (BeginRegion/EndRegion/EventString/
    /// ValueString), the name string read from guest memory at rs1_val at
    /// runtime. This is required when the name lives on the stack (e.g. Fortran
    /// passes a copy of a string literal) rather than in the ELF image.
    pub name: Option<String>,
}
