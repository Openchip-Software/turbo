//! RISC-V vector configuration types: [`Lmul`] and the global [`SYSTEM_VLEN`].

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU16, Ordering};

/// RISC-V Vector LMUL (Length MULtiplier) encoding.
///
/// Represents the vlmul field from the vtype CSR (bits [2:0]).
/// Controls how many vector registers are grouped together.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Lmul {
    /// m1 - single register (vlmul=0)
    M1,
    /// m2 - 2 registers grouped (vlmul=1)
    M2,
    /// m4 - 4 registers grouped (vlmul=2)
    M4,
    /// m8 - 8 registers grouped (vlmul=3)
    M8,
    /// mf8 - 1/8 of a register (vlmul=5)
    Mf8,
    /// mf4 - 1/4 of a register (vlmul=6)
    Mf4,
    /// mf2 - 1/2 of a register (vlmul=7)
    Mf2,
}

impl Lmul {
    /// Decode from the raw vlmul field (bits [2:0] of vtype).
    /// Returns `None` for reserved encoding (4).
    pub fn from_vlmul(bits: u8) -> Option<Self> {
        match bits & 0x7 {
            0 => Some(Lmul::M1),
            1 => Some(Lmul::M2),
            2 => Some(Lmul::M4),
            3 => Some(Lmul::M8),
            5 => Some(Lmul::Mf8),
            6 => Some(Lmul::Mf4),
            7 => Some(Lmul::Mf2),
            _ => None, // 4 is reserved
        }
    }

    /// Convert back to the raw vlmul encoding (0..=7).
    pub fn to_vlmul(self) -> u8 {
        match self {
            Lmul::M1 => 0,
            Lmul::M2 => 1,
            Lmul::M4 => 2,
            Lmul::M8 => 3,
            Lmul::Mf8 => 5,
            Lmul::Mf4 => 6,
            Lmul::Mf2 => 7,
        }
    }

    /// The multiplier as a float (e.g. M2 -> 2.0, Mf4 -> 0.25).
    pub fn multiplier(self) -> f64 {
        match self {
            Lmul::M1 => 1.0,
            Lmul::M2 => 2.0,
            Lmul::M4 => 4.0,
            Lmul::M8 => 8.0,
            Lmul::Mf2 => 0.5,
            Lmul::Mf4 => 0.25,
            Lmul::Mf8 => 0.125,
        }
    }
}

impl std::fmt::Display for Lmul {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Lmul::M1 => write!(f, "m1"),
            Lmul::M2 => write!(f, "m2"),
            Lmul::M4 => write!(f, "m4"),
            Lmul::M8 => write!(f, "m8"),
            Lmul::Mf8 => write!(f, "mf8"),
            Lmul::Mf4 => write!(f, "mf4"),
            Lmul::Mf2 => write!(f, "mf2"),
        }
    }
}

/// System VLEN in bytes. Set only during QemuRunner construction.
/// 0 is the sentinel for "not yet set".
///
/// Bytes, not bits, is what keeps this a `u16`: the RVV maximum VLEN of 65536
/// bits is 8192 bytes. Every VLEN *in bits* is a `u32` for exactly that reason.
pub struct SystemVlen {
    vlen_bytes: AtomicU16,
}

impl SystemVlen {
    // Module-private to prevent initialisation after startup
    const fn new() -> Self {
        SystemVlen {
            vlen_bytes: AtomicU16::new(DEFAULT_VLEN_BYTES),
        }
    }

    /// Returns the current system VLEN in bytes.
    ///
    /// # Panics
    /// Panics if VLEN has not been set (programming error:
    /// QemuRunner must be constructed before calling this).
    pub fn get_bytes(&self) -> u16 {
        let v = self.vlen_bytes.load(Ordering::Acquire);
        assert!(
            v != 0,
            "SYSTEM_VLEN has not been set; QemuRunner must be constructed first"
        );
        v
    }

    /// Whether a VLEN has been set (non-sentinel).
    pub fn is_set(&self) -> bool {
        self.vlen_bytes.load(Ordering::Acquire) != 0
    }

    /// Set the system VLEN in bytes
    pub fn set_bytes(&self, vlen_bytes: u16) {
        self.vlen_bytes.store(vlen_bytes, Ordering::Release);
    }

    /// Reset VLEN to unset state (e.g. between multiple runs).
    pub fn reset(&self) {
        self.vlen_bytes.store(0, Ordering::Release);
    }
}

/// The VLEN, in **bits**, that a run models when neither `--vlen` nor
/// `cpu_config.json` sets one. It is the `vlen=` QEMU is launched with, so it is
/// the single source of the default: every "default VLEN" elsewhere is this
/// constant, never a value read back out of [`SYSTEM_VLEN`].
///
/// It is in bits, like every VLEN a user sees, and [`DEFAULT_VLEN_BYTES`] is
/// derived from it rather than written out. The two used to be one hand-written
/// `512`, meant as bits and stored as bytes, which made the default 4096 bits.
pub const DEFAULT_VLEN_BITS: u32 = 1024;

/// [`DEFAULT_VLEN_BITS`] in bytes, the unit [`SystemVlen`] stores.
const DEFAULT_VLEN_BYTES: u16 = (DEFAULT_VLEN_BITS / 8) as u16;
const _: () = assert!(
    DEFAULT_VLEN_BITS.is_power_of_two()
        && DEFAULT_VLEN_BITS >= 128
        && DEFAULT_VLEN_BITS <= 65536
        && DEFAULT_VLEN_BYTES as u32 * 8 == DEFAULT_VLEN_BITS,
    "DEFAULT_VLEN_BITS must be a VLEN RVV v1.0 permits"
);
pub static SYSTEM_VLEN: SystemVlen = SystemVlen::new();

/// Complain, loudly, when a configured VLEN disagrees with the one the guest
/// actually executed at.
///
/// The guest's VLEN is not something a config file can assert: the tracer plugin
/// reads QEMU's own `vlenb` CSR at vCPU init and stamps it into
/// `turbo_metadata.json`, so that value is what the program *ran* at, whatever
/// was asked for. The two can diverge in both directions -- a QEMU build that
/// caps VLEN below the requested one silently clamps it, and re-processing a
/// recording with a `cpu_config.json` that has since been edited configures a
/// VLEN the recording never used.
///
/// Either way the recorded value is the one to keep, because VLEN sizes every
/// vector operation: element counts, bytes moved and cache footprints are all
/// derived from it, so a mismatch does not degrade the report, it invalidates
/// the vector half of it. Hence a loud message rather than a quiet one -- and no
/// silent adjustment, so the numbers can never be right by accident.
///
/// `configured` is `None` when the invocation asserted no VLEN at all, which is
/// not a mismatch: the recording is then the only claim about it.
pub fn warn_on_vlen_mismatch(configured_bits: Option<u32>, recorded_bits: u32, source: &str) {
    let Some(configured_bits) = configured_bits else {
        return;
    };
    if configured_bits == recorded_bits {
        return;
    }
    // A recorded 0 is not a VLEN the guest ran at -- it is the plugin saying it
    // never read `vlenb`. Reporting it as "actually ran: 0 bits" sends the reader
    // after a vector-configuration problem when the run itself is what failed
    // (QEMU exiting before any vCPU init, e.g. a dynamically-linked guest whose
    // interpreter is not under `-L`) or when this QEMU build has no vector
    // support at all. Neither is fixed by editing a VLEN.
    if recorded_bits == 0 {
        log::warn!(
            "vlen: {} records no vlenb (0), so the configured {configured_bits} bits cannot \
             be checked against it: either QEMU never reached a vCPU init (the guest did not \
             start) or this QEMU has no vector support",
            turbo_tracer_shared::TURBO_METADATA_FILE,
        );
        return;
    }
    // One call, not several: another thread's line landing in the middle of a
    // banner is how a loud warning becomes an unreadable one.
    log::error!(
        "\n\
         ========================= VLEN MISMATCH =========================\n\
         configured : {configured_bits} bits (from {source})\n\
         actually ran: {recorded_bits} bits (QEMU's vlenb CSR, recorded in {})\n\
         VLEN sizes every vector op, so mixing the two makes every vector\n\
         element count, byte count and cache footprint wrong.\n\
         Continuing at {recorded_bits} bits -- what the guest actually ran.\n\
         Fix the configured VLEN, or re-record at the configured one.\n\
         =================================================================",
        turbo_tracer_shared::TURBO_METADATA_FILE,
    );
}
