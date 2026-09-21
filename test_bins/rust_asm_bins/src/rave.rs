//! RAVE user event helpers for RISC-V.
//!
//! These mirror the C macros in `rave_user_events_v2.h`, emitting the
//! special instruction encodings that the RAVE tracing infrastructure
//! recognises.

use core::arch::asm;

/// `rave_begin_region(name)` -- mark the start of a named region.
/// The name must be a byte-string literal or slice that lives long enough.
#[inline(always)]
pub fn begin_region(name: &[u8]) {
    let ptr = name.as_ptr();
    let len: isize = -1;
    unsafe {
        asm!(
            "add x0, {ptr}, {len}",
            ptr = in(reg) ptr,
            len = in(reg) len,
            options(nomem, nostack, preserves_flags),
        );
    }
}

/// `rave_end_region(name)` -- mark the end of a named region.
/// The name **must** be the same string passed to `begin_region`.
#[inline(always)]
pub fn end_region(name: &[u8]) {
    let ptr = name.as_ptr();
    let len: isize = -1;
    unsafe {
        asm!(
            "sub x0, {ptr}, {len}",
            ptr = in(reg) ptr,
            len = in(reg) len,
            options(nomem, nostack, preserves_flags),
        );
    }
}

/// `rave_begin_region_len(name, len)` -- start region with explicit length.
#[inline(always)]
pub fn begin_region_len(name: &[u8], length: usize) {
    let ptr = name.as_ptr();
    unsafe {
        asm!(
            "add x0, {ptr}, {len}",
            ptr = in(reg) ptr,
            len = in(reg) length,
            options(nomem, nostack, preserves_flags),
        );
    }
}

/// `rave_end_region_len(name, len)` -- end region with explicit length.
#[inline(always)]
pub fn end_region_len(name: &[u8], length: usize) {
    let ptr = name.as_ptr();
    unsafe {
        asm!(
            "sub x0, {ptr}, {len}",
            ptr = in(reg) ptr,
            len = in(reg) length,
            options(nomem, nostack, preserves_flags),
        );
    }
}

/// `rave_enable_trace()`
#[inline(always)]
pub fn enable_trace() {
    unsafe {
        asm!("li x0, -3", options(nomem, nostack, preserves_flags));
    }
}

/// `rave_disable_trace()`
#[inline(always)]
pub fn disable_trace() {
    unsafe {
        asm!("li x0, -4", options(nomem, nostack, preserves_flags));
    }
}

/// `rave_enable_regions()`
#[inline(always)]
pub fn enable_regions() {
    unsafe {
        asm!("li x0, -7", options(nomem, nostack, preserves_flags));
    }
}

/// `rave_disable_regions()`
#[inline(always)]
pub fn disable_regions() {
    unsafe {
        asm!("li x0, -8", options(nomem, nostack, preserves_flags));
    }
}

/// `rave_enable()` -- enable both regions and tracing.
#[inline(always)]
pub fn enable() {
    enable_regions();
    enable_trace();
}

/// `rave_disable()` -- disable both regions and tracing.
#[inline(always)]
pub fn disable() {
    disable_regions();
    disable_trace();
}

/// `rave_restart_trace()`
#[inline(always)]
pub fn restart_trace() {
    unsafe {
        asm!("li x0, -2", options(nomem, nostack, preserves_flags));
    }
}

/// `rave_name_event(x, name)` -- associate an event id with a name.
#[inline(always)]
pub fn name_event(x: usize, name: &[u8]) {
    let ptr = name.as_ptr();
    let len: isize = -1;
    unsafe {
        asm!(
            "and x0, {x}, x0",
            "sll x0, {ptr}, {len}",
            x = in(reg) x,
            ptr = in(reg) ptr,
            len = in(reg) len,
            options(nomem, nostack, preserves_flags),
        );
    }
}

/// `rave_name_event_len(x, name, len)`
#[inline(always)]
pub fn name_event_len(x: usize, name: &[u8], length: usize) {
    let ptr = name.as_ptr();
    unsafe {
        asm!(
            "and x0, {x}, x0",
            "sll x0, {ptr}, {len}",
            x = in(reg) x,
            ptr = in(reg) ptr,
            len = in(reg) length,
            options(nomem, nostack, preserves_flags),
        );
    }
}

/// `rave_name_value(x, y, name)` -- associate an event+value pair with a name.
#[inline(always)]
pub fn name_value(x: usize, y: usize, name: &[u8]) {
    let ptr = name.as_ptr();
    let len: isize = -1;
    unsafe {
        asm!(
            "and x0, {x}, {y}",
            "srl x0, {ptr}, {len}",
            x = in(reg) x,
            y = in(reg) y,
            ptr = in(reg) ptr,
            len = in(reg) len,
            options(nomem, nostack, preserves_flags),
        );
    }
}

/// `rave_name_value_len(x, y, name, len)`
#[inline(always)]
pub fn name_value_len(x: usize, y: usize, name: &[u8], length: usize) {
    let ptr = name.as_ptr();
    unsafe {
        asm!(
            "and x0, {x}, {y}",
            "srl x0, {ptr}, {len}",
            x = in(reg) x,
            y = in(reg) y,
            ptr = in(reg) ptr,
            len = in(reg) length,
            options(nomem, nostack, preserves_flags),
        );
    }
}

/// `rave_event_and_value(x, y)`
#[inline(always)]
pub fn event_and_value(x: usize, y: usize) {
    unsafe {
        asm!(
            "or x0, {x}, {y}",
            x = in(reg) x,
            y = in(reg) y,
            options(nomem, nostack, preserves_flags),
        );
    }
}

/// RAII guard that calls `end_region` on drop.
pub struct Region {
    name: &'static [u8],
}

impl Region {
    /// Begin a RAVE region and return a guard that ends it on drop.
    /// The name should be a byte string like `b"my_region"` (no null terminator needed).
    #[inline(always)]
    pub fn new(name: &'static [u8]) -> Self {
        begin_region_len(name, name.len());
        Self { name }
    }
}

impl Drop for Region {
    #[inline(always)]
    fn drop(&mut self) {
        end_region_len(self.name, self.name.len());
    }
}
