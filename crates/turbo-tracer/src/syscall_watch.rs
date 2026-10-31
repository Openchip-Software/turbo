//! Shared-object discovery from the guest's `openat`/`mmap`/`close` syscalls.
//!
//! These callbacks are registered with QEMU directly rather than through the
//! plugin crate's `HasCallbacks`, because the crate runs every callback under
//! one global plugin mutex, and these read guest memory: the `openat` path and
//! the ELF magic of an `mmap`ed segment. In user mode that read takes QEMU's
//! `mmap_lock`, which a vCPU translating a block already holds while it waits
//! for the plugin mutex to run the translate callback -- an AB-BA deadlock
//! between any two guest threads, one in `openat` while another translates.
//!
//! So here no lock is held while guest memory is read: the read happens first,
//! and only then is [`SyscallWatch::state`] taken to record what it found.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use qemu_plugin::{PluginId, VCPUIndex};

use super::{PerVCPUData, VCPUArray};

// RISC-V (asm-generic) syscall numbers. RISC-V has no `open`/`mmap2`; the
// loader uses `openat` + `mmap`.
const SYS_OPENAT: i64 = 56;
const SYS_CLOSE: i64 = 57;
const SYS_MMAP: i64 = 222;

pub(crate) struct SyscallWatch {
    per_vcpu_data: Arc<VCPUArray<PerVCPUData>>,
    state: Mutex<SyscallState>,
    /// Set when `libs` gained or moved an entry; see
    /// [`SyscallWatch::libs_changed`].
    libs_changed: AtomicBool,
}

#[derive(Default)]
struct SyscallState {
    /// Open file descriptors observed via `openat`: fd -> guest path. Cleared
    /// on `close` so a later reuse of the same fd number is attributed
    /// correctly.
    open_fds: HashMap<i64, String>,
    /// Per-vCPU `openat` path captured at syscall entry, resolved to its fd on
    /// syscall return.
    pending_openat: HashMap<VCPUIndex, String>,
    /// Per-vCPU `(fd, offset)` captured at `mmap` entry, resolved to a base
    /// address on syscall return.
    pending_mmap: HashMap<VCPUIndex, (i64, u64)>,
    /// Discovered shared objects: guest path -> runtime load base.
    libs: HashMap<String, u64>,
}

static WATCH: OnceLock<SyscallWatch> = OnceLock::new();

impl SyscallWatch {
    /// Register the syscall callbacks. Call once, from plugin registration.
    pub(crate) fn install(id: PluginId, per_vcpu_data: Arc<VCPUArray<PerVCPUData>>) {
        let watch = SyscallWatch {
            per_vcpu_data,
            state: Mutex::new(SyscallState::default()),
            libs_changed: AtomicBool::new(false),
        };
        if WATCH.set(watch).is_err() {
            return;
        }
        // SAFETY: plain registration of `extern "C"` functions with QEMU.
        unsafe {
            qemu_plugin::sys::qemu_plugin_register_vcpu_syscall_cb(id, Some(on_syscall));
            qemu_plugin::sys::qemu_plugin_register_vcpu_syscall_ret_cb(id, Some(on_syscall_ret));
        }
    }

    /// Whether a shared object was added or moved since the last call.
    pub(crate) fn libs_changed() -> bool {
        WATCH
            .get()
            .is_some_and(|w| w.libs_changed.swap(false, Ordering::Relaxed))
    }

    /// The shared objects discovered so far: guest path -> load base.
    pub(crate) fn libs() -> Vec<(String, u64)> {
        WATCH
            .get()
            .and_then(|w| w.state.lock().ok())
            .map(|s| s.libs.iter().map(|(p, &b)| (p.clone(), b)).collect())
            .unwrap_or_default()
    }

    /// Read guest memory on `vcpu`. No lock may be held by the caller: see the
    /// module docs.
    fn read(&self, vcpu: VCPUIndex, addr: u64, len: u64) -> Option<Vec<u8>> {
        // SAFETY: syscall callbacks run on this vCPU's own thread.
        unsafe { self.per_vcpu_data.get_mut(vcpu) }?.read_guest_bytes(addr, len)
    }

    /// Read a NUL-terminated guest C string. The readable length is unknown,
    /// so try progressively smaller reads (a long path near the end of a
    /// mapping would fail an over-long all-or-nothing read).
    fn read_cstr(&self, vcpu: VCPUIndex, ptr: u64) -> Option<String> {
        if ptr == 0 {
            return None;
        }
        [256u64, 128, 64, 32].iter().find_map(|&len| {
            let bytes = self.read(vcpu, ptr, len)?;
            let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
            Some(String::from_utf8_lossy(&bytes[..end]).to_string())
        })
    }
}

/// Syscall entry: stash what the outcome will need.
extern "C" fn on_syscall(
    _id: PluginId,
    vcpu: VCPUIndex,
    num: i64,
    a1: u64,
    a2: u64,
    _a3: u64,
    _a4: u64,
    a5: u64,
    a6: u64,
    _a7: u64,
    _a8: u64,
) {
    let Some(watch) = WATCH.get() else { return };
    match num {
        SYS_OPENAT => {
            // a2 = const char *pathname (a1 = dirfd). Read before locking.
            if let Some(path) = watch.read_cstr(vcpu, a2) {
                if let Ok(mut s) = watch.state.lock() {
                    s.pending_openat.insert(vcpu, path);
                }
            }
        }
        SYS_MMAP => {
            // a5 = fd, a6 = offset.
            if let Ok(mut s) = watch.state.lock() {
                s.pending_mmap.insert(vcpu, (a5 as i64, a6));
            }
        }
        SYS_CLOSE => {
            if let Ok(mut s) = watch.state.lock() {
                s.open_fds.remove(&(a1 as i64));
            }
        }
        _ => {}
    }
}

/// Syscall return: map fds to paths on `openat`, and record a shared object's
/// load base when its file-offset-0 segment is `mmap`ed (validated by ELF
/// magic).
extern "C" fn on_syscall_ret(_id: PluginId, vcpu: VCPUIndex, num: i64, ret: i64) {
    let Some(watch) = WATCH.get() else { return };
    match num {
        SYS_OPENAT => {
            if let Ok(mut s) = watch.state.lock() {
                if let Some(path) = s.pending_openat.remove(&vcpu) {
                    if ret >= 0 {
                        s.open_fds.insert(ret, path);
                    }
                }
            }
        }
        SYS_MMAP => {
            let Some((fd, offset)) = watch
                .state
                .lock()
                .ok()
                .and_then(|mut s| s.pending_mmap.remove(&vcpu))
            else {
                return;
            };
            // mmap failure returns a small negative errno; a success is an
            // address. Only the file-offset-0 segment lands at the load base.
            if (-4095..0).contains(&ret) || offset != 0 || fd < 0 {
                return;
            }
            let base = ret as u64;
            // Read before locking (see the module docs).
            if watch.read(vcpu, base, 4).as_deref() != Some(b"\x7fELF") {
                return;
            }
            let Ok(mut s) = watch.state.lock() else {
                return;
            };
            let Some(path) = s.open_fds.get(&fd).cloned() else {
                return;
            };
            // A small object can map several segments at page-aligned file
            // offset 0 (each starting with the ELF header); the load base is
            // the lowest such address.
            if s.libs.get(&path).is_none_or(|&existing| base < existing) {
                log::debug!("libmap: {path} @ 0x{base:x}");
                s.libs.insert(path, base);
                watch.libs_changed.store(true, Ordering::Relaxed);
            }
        }
        _ => {}
    }
}
