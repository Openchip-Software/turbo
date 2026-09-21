Turbo Tracer Advanced User Notes
================================

`turbo-tracer` is the QEMU TCG plugin at the front of the turbo pipeline. It is
built as a `cdylib` (`libturbo_tracer.so`), loaded into `qemu-riscv64` with
`-plugin`, and its job is to record *what the guest executed* as cheaply as
possible, leaving all interpretation to the consumer (`turbo`).


What it does
------------

The plugin instruments every translated block once, at translate time, and then
does almost nothing per execution:

* **Basic-block trace (BBT).** Each block gets a dictionary entry (id, start PC,
  per-instruction sizes) the first time it is translated, and its execution
  callback appends a single precomputed 32-bit word. That word — one store and
  two counter bumps — is the entire per-block cost of tracing, and it is why the
  format exists. The dictionary is shared by all harts and interned once, so a
  block translated while one hart was running can be executed by another.
* **Vector `vl` capture.** `vsetvl*` instructions get an extra record carrying
  the resulting `vl`, so the consumer can attribute real vector work rather than
  guessing element counts. `vl` is *computed* from the instruction operands per
  the RISC-V V spec, and cross-checked against the register QEMU actually wrote
  (see `vl_verify=` below).
* **ROI markers.** Region-of-interest instrumentation in the guest is picked up
  and shipped as a per-batch side channel, which is what lets the consumer scope
  counters to a named region instead of the whole run.
* **Run metadata.** `turbo_metadata.json` (main binary path, code range, entry
  point, shared-object mappings, `run_id`) and a `vdso_mem.dump` taken straight
  out of guest memory, since the consumer has no guest-memory access but still
  needs the vDSO for symbol resolution.

Everything is written as an append-only, per-hart record log that a consumer can
tail *while the run is in flight*, which is what the live path depends on.

### Files it writes into `workdir`

| File | Contents |
| --- | --- |
| `turbo_trace_vcpu<N>.bin` | Per-hart BBT record log (deflate-compressed batches) |
| `turbo_metadata.json` | Binary path, code range, entry point, lib mappings, `run_id` |
| `vdso_mem.dump` | Raw vDSO ELF image dumped from guest memory |
| `turbo_trace_debug.log` | Encoder diagnostics — only when `debug=true` (see below) |


Turning features on and off
---------------------------

All configuration arrives as comma-separated `key=value` pairs appended to the
`-plugin` argument, parsed in `ControlFlow::register` (`src/lib.rs`). The plugin
prints its resolved configuration to stdout at init, so the first few lines of a
run tell you exactly what is enabled:

```
turbo_tracer plugin configuration:
  workdir: /tmp/turbo-output
  flush_interval: 10000
  compress: deflate level 6
  debug: false
  run_id: 6f1c9a0b2d3e4f50
  vl_verify: Abort
```

A full manual invocation, with everything spelled out:

```bash
qemu-riscv64 \
  -L "$SYSROOT" \
  -plugin /path/to/libturbo_tracer.so,workdir=/tmp/turbo-output,run_id=manual-1,flush_interval=10000,compress=6,vl_verify=warn,debug=true \
  ./my_bin
```

| Argument | Default | Effect |
| --- | --- | --- |
| `workdir=<dir>` | `.` | Where every output file above is written; created if missing |
| `flush_interval=<n>` | `10000` | Block records per flush window (one batch on disk per window). Smaller = lower live-path latency, more envelope overhead |
| `compress=<0..9>` | `6` | Deflate level for record payloads. `0` disables compression — note level 1 is usually *faster end to end* than 0, because the writer does less I/O |
| `debug=true` | `false` | Routes encoder diagnostics to `workdir/turbo_trace_debug.log` (see below) |
| `vl_verify=abort\|warn\|off` | `abort` | The computed-`vl` oracle: compare the computed `vl` against the value QEMU wrote |
| `run_id=<string>` | *(empty)* | Opaque per-run id stamped into every metadata write, so a consumer can tell this run's files from a stale leftover in the same `workdir` |

Booleans go through QEMU's `qemu_plugin_bool_parse`, so `debug=true`, `debug=on`
and `debug=yes` are all equivalent.

### What `turbo run` passes for you

`turbo` builds the plugin argument in
`crates/turbo/src/source/qemu/qemu_runner.rs` and currently sets `workdir=`,
`run_id=` and `vl_verify=warn` — nothing else. So `flush_interval`,
`compress` and `debug` are **not** reachable from the `turbo` CLI:

* `compress` has an environment fallback for exactly this reason. QEMU inherits
  the launching process's environment, so `TURBO_TRACE_COMPRESS=9 turbo run ...`
  reaches the plugin. The plugin argument wins when both are set.
* `flush_interval` and `debug` have no fallback. To exercise them, either run
  `qemu-riscv64` by hand as above, or add the argument at
  `qemu_runner.rs:249` alongside the existing `vl_verify=warn` push.


Debug logging
-------------

There are two independent output channels, and knowing which one you are looking
at saves a lot of confusion.

**1. The `log` crate / `RUST_LOG`.** The plugin is a `.so` that QEMU loads, so
nothing else would ever initialise a logger for it — `register` calls
`env_logger::try_init()` itself. Without that, every `log::` call in the plugin
is a silent no-op. With it, `RUST_LOG` works as usual:

```bash
RUST_LOG=turbo_tracer=debug turbo run ...    # plugin diagnostics to stderr
RUST_LOG=turbo_tracer=trace turbo run ...    # + per-marker / per-instruction detail
```

This covers CPU-property audits, register-handle discovery, `vl` divergence
warnings, metadata capture and file-creation errors.

**2. `debug=true` → `workdir/turbo_trace_debug.log`.** The trace encoder
(`src/trace_encoder.rs`) emits per-batch accounting that is far too noisy for a
terminal, so it gets its own file. When `debug=true`, *every* diagnostic the
encoder produces — the per-batch size breakdown, the end-of-run write summary,
ROI marker traces, and record-log append/flush/finish errors — is written there
instead of going through the `log` crate. That means `RUST_LOG` does **not**
gate these lines; the file is the switch. Lines are written as single unbuffered
appends, so concurrent harts share the file without interleaving mid-message,
and the first encoder in the process truncates any log left over from a previous
run.

```
[DEBUG] TraceEncoder[hart=0]: BBT batch #1: 40960 B on disk (4.2x) from 172032 B uncompressed = 163840 B block records (40960) + 512 B vl records (128) + 7668 B dict (219 descriptors) + 12 B chunk header + 40 B envelope (0 roi markers)
[TRACE] ROI marker recorded: RegionStart at PC 0x10a34 rs1=0x1 rs2=0x0
[DEBUG] TraceEncoder[hart=0]: wrote 1201312 B on disk in 29 batches, 5013504 B uncompressed (4.2x compression):
[DEBUG] TraceEncoder[hart=0]:   bbt block executions        4759552 B ( 94.9%) over    1189888 item(s)
```

With `debug=false` (the default) the per-batch line is suppressed entirely rather
than downgraded to `log::debug!` — that spam is opt-in by design. The end-of-run
summary, marker traces and encoder errors still go to the `log` crate, so
`RUST_LOG=turbo_tracer=debug` gets you the summary without the per-batch flood.

Every byte in the summary falls into exactly one bucket and the buckets sum to
the file size, so it is a real audit rather than an estimate. `turbo inspect`
recomputes the same numbers from the file on disk — if the two disagree, one of
them is wrong:

```bash
turbo inspect /tmp/turbo-output            # every hart in the spool directory
turbo inspect /tmp/turbo-output --vcpu 0   # just hart 0
```


The `vl` oracle
---------------

`vl_verify` is the gate that makes *computing* `vl` (instead of reading it back)
safe. It is near-free — one extra register read at sites that already carry
`R_REGS` — so it is on by default:

* `abort` (default): panic on the first divergence between computed and actual
  `vl`. This is what you want for correctness work.
* `warn`: log and continue. This is what `turbo run` passes, because
  fault-only-first vector loads change `vl` without a `vsetvl*` and the oracle
  cannot yet model them; aborting would break otherwise-valid runs.
* `off` / `false`: no checking at all.

Related, and not configurable: at init the plugin reads `/proc/self/cmdline` and
**hard-fails** if QEMU was started with a `-cpu` property that changes how `vl`
is derived (`rvv_vl_half_avl`, `rvv_vsetvl_x0_vill`). The plugin API cannot read
CPU properties, so the command line is the only place to see them, and every
recorded `vl` would be silently wrong. Explicitly disabling the property
(`rvv_vl_half_avl=false`) is fine.

If `vl` capture is unavailable — usually a plugin-API version mismatch, so the
register handles are not found — the plugin logs an error naming the missing
handles, and the consumer will report `Ran out of VL values`. That is the symptom
to look for.


Building
--------

```bash
cargo build --release -p turbo-tracer     # -> target/release/libturbo_tracer.so
```

Note that the tests load the **release** `.so`, so a plugin change needs a
release rebuild before testing. The `plugin-api-v2` … `plugin-api-v6` cargo
features select which QEMU plugin API the crate is compiled against
(`plugin-api-v6` by default); it must match the QEMU binary you load it into, or
API-dependent features such as `vl` capture will silently fail to initialise.
The crate has `test = false` / `doctest = false` — it is a C dynamic library for
QEMU, not a Rust binary, so building tests for it just fails on missing symbols.
