# Openchip TURBO: Tuning Utilities for RISC-V Benchmarking and Optimization

TURBO profiles RISC-V binaries by running them under QEMU user-mode emulation
and reporting what the program actually executed. Metrics reported include instruction
counts and instruction mix, per-function and per-region hotspots, and a histogram
of `VL` values for all RISC-V Vector instructions.

## Quickstart

These commands take a fresh Ubuntu 24.04 (x86-64) machine from nothing to a
profiled run. Run them in order from one shell. Each step is explained in
[Prerequisites](#prerequisites), [Building](#building) and
[Configuration](#configuration).

```bash
# 1. System packages: build tools, protoc, lld, the RISC-V guest sysroot,
#    and what QEMU needs to build.
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    build-essential ca-certificates curl git patch pkg-config \
    lld protobuf-compiler libc6-riscv64-cross \
    python3 python3-venv ninja-build flex bison xz-utils \
    libglib2.0-dev zlib1g-dev

# 2. Rust 1.89 or newer, from rustup. The distribution's rustc is too old.
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
. "$HOME/.cargo/env"

# 3. QEMU 11.0.0 with plugin support. No distribution packages it, and
#    11.1 or later will not load the tracer plugin as the API/ABI changed
curl -fsSL https://download.qemu.org/qemu-11.0.0.tar.xz | tar -xJ
cd qemu-11.0.0
./configure --target-list=riscv64-linux-user --enable-plugins \
    --without-default-features --disable-system --disable-tools --disable-docs
ninja -C build qemu-riscv64
sudo install -m 755 build/qemu-riscv64 /usr/local/bin/qemu-riscv64
cd ..

# 4. TURBO itself.
git clone https://github.com/openchip-software/turbo
cd turbo
./third-party/initialize_dependency.sh
cargo build --release

# 5. Write a configuration for this machine, then verify it end to end.
./target/release/turbo config init
./target/release/turbo config check

# 6. Profile a program: a test binary from the repository here. This serves
#    the web UI on http://127.0.0.1:9829 when done; Ctrl-C stops it.
./target/release/turbo run test_bins/basic_branch_asm/add_1k.riscv
```

`turbo config check` logs one line per check and should end with
`All checks passed.`. If it does not, the `ERROR` line names what to fix.

## Prerequisites

| Requirement | Version | Why | Check with |
| --- | --- | --- | --- |
| Rust toolchain, via [rustup](https://rustup.rs) | 1.89 or newer | Minimum supported Rust version (`rust-version` in `Cargo.toml`). Distribution packages are usually older (Ubuntu 24.04 ships 1.75). | `rustc --version` |
| `protoc` (`protobuf-compiler`) | any recent | Compiles the Perfetto protobufs at build time | `protoc --version` |
| `lld` | any recent | `.cargo/config.toml` links with `-fuse-ld=lld` | `ld.lld --version` |
| C toolchain, `git`, `curl`, `patch` | — | Building native dependencies, and `third-party/initialize_dependency.sh` | `cc --version` |
| QEMU user-mode `qemu-riscv64`, built with `--enable-plugins` | **11.0.x exactly** | Runs the guest and loads the tracer plugin. The plugin uses QEMU plugin API v6; QEMU 11.1 and later only accept v7. | `qemu-riscv64 --version` |
| riscv64 glibc sysroot (`libc6-riscv64-cross`) | — | Dynamically linked guests need its dynamic linker; statically linked ones do not | `ls /usr/riscv64-linux-gnu/lib/ld-linux-riscv64-lp64d.so.1` |
| x86-64 host with AVX2 | x86-64-v3 | `.cargo/config.toml` builds with `-Ctarget-cpu=x86-64-v3` | `grep -m1 -o avx2 /proc/cpuinfo` |

Once TURBO is built, `turbo config check` checks all the runtime requirements
in one go: QEMU is found and launches, its version, plugin support, the tracer
plugin, and the sysroot. It then does a real traced test run.

### Building QEMU 11.0.0

QEMU 11.0 is built from the release tarball. Only the riscv64 user-mode emulator is needed:

```bash
curl -fsSL https://download.qemu.org/qemu-11.0.0.tar.xz | tar -xJ
cd qemu-11.0.0
./configure --target-list=riscv64-linux-user --enable-plugins \
    --without-default-features --disable-system --disable-tools --disable-docs
ninja -C build qemu-riscv64
```

The build needs `python3`, `python3-venv`, `ninja-build`, `flex`, `bison`,
`pkg-config`, `libglib2.0-dev` and `zlib1g-dev`. Put the resulting
`build/qemu-riscv64` on your `PATH` (e.g. `/usr/local/bin`), or leave it where
it is and point `qemu_bin` at it (see [Configuration](#configuration)).

## Building

`third-party/riscv-isa` must exist before any `cargo` command, because
`Cargo.toml` patches the `riscv-isa` crate to that path:

```bash
./third-party/initialize_dependency.sh   # run once after cloning
```

Then:

```bash
cargo build --release                       # turbo CLI + tracer plugin
cargo build --release -p turbo-tracer       # plugin only -> target/release/libturbo_tracer.so
```

The CLI binary is built as `target/release/turbo`, the same name the
distributed packages install it under. The tracer plugin is
`target/release/libturbo_tracer.so`, and a source-built `turbo` uses that one by
default.

The tests load the **release** plugin, so rebuild it in release after any
plugin change.

### Running the tests

```bash
cargo build --release    # the tests load target/release/libturbo_tracer.so
cargo test --release
```

Some tests need more than the Quickstart installs. Without it they skip, and print why
with `SKIP:` (libtest still reports them as `ok`; add `-- --nocapture` to see the messages).
Set `TURBO_TEST_NO_SKIP=1` to make every skip a failure instead.

## Configuration

TURBO reads two JSON files:

* **`environment_config.json`**: where the tools on *this machine* are
  (QEMU, the tracer plugin, the guest sysroot). Only `record` and `run` use
  it, since they are the commands that launch QEMU.
* **`cpu_config.json`**: what the *modelled CPU* is, e.g. its `vlen`.

Each file is looked up in `~/.config/openchip/` first, then in
`/usr/share/turbo/`. A file passed with `--env-config <path>` or
`--cpu-config <path>` is used instead.

Neither file is required. Without them, TURBO uses built-in defaults, and a
`run` prints which paths those are and whether each one exists.

### Setting up a source build

There is no install step for a source build. `/usr/share/turbo/` is where the
distributed packages put their files, so nothing is there in a checkout. Pick
one of:

1. **Nothing.** If `qemu-riscv64` 11.0.x is on your `PATH` and you built with
   `cargo build --release`, the built-in defaults already point at the right
   QEMU and at `target/release/libturbo_tracer.so`.
2. **`turbo config init`** writes both files to `~/.config/openchip/`. The
   QEMU path is `qemu-riscv64` as found on your `PATH`, and the plugin path is
   the one from this build. It prints a note for anything it could not find,
   and never overwrites an existing file unless you pass `--force`
   (`--dir <DIR>` writes somewhere else).
3. **Write the files yourself**, using the examples below.

Then run `turbo config check`. It resolves the configuration exactly as `run`
does and exits non-zero if anything is wrong.

The files in `crates/turbo/src/` (`environment_config.json`,
`cpu_config.json`) are the ones the **packages** install. Their paths
(`/usr/local/bin/qemu-riscv64`, `/usr/lib/libturbo_tracer.so`) belong to a
packaged install, so do not copy them into a source build as they are.

### `environment_config.json`

Example for a source build with QEMU built in `~/qemu-11.0.0` and TURBO cloned
to `~/turbo`. JSON has no `~`, so paths must be spelled out in full:

```json
{
  "qemu_bin": "/home/me/qemu-11.0.0/build/qemu-riscv64",
  "turbo_tracer_plugin": "/home/me/turbo/target/release/libturbo_tracer.so",
  "loader_path": "/usr/riscv64-linux-gnu/"
}
```

Every field is optional. Unknown fields are ignored.

| Field | Meaning | Default when omitted |
| --- | --- | --- |
| `qemu_bin` | The QEMU 11.0.x `qemu-riscv64` to run. A bare name is looked up on `PATH`; anything with a `/` in it is a path. | `qemu-riscv64`, from `PATH` |
| `turbo_tracer_plugin` | The tracer plugin QEMU loads (`libturbo_tracer.so`) | `target/release/libturbo_tracer.so` in the checkout `turbo` was built from |
| `loader_path` | Guest sysroot, passed to QEMU as `-L`, in which dynamically linked guests find their dynamic linker and libraries | `/usr/riscv64-linux-gnu/` |
| `default_cpu_config` | The `cpu_config.json` to use when `--cpu-config` is not given, e.g. one file per modelled core | `cpu_config.json` from the default location |

Use absolute paths throughout: a relative path is resolved against the
directory you run `turbo` from, not against the config file.

### `cpu_config.json`

```json
{
  "vlen": 1024
}
```

| Field | Meaning | Default when omitted |
| --- | --- | --- |
| `vlen` | Vector register length in bits: a power of two, from 128 up to the most your QEMU supports. QEMU 11.0.0 as built above stops at **1024**; the RVV format itself allows up to 65536. `--vlen` overrides it. | 1024 |

`turbo config check` asks QEMU which VLENs it accepts and fails if `vlen` is
outside that range, e.g. `QEMU VLEN support: 128 to 1024 bits (configured: 1024)`.


## Usage

```bash
# check the configuration, and write a starter one
turbo config check
turbo config init

# record, decode and serve the UI, in one go
turbo run ./my_bin -- arg1 arg2

# the same three steps, separately
turbo record  ./my_bin -- arg1 arg2      # trace to disk, no decoding
turbo process ./my_bin output/<run>      # decode + write reports, no server
turbo load    output/<run>               # serve an already-processed directory

# what is in a recorded trace: sizes, compression, hottest blocks
turbo inspect output/<run>
```

Useful flags on `run` / `process`:

| Flag | Effect |
| --- | --- |
| `--summary[=program\|function\|regions]` | Print a summary table after processing |
| `--src-dir <path>` | Where to find sources for the annotated report |
| `--no-server` | Finish without starting the web UI (see [Scripting](#scripting)) |
| `--vcpu <N>` | Restrict processing to specific harts |
| `--decode-threads <N>` | Cap decode/enrich workers (1..=8) |
| `--vlen <bits>` | Vector register length to emulate, overriding `cpu_config.json` (at most 1024 on QEMU 11.0.0) |

A recording can be re-processed as often as you like — with different ROI
or annotation settings — without re-running the guest. VLEN is the
exception: it is stamped into the recording by the plugin and cannot be changed
after the fact, because it is what the recorded instructions *meant*.

### Scripting

By default `turbo run` ends by serving the web UI, and it does not return until
you press Ctrl-C. In scripts, CI jobs and batch runs, pass `--no-server`. The
command then exits as soon as the reports are written, with exit status 0 on
success and non-zero on failure:

```bash
for n in 64 128 256; do
    turbo run --no-server --summary --output-dir "sweep_$n" ./my_bin -- "$n" || exit 1
done
turbo load output/sweep_128      # browse any one of them afterwards
```

`turbo record` and `turbo process` never start the server, so they need no
flag. `turbo config check` also exits non-zero when a check fails, so a
setup script can run it first.

The web UI binds to `127.0.0.1` only. To view it from another machine, forward
the port (e.g. `ssh -L 9829:127.0.0.1:9829 <host>`).

## Data Files Produced by TURBO

Every run writes, in its output directory:

* `perf_data_final.json` — the full aggregated result (instruction mix, per
  function/region counters, vector stats).
* `turbo_vcpu_<N>_perf.json` — the same result for each hart on its own.
* `perf_data_schema.json` — the JSON schema of the two files above.
* `html/annotate_*.html.gz` — source and assembly annotated with
  per-instruction counts, gzip-compressed (`annotate_index.html.gz` is the
  entry point; the web UI serves them directly).
* `turbo_trace_vcpu<N>.bin`, `turbo_metadata.json`, `vdso_mem.dump` — the recorded trace, which
  `turbo process` can decode again.
* `qemu_stdout.txt`, `qemu_stderr.txt` — the guest's console output, and QEMU's.

Only when the program executes ROI markers:

* `roi_flamegraph.html` / `.folded` / `.json` — flamegraph over the ROI call tree.
* `roi_flamegraph_hart<N>.folded` / `.json` and `roi_flamegraph_harts.json` —
  one flamegraph per hart, plus their index, when more than one hart ran ROIs.

`turbo run` then serves an interactive web UI over these files on port 9829
(`--no-server` skips it, for scripting).

ROI markers let counters be scoped to a named region of the guest program
instead of the whole run; they are plain R-type instructions writing to `x0`, so
an uninstrumented QEMU or real hardware just ignores them.


## How it works

No hardware is required, and the guest binary needs no recompilation for basic
profiling. To enable the ROI markers, recompilation might be required.

```
guest binary
    |
    v
qemu-riscv64 + turbo-tracer plugin   ->  turbo_trace_vcpu<N>.bin  (per-hart trace)
    |                                    turbo_metadata.json
    v
turbo (decode + enrich)              ->  reports on disk
    |
    v
web UI on http://127.0.0.1:9829
```

`turbo-tracer` is a QEMU TCG plugin that tracks execution in a novel trace format
called Basic-Block-Trace (BBT). BBT records execution metadata like the `vl` produced
by every `vsetvl*` or `vl*ff`, region-of-interest (ROI) markers executed. This metadata
allows turning basic execution traces into insight rich information for turbo to display!

`turbo` decodes the BBT trace files, resolves symbols and DWARF line info against the
binary, and aggregates the results. Decoding runs *while* QEMU is still
executing by tailing the trace as it is written, so results appear live in the
browser as the run progresses. Cross referencing the BBT metadata with the execution trace
allows showing `vl` histograms per Region Of Interest, instruction mixes, bytes moved per
function, and lots more.

In short, TURBO measures the interesting stuff at runtime, turns that data into insight,
and display is in real time for you to understand and optimize your code with!

## Licensing

The repository is REUSE-compliant; refer to `REUSE.toml` and `LICENSES/` for details.
