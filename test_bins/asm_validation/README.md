# RVV Assembly Validation Tests

This directory contains hand-crafted RISC-V Vector (RVV) assembly files designed to validate the accuracy of the profiling tool's metric collection for HPC workloads.

## Building Tests

From this directory:

```bash
make        # Build all tests
make clean  # Remove the bins/ directory
make help   # Show available targets
```

Binaries are output to `./bins/`, one `.riscv` per source: `bins/vfmacc_simple.riscv`.

## Running Tests

After building, run the Rust integration tests:

```bash
cargo test --test integration asm_validation::
```

Individual tests can be run:

```bash
cargo test --test integration asm_validation::vfmacc_simple
cargo test --test integration asm_validation::stream_triad
```

## The VLEN sweep

`vector_arithmetic/vadd_vlmax.S` is not part of the fixture suite above. It is
`vadd_simple.S` with `vsetvli` taking its AVL from `x0`, so `vl` is VLMAX and the
kernel's element counts are a function of VLEN rather than of the source. It is
driven by `crates/turbo/tests/vlen_sweep.rs`, which runs it once per power-of-two
VLEN -- all ten of them, 128 bits up to the 65536-bit RVV maximum -- and computes
the expected counters from the width instead of reading them from
`expected_metrics.json`, which can only state one VLEN's worth of them.

```bash
cargo test -p turbo --test integration vlen_sweep::
```

## Adding Tests

1. Create a test in this directory 
2. Add a `$(OUTPUT_DIR)/<name>.riscv` target to the Makefile
3. Add ground truth to expected_metrics.json 
4. Add the name to the `turbo_tests_named!` table in
   `crates/turbo/tests/asm_validation.rs`
