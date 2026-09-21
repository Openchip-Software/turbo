# Makefile for RVV Assembly Validation Tests
# Builds hand-crafted assembly files to validate profiling metrics

# Compiler and flags
CC := clang
CFLAGS := -nostdlib -static

# All tests use vector extensions
ARCH := -march=rv64gv --target=riscv64-linux-gnu
# Zvbb (vector bit-manipulation) needs an explicit extension in -march
ARCH_ZVBB := -march=rv64gv_zvbb --target=riscv64-linux-gnu

# Output directory
OUTPUT_DIR := ./bins

# Source files organized by category
VECTOR_ARITH_SRCS := \
	vector_arithmetic/vfmacc_simple \
	vector_arithmetic/vadd_simple \
	vector_arithmetic/vadd_vlmax \
	vector_arithmetic/lmul_stress \
	vector_arithmetic/vl_variation \
	vector_arithmetic/vfmul_simple \
	vector_arithmetic/compute_intensive \
	vector_arithmetic/vzvbb_ops

MEMORY_SRCS := \
	memory/vload_store_simple \
	memory/full_reg_ops \
	memory/vseg_load_store \
	memory/vseg_strided_indexed \
	memory/vindexed_eew_mismatch \
	memory/vload_widths \
	memory/vseg_nf_sweep \
	memory/vindexed_widths \
	memory/vindexed_vl_dedup \
	memory/vleff_trim \
	memory/membench_loads \
	memory/membench_stores

MIXED_SRCS := \
	mixed/stream_triad \
	mixed/daxpy

# Guests that spawn a second guest thread, i.e. a second vCPU.
THREAD_SRCS := \
	threads/clone_vl_inherit

# All source files
ALL_SRCS := $(VECTOR_ARITH_SRCS) $(MEMORY_SRCS) $(MIXED_SRCS) $(THREAD_SRCS)

# Generate target names (just the basename for output)
ALL_TARGETS := $(foreach src,$(ALL_SRCS),$(OUTPUT_DIR)/$(notdir $(src)).riscv)

# Default target
all: $(OUTPUT_DIR) $(ALL_TARGETS)
	@echo "Built all assembly validation tests"
	@echo "Binaries in: $(OUTPUT_DIR)"

# Create output directory
$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)

# Pattern rule to build each test
# Match any source file and extract its path
$(OUTPUT_DIR)/vfmacc_simple.riscv: vector_arithmetic/vfmacc_simple.S | $(OUTPUT_DIR)
	$(CC) $(ARCH) $(CFLAGS) $< -o $@

$(OUTPUT_DIR)/vadd_simple.riscv: vector_arithmetic/vadd_simple.S | $(OUTPUT_DIR)
	$(CC) $(ARCH) $(CFLAGS) $< -o $@

$(OUTPUT_DIR)/vadd_vlmax.riscv: vector_arithmetic/vadd_vlmax.S | $(OUTPUT_DIR)
	$(CC) $(ARCH) $(CFLAGS) $< -o $@

$(OUTPUT_DIR)/vl_variation.riscv: vector_arithmetic/vl_variation.S | $(OUTPUT_DIR)
	$(CC) $(ARCH) $(CFLAGS) $< -o $@

$(OUTPUT_DIR)/vfmul_simple.riscv: vector_arithmetic/vfmul_simple.S | $(OUTPUT_DIR)
	$(CC) $(ARCH) $(CFLAGS) $< -o $@

$(OUTPUT_DIR)/lmul_stress.riscv: vector_arithmetic/lmul_stress.S | $(OUTPUT_DIR)
	$(CC) $(ARCH) $(CFLAGS) $< -o $@

$(OUTPUT_DIR)/compute_intensive.riscv: vector_arithmetic/compute_intensive.S | $(OUTPUT_DIR)
	$(CC) $(ARCH) $(CFLAGS) $< -o $@

$(OUTPUT_DIR)/vload_store_simple.riscv: memory/vload_store_simple.S | $(OUTPUT_DIR)
	$(CC) $(ARCH) $(CFLAGS) $< -o $@

$(OUTPUT_DIR)/stream_triad.riscv: mixed/stream_triad.S | $(OUTPUT_DIR)
	$(CC) $(ARCH) $(CFLAGS) $< -o $@

$(OUTPUT_DIR)/daxpy.riscv: mixed/daxpy.S | $(OUTPUT_DIR)
	$(CC) $(ARCH) $(CFLAGS) $< -o $@

$(OUTPUT_DIR)/full_reg_ops.riscv: memory/full_reg_ops.S | $(OUTPUT_DIR)
	$(CC) $(ARCH) $(CFLAGS) $< -o $@

$(OUTPUT_DIR)/vseg_load_store.riscv: memory/vseg_load_store.S | $(OUTPUT_DIR)
	$(CC) $(ARCH) $(CFLAGS) $< -o $@

$(OUTPUT_DIR)/vseg_strided_indexed.riscv: memory/vseg_strided_indexed.S | $(OUTPUT_DIR)
	$(CC) $(ARCH) $(CFLAGS) $< -o $@

$(OUTPUT_DIR)/vindexed_eew_mismatch.riscv: memory/vindexed_eew_mismatch.S | $(OUTPUT_DIR)
	$(CC) $(ARCH) $(CFLAGS) $< -o $@

$(OUTPUT_DIR)/vload_widths.riscv: memory/vload_widths.S | $(OUTPUT_DIR)
	$(CC) $(ARCH) $(CFLAGS) $< -o $@

$(OUTPUT_DIR)/vseg_nf_sweep.riscv: memory/vseg_nf_sweep.S | $(OUTPUT_DIR)
	$(CC) $(ARCH) $(CFLAGS) $< -o $@

$(OUTPUT_DIR)/vindexed_widths.riscv: memory/vindexed_widths.S | $(OUTPUT_DIR)
	$(CC) $(ARCH) $(CFLAGS) $< -o $@

$(OUTPUT_DIR)/vindexed_vl_dedup.riscv: memory/vindexed_vl_dedup.S | $(OUTPUT_DIR)
	$(CC) $(ARCH) $(CFLAGS) $< -o $@

$(OUTPUT_DIR)/vleff_trim.riscv: memory/vleff_trim.S | $(OUTPUT_DIR)
	$(CC) $(ARCH) $(CFLAGS) $< -o $@

$(OUTPUT_DIR)/clone_vl_inherit.riscv: threads/clone_vl_inherit.S | $(OUTPUT_DIR)
	$(CC) $(ARCH) $(CFLAGS) $< -o $@

$(OUTPUT_DIR)/membench_loads.riscv: memory/membench_loads.S | $(OUTPUT_DIR)
	$(CC) $(ARCH) $(CFLAGS) $< -o $@

$(OUTPUT_DIR)/membench_stores.riscv: memory/membench_stores.S | $(OUTPUT_DIR)
	$(CC) $(ARCH) $(CFLAGS) $< -o $@

$(OUTPUT_DIR)/vzvbb_ops.riscv: vector_arithmetic/vzvbb_ops.S | $(OUTPUT_DIR)
	$(CC) $(ARCH_ZVBB) $(CFLAGS) $< -o $@

# The whole $(OUTPUT_DIR) goes, not just $(ALL_TARGETS): it is entirely
# generated, so this also sweeps binaries left by an older target list.
clean:
	rm -rf $(OUTPUT_DIR)
	@echo "Cleaned assembly validation test binaries"

# Help target
help:
	@echo "Assembly Validation Test Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all     - Build all assembly validation tests (default)"
	@echo "  clean   - Remove all built binaries"
	@echo "  help    - Show this help message"
	@echo ""
	@echo "Output directory: $(OUTPUT_DIR)"
	@echo ""
	@echo "Tests:"
	@echo "  Vector Arithmetic:"
	@echo "    - vfmacc_simple: FMA operations"
	@echo "    - vadd_simple: Vector addition"
	@echo "    - vadd_vlmax: Vector addition at vl=VLMAX, for the VLEN sweep"
	@echo "    - vfmul_simple: Vector multiplication"
	@echo "    - compute_intensive: Pure compute FMA"
	@echo "    - vzvbb_ops: Zvbb bit-manip + vsext/vzext"
	@echo "  Memory:"
	@echo "    - vload_store_simple: Memory bandwidth"
	@echo "    - vseg_load_store: Unit-stride segmented load/store"
	@echo "    - vseg_strided_indexed: Strided + indexed segmented load/store"
	@echo "    - vindexed_eew_mismatch: Indexed ops with index EEW != data SEW"
	@echo "    - vload_widths: Unit-stride/strided/ff/mask across all EEWs"
	@echo "    - vseg_nf_sweep: Segment ops at every NFIELDS, incl ff+strided"
	@echo "    - vindexed_widths: All index EEWs, ordered and unordered"
	@echo "    - vindexed_vl_dedup: indexed at vl < VLMAX, with and without line reuse"
	@echo "    - vleff_trim: vle8ff.v across a page hole, so VL really is trimmed"
	@echo "    - membench_loads: mem-capture benchmark, unit/strided/indexed loads"
	@echo "    - membench_stores: mem-capture benchmark, the same as stores"
	@echo "  Threads:"
	@echo "    - clone_vl_inherit: a cloned thread inherits the spawner's VL"
	@echo "  Mixed/HPC:"
	@echo "    - stream_triad: STREAM Triad benchmark"
	@echo "    - daxpy: DAXPY (BLAS level 1)"

.PHONY: all clean help
