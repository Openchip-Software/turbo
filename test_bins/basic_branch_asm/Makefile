# Compiler and flags
CC := clang
CFLAGS := --target=riscv64-unknown-linux-gnu -nostdlib -static

# Architecture flags - most files use rv64g, but some need rv64gv for vector operations
ARCH_RV64G := -march=rv64g
ARCH_RV64GV := -march=rv64gv

# List of source files (without .S extension)
SRCS_RV64G := \
	loop_10x \
	branch_complex \
	branch_minimal \
	branch_call \
	thirty_one_branches_jalr \
	simple_31_branches_jalr \
	four_branches_jalr \
	unprocessed_branches_test \
	unprocessed_branches_ret_test \
	branches_before_ecall_test \
	three_branches_jalr_test

SRCS_RV64GV := \
	vld_vst \
	flops_test \
	add_1k

# Generate target names
TARGETS_RV64G := $(addsuffix .riscv,$(SRCS_RV64G))
TARGETS_RV64GV := $(addsuffix .riscv,$(SRCS_RV64GV))
ALL_TARGETS := $(TARGETS_RV64G) $(TARGETS_RV64GV)

# Default target
all: $(ALL_TARGETS)

# Pattern rule for rv64g targets
$(TARGETS_RV64G): %.riscv: %.S
	$(CC) $(ARCH_RV64G) $(CFLAGS) $< -o $@

# Pattern rule for rv64gv targets
$(TARGETS_RV64GV): %.riscv: %.S
	$(CC) $(ARCH_RV64GV) $(CFLAGS) $< -o $@

# Clean target
clean:
	rm -f $(ALL_TARGETS) output.txt

qemu:
	@if [ -z "$(QEMU_PLUGIN_DIR)" ]; then \
		echo "Error: QEMU_PLUGIN_DIR is not set."; \
		echo "Please export QEMU_PLUGIN_DIR to a valid directory containing QEMU plugins."; \
		echo "Example: export QEMU_PLUGIN_DIR=/usr/lib/qemu"; \
		echo "   or point to <QEMU build dir>/contrib/plugins/"; \
		exit 1; \
	fi
	qemu-riscv64 -d plugin -D output.txt -plugin $(QEMU_PLUGIN_DIR)/libexeclog.so ./loop_10x.riscv
	cat output.txt

rave:
	# using "DevContainer" @ 5559874a3048 (manually ran command, not via Makefile here)
	# Note that "PRINT_SCALAR" has been renamed/deprecated in newer RAVE's for TRACE_SCALAR instead
	RAVE_PRINT_SCALAR=1 RAVE_TRACE_SCALAR=1 RAVE_PRINT_REPORT=1 RAVE_PRINT_PRV=1 rave test_bins/basic_branch_asm/loop_10x.riscv

.PHONY: all clean qemu rave
