# Minimal test case to reproduce UnprocessedBranches(2) error
# This test has exactly 2 branches followed by an uninferable discontinuity (JALR)
# Expected behavior: Should trigger UnprocessedBranches(2) when traced

.section .text
.globl _start

_start:
    # Branch 1: Simple branch NOT taken
    addi  a1, x0, 5
    beq   a1, x0, skip1         # Not taken (a1=5, x0=0)
    addi  a2, x0, 100           # This executes

skip1:
    # Branch 2: Simple branch TAKEN
    addi  a3, x0, 0
    beq   a3, x0, skip2         # Taken (a3=0, x0=0)
    addi  a4, x0, 200           # This does NOT execute

skip2:
    # Set up register for JALR using immediate calculation
    # We'll use auipc + addi to get target address
    auipc a0, 0                 # a0 = current PC
    addi  a0, a0, 16            # a0 = PC + 16 (pointing to target_label)

    # Now perform an uninferable discontinuity (indirect jump via JALR)
    # This should cause the encoder to generate a Branch packet with:
    # - 2 branches in the branch map
    # - An address pointing to target_label
    # The decoder should fail with UnprocessedBranches(2) because it can't
    # consume the 2 branches after jumping to target_label
    jalr  x0, a0, 0             # Jump to address in a0 (target_label)

    # This code should never execute
    addi  a5, x0, 999

target_label:
    # Landing point after the indirect jump
    addi  a6, x0, 42

    # Exit cleanly
    li    a0, 0
    li    a7, 93                # syscall number for exit
    ecall
