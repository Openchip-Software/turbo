# Test case with 3 branches followed by JALR
# Testing if more branches before an uninferable discontinuity triggers UnprocessedBranches error

.section .text
.globl _start

_start:
    # Branch 1: NOT taken
    addi  a1, x0, 5
    beq   a1, x0, skip1         # Not taken (a1=5, x0=0)
    addi  a2, x0, 100           # This executes

skip1:
    # Branch 2: TAKEN
    addi  a3, x0, 0
    beq   a3, x0, skip2         # Taken (a3=0, x0=0)
    addi  a4, x0, 200           # This does NOT execute

skip2:
    # Branch 3: NOT taken
    addi  a5, x0, 10
    beq   a5, x0, skip3         # Not taken (a5=10, x0=0)
    addi  a6, x0, 300           # This executes

skip3:
    # Set up target for JALR
    auipc a0, 0                 # a0 = current PC
    addi  a0, a0, 16            # a0 points to target_label

    # Uninferable discontinuity (indirect jump via JALR)
    # At this point we have 3 branches in the branch map
    jalr  x0, a0, 0             # Jump to address in a0

    # Should not execute
    addi  a7, x0, 999

target_label:
    # Landing point
    addi  t0, x0, 42

    # Exit
    li    a0, 0
    li    a7, 93
    ecall
