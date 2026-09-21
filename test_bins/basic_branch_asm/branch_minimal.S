# Minimal RV64 branching program to test E-Trace PC tracking
.section .text
.globl _start

_start:
    # Test 1: Simple forward branch NOT taken
    addi  a0, x0, 5
    beq   a0, x0, skip1     # Not taken (a0=5, x0=0)
    addi  a1, x0, 100       # This executes

skip1:
    # Test 2: Simple forward branch TAKEN
    addi  a2, x0, 0
    beq   a2, x0, skip2     # Taken (a2=0, x0=0)
    addi  a3, x0, 200       # This does NOT execute

skip2:
    # Test 3: Simple backward loop (branch taken, then not taken)
    addi  a4, x0, 3

loop:
    addi  a4, a4, -1
    bnez  a4, loop          # Taken twice, not taken once

    # Exit cleanly
    li    a0, 0
    li    a7, 93            # syscall number for exit
    ecall
