# Minimal test case to reproduce UnprocessedBranches(2) error using RET
# This test has exactly 2 branches followed by a return instruction (uninferable discontinuity)
# Expected behavior: Should trigger UnprocessedBranches(2) when traced

.section .text
.globl _start

_start:
    # Call a function that will have 2 branches before returning
    call  function_with_branches

    # After return, do some work
    addi  a5, x0, 99

    # Exit cleanly
    li    a0, 0
    li    a7, 93                # syscall number for exit
    ecall

function_with_branches:
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
    # Now return - this is an uninferable discontinuity
    # The encoder should generate a Branch packet with:
    # - 2 branches in the branch map
    # - An address pointing to the return address (after the call)
    # The decoder should fail with UnprocessedBranches(2) because it can't
    # consume the 2 branches after jumping to the return address
    ret                         # Return to caller (uninferable jump)

    # This code should never execute
    addi  a6, x0, 999
