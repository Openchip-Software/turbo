# Test case with branches followed by ecall (syscall)
# Testing if branches before a trap/exception trigger UnprocessedBranches error
# ecall is a synchronizing exception that causes a control flow change

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
    # Immediately perform ecall after branches
    # This is a trap/exception which is a form of uninferable discontinuity
    # The encoder should handle the branch map before the ecall
    li    a0, 0                 # exit code
    li    a7, 93                # syscall number for exit
    ecall                       # This should cause special handling in trace

    # This code should never execute
    addi  a5, x0, 999
