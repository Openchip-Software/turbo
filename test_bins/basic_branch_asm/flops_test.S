# RV64 assembly program for testing Flops/Byte counters
.section .rodata
flops_test_name:
    .asciz "flops_test"

.section .data
    data: .zero 2048

.section .text
.globl _start
# required for RAVE markers to be found.
.option nopic

_start:
    # Initialize vector unit, avl = -1 (set to VLMAX)
    li    a1, 256
    vsetvli a2, a1, e64, m1, ta, ma

    # load address of data for vector load into a3
    la a3, data

    # Begin RAVE region
    la t0, flops_test_name
    li t1, -1
    add x0, t0, t1

    # Set loop counter to 250 iterations
    addi  a0, x0, 250

    fadd.s fa0, fa1, fa2

loop:
    # load: 8 bytes per element
    vle64.v v1, (a3)
    # add: 1x per element
    vfadd.vv v0, v1, v2
    # FMA: 2x per element
    vfmacc.vv v0, v1, v2

    # Decrement loop counter and branch if not zero
    addi  a0, a0, -1
    bnez  a0, loop

    # End RAVE region
    la t0, flops_test_name
    li t1, -1
    sub x0, t0, t1

    li a7, 93 # syscall number for exit, a0 is already zero
    ecall
