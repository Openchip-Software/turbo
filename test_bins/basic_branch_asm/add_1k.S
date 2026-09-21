# RV64 assembly program with 4x vadd.vv in 250-iteration loop (1000 vector adds total)
.section .text
.globl _start

_start:
    # Initialize vector unit
    li    a1, -1      # avl = -1 (set to VLMAX)
    vsetvli a2, a1, e8, m1, ta, ma
    vsetvli a2, a1, e8, m1, ta, ma

    # Set loop counter to 250 iterations
    addi  a0, x0, 250

loop:
    # 4x vadd.vv instructions - with 250 iterations = 1000 vector adds
    vadd.vv v0, v1, v2
    vadd.vv v3, v4, v5
    vadd.vv v6, v7, v8
    vadd.vv v9, v10, v11

    # Decrement loop counter and branch if not zero
    addi  a0, a0, -1
    bnez  a0, loop

    li a7, 93 # syscall number for exit, a0 is already zero
    ecall
