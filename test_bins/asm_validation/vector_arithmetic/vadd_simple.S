.section .text
.globl _start

_start:
    # Set VL
    li    a1, 32
    vsetvli a2, a1, e8, m1, ta, ma

    # Set loop counter
    addi  a0, x0, 1000

loop:
    vadd.vv v0, v1, v2

    # Decrement loop counter and branch if not zero
    addi  a0, a0, -1
    bnez  a0, loop

end:
    li a7, 93 # syscall number for exit, a0 is already zero
    ecall
