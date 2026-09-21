# vadd at VLMAX, for the VLEN sweep (`vlen_sweep.rs`).
#
# `vadd_simple.S` with one deliberate change: `vsetvli` takes its AVL from `x0`
# rather than from an immediate. With `rs1 == x0` and `rd != x0` the spec sets
# `vl = VLMAX`, so this kernel names no vector length of its own -- the width it
# runs at is whatever VLEN QEMU was configured with. That is the point: the
# retired instruction count is fixed at every VLEN, and every element counter
# scales with it, so a VL decode that ignores or mis-scales VLEN cannot pass.
#
# e8/m1 keeps VLMAX == VLEN/8 exactly, with no LMUL grouping and no spill
# hazard, up to and including the 64Ki-bit maximum the RVV spec allows.

.section .text
.globl _start

_start:
    # vl = VLMAX = VLEN/8 elements at e8, m1.
    vsetvli a2, x0, e8, m1, ta, ma

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
