.section .text
.globl _start

_start:
    # large enough to (mostly..) fill LMUL=2 at e8 SEW.
    li    a0, 4000
    li    t0, 31 # bitmask of 5
    NOT   t0, t0

lmul2:
    vsetvli a2, a0, e8, m2, ta, ma
    vadd.vv v0, v2, v2
    vsetvli a2, a0, e16, m2, ta, ma
    vadd.vv v0, v2, v2
    vsetvli a2, a0, e32, m2, ta, ma
    vadd.vv v0, v2, v2

lmul1:
    # Set SEW and VL, and do compute
    vsetvli a2, a0, e8, m1, ta, ma
    vadd.vv v0, v1, v2
    vsetvli a2, a0, e16, m1, ta, ma
    vadd.vv v0, v1, v2
    vsetvli a2, a0, e32, m1, ta, ma
    vadd.vv v0, v1, v2


    # Set VL, and do compute
    li    a0, 300
    vsetvli a2, a0, e64, m1, ta, ma
    vadd.vv v0, v1, v2
loop:
    # now we do compute at e64, for 256->300 it'll all bucket into 256 (max VLEN)
    and t1, t0, a0
    vsetvli a2, t1, e64, m1, ta, ma
    vadd.vv v0, v1, v2

    # Decrement loop counter and branch if not zero
    addi  a0, a0, -1
    bnez  a0, loop

end:
    li a7, 93 # syscall number for exit, a0 is already zero
    ecall
