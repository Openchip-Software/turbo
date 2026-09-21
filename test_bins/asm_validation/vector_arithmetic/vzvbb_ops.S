# Zvbb vector bit-manipulation and integer extension validation.
#
# Exercises the Zvbb opcodes (vandn, vbrev, vbrev8, vrev8, vclz, vctz, vcpop,
# vrol, vror incl. the 6-bit vror.vi immediate, vwsll) and the vsext/vzext
# integer extension family. Pure compute: no loads or stores, no float ops.
#
# AVL is fixed at 64, below VLMAX for e32/m1 and e64/m1, so vl == 64.

.section .text
.global _start

_start:
    li t0, 64                       # AVL
    li a0, 5                        # scalar operand for the .vx forms

    vsetvli t1, t0, e32, m1, ta, ma
    vid.v v2
    vid.v v3

    li t4, 100

# Per iteration: 2 vsetvli + 16 vector ops + addi + bnez = 20 instructions,
# 16 vector instructions * vl 64 = 1024 vector elements.
#   e32 block: 13 vector instructions
#   e64 block:  3 vector instructions
loop:
    vsetvli t1, t0, e32, m1, ta, ma
    vandn.vv v1, v2, v3
    vandn.vx v1, v2, a0
    vbrev.v v1, v2
    vbrev8.v v1, v2
    vrev8.v v1, v2
    vclz.v v1, v2
    vctz.v v1, v2
    vcpop.v v1, v2
    vrol.vv v1, v2, v3
    vrol.vx v1, v2, a0
    vror.vi v1, v2, 5
    vwsll.vi v4, v2, 3              # widening: vd is EMUL=2 (v4, v5)
    vsext.vf2 v6, v2

    vsetvli t1, t0, e64, m1, ta, ma
    vsext.vf4 v10, v2
    vzext.vf8 v12, v2
    vror.vi v14, v2, 40             # imm > 31 exercises the 6-bit immediate

    addi t4, t4, -1
    bnez t4, loop

end:
    li a7, 93
    li a0, 0
    ecall
