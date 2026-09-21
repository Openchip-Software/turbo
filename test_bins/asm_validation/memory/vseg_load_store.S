# Segmented unit-stride vector load/store validation.
#
# Exercises VLSEG_V / VSSEG_V decode + bytes_moved (nf * eew/8 * vl) across
# several nf values and element widths.
#
# AVL is fixed at 64, which is below VLMAX for every (SEW, LMUL=1) pair used
# here at the 16k VLEN the validation suite runs with, so vl == 64 throughout.

.section .text
.global _start

_start:
    li t0, 64                       # AVL

    la t2, source_data
    la t3, dest_data

    li t4, 100

# Per iteration:
#   e64 nf=2 : 2 * 8 * 64 = 1024 bytes loaded, 1024 stored
#   e64 nf=3 : 3 * 8 * 64 = 1536 bytes loaded, 1536 stored
#   e32 nf=4 : 4 * 4 * 64 = 1024 bytes loaded, 1024 stored
#   e32 nf=2 : 2 * 4 * 64 =  512 bytes loaded,  512 stored
#   e8  nf=8 : 8 * 1 * 64 =  512 bytes loaded,  512 stored
#   => 4608 bytes loaded, 4608 bytes stored per iteration
loop:
    vsetvli t1, t0, e64, m1, ta, ma
    vlseg2e64.v v0, (t2)
    vsseg2e64.v v0, (t3)
    vlseg3e64.v v4, (t2)
    vsseg3e64.v v4, (t3)

    vsetvli t1, t0, e32, m1, ta, ma
    vlseg4e32.v v8, (t2)
    vsseg4e32.v v8, (t3)
    # Same nf as vlseg2e64.v above but a different eew: these two share both
    # an enum variant and an nf, so they only stay in separate instruction-mix
    # buckets if the subkey encodes eew as well.
    vlseg2e32.v v12, (t2)
    vsseg2e32.v v12, (t3)

    vsetvli t1, t0, e8, m1, ta, ma
    vlseg8e8.v v16, (t2)
    vsseg8e8.v v16, (t3)

    addi t4, t4, -1
    bnez t4, loop

end:
    li a7, 93
    li a0, 0
    ecall

.section .data
.align 8
source_data:
    .rept 512
    .dword 0x0123456789abcdef
    .endr

.section .bss
.align 8
dest_data:
    .skip 4096
