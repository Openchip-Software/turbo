# Strided and indexed segmented vector load/store validation.
#
# Exercises VLSSEG_V / VSSSEG_V (strided) and VLUXSEG_V / VLOXSEG_V /
# VSUXSEG_V / VSOXSEG_V (indexed, unordered and ordered) decode plus
# bytes_moved (nf * eew/8 * vl).
#
# AVL is fixed at 64 and stays below VLMAX for e64/m1, so vl == 64.

.section .text
.global _start

_start:
    li t0, 64                       # AVL
    vsetvli t1, t0, e64, m1, ta, ma

    la t2, source_data
    la t3, dest_data
    li t5, 32                       # byte stride for the strided segment ops

    # Index vector: element i -> byte offset i * 16 (one 2 x e64 segment).
    # Max offset is 63 * 16 + 16 = 1024 bytes, inside both buffers.
    vid.v v8
    vsll.vi v8, v8, 4

    li t4, 100

# Per iteration, all with nf=2, eew=64, vl=64 => 1024 bytes each:
#   3 loads  (vlsseg2e64, vluxseg2ei64, vloxseg2ei64) => 3072 bytes loaded
#   3 stores (vssseg2e64, vsuxseg2ei64, vsoxseg2ei64) => 3072 bytes stored
loop:
    vlsseg2e64.v v0, (t2), t5
    vssseg2e64.v v0, (t3), t5

    vluxseg2ei64.v v4, (t2), v8
    vloxseg2ei64.v v6, (t2), v8

    vsuxseg2ei64.v v4, (t3), v8
    vsoxseg2ei64.v v6, (t3), v8

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
