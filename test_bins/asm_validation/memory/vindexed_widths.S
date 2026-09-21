# Indexed load/store across every index EEW, ordered and unordered.
#
# vindexed_eew_mismatch.S proved the data width comes from SEW rather than the
# index suffix, but only for ei32/ei16 and only three of the eight non-segment
# forms. This sweeps all four index widths x {unordered, ordered} x
# {load, store}, so every VL*XEI*_V variant is decoded and byte-counted, plus
# two indexed segment forms at different index widths.
#
# SEW is e64 (8-byte data) while indices are e8/e16/e32/e64, so every non-e64
# case has index width != data width. Index EMUL = (EEW/SEW)*LMUL bottoms out
# at 1/8 for ei8, which is the legal minimum.
#
# All indices are zero, so accesses land on the base address. Byte accounting
# depends on SEW, NFIELDS and VL, not on the addresses.

.section .text
.global _start

_start:
    li t0, 64                       # AVL -> vl = 64 at e64/m1
    vsetvli t1, t0, e64, m1, ta, ma

    la t2, source_data
    la t3, dest_data

    vmv.v.i v8, 0                   # zero index vector, valid at any index EEW

    li t4, 100

# Per iteration, vl = 64 with 8-byte data elements:
#   8 non-segment loads  * 8 * 64 = 4096 B loaded
#   8 non-segment stores * 8 * 64 = 4096 B stored
#   vluxseg3ei8  : 3 * 8 * 64     = 1536 B loaded
#   vsoxseg3ei32 : 3 * 8 * 64     = 1536 B stored
#   => 5632 bytes loaded, 5632 bytes stored per iteration
loop:
    vluxei8.v v0, (t2), v8
    vloxei8.v v1, (t2), v8
    vsuxei8.v v0, (t3), v8
    vsoxei8.v v1, (t3), v8

    vluxei16.v v2, (t2), v8
    vloxei16.v v3, (t2), v8
    vsuxei16.v v2, (t3), v8
    vsoxei16.v v3, (t3), v8

    vluxei32.v v4, (t2), v8
    vloxei32.v v5, (t2), v8
    vsuxei32.v v4, (t3), v8
    vsoxei32.v v5, (t3), v8

    vluxei64.v v6, (t2), v8
    vloxei64.v v7, (t2), v8
    vsuxei64.v v6, (t3), v8
    vsoxei64.v v7, (t3), v8

    # Indexed segment, nf=3, at two different index widths.
    vluxseg3ei8.v v16, (t2), v8
    vsoxseg3ei32.v v16, (t3), v8

    addi t4, t4, -1
    bnez t4, loop

end:
    # Exit
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
