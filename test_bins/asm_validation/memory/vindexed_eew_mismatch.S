# Indexed load/store byte accounting when index EEW != data SEW.
#
# The numeric suffix of an indexed op is the *index* element width, while the
# data width is SEW from vtype. `vluxei32.v` under SEW=64 therefore reads
# 32-bit offsets to move 64-bit data, and any byte count derived from the
# suffix alone undercounts it by 2x.
#
# vseg_strided_indexed.S cannot catch that: it uses ei64 indices under SEW=64,
# so the two widths coincide and the wrong formula gives the right answer. Here
# they deliberately differ, with one matching-width op kept as a control.
#
# All indices are zero, so every access targets the base address. Byte
# accounting depends on the widths and VL, not on the addresses.

.section .text
.global _start

_start:
    li t0, 64                       # AVL
    vsetvli t1, t0, e64, m1, ta, ma # SEW = 64 -> 8-byte data elements, vl = 64

    la t2, source_data
    la t3, dest_data

    # Index vector. Written as e64 zeros, which is all-zero for any narrower
    # index width the instructions below reinterpret it as.
    vmv.v.i v8, 0

    li t4, 100

# Per iteration, all at vl=64 with 8-byte data elements:
#   vluxei32.v     index e32, nf=1 =>       8 * 64 =  512 bytes loaded
#   vsuxei32.v     index e32, nf=1 =>       8 * 64 =  512 bytes stored
#   vluxseg2ei16.v index e16, nf=2 =>   2 * 8 * 64 = 1024 bytes loaded
#   vsoxseg2ei16.v index e16, nf=2 =>   2 * 8 * 64 = 1024 bytes stored
#   vloxei64.v     index e64, nf=1 =>       8 * 64 =  512 bytes loaded (control)
#   => 2048 bytes loaded, 1536 bytes stored per iteration
loop:
    vluxei32.v v4, (t2), v8
    vsuxei32.v v4, (t3), v8

    vluxseg2ei16.v v6, (t2), v8
    vsoxseg2ei16.v v6, (t3), v8

    vloxei64.v v10, (t2), v8

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
