# Segment load/store across every NFIELDS value, plus the ff and strided forms.
#
# vseg_load_store.S covers nf = 2, 3, 4, 8; nf = 5, 6, 7 were never decoded, and
# neither VLSEGFF_V nor the strided segment forms appeared in any test. Since
# one enum variant spans the whole mnemonic family, an nf-handling error at an
# untested value shows up in both byte counts and mix bucketing.
#
# e8 throughout keeps EMUL = 1 so EMUL * NFIELDS <= 8 holds at nf = 8, and keeps
# the destination register groups inside v2..v31.

.section .text
.global _start

_start:
    li t0, 64                       # AVL -> vl = 64 at e8/m1
    vsetvli t1, t0, e8, m1, ta, ma

    la t2, source_data
    la t3, dest_data
    li t5, 16                       # byte stride for the strided segment ops

    li t4, 100

# Per iteration, vl = 64, eew = 8 so bytes == elements:
#   vlseg2e8 .. vlseg8e8 : (2+3+4+5+6+7+8) * 64 = 2240 B loaded
#   vlseg2e8ff           :             2 * 64 =  128 B loaded
#   vlsseg2e8            :             2 * 64 =  128 B loaded
#   vsseg2e8             :             2 * 64 =  128 B stored
#   vssseg2e8            :             2 * 64 =  128 B stored
#   => 2496 bytes loaded, 256 bytes stored per iteration
loop:
    vlseg2e8.v v2, (t2)
    vlseg3e8.v v4, (t2)
    vlseg4e8.v v8, (t2)
    vlseg5e8.v v12, (t2)
    vlseg6e8.v v16, (t2)
    vlseg7e8.v v20, (t2)
    vlseg8e8.v v24, (t2)

    vlseg2e8ff.v v2, (t2)
    vlsseg2e8.v v2, (t2), t5

    vsseg2e8.v v2, (t3)
    vssseg2e8.v v2, (t3), t5

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
    .rept 256
    .dword 0x0123456789abcdef
    .endr

.section .bss
.align 8
dest_data:
    .skip 2048
