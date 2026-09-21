# Indexed loads at vl below VLMAX, with and without line reuse.
#
# Two properties of the indexed capture that nothing else pins, because every
# other indexed fixture here uses an all-zero index vector -- which collapses to
# one line whether the expansion honours vl or walks the whole register:
#
#   * the expansion stops at vl. vl is 20 here against a VLMAX of 256 at e64/m1,
#     and the first op's indices are i * 64, so each active element reaches a
#     distinct cache line: exactly 20 of them, not 256 and not VLEN/SEW.
#   * lines touched by more than one element are reported once. The second op's
#     indices are (i * 8) & 63 -- 20 distinct byte offsets, all inside the same
#     64-byte line as the (64-byte aligned) base -- so its whole footprint is a
#     single line.
#
# The two together also separate the two counts that a footprint trace must not
# confuse: the byte total is 20 elements x 8 bytes for *both* ops, while the line
# count is 20 for one and 1 for the other.

.section .text
.global _start

_start:
    li t0, 20                       # AVL = 20, well below VLMAX (256 at e64/m1)
    vsetvli t1, t0, e64, m1, ta, ma

    la t2, source_data

    vid.v v8
    vsll.vi v8, v8, 6               # i * 64: one distinct cache line per element

    vid.v v9
    vsll.vi v9, v9, 3               # i * 8: distinct elements...
    li t6, 63
    vand.vx v9, v9, t6              # ...wrapped into a single 64-byte line

    li t4, 100

# Per iteration, vl = 20 with 8-byte elements: 2 loads * 20 * 8 = 320 bytes
# loaded, over 20 + 1 = 21 distinct cache lines.
loop:
    vluxei64.v v4, (t2), v8
    vloxei64.v v5, (t2), v9

    addi t4, t4, -1
    bnez t4, loop

end:
    # Exit
    li a7, 93
    li a0, 0
    ecall

.section .data
# 64-byte aligned, so the second op's indices stay inside one line rather than
# straddling two -- the alignment is load-bearing, not tidiness.
.align 6
source_data:
    .rept 512
    .dword 0x0123456789abcdef
    .endr
