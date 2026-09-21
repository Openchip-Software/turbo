# Benchmark kernel for the vector memory address capture (MEM_IO_PLAN.md).
#
# Unlike its neighbours here, this file is not a correctness fixture: it exists
# to be run twice, with `mem_capture` on and off, so the capture's runtime cost
# and its contribution to the trace can both be measured against a baseline that
# is the same guest execution. Everything it does is therefore chosen to be
# expensive *for the tracer* rather than interesting for the guest.
#
# The loop mixes the three capture kinds, because their costs are nothing alike:
#
#   * unit-stride (`Base`)     -- one GP read, 3 record words.
#   * strided (`BaseStride`)   -- two GP reads, 5 record words.
#   * indexed (`Lines`)        -- a GP read *and* a whole vector register read,
#                                 an expansion over `vl` elements, and one
#                                 record pair per distinct line touched.
#
# The indexed ops are the stressor, so their indices are `i * 64`: every element
# lands on its own cache line, which is the worst case both for the expansion
# and for the trace (`vl` = 64 lines = 129 record words = 516 B per execution,
# against 12 B for a unit-stride op). The strided ops walk 64 B for the same
# reason, but cost the trace nothing extra -- their footprint is derived offline,
# which is exactly the asymmetry this benchmark is meant to show.
#
# 10000 iterations: enough that the per-iteration cost dominates process startup
# (the run is still well under a second), and enough that the trace reaches a
# double-digit number of megabytes uncompressed, which is where the deflate
# ratio stops being a startup artefact.
#
# Loads only; membench_stores.S is the same kernel with the store forms.

.section .text
.global _start

_start:
    li t0, 64                       # AVL -> vl = 64 at e64/m1
    vsetvli t1, t0, e64, m1, ta, ma

    la t2, source_data
    li t5, 64                       # stride: one cache line per element

    # Index vector: i * 64, so each of the 64 elements gathers from its own
    # cache line. Built once -- it is loop-invariant, so every iteration
    # re-reads the same register and re-derives the same 64 lines, which is what
    # makes the runs comparable.
    vid.v v8
    vsll.vi v8, v8, 6

    li t4, 10000

# Per iteration, vl = 64 at 8-byte elements:
#   2 unit-stride loads * 512 B                    = 1024 B
#   2 strided loads     * 64 elements * 8 B        = 1024 B
#   2 indexed loads     * 64 elements * 8 B        = 1024 B
#   => 3072 bytes loaded, over 2*8 + 2*64 + 2*64 = 272 distinct lines
loop:
    vle64.v v0, (t2)
    vle64.v v1, (t2)

    vlse64.v v2, (t2), t5
    vlse64.v v3, (t2), t5

    vluxei64.v v4, (t2), v8
    vloxei64.v v5, (t2), v8

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
    # 8 KiB: the strided and indexed ops both reach 63 * 64 + 8 bytes past the
    # base, so the whole footprint stays inside this buffer.
    .rept 1024
    .dword 0x0123456789abcdef
    .endr
