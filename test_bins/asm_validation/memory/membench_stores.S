# Benchmark kernel for the vector memory address capture: the store half.
#
# The mirror of membench_loads.S -- same shape, same iteration count, same
# indices, so the two are directly comparable and any difference between them is
# the store path rather than the workload. See that file for why the loop is
# built the way it is.
#
# Stores are worth measuring separately because the capture treats them
# identically (a store's base register is read exactly as a load's is), and a
# measured cost that differs between the two would mean something in the tracer
# does not, in fact, treat them identically.

.section .text
.global _start

_start:
    li t0, 64                       # AVL -> vl = 64 at e64/m1
    vsetvli t1, t0, e64, m1, ta, ma

    la t3, dest_data
    li t5, 64                       # stride: one cache line per element

    # Index vector: i * 64, so each element scatters to its own line.
    vid.v v8
    vsll.vi v8, v8, 6

    # Something to store. The values are irrelevant; only the addresses are.
    vmv.v.i v0, 1
    vmv.v.i v1, 2
    vmv.v.i v2, 3
    vmv.v.i v3, 4
    vmv.v.i v4, 5
    vmv.v.i v5, 6

    li t4, 10000

# Per iteration, vl = 64 at 8-byte elements:
#   2 unit-stride stores * 512 B                   = 1024 B
#   2 strided stores     * 64 elements * 8 B       = 1024 B
#   2 indexed stores     * 64 elements * 8 B       = 1024 B
#   => 3072 bytes stored, over 2*8 + 2*64 + 2*64 = 272 distinct lines
loop:
    vse64.v v0, (t3)
    vse64.v v1, (t3)

    vsse64.v v2, (t3), t5
    vsse64.v v3, (t3), t5

    vsuxei64.v v4, (t3), v8
    vsoxei64.v v5, (t3), v8

    addi t4, t4, -1
    bnez t4, loop

end:
    # Exit
    li a7, 93
    li a0, 0
    ecall

.section .bss
.align 8
dest_data:
    # 8 KiB, as in membench_loads.S: the strided and indexed stores reach
    # 63 * 64 + 8 bytes past the base.
    .skip 8192
