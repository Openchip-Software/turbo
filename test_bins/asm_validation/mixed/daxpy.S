# daxpy.S
# Test: DAXPY (Double-precision A*X Plus Y)
# Pattern: y[i] = a * x[i] + y[i]
#
# The loop runs 100 iterations at a fixed vl = 16 (e64/m1, so 128 bytes per
# vector op). The x/y pointers are never advanced, so the same first 16
# elements of the 128-element arrays are updated on every iteration. That is
# deliberate: this is a profiler-validation kernel whose per-iteration
# metrics are hand-computable and must match the "daxpy" entry in
# test_bins/asm_validation/expected_metrics.json.

.section .data
.align 8
scalar_a:
    .double 2.5

array_x:
    .rept 128
    .double 1.0
    .endr

array_y:
    .rept 128
    .double 0.5
    .endr
.section .text
.global _start

_start:
test_entry:
    # Pin vl = 16 elements via the AVL
    li t0, 16
    vsetvli t1, t0, e64, m1, ta, ma

    # Load scalar 'a'
    la t2, scalar_a
    fld f0, 0(t2)

    # Broadcast scalar to vector
    vfmv.v.f v3, f0

    # Prepare addresses
    la t2, array_x
    la t3, array_y

    # 100 fixed iterations (pointers above are never advanced)
    li t4, 100

loop:
    vle64.v v0, (t2)         # Load x[i]
    vle64.v v1, (t3)         # Load y[i]
    vfmacc.vv v1, v0, v3     # y = y + a * x
    vse64.v v1, (t3)         # Store back to y[i]

    addi t4, t4, -1
    bnez t4, loop

end:
    # Exit
    li a7, 93
    li a0, 0
    ecall
