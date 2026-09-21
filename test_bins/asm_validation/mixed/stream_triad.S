.section .data
.align 8
scalar:
    .double 3.0

array_b:
    .rept 64
    .double 1.0
    .endr

array_c:
    .rept 64
    .double 2.0
    .endr

.section .bss
.align 8
array_a:
    .skip 512  # 64 elements * 8 bytes

.section .text
.global _start

_start:
test_entry:
    # Set vl
    li t0, 16
    vsetvli t1, t0, e64, m1, ta, ma
    
    # Load scalar multiplier
    la t2, scalar
    fld f0, 0(t2)
    
    # Prepare addresses
    la t2, array_b
    la t3, array_c
    la t4, array_a
    
    # Loop: 20 STREAM Triad operations
    # a = b + s * c
    li t5, 100
    mv t6, t2  # Reset b pointer
    mv a0, t3  # Reset c pointer
    mv a1, t4  # Reset a pointer
    
loop:
    vle64.v v1, (t6)         # Load b[i]
    vle64.v v2, (a0)         # Load c[i]
    vfmacc.vv v1, v2, v0     # v1 = b + s * c (broadcast scalar to v0 first)
    vse64.v v1, (a1)         # Store to a[i]
    
    addi t5, t5, -1
    bnez t5, loop
    
end:
    # Exit
    li a7, 93
    li a0, 0
    ecall

