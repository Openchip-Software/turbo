.section .text
.global _start

_start:
test_entry:
    # Setup vector configuration
    li t0, 256
    vsetvli t1, t0, e32, m1, ta, ma
    
    # Prepare addresses
    la t2, operand_a
    la t3, operand_b
    la t4, operand_c
    
    # Load constant operands (outside measured loop)
    vle32.v v0, (t2)     # operand_a (constant)
    vle32.v v2, (t4)     # operand_c (accumulator)

    # Loop: Each iteration performs 1 load + 10 FMAs
    # Memory: 1 vector load = 128 bytes (32 elements * 4 bytes)
    # Compute: 10 FMAs * 64 FLOPS each = 640 FLOPS
    # Arithmetic Intensity: 640 FLOPS / 128 bytes = 5 FLOPS/byte
    li t5, 100
loop:
    vle32.v v1, (t3)          # 1 load: 128 bytes
    vfmacc.vv v2, v0, v1      # FMA 1: 64 FLOPS
    vfmacc.vv v2, v0, v1      # FMA 2: 64 FLOPS
    vfmacc.vv v2, v0, v1      # FMA 3: 64 FLOPS
    vfmacc.vv v2, v0, v1      # FMA 4: 64 FLOPS
    vfmacc.vv v2, v0, v1      # FMA 5: 64 FLOPS
    vfmacc.vv v2, v0, v1      # FMA 6: 64 FLOPS
    vfmacc.vv v2, v0, v1      # FMA 7: 64 FLOPS
    vfmacc.vv v2, v0, v1      # FMA 8: 64 FLOPS
    vfmacc.vv v2, v0, v1      # FMA 9: 64 FLOPS
    vfmacc.vv v2, v0, v1      # FMA 10: 64 FLOPS
    addi t5, t5, -1
    bnez t5, loop

end:
    # Exit
    li a7, 93
    li a0, 0
    ecall

.section .data
.align 4
operand_a:
    .rept 32
    .float 1.5
    .endr

operand_b:
    .rept 32
    .float 2.0
    .endr

operand_c:
    .rept 32
    .float 0.1
    .endr
