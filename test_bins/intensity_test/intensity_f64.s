.section .data
    .align 4        # Ensure alignment
    var1: .word 10
    var2: .word 20
    result: .word 0

.section .text
.globl _start

.type _start, @function
_start:
    .option push
    .option norelax # Prevent gp-relative optimizations requiring gp init
    la gp, __global_pointer$
    .option pop
    
    # Set up stack pointer (optional in some sims, needed in OS environment)
    # Just calling main here assuming standard entry
    call main

    # Exit syscall (Linux)
    li a7, 93       # syscall: exit
    li a0, 0        # status: 0
    ecall

.type main, @function
main:
    addi sp, sp, -16
    sw ra, 12(sp)
    
    call do_work
    
    lw ra, 12(sp)
    addi sp, sp, 16
    ret

# Flops: 1 * 24 = 24 flops
# Bytes moved: 8 * 3 + 8 * 3 = 48 bytes 
# Intensity: 24 / 48 = 0.5
.type do_work, @function
do_work:
    # Load addresses
    # Use lui/addi pair manually or allow assembler to do it, 
    # but ensure we are accessing .data correctly.
    la t0, var1
    la t1, var2
    la t2, result

    # Doubles
    fld f0, 0(t0)        
    
    fld f1, 0(t1)        

    # 24 flops
    fadd.d f2, f0, f1    # f2 = var1 + var2
    fadd.d f2, f0, f1    # f2 = var1 + var2
    fadd.d f2, f0, f1    # f2 = var1 + var2
    fadd.d f2, f0, f1    # f2 = var1 + var2
    fadd.d f2, f0, f1    # f2 = var1 + var2
    fadd.d f2, f0, f1    # f2 = var1 + var2
    fadd.d f2, f0, f1    # f2 = var1 + var2
    fadd.d f2, f0, f1    # f2 = var1 + var2
    fadd.d f2, f0, f1    # f2 = var1 + var2
    fadd.d f2, f0, f1    # f2 = var1 + var2
    fadd.d f2, f0, f1    # f2 = var1 + var2
    fadd.d f2, f0, f1    # f2 = var1 + var2
    fadd.d f2, f0, f1    # f2 = var1 + var2
    fadd.d f2, f0, f1    # f2 = var1 + var2
    fadd.d f2, f0, f1    # f2 = var1 + var2
    fadd.d f2, f0, f1    # f2 = var1 + var2
    fadd.d f2, f0, f1    # f2 = var1 + var2
    fadd.d f2, f0, f1    # f2 = var1 + var2
    fadd.d f2, f0, f1    # f2 = var1 + var2
    fadd.d f2, f0, f1    # f2 = var1 + var2
    fadd.d f2, f0, f1    # f2 = var1 + var2
    fadd.d f2, f0, f1    # f2 = var1 + var2
    fadd.d f2, f0, f1    # f2 = var1 + var2
    fadd.d f2, f0, f1    # f2 = var1 + var2

    fsd f2, 0(t2)        

    ret
