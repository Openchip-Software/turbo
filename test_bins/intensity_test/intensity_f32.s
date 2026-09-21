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

# Flops: 2 * 18 = 36 flops
# Bytes moved: 8 * 3 + 4 * 3 = 36 bytes 
# Intensity: 36 / 36 = 1
.type do_work, @function
do_work:
    # Load addresses
    # 8 bytes
    la t0, var1
    # 8 bytes
    la t1, var2
    # 8 bytes
    la t2, result

    # 4 bytes
    flw f0, 0(t0)        
    
    # 4 bytes
    flw f1, 0(t1)        

    # 2 * 18 = 36 flops
    fmadd.s f2, f0, f0, f1    # f2 = var1 + var2
    fmadd.s f2, f0, f0, f1    # f2 = var1 + var2
    fmadd.s f2, f0, f0, f1    # f2 = var1 + var2
    fmadd.s f2, f0, f0, f1    # f2 = var1 + var2
    fmadd.s f2, f0, f0, f1    # f2 = var1 + var2
    fmadd.s f2, f0, f0, f1    # f2 = var1 + var2
    fmadd.s f2, f0, f0, f1    # f2 = var1 + var2
    fmadd.s f2, f0, f0, f1    # f2 = var1 + var2
    fmadd.s f2, f0, f0, f1    # f2 = var1 + var2
    fmadd.s f2, f0, f0, f1    # f2 = var1 + var2
    fmadd.s f2, f0, f0, f1    # f2 = var1 + var2
    fmadd.s f2, f0, f0, f1    # f2 = var1 + var2
    fmadd.s f2, f0, f0, f1    # f2 = var1 + var2
    fmadd.s f2, f0, f0, f1    # f2 = var1 + var2
    fmadd.s f2, f0, f0, f1    # f2 = var1 + var2
    fmadd.s f2, f0, f0, f1    # f2 = var1 + var2
    fmadd.s f2, f0, f0, f1    # f2 = var1 + var2
    fmadd.s f2, f0, f0, f1    # f2 = var1 + var2

    # 4 bytes
    fsw f2, 0(t2)        

    ret
    
