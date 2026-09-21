# Minimal test case to reproduce UnprocessedBranches(2) error
# Pattern: 4 branches followed by jalr (indirect jump)
# Based on flash_attn failure at PC sequence around 0xf1b10 and 0xb46b0

.section .text
.globl _start

_start:
    # Initialize registers
    li      a0, 0           # Will be zero for beqz
    li      a1, 1           # Non-zero value
    li      a2, 10          # Value for comparison
    li      a3, 10          # Equal to a2
    la      a5, target1     # Target address for jalr

    # Branch 1: TAKEN (beqz when a0 is zero)
    beqz    a0, label1      # Taken: a0 == 0
    addi    t0, t0, 100     # Should NOT execute

label1:
    # Branch 2: TAKEN (beq when values are equal)
    beq     a2, a3, label2  # Taken: a2 == a3
    addi    t1, t1, 200     # Should NOT execute

label2:
    # Branch 3: NOT TAKEN (bne when values are equal)
    bne     a2, a3, label3  # Not taken: a2 == a3
    addi    t2, t2, 300     # Should execute

label3:
    # Branch 4: TAKEN (bnez when value is non-zero)
    bnez    a1, label4      # Taken: a1 != 0
    addi    t3, t3, 400     # Should NOT execute

label4:
    # JALR - indirect jump (uninferable discontinuity)
    # This should cause the encoder to flush the branch map
    jalr    a5              # Jump to address in a5

target1:
    # Landing point after jalr
    addi    a4, a4, 1

    # Another sequence to ensure proper termination
    li      a6, 5
    li      a7, 5
    beq     a6, a7, target2
    addi    t4, t4, 500

target2:
    # Exit cleanly
    li      a0, 0
    li      a7, 93          # syscall number for exit
    ecall

.section .data
# Empty data section
