# Test case to verify UnexpectedUninferableDiscon fix
# Pattern: Exactly 31 branches followed by jalr (indirect jump)
# This tests that the encoder properly includes the target address
# in the Branch packet when an uninferable discontinuity follows.

.section .text
.globl _start

_start:
    # Initialize registers for branch conditions
    li      a0, 0           # Zero for beqz
    li      a1, 1           # Non-zero for bnez
    li      a2, 10          # Value for comparisons
    li      a3, 10          # Equal to a2
    li      a4, 5           # Different value

    # Branch 1: TAKEN (beqz when a0 is zero)
    beqz    a0, b1
    nop
b1:
    # Branch 2: TAKEN (beq when values are equal)
    beq     a2, a3, b2
    nop
b2:
    # Branch 3: NOT TAKEN (bne when values are equal)
    bne     a2, a3, b3
    addi    t0, t0, 1       # Execute (branch not taken)
b3:
    # Branch 4: TAKEN (bnez when value is non-zero)
    bnez    a1, b4
    nop
b4:
    # Branch 5: NOT TAKEN (beq with different values)
    beq     a2, a4, b5
    addi    t1, t1, 1       # Execute
b5:
    # Branch 6: TAKEN
    beqz    a0, b6
    nop
b6:
    # Branch 7: TAKEN
    beq     a2, a3, b7
    nop
b7:
    # Branch 8: NOT TAKEN
    bne     a2, a3, b8
    addi    t2, t2, 1
b8:
    # Branch 9: TAKEN
    bnez    a1, b9
    nop
b9:
    # Branch 10: NOT TAKEN
    beq     a2, a4, b10
    addi    t3, t3, 1
b10:
    # Branch 11: TAKEN
    beqz    a0, b11
    nop
b11:
    # Branch 12: TAKEN
    beq     a2, a3, b12
    nop
b12:
    # Branch 13: NOT TAKEN
    bne     a2, a3, b13
    addi    t4, t4, 1
b13:
    # Branch 14: TAKEN
    bnez    a1, b14
    nop
b14:
    # Branch 15: NOT TAKEN
    beq     a2, a4, b15
    addi    t5, t5, 1
b15:
    # Branch 16: TAKEN
    beqz    a0, b16
    nop
b16:
    # Branch 17: TAKEN
    beq     a2, a3, b17
    nop
b17:
    # Branch 18: NOT TAKEN
    bne     a2, a3, b18
    addi    t6, t6, 1
b18:
    # Branch 19: TAKEN
    bnez    a1, b19
    nop
b19:
    # Branch 20: NOT TAKEN
    beq     a2, a4, b20
    addi    s0, s0, 1
b20:
    # Branch 21: TAKEN
    beqz    a0, b21
    nop
b21:
    # Branch 22: TAKEN
    beq     a2, a3, b22
    nop
b22:
    # Branch 23: NOT TAKEN
    bne     a2, a3, b23
    addi    s1, s1, 1
b23:
    # Branch 24: TAKEN
    bnez    a1, b24
    nop
b24:
    # Branch 25: NOT TAKEN
    beq     a2, a4, b25
    addi    s2, s2, 1
b25:
    # Branch 26: TAKEN
    beqz    a0, b26
    nop
b26:
    # Branch 27: TAKEN
    beq     a2, a3, b27
    nop
b27:
    # Branch 28: NOT TAKEN
    bne     a2, a3, b28
    addi    s3, s3, 1
b28:
    # Branch 29: TAKEN
    bnez    a1, b29
    nop
b29:
    # Branch 30: NOT TAKEN
    beq     a2, a4, b30
    addi    s4, s4, 1
b30:
    # Branch 31: TAKEN (the 31st and final branch)
    beqz    a0, b31
    nop
b31:
    # Load landing address directly into a5 using auipc+addi
    auipc   a5, 0           # Get current PC
    addi    a5, a5, 12      # Add offset to landing (jalr is next, then landing)

    # CRITICAL: JALR - uninferable discontinuity immediately after 31 branches
    # This is where the bug would occur if not fixed
    # The encoder MUST emit: Branch { count: 31, address: Some(landing) }
    jalr    zero, a5, 0     # Jump to address in a5 (landing)

landing:
    # Successfully landed after the indirect jump
    # Add a marker instruction
    addi    a6, zero, 99

    # Clean exit
    li      a0, 0           # Exit code 0
    li      a7, 93          # syscall number for exit
    ecall
