# Simple test case: Exactly 31 branches followed immediately by jalr
# This test loads the target address early, then executes 31 branches,
# then immediately does jalr without any intermediate address computation.

.section .text
.globl _start

_start:
    # Load target address into a5 BEFORE any branches
    la      a5, landing

    # Initialize registers for branch conditions
    li      a0, 0           # Zero for beqz
    li      a1, 1           # Non-zero for bnez

    # Branch 1: TAKEN
    beqz    a0, b1
    nop
b1:
    # Branch 2: TAKEN
    bnez    a1, b2
    nop
b2:
    # Branch 3: TAKEN
    beqz    a0, b3
    nop
b3:
    # Branch 4: TAKEN
    bnez    a1, b4
    nop
b4:
    # Branch 5: TAKEN
    beqz    a0, b5
    nop
b5:
    # Branch 6: TAKEN
    bnez    a1, b6
    nop
b6:
    # Branch 7: TAKEN
    beqz    a0, b7
    nop
b7:
    # Branch 8: TAKEN
    bnez    a1, b8
    nop
b8:
    # Branch 9: TAKEN
    beqz    a0, b9
    nop
b9:
    # Branch 10: TAKEN
    bnez    a1, b10
    nop
b10:
    # Branch 11: TAKEN
    beqz    a0, b11
    nop
b11:
    # Branch 12: TAKEN
    bnez    a1, b12
    nop
b12:
    # Branch 13: TAKEN
    beqz    a0, b13
    nop
b13:
    # Branch 14: TAKEN
    bnez    a1, b14
    nop
b14:
    # Branch 15: TAKEN
    beqz    a0, b15
    nop
b15:
    # Branch 16: TAKEN
    bnez    a1, b16
    nop
b16:
    # Branch 17: TAKEN
    beqz    a0, b17
    nop
b17:
    # Branch 18: TAKEN
    bnez    a1, b18
    nop
b18:
    # Branch 19: TAKEN
    beqz    a0, b19
    nop
b19:
    # Branch 20: TAKEN
    bnez    a1, b20
    nop
b20:
    # Branch 21: TAKEN
    beqz    a0, b21
    nop
b21:
    # Branch 22: TAKEN
    bnez    a1, b22
    nop
b22:
    # Branch 23: TAKEN
    beqz    a0, b23
    nop
b23:
    # Branch 24: TAKEN
    bnez    a1, b24
    nop
b24:
    # Branch 25: TAKEN
    beqz    a0, b25
    nop
b25:
    # Branch 26: TAKEN
    bnez    a1, b26
    nop
b26:
    # Branch 27: TAKEN
    beqz    a0, b27
    nop
b27:
    # Branch 28: TAKEN
    bnez    a1, b28
    nop
b28:
    # Branch 29: TAKEN
    beqz    a0, b29
    nop
b29:
    # Branch 30: TAKEN
    bnez    a1, b30
    nop
b30:
    # Branch 31: TAKEN (the 31st and final branch)
    beqz    a0, b31
    nop
b31:
    # CRITICAL: JALR immediately after 31st branch
    # The target was loaded into a5 at the beginning
    jalr    zero, a5, 0

landing:
    # Successfully landed after the indirect jump
    addi    a6, zero, 99

    # Clean exit
    li      a0, 0
    li      a7, 93
    ecall
