# Complex RV64 branching program, to stress taken/not-taken branches
.section .text
.globl _start

_start:
    # Test 1: Simple countdown loop (branch taken multiple times, then not taken)
    addi  a0, x0, 10

loop:
    addi  a0, a0, -1
    bnez  a0, loop          # Branch taken 9 times, not taken once

    # Test 2: Forward branch not taken
    addi  a1, x0, 5
    beq   a1, x0, skip1     # Not taken (a1=5, x0=0)
    addi  a2, x0, 100       # This executes

skip1:
    # Test 3: Forward branch taken
    addi  a1, x0, 0
    beq   a1, x0, skip2     # Taken (a1=0, x0=0)
    addi  a2, x0, 200       # This does NOT execute

skip2:
    # Test 4: BLT (branch less than) - taken
    addi  a3, x0, 5
    addi  a4, x0, 10
    blt   a3, a4, skip3     # Taken (5 < 10)
    addi  a5, x0, 1

skip3:
    # Test 5: BLT - not taken
    addi  a3, x0, 15
    blt   a4, a3, skip4     # Not taken (10 >= 15 is false, so 10 < 15 is true) - TAKEN
    addi  a5, x0, 2

skip4:
    # Test 6: BGE (branch greater or equal) - taken
    addi  a3, x0, 10
    addi  a4, x0, 5
    bge   a3, a4, skip5     # Taken (10 >= 5)
    addi  a5, x0, 3

skip5:
    # Test 7: Call function with different values
    addi  a0, x0, 0
    jal   ra, check_value   # Call with a0=0

    addi  a0, x0, 5
    jal   ra, check_value   # Call with a0=5

    addi  a0, x0, 10
    jal   ra, check_value   # Call with a0=10

    # Test 8: Backward branch taken then not taken
    addi  a0, x0, 3

loop2:
    addi  a0, a0, -1
    bgtz  a0, loop2         # Branch taken while a0 > 0

    # Test 9: BNE (branch not equal) test
    addi  a1, x0, 7
    addi  a2, x0, 7
    bne   a1, a2, skip6     # Not taken (7 == 7)
    addi  a3, x0, 42

skip6:
    addi  a1, x0, 7
    addi  a2, x0, 8
    bne   a1, a2, skip7     # Taken (7 != 8)
    addi  a3, x0, 99

skip7:
    # Exit cleanly
    li    a0, 0
    li    a7, 93            # syscall number for exit
    ecall

# Function: check_value
# Input: a0 = value to check
# Branches based on value ranges
check_value:
    # Save return address
    addi  sp, sp, -8
    sd    ra, 0(sp)

    # Check if a0 == 0
    beqz  a0, is_zero       # Branch if zero

    # Check if a0 < 5
    addi  t0, x0, 5
    blt   a0, t0, is_small  # Branch if less than 5

    # Check if a0 >= 10
    addi  t0, x0, 10
    bge   a0, t0, is_large  # Branch if >= 10

    # Default case (5 <= a0 < 10)
    addi  a1, x0, 2         # Mark as medium
    j     check_done

is_zero:
    addi  a1, x0, 0         # Mark as zero
    j     check_done

is_small:
    addi  a1, x0, 1         # Mark as small
    j     check_done

is_large:
    addi  a1, x0, 3         # Mark as large

check_done:
    # Restore return address and return
    ld    ra, 0(sp)
    addi  sp, sp, 8
    jalr  x0, 0(ra)         # Return
