# Simple RV64 program with function call to test E-Trace JAL handling
.section .text
.globl _start

_start:
    # Test: Simple function call using JAL
    addi  a0, x0, 5
    jal   ra, add_ten       # Call function
    addi  a1, a0, 0         # Move result to a1

    # Exit cleanly
    li    a0, 0
    li    a7, 93            # syscall number for exit
    ecall

# Function: add_ten
# Input: a0
# Output: a0 = a0 + 10
add_ten:
    addi  a0, a0, 10
    ret                     # Return (jalr x0, 0(ra))
