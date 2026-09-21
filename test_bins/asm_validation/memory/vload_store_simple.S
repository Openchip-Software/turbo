.section .text
.global _start

_start:
    # Set vl
    li t0, 256
    vsetvli t1, t0, e64, m1, ta, ma

    # Prepare addresses
    # Loads four bytes
    la t2, source_data
    la t3, dest_data

    # Loop
    li t4, 100

# 100 iterations * 8 bytes per element * 16 elements * 2 instructions = 25600 bytes moved
# 100 iterations * 16 elements * 2 insturctions = 3200 memory ops
loop:
    # changing t2 to t3 here allows testing "l1-hit-only-store"
    vle64.v v0, (t2)
    vse64.v v0, (t3)
    addi t4, t4, -1
    bnez t4, loop

end:
    # Exit
    li a7, 93
    li a0, 0
    ecall

.section .data
.align 8
source_data:
    .rept 256
    .double 7.0
    .endr

.section .bss
.align 8
dest_data:
    .skip 2048  # 256 elements at 8 bytes each
