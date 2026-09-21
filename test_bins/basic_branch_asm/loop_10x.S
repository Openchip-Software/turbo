# Minimal RV64 assembly program with 10-iteration loop
.section .text
.globl _start

_start:
    addi  a0, x0, 10

loop:
    addi  a0, a0, -1
    bnez  a0, loop

    li a7, 93 # syscall number for exit, a0 is already zero
    ecall
