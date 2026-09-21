# Fault-only-first load that really is trimmed, and a vector op that reads the
# trimmed VL.
#
# Every other fault-only-first test in this directory (vload_widths.S,
# vseg_nf_sweep.S) runs its ff loads entirely inside a mapped buffer, so they
# never fault and VL is never trimmed -- which means none of them could tell
# whether the tool tracked a vl*ff's effect on VL at all. This one can.
#
# The setup is a guaranteed page hole: mmap two anonymous pages, munmap the
# second, and point the load at the last 8 bytes of the first. `vle8ff.v` then
# faults on element 8 (the first byte in the unmapped page) and, per V spec 8.2,
# writes VL = 8 instead of raising -- QEMU's `vext_ldff` does literally
# `env->vl = i`. The following `vse8.v` runs at that VL, so it stores 8 bytes,
# not the 16 the vsetivli asked for.
#
# That store is the assertion. A tool that misses the trim reports the store at
# VL = 16 and doubles its byte count, which is exactly what happened before
# ADD_VLEFF.md.
#
# The `vsetvli zero, zero, e8, m1` after it is the second half of the test. That
# form takes its AVL from the *current* VL, which the tracer supplies from its
# own shadow rather than reading the CSR -- so it is only right if the trim was
# folded into that shadow. It keeps VL at 8, and the store behind it moves 8 more
# bytes; a stale shadow of 16 would make the tracer claim 16 for that store while
# the guest moved 8.
#
# The requested VL is a vsetivli immediate rather than a register, so it is 16
# regardless of VLEN (any VLEN >= 128 bits gives VLMAX >= 16 at e8/m1) and the
# expected counters are a property of this file alone.

.section .text
.global _start

_start:
    # mmap(NULL, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0)
    li a0, 0
    li a1, 8192
    li a2, 3                        # PROT_READ|PROT_WRITE
    li a3, 0x22                     # MAP_PRIVATE|MAP_ANONYMOUS
    li a4, -1
    li a5, 0
    li a7, 222                      # __NR_mmap
    ecall
    bltz a0, fail                   # a negative return is -errno
    mv s0, a0                       # s0 = base of the two-page mapping

    # munmap(base + 4096, 4096): the hole the ff load must fault into. Punching
    # it out of a mapping we own is what makes the layout deterministic --
    # "whatever follows a one-page mmap" is not.
    li t0, 4096                     # via a register: addi immediates are 12-bit
    add a0, s0, t0
    li a1, 4096
    li a7, 215                      # __NR_munmap
    ecall
    bnez a0, fail

    # s1 = base + 4088: eight valid bytes, then the hole.
    li t0, 4088
    add s1, s0, t0

    li t4, 100

# Per iteration:
#   vsetivli    : requests vl = 16
#   vle8ff.v    : faults on element 8, writes vl = 8, loads 8 B
#   vse8.v      : runs at vl = 8, stores 8 B
#   vsetvli 0,0 : AVL = current vl = 8, so vl stays 8
#   vse8.v      : runs at vl = 8, stores 8 B
#   addi/bnez   : the loop counter
# => 100 * (8 B loaded, 16 B stored) over 300 vector instructions, all at vl = 8.
loop:
    vsetivli t1, 16, e8, m1, ta, ma
    vle8ff.v v0, (s1)
    vse8.v v0, (s0)
    vsetvli zero, zero, e8, m1, ta, ma
    vse8.v v0, (s0)
    addi t4, t4, -1
    bnez t4, loop

end:
    li a7, 93                       # __NR_exit
    li a0, 0
    ecall

fail:
    # A failed mmap/munmap would leave the ff load never faulting, i.e. a test
    # that passes for the wrong reason. Exit non-zero so the harness sees it.
    li a7, 93
    li a0, 1
    ecall
