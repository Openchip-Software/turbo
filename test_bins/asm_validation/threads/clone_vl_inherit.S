# A spawned guest thread inherits the spawning thread's VL, and the tool must
# know it.
#
# QEMU linux-user makes a new thread's vCPU with cpu_copy(), which is a memcpy
# of the whole CPUArchState; cpu_clone_regs_child() then fixes up only sp and
# a0. So the child starts with the parent's `vl` and `vtype` verbatim, as of the
# instant of the clone -- a thread's vector state at entry is whatever the
# spawning thread happened to be running at, not zero.
#
# The tracer computes `vl` rather than reading the CSR back, from a per-vCPU
# shadow. That shadow used to be *zeroed* at vcpu init on the assumption that a
# new vCPU starts at vl = 0, which made two things go wrong on every threaded
# guest whose spawning thread left `vl` non-zero (glibc's vector memcpy inside
# printf is enough, so cloudsc tripped it):
#
#   * `vsetvli zero, zero` takes its AVL from the *current* vl, which the plugin
#     supplies from that shadow. From a zeroed shadow it computes vl = 0, so
#     every vector op behind it is counted at length 0.
#   * the vl oracle compares the shadow against the real CSR at each vsetvl*
#     site that reads a register, and a zeroed shadow disagrees -- aborting the
#     run outright under the vl_verify=abort default.
#
# This fixture pins both. The parent sets vl = 2 and clones with raw clone(2)
# rather than pthread_create, so the child's first executed instruction is ours:
# glibc's thread startup would run its own vsetvl* and overwrite the inherited
# value before we could observe it.

.section .text
.global _start

_start:
    # A page for the child to store into. CLONE_VM shares the address space, so
    # mapping it here is enough. Done before the vsetivli so that nothing
    # between "vl is set" and the clone can disturb vl.
    li a0, 0
    li a1, 4096
    li a2, 3                        # PROT_READ|PROT_WRITE
    li a3, 0x22                     # MAP_PRIVATE|MAP_ANONYMOUS
    li a4, -1
    li a5, 0
    li a7, 222                      # __NR_mmap
    ecall
    bltz a0, fail
    mv s2, a0                       # s2 = scratch page, for the child's stores

    # A 64 KiB stack for the child.
    li a0, 0
    li a1, 65536
    li a2, 3
    li a3, 0x22
    li a4, -1
    li a5, 0
    li a7, 222
    ecall
    bltz a0, fail
    li t0, 65536
    add s0, a0, t0                  # s0 = child stack top

    # vl = 2. This is the value the child must be seen to inherit; e8/m1 makes
    # it 2 for any VLEN, and the AVL is a vsetivli immediate so it does not
    # depend on the register file either.
    vsetivli t0, 2, e8, m1, ta, ma

    # clone(CLONE_VM|CLONE_FS|CLONE_FILES|CLONE_SIGHAND|CLONE_THREAD|CLONE_SYSVSEM,
    #       child_stack, NULL, 0, NULL).
    # QEMU's do_fork requires exactly that set (its CLONE_THREAD_FLAGS) to take
    # the cpu_copy() thread path; a subset is EINVAL, and no CLONE_VM at all is
    # a fork, which is a separate process and not what is under test here.
    li a0, 0x50f00
    mv a1, s0
    li a2, 0
    li a3, 0
    li a4, 0
    li a7, 220                      # __NR_clone
    ecall
    bltz a0, fail
    beqz a0, child

parent:
    # Spin until the child is done, then take the whole process down. The parent
    # runs no vector code from here on, so it cannot perturb the child's ROI.
    la s1, done_flag
1:
    lw t2, 0(s1)
    beqz t2, 1b
    li a7, 94                       # __NR_exit_group
    li a0, 0
    ecall

child:
    li t3, 100

# 100 iterations at the *inherited* vl, and nothing in the body ever names it:
# `vsetvli zero, zero` re-asserts the same vtype with AVL = the current vl, so
# the length the store runs at can only come from what the thread started with.
# Per iteration one vse8.v at vl = 2, i.e. 2 bytes and 2 elements.
loop:
    vsetvli zero, zero, e8, m1, ta, ma
    vse8.v v0, (s2)
    addi t3, t3, -1
    bnez t3, loop

end:
    # An R_REGS vsetvl* site, outside the ROI: this form reads its AVL from a
    # register, which is where the vl oracle compares the shadow against the
    # real CSR. With the shadow zeroed at vcpu init this is the abort. Its AVL
    # is 2, the value already in force, so it changes nothing it observes.
    li a5, 2
    vsetvli t1, a5, e8, m1, ta, ma

    la s1, done_flag
    li t2, 1
    sw t2, 0(s1)
    li a7, 93                       # __NR_exit
    li a0, 0
    ecall

fail:
    # A failed mmap or clone would leave a single-threaded run whose counters
    # look fine, i.e. a test passing for the wrong reason. Exit non-zero.
    li a7, 94
    li a0, 1
    ecall

.section .data
done_flag:
    .word 0
