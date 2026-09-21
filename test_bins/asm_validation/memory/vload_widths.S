# Element-wise vector load/store byte accounting across every EEW.
#
# Sweeps e8/e16/e32/e64 for unit-stride, strided and fault-only-first, plus the
# mask load/store pair. Existing tests only exercised e64 (and e32 incidentally),
# so a per-width byte-count error was invisible; likewise nothing covered
# vlse/vsse, the ff variants, or vlm/vsm at all.
#
# All indices/strides keep accesses inside the buffers. The ff loads never
# fault, so they count exactly as their non-ff counterparts.

.section .text
.global _start

_start:
    li t0, 64                       # AVL; below VLMAX for every SEW here
    la t2, source_data
    la t3, dest_data
    li t5, 8                        # byte stride for the strided ops

    li t4, 100

# Per iteration, vl = 64 throughout:
#   e8  : 3 loads * 64 =  192 B loaded, 2 stores * 64 =  128 B stored
#   e16 : 3 loads * 128 = 384 B loaded, 2 stores * 128 = 256 B stored
#   e32 : 3 loads * 256 = 768 B loaded, 2 stores * 256 = 512 B stored
#   e64 : 3 loads * 512 = 1536 B loaded, 2 stores * 512 = 1024 B stored
#   mask: ceil(64/8) = 8 B loaded, 8 B stored
#   => 2888 bytes loaded, 1928 bytes stored per iteration
loop:
    vsetvli t1, t0, e8, m1, ta, ma
    vle8.v v0, (t2)
    vse8.v v0, (t3)
    vlse8.v v1, (t2), t5
    vsse8.v v1, (t3), t5
    vle8ff.v v2, (t2)

    vsetvli t1, t0, e16, m1, ta, ma
    vle16.v v3, (t2)
    vse16.v v3, (t3)
    vlse16.v v4, (t2), t5
    vsse16.v v4, (t3), t5
    vle16ff.v v5, (t2)

    vsetvli t1, t0, e32, m1, ta, ma
    vle32.v v6, (t2)
    vse32.v v6, (t3)
    vlse32.v v7, (t2), t5
    vsse32.v v7, (t3), t5
    vle32ff.v v8, (t2)

    vsetvli t1, t0, e64, m1, ta, ma
    vle64.v v9, (t2)
    vse64.v v9, (t3)
    vlse64.v v10, (t2), t5
    vsse64.v v10, (t3), t5
    vle64ff.v v11, (t2)

    # Mask load/store: encoded as EEW=8 but run at evl = ceil(vl/8) = 8.
    vlm.v v12, (t2)
    vsm.v v12, (t3)

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
    .rept 512
    .dword 0x0123456789abcdef
    .endr

.section .bss
.align 8
dest_data:
    .skip 4096
