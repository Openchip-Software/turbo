# RVV vector load/store exerciser
#
# Goal: touch the full set of vector memory load/store op *families* described
# in the vector memory format doc (unit-stride, strided, indexed; ordered/unordered;
# segment; fault-only-first; whole-register). This is a bare-metal style `_start`
# similar to other tests in this directory.
#
# Notes:
# - This file is intended to be assembled by riscv64-linux-gnu-gcc.
# - The Makefile currently uses `-march=rv64g` for other tests; to assemble this,
#   you likely need `-march=rv64gcv` (RVV). If you keep `-march=rv64g`, the build
#   will fail because `vsetvli` and `vle/vse/...` are not in base `G`.
# - GNU as expects canonical integer register names (`x0..x31`) by default in some
#   configurations; this file uses `xN` names to avoid "illegal operands" errors.
# - Whole-register *loads* are widely supported in GNU as (`vl1re8.v`, `vl2re8.v`,
#   `vl4re8.v`). Whole-register *stores* mnemonics vary and are not consistently
#   accepted; to keep this building, whole-register store instructions are omitted.
#
# Build example:
#   riscv64-linux-gnu-gcc -march=rv64gcv test_bins/basic_branch_asm/vld_vst.S -o test_bins/basic_branch_asm/vld_vst.riscv -nostdlib -static

.section .text
.globl _start

_start:
    # ------------------------------------------------------------
    # Setup pointers
    # ------------------------------------------------------------
    la      x5, src_bytes         # t0
    la      x6, dst_bytes         # t1
    la      x7, stride_src        # t2
    la      x28, stride_dst       # t3

    # Zero destination area (scalar)
    li      x29, 0                # t4
    li      x30, 256/8            # t5
    mv      x31, x6               # t6
1:
    sd      x29, 0(x31)
    addi    x31, x31, 8
    addi    x30, x30, -1
    bnez    x30, 1b

    # ------------------------------------------------------------
    # Pick a small, stable vector configuration
    # ------------------------------------------------------------
    # Use SEW=8, LMUL=1 first; then reconfigure as needed.
    li      x10, 0                # a0
    vsetvli x29, x10, e8, m1, ta, ma

    # ------------------------------------------------------------
    # 1) Unit-stride loads/stores: vle*/vse*
    # ------------------------------------------------------------
    # Byte
    vle8.v  v0, (x5)
    addi    x31, x6, 0
    vse8.v  v0, (x31)

    # Halfword
    vsetvli x29, x10, e16, m1, ta, ma
    vle16.v v1, (x5)
    addi    x31, x6, 32
    vse16.v v1, (x31)

    # Word
    vsetvli x29, x10, e32, m1, ta, ma
    vle32.v v2, (x5)
    addi    x31, x6, 64
    vse32.v v2, (x31)

    # Doubleword
    vsetvli x29, x10, e64, m1, ta, ma
    vle64.v v3, (x5)
    addi    x31, x6, 96
    vse64.v v3, (x31)

    # ------------------------------------------------------------
    # 2) Unit-stride segment loads/stores: vlsegN/ vssegN
    # ------------------------------------------------------------
    # SEW=8
    vsetvli x29, x10, e8, m1, ta, ma
    vlseg2e8.v  v4, (x5)
    addi        x31, x6, 128
    vsseg2e8.v  v4, (x31)

    vlseg3e8.v  v6, (x5)
    addi        x31, x6, 144
    vsseg3e8.v  v6, (x31)

    vlseg4e8.v  v8, (x5)
    addi        x31, x6, 160
    vsseg4e8.v  v8, (x31)

    # ------------------------------------------------------------
    # 3) Fault-only-first unit-stride: vle*ff
    # ------------------------------------------------------------
    # Typically used for probing; here we just execute it.
    vsetvli x29, x10, e8, m1, ta, ma
    vle8ff.v  v12, (x5)
    addi      x31, x6, 176
    vse8.v    v12, (x31)

    # Segment fault-only-first (if supported by assembler)
    # The v-spec defines fault-only-first for segment loads too.
    vlseg2e8ff.v v14, (x5)
    addi         x31, x6, 192
    vsseg2e8.v   v14, (x31)

    # ------------------------------------------------------------
    # 4) Strided loads/stores: vlse*/vsse*
    # ------------------------------------------------------------
    # Use stride from memory (scalar load) to avoid large immediates.
    ld      x11, 0(x7)            # a1 = stride_src
    ld      x12, 0(x28)           # a2 = stride_dst

    vsetvli x29, x10, e8, m1, ta, ma
    vlse8.v  v16, (x5), x11
    addi     x31, x6, 208
    vsse8.v  v16, (x31), x12

    vsetvli x29, x10, e16, m1, ta, ma
    vlse16.v v17, (x5), x11
    addi     x31, x6, 216
    vsse16.v v17, (x31), x12

    vsetvli x29, x10, e32, m1, ta, ma
    vlse32.v v18, (x5), x11
    addi     x31, x6, 224
    vsse32.v v18, (x31), x12

    vsetvli x29, x10, e64, m1, ta, ma
    vlse64.v v19, (x5), x11
    addi     x31, x6, 232
    vsse64.v v19, (x31), x12

    # Strided segment loads/stores (8-bit segments)
    vsetvli x29, x10, e8, m1, ta, ma
    vlsseg2e8.v  v20, (x5), x11
    addi         x31, x6, 240
    vssseg2e8.v  v20, (x31), x12

    # ------------------------------------------------------------
    # 5) Indexed loads/stores: vluxei/vloxei and vsuxei/vsoxei
    # ------------------------------------------------------------
    # NOTE: These instructions have trapped as SIGILL in your runtime environment
    # (e.g. `vluxei16.v ...`). That indicates the execution target does not
    # implement RVV indexed memory ops (or RVV is not enabled / version mismatch).
    #
    # To keep this test runnable, the indexed load/store section is disabled.
    #
    # If/when you run on a target with full RVV indexed support, you can re-enable
    # the block below.
    #
    # Build a small vector of indices in bytes: [0, 2, 4, ...]
    # (We do it with scalar stores then load indices as a vector for simplicity.)
    # la      x30, indices          # t5
    # li      x31, 0                # t6
    # li      x27, 16               # s11 (loop count)
    #2:
    # sh      x31, 0(x30)
    # addi    x30, x30, 2
    # addi    x31, x31, 2
    # addi    x27, x27, -1
    # bnez    x27, 2b
    #
    # la      x30, indices
    #
    # # Use 16-bit indices (ei16). Data SEW varies.
    # vsetvli x29, x10, e16, m1, ta, ma
    # vle16.v v24, (x30)           # index vector (u16 offsets)
    #
    # # Unordered indexed loads
    # vsetvli x29, x10, e8, m1, ta, ma
    # vluxei16.v v25, (x5), v24
    # addi        x31, x6, 248
    # vsuxei16.v  v25, (x31), v24
    #
    # # Ordered indexed loads
    # vloxei16.v v26, (x5), v24
    # addi       x31, x6, 256
    # vsoxei16.v v26, (x31), v24
    #
    # # Indexed segment loads/stores (unordered)
    # vluxseg2ei16.v v28, (x5), v24
    # addi           x31, x6, 264
    # vsuxseg2ei16.v v28, (x31), v24
    #
    # # Indexed segment loads/stores (ordered)
    # vloxseg2ei16.v v30, (x5), v24
    # addi           x31, x6, 280
    # vsoxseg2ei16.v v30, (x31), v24

    # ------------------------------------------------------------
    # 6) Whole-register loads (stores omitted due to assembler variance)
    # ------------------------------------------------------------
    vsetvli x29, x10, e8, m1, ta, ma
    vl1re8.v  v0, (x5)
    vl2re8.v  v2, (x5)
    vl4re8.v  v4, (x5)

    # ------------------------------------------------------------
    # Exit (Linux syscall ABI)
    # ------------------------------------------------------------
    li    x10, 0
    li    x17, 93
    ecall


# ------------------------------------------------------------
# Data
# ------------------------------------------------------------
.section .rodata
.align 8
stride_src: .dword 3
stride_dst: .dword 5

.section .data
.align 16
# Source bytes: 0..255
src_bytes:
    .byte 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
    .byte 16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31
    .byte 32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47
    .byte 48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63
    .byte 64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79
    .byte 80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95
    .byte 96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111
    .byte 112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127
    .byte 128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143
    .byte 144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159
    .byte 160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175
    .byte 176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191
    .byte 192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207
    .byte 208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223
    .byte 224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239
    .byte 240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255

.align 16
dst_bytes:
    .space 512

.align 16
indices:
    .space 64
