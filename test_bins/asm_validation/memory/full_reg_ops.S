        .section .data
        .align 8
buf:    .fill 512, 1, 0         # 256-byte scratch buffer (2*vlenb for VLEN=1024)

        .text
        .globl _start
_start:
        la      a0, buf

load_whole_reg:
        vsetvli t2, zero, e32, m1, ta, ma # 16384 VLEN / SEW 32 = vl 512
        # vl and SEW not considered for register load
        vl2re64.v   v8, (a0)        # LOAD whole-reg : 2 registers * 2048 vlenb = 4096 bytes

store_whole_reg:
        vsetvli t2, zero, e32, m1, ta, ma # 16384 VLEN / SEW 32 = vl 512
        vs1r.v      v8, (a0)        # STORE whole-reg : 1 register * 2048 vlenb = 2048 bytes

load_normal:
        vsetvli t2, zero, e64, m1, ta, ma   # 16384 VLEN / SEW 64 = vl 256 
        vle64.v     v8, (a0)        # LOAD normal : 256 vl * 64 sew = 16384 bits = 2048 bytes

store_normal:
        vsetvli t2, zero, e32, m1, ta, ma   # 16384 VLEN / SEW 32 = vl 512 
        vse32.v     v8, (a0)        # STORE normal : 512 vl * 32 sew = 16384 bits = 2048 bytes

exit:
        # ── exit(0) ───────────────────────────────────────────────────
        li      a7, 93
        li      a0, 0
        ecall
