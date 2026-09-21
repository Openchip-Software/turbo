CC      = clang
OBJDUMP = riscv64-linux-gnu-objdump

# clang emits RISC-V objects but defaults to the *host* linker (which cannot
# link elf64lriscv) and the *host* headers, so both have to be pointed at a
# cross toolchain. Which one depends on where we are building:
#
#   CI  - a full riscv64-unknown-linux-gnu GNU toolchain is unpacked under
#         $TOOLS_DIR and clang picks up its sysroot + gcc-toolchain + ld from
#         $TOOLS_DIR/bin/riscv64-unknown-linux-gnu.cfg automatically. Adding
#         our own --sysroot on top of that pointed at Debian's cross packages
#         is what broke the dynamic flavours: Debian's /usr/riscv64-linux-gnu
#         is not a real sysroot (the shared libc lives in .../lib, the linker
#         scripts in .../usr/lib refer to unprefixed /lib paths), so ld could
#         not resolve libc.so.6 / libc_nonshared.a / ld-linux-*.so.1 for any
#         non-static link.
#   dev - Debian's gcc-riscv64-linux-gnu + libc6[-dev]-riscv64-cross, which
#         does need the explicit --sysroot/--ld-path below.
#
# Detect a self-contained toolchain and stay out of its way; otherwise fall
# back to the Debian layout. A self-contained one ships its own sysroot and
# clang finds it via bin/<triple>.cfg, so adding our flags would only override
# it: CI publishes it as $TOOLS_DIR/riscv64-unknown-linux-gnu, the
# riscv_tools_env container exports $GNU_TOOLCHAIN_ROOT from ~/environment.sh.
# Either can be forced explicitly, e.g.
#   make SYSROOT=/path/to/sysroot LDPATH=/path/to/riscv64-ld
#   make SYSROOT= LDPATH=            # trust clang's own config file
ifeq ($(wildcard $(TOOLS_DIR)/riscv64-unknown-linux-gnu/sysroot $(GNU_TOOLCHAIN_ROOT)/sysroot),)
SYSROOT ?= /usr/riscv64-linux-gnu
LDPATH  ?= /usr/bin/riscv64-linux-gnu-ld
else
SYSROOT ?=
LDPATH  ?=
endif

TCFLAGS = $(if $(SYSROOT),--sysroot=$(SYSROOT)) $(if $(LDPATH),--ld-path=$(LDPATH))
CFLAGS  = --target=riscv64-unknown-linux-gnu $(TCFLAGS) \
          -O2 -g -march=rv64gcv -Wall -Wextra

# Flavours of the same program, all built from main.c (+ add.c), chosen to
# exercise every RISC-V ELF linking mode the etrace plugin's memory-map
# discovery has to cope with (main-binary base, shared libs, dynamic linker,
# vDSO). See crates/turbo/tests/basic.rs for the matching decode tests.
#
#   1. static_plain    - static ET_EXEC, no vDSO, no shared lib (simplest case)
#   2. static_gtod     - static ET_EXEC + gettimeofday()  (vDSO from a static binary)
#   3. dyn_gtod        - dynamic PIE + gettimeofday()      (PIE main + ld.so + vDSO)
#   4. dyn_addso       - dynamic PIE + libadd.so (NEEDED)  (shared lib mapped at loader init)
#   5. dyn_addso_gtod  - dynamic PIE + libadd_gtod.so      (gettimeofday in BOTH main and add.so)
#   6. static_pie      - static ET_DYN, NO interpreter     (PIE-bias main, no ld.so, no NEEDED)
#   7. dyn_nopie       - dynamic ET_EXEC (-no-pie) + gtod   (fixed-address main + ld.so + vDSO)
#   8. dyn_sepcode     - dynamic PIE, -z separate-code      (split code/rodata segment layout)
#   9. dyn_dlopen      - dynamic PIE, dlopen(libadd.so)     (shared lib mapped AFTER startup)
#  10. dyn_addso_reloc - dynamic PIE + libadd.so, NO rpath  (lib lives in libs/, needs LD_LIBRARY_PATH)
#
TARGETS = static_plain.riscv \
          static_gtod.riscv \
          dyn_gtod.riscv \
          dyn_addso.riscv \
          dyn_addso_gtod.riscv \
          static_pie.riscv \
          dyn_nopie.riscv \
          dyn_sepcode.riscv \
          dyn_dlopen.riscv \
          dyn_addso_reloc.riscv

SHLIBS  = libadd.so libadd_gtod.so libs/libadd.so

.PHONY: all run objdump clean
all: $(TARGETS)

# --- shared libraries ----------------------------------------------------
# libadd.so       : add() only
# libadd_gtod.so  : add() that also calls gettimeofday()
libadd.so: add.c
	$(CC) $(CFLAGS) -fPIC -shared -o $@ $<

libadd_gtod.so: add.c
	$(CC) $(CFLAGS) -DADD_USE_GETTIMEOFDAY -fPIC -shared -o $@ $<

# --- 1. static, plain ----------------------------------------------------
static_plain.riscv: main.c
	$(CC) $(CFLAGS) -static -o $@ $<

# --- 2. static + gettimeofday() ------------------------------------------
static_gtod.riscv: main.c
	$(CC) $(CFLAGS) -DUSE_GETTIMEOFDAY -static -o $@ $<

# --- 3. dynamic (PIE) + gettimeofday() -----------------------------------
dyn_gtod.riscv: main.c
	$(CC) $(CFLAGS) -DUSE_GETTIMEOFDAY -o $@ $<

# --- 4. dynamic (PIE) + libadd.so (no gettimeofday) ----------------------
# -L. to find the lib at link time; -Wl,-rpath,'$$ORIGIN' so the loader finds
# it next to the binary at run time without needing LD_LIBRARY_PATH.
dyn_addso.riscv: main.c libadd.so
	$(CC) $(CFLAGS) -DUSE_ADD_SO -o $@ $< -L. -ladd -Wl,-rpath,'$$ORIGIN'

# --- 5. dynamic (PIE) + libadd_gtod.so (gettimeofday in main AND add.so) -
dyn_addso_gtod.riscv: main.c libadd_gtod.so
	$(CC) $(CFLAGS) -DUSE_ADD_SO -DUSE_GETTIMEOFDAY -o $@ $< -L. -ladd_gtod -Wl,-rpath,'$$ORIGIN'

# --- 6. static-PIE (ET_DYN, no interpreter) ------------------------------
# Loaded at a nonzero bias like a shared object, but has no PT_INTERP and pulls
# in no NEEDED libs: the main binary's PCs are only resolvable if its runtime
# load base (start_code vs link-time vaddr) is recovered correctly.
static_pie.riscv: main.c
	$(CC) $(CFLAGS) -DUSE_GETTIMEOFDAY -static-pie -o $@ $<

# --- 7. non-PIE dynamic (ET_EXEC + interpreter) --------------------------
# Main binary sits at fixed absolute addresses (base 0), while ld.so, libc and
# the vDSO are still mapped elsewhere. All prior dynamic flavours are PIE, so
# this is the only one whose main binary has a zero load bias yet still runs a
# dynamic linker.
dyn_nopie.riscv: main.c
	$(CC) $(CFLAGS) -DUSE_GETTIMEOFDAY -no-pie -o $@ $<

# --- 8. dynamic (PIE), -z separate-code ----------------------------------
# Forces separate R / R-E / R-W PT_LOAD segments with alignment gaps, so the
# executable text no longer sits at file offset 0 of the mapping -- stresses
# the main-binary code-range/offset derivation.
dyn_sepcode.riscv: main.c
	$(CC) $(CFLAGS) -DUSE_GETTIMEOFDAY -Wl,-z,separate-code -o $@ $<

# --- 9. dynamic (PIE), dlopen(libadd.so) at run time ---------------------
# Unlike dyn_addso (NEEDED, mapped by ld.so at init), this maps libadd.so only
# when main() calls dlopen(), i.e. mid-run -- the case the plugin's openat/mmap
# syscall discovery must catch on the live path.
dyn_dlopen.riscv: main.c libadd.so
	$(CC) $(CFLAGS) -DUSE_DLOPEN -o $@ $< -ldl -Wl,-rpath,'$$ORIGIN'

# --- 10. dynamic (PIE) + libadd.so in libs/, NO rpath ---------------------
# The .so is deliberately NOT next to the binary and the binary has NO rpath,
# so the dynamic loader can only find it via LD_LIBRARY_PATH. Used by the
# LD_LIBRARY_PATH integration test: without the env var the guest fails to
# start (libadd.so unresolvable); with it pointing at libs/ it runs.
libs/libadd.so: add.c
	mkdir -p libs
	$(CC) $(CFLAGS) -fPIC -shared -o $@ $<

dyn_addso_reloc.riscv: main.c libs/libadd.so
	$(CC) $(CFLAGS) -DUSE_ADD_SO -o $@ $< -Llibs -ladd

# --- run all flavours under qemu -----------------------------------------
run: all
	@for t in $(TARGETS); do \
	  echo "=== $$t ==="; \
	  qemu-riscv64 $(if $(SYSROOT),-L $(SYSROOT)) ./$$t; \
	done

# --- linkage inspection --------------------------------------------------
# Quick proof each flavour is linked as intended (static vs dynamic, NEEDED libs).
objdump: all
	@for t in $(TARGETS); do \
	  echo "=== $$t ==="; \
	  $(OBJDUMP) -p ./$$t | grep -E 'NEEDED|INTERP' || echo "  (no INTERP/NEEDED: static)"; \
	done

clean:
	rm -f $(TARGETS) $(SHLIBS)
	rm -rf libs
