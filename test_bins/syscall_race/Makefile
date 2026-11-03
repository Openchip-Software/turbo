CC = clang

# clang emits RISC-V objects but defaults to the *host* linker (which cannot
# link elf64lriscv) and the *host* headers, so both have to be pointed at a
# cross toolchain. Which one depends on where we are building -- see
# test_bins/linker_flavours/Makefile for the full story:
#
#   CI  - a self-contained riscv64-unknown-linux-gnu GNU toolchain that clang
#         picks up by itself via bin/<triple>.cfg, so adding our own flags on
#         top would only override it.
#   dev - Debian's gcc-riscv64-linux-gnu + libc6[-dev]-riscv64-cross, which
#         does need the explicit --sysroot/--ld-path below.
#
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
CFLAGS  = --target=riscv64-unknown-linux-gnu $(TCFLAGS) -O2 -g -static

all: syscall_translate_race.riscv

syscall_translate_race.riscv: syscall_translate_race.c
	$(CC) $(CFLAGS) -o $@ $< -lpthread

clean:
	rm -f syscall_translate_race.riscv

.PHONY: all clean
