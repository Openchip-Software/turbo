CC      = clang
CFLAGS  = -march=rv64gcv --target=riscv64-linux-gnu -O3 -g -Wall -Wextra -Irave -static
ASFLAGS = -march=rv64gcv --target=riscv64-linux-gnu -g -fno-pie -fno-pic
LDFLAGS = -static -lm
ASLDFLAGS = -nostdlib -static
OUTPUT_DIR := bins

# V1 tests (old API)
V1_SRCS    := $(wildcard v1/*.c)
V1_BINS    := $(patsubst v1/%.c,$(OUTPUT_DIR)/v1_%.riscv,$(V1_SRCS))
V1_AS_SRCS := $(wildcard v1/*.S)
V1_AS_BINS := $(patsubst v1/%.S,$(OUTPUT_DIR)/v1_%.riscv,$(V1_AS_SRCS))

# V2 tests (new API)
V2_SRCS    := $(wildcard v2/*.c)
V2_BINS    := $(patsubst v2/%.c,$(OUTPUT_DIR)/v2_%.riscv,$(V2_SRCS))
V2_AS_SRCS := $(wildcard v2/*.S)
V2_AS_BINS := $(patsubst v2/%.S,$(OUTPUT_DIR)/v2_%.riscv,$(V2_AS_SRCS))

ALL_BINS := $(V1_BINS) $(V1_AS_BINS) $(V2_BINS) $(V2_AS_BINS)

.PHONY: all clean v1 v2

all: $(ALL_BINS)

v1: $(V1_BINS) $(V1_AS_BINS)

v2: $(V2_BINS) $(V2_AS_BINS)

$(OUTPUT_DIR)/v1_%.riscv: v1/%.c | $(OUTPUT_DIR)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(OUTPUT_DIR)/v2_%.riscv: v2/%.c | $(OUTPUT_DIR)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(OUTPUT_DIR)/v1_%.riscv: v1/%.S | $(OUTPUT_DIR)
	$(CC) $(ASFLAGS) -o $@ $< $(ASLDFLAGS)

$(OUTPUT_DIR)/v2_%.riscv: v2/%.S | $(OUTPUT_DIR)
	$(CC) $(ASFLAGS) -o $@ $< $(ASLDFLAGS)

$(OUTPUT_DIR):
	mkdir -p $@

clean:
	rm -f $(OUTPUT_DIR)/v1_*.riscv $(OUTPUT_DIR)/v2_*.riscv
