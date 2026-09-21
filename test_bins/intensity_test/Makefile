CC      := clang
TARGET  := --target=riscv64-unknown-linux-gnu
MARCH   := -march=rv64gcv
CFLAGS  := $(TARGET) $(MARCH) -nostartfiles -static

.PHONY: all clean

all: intensity_f64.riscv intensity_f32.riscv

intensity_f32.riscv: intensity_f32.s
	$(CC) $(CFLAGS) $< -o $@

intensity_f64.riscv: intensity_f64.s
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f *.riscv
