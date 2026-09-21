CC     = clang
CFLAGS = --target=riscv64-unknown-linux-gnu --sysroot=/usr/riscv64-linux-gnu \
         --ld-path=/usr/bin/riscv64-linux-gnu-ld -O2 -g -static

all: loop.riscv pthreads.riscv openmp.riscv load_1gb.riscv

loop.riscv: loop.c
	$(CC) $(CFLAGS) -o $@ $<

pthreads.riscv: pthreads.c
	$(CC) $(CFLAGS) -o $@ $< -lpthread

# The ld warning about 'dlopen' in statically linked applications is expected.
# There is no clean way to silence it while keeping the binary statically linked.
openmp.riscv: openmp.c
	riscv64-linux-gnu-gcc -O2 -g -static -fopenmp -o $@ $<

load_1gb.riscv: load_1gb.c
	$(CC) $(CFLAGS) -march=rv64gcv -o $@ $<

clean:
	rm -f loop.riscv pthreads.riscv openmp.riscv load_1gb.riscv

.PHONY: all clean
