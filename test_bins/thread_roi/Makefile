CC = clang
CFLAGS = --target=riscv64-linux-gnu --sysroot=/usr/riscv64-linux-gnu -O2 -g -static -I../roi_validation/rave
TARGET = pthreads_roi.riscv

all: $(TARGET)

$(TARGET): pthreads_roi.c
	$(CC) $(CFLAGS) -o $(TARGET) pthreads_roi.c -lpthread

clean:
	rm -f $(TARGET)

.PHONY: all clean
