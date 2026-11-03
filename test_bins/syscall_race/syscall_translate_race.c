/*
 * Stress test for an AB-BA deadlock between the tracer's syscall callbacks
 * and translation, fixed in turbo-tracer commit:
 * "fix deadlock between syscall callbacks and translation"
 *
 * Under QEMU user mode every guest-memory read the plugin makes takes
 * mmap_lock, and every translation holds mmap_lock while it waits for the
 * plugin to run its translate callback. If a syscall callback reads guest
 * memory while holding a lock the translate callback also needs, one guest
 * thread in openat/mmap plus one translating thread deadlock QEMU.
 *
 * So this program keeps both sides busy at the same moment:
 *
 *   - SYSCALL threads loop openat + mmap(offset 0) + munmap + close on the
 *     binary itself: the openat path and the ELF magic of the mapping are
 *     exactly the two reads the tracer makes.
 *   - JIT threads rewrite their own code page and call it, every iteration.
 *     The write invalidates the translated block, so the next call
 *     translates it again: a translation per iteration rather than only
 *     on first execution. The bytes written are always the same, so the
 *     tracer's block dictionary does not grow.
 *
 * All threads start together on a barrier to maximise the overlap.
 * Exit code 0 means every iteration ran; the test's watchdog catches a hang.
 */
#define _GNU_SOURCE
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define SYSCALL_THREADS 4
#define JIT_THREADS 4
#define ITERS 3000

static pthread_barrier_t start;

static void *syscall_thread(void *arg) {
    (void)arg;
    pthread_barrier_wait(&start);
    for (int i = 0; i < ITERS; i++) {
        int fd = open("/proc/self/exe", O_RDONLY);
        if (fd < 0) {
            perror("open");
            exit(2);
        }
        void *p = mmap(NULL, 4096, PROT_READ, MAP_PRIVATE, fd, 0);
        if (p == MAP_FAILED) {
            perror("mmap");
            exit(2);
        }
        munmap(p, 4096);
        close(fd);
    }
    return NULL;
}

/* addi a0, a0, 1 ; ret */
static const uint32_t code[] = {0x00150513, 0x00008067};

static void *jit_thread(void *arg) {
    (void)arg;
    uint32_t *page = mmap(NULL, 4096, PROT_READ | PROT_WRITE | PROT_EXEC,
                          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (page == MAP_FAILED) {
        perror("mmap jit");
        exit(2);
    }
    long (*fn)(long) = (long (*)(long))(uintptr_t)page;
    long acc = 0;
    pthread_barrier_wait(&start);
    for (int i = 0; i < ITERS; i++) {
        memcpy(page, code, sizeof code);
        __builtin___clear_cache((char *)page, (char *)page + sizeof code);
        acc = fn(acc);
    }
    if (acc != ITERS) {
        fprintf(stderr, "jit: acc %ld != %d\n", acc, ITERS);
        exit(3);
    }
    munmap(page, 4096);
    return NULL;
}

int main(void) {
    pthread_t t[SYSCALL_THREADS + JIT_THREADS];
    pthread_barrier_init(&start, NULL, SYSCALL_THREADS + JIT_THREADS);
    for (int i = 0; i < SYSCALL_THREADS + JIT_THREADS; i++)
        pthread_create(&t[i], NULL, i % 2 ? jit_thread : syscall_thread, NULL);
    for (int i = 0; i < SYSCALL_THREADS + JIT_THREADS; i++)
        pthread_join(t[i], NULL);
    puts("syscall_translate_race: ok");
    return 0;
}
