#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <riscv_vector.h>

#define GB_SIZE (1UL * 1024 * 1024 * 1024)  // 1 GB in bytes, default size

__attribute__((noinline))
__attribute__((target("arch=rv64g")))
uint64_t scalar_sum(const uint64_t *data, size_t count) {
    uint64_t total = 0;
    for (size_t i = 0; i < count; i++)
        total += data[i];
    return total;
}

__attribute__((noinline))
uint64_t scalar_v_isa_sum(const uint64_t *data, size_t count) {
    uint64_t total = 0;
    for (size_t i = 0; i < count; i++)
        total += data[i];
    return total;
}

__attribute__((noinline))
uint64_t vector_sum(const uint64_t *data, size_t count) {
    size_t remaining = count;
    const uint64_t *ptr = data;

    // Zero-initialised u64 accumulator across all max_vl lanes
    size_t max_vl = __riscv_vsetvlmax_e64m1();
    vuint64m1_t acc = __riscv_vmv_v_x_u64m1(0, max_vl);

    // Single loop: vsetvl handles tail naturally. _tu leaves upper acc lanes
    // undisturbed (already zero) so they don't pollute the final reduction.
    while (remaining > 0) {
        size_t vl     = __riscv_vsetvl_e64m1(remaining);
        vuint64m1_t v = __riscv_vle64_v_u64m1(ptr, vl);
        acc           = __riscv_vadd_vv_u64m1_tu(acc, acc, v, vl);
        ptr          += vl;
        remaining    -= vl;
    }

    // Single reduction across all max_vl lanes
    vuint64m1_t zero = __riscv_vmv_s_x_u64m1(0, 1);
    vuint64m1_t red  = __riscv_vredsum_vs_u64m1_u64m1(acc, zero, max_vl);
    return __riscv_vmv_x_s_u64m1_u64(red);
}

int main(int argc, char **argv) {
    // Default behaviour (no args): allocate 1GB, run only the vector sum.
    // --size <MB>  : allocate <MB> megabytes instead (e.g. 2 for CI runs).
    // --validate   : also run the scalar sums and check they agree.
    size_t bytes = GB_SIZE;
    int validate = 0;

    for (int a = 1; a < argc; a++) {
        if (strcmp(argv[a], "--size") == 0 && a + 1 < argc) {
            bytes = (size_t)strtoull(argv[++a], NULL, 10) * 1024 * 1024;
        } else if (strcmp(argv[a], "--validate") == 0) {
            validate = 1;
        }
    }

    size_t elem_count = bytes / sizeof(uint64_t);

    uint64_t *data = (uint64_t *)malloc(bytes);
    if (!data) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    // initialize data and validate only if requested
    if (validate) {
        for (size_t i = 0; i < elem_count; i++) {
            data[i] = (uint64_t)i & 0xF;
        }
    }

    // ~3.67M instructions in this function for 1GB
    uint64_t vsum = vector_sum(data, elem_count);

    if (validate) {
        printf("validating with scalar and scalar_v_isa functions\n");
        uint64_t ssum = scalar_sum(data, elem_count);
        uint64_t s_isa_sum = scalar_v_isa_sum(data, elem_count);

        if (vsum != ssum || vsum != s_isa_sum) {
            fprintf(stderr, "MISMATCH: vector=%lu scalar=%lu scalar_v_isa=%lu\n", vsum, ssum, s_isa_sum);
            free(data);
            return 1;
        }
        printf("sum = %lu  (vector and scalar agree)\n", vsum);
    } else {
        printf("sum = %lu  (only vector executed)\n", vsum);
    }

    free(data);
    return 0;
}
