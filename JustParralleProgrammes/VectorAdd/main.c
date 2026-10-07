#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "common.h"

#define WARMUPS 2
#define RUNS 10 

/* Define a kernel table for all kernels so we can loop easily */ 
typedef void (*vadd_fn)(const float *, const float *, float *, size_t);

typedef struct {
    const char *name;
    vadd_fn fn;
} kernel_t;

static const kernel_t kernels[] = {
    { "scalar", vadd_scalar },
    { "neon", vadd_neon },
    // { "threads", vadd_thread }
};

static const size_t num_kernels = sizeof(kernels) / sizeof(kernels[0]);

static const size_t lens[] = { 8u<<10, 512u<<10, 4u<<20, 32u<<20 };
static const size_t num_lens = sizeof(lens) / sizeof(lens[0]);

/* Populate input vectors with deterministic test data */
static void fill_inputs(float *a, float *b, size_t len) {
    for (size_t i = 0; i < len; i++) {
        a[i] = (float)(i % 1000);
        b[i] = 2.0f * a[i];
    }
}

/* Compute expected results using plain scalar addition */
static void compute_ref(const float *a, const float *b, float *ref, size_t len) {
    for (size_t i = 0; i < len; i++) {
        ref[i] = a[i] + b[i];
    }
}

/* Fill destination buffer with quiet NaNs to ensure true overwriting */
static void poison(float *c, size_t len) {
    for (size_t i = 0; i < len; i++) {
        c[i] = NAN;
    }
}

/* Time ONE call of a kernel, return elapsed nanoseconds */
static uint64_t time_kernel(vadd_fn fn, const float *a, const float *b,
                            float *c, size_t len) {
    uint64_t t0 = now_ns();
    fn(a, b, c, len);
    uint64_t t1 = now_ns();
    return t1 - t0;
}

int main (void) {
    // Write CSV header to stdout
    printf("kernel,len,nthreads,run,time_ns\n");

    for (size_t l = 0; l < num_lens; l++) {
        size_t len = lens[l];

        /* Allocate aligned memory buffers */
        float *a   = alloc_aligned(len);
        float *b   = alloc_aligned(len);
        float *c   = alloc_aligned(len);
        float *ref = alloc_aligned(len);

        fill_inputs(a, b, len);
        compute_ref(a, b, ref, len);

        for (size_t k = 0; k < num_kernels; k++) {
            kernel_t kernel = kernels[k];

            /* Warm-up iterations: fill cache lines, not recorded */
            for (int w = 0; w < WARMUPS; w++) {
                poison(c, len);
                time_kernel(kernel.fn, a, b, c, len);
            }
            /* Timed benchmark runs */
            for (int r = 0; r < RUNS; r++) {
                poison(c, len); // Outside timed window

                uint64_t ns = time_kernel(kernel.fn, a, b, c, len);

                /* Verify output correctness */
                if (verify(c, ref, len) != 0) {
                    fprintf(stderr, "FATAL: Correctness failure in kernel '%s' at len=%zu!\n",
                            kernel.name, len);
                    free(a); free(b); free(c); free(ref);
                    return EXIT_FAILURE;
                }

                /* Output CSV row directly to stdout */
                printf("%s,%zu,1,%d,%llu\n", kernel.name, len, r, (unsigned long long)ns);

                /* Progress line to stderr (won't pollute the redirected CSV file) */
                fprintf(stderr, "[BENCH] Kernel: %-8s | Len: %-8zu | Run: %2d/%d | Time: %10llu ns\n",
                        kernel.name, len, r + 1, RUNS, (unsigned long long)ns);
            }
            }

        /* Free memory for this vector size */
        free(a); free(b); free(c); free(ref);
    }
    fprintf(stderr, "\nBenchmark complete.\n");
    return EXIT_SUCCESS;
}