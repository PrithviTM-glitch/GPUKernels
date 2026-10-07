#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "common.h"

#define WARMUPS 2
#define RUNS 10 
#define MAX_THREADS 12

/* Define a kernel table for all single-threaded kernels so we can loop easily */
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

/* Chunk sizes (in elements) for the dynamic threaded scheduler */
static const size_t chunks[] = { 64u<<10, 256u<<10 };
static const size_t num_chunks = sizeof(chunks) / sizeof(chunks[0]);

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

/* Threaded kernel has extra arguments, so wrap it to fit vadd_fn.
 * The thread count and chunk size are set through these two variables. */
static int g_nthreads = 1;
static size_t g_chunk = 0;

static void vadd_threads_adapter(const float *a, const float *b, float *c, size_t len) {
    vadd_threads(a, b, c, len, g_nthreads, g_chunk);
}

/* Warm up, then time RUNS runs of one configuration.
 * Writes one CSV row per run to stdout, progress to stderr.
 * Returns 0 on success, 1 on a correctness failure. */
static int run_config(const char *name, vadd_fn fn, int nthreads, size_t chunk,
                      const float *a, const float *b, float *c,
                      const float *ref, size_t len) {
    /* Warm-up iterations: absorb page faults and clock ramp-up, not recorded */
    for (int w = 0; w < WARMUPS; w++) {
        poison(c, len);
        time_kernel(fn, a, b, c, len);
    }

    /* Timed benchmark runs */
    for (int r = 0; r < RUNS; r++) {
        poison(c, len); // Outside timed window

        uint64_t ns = time_kernel(fn, a, b, c, len);

        /* Verify output correctness */
        if (verify(c, ref, len) != 0) {
            fprintf(stderr,
                    "FATAL: correctness failure in '%s' len=%zu nthreads=%d chunk=%zu\n",
                    name, len, nthreads, chunk);
            return 1;
        }

        /* CSV row to stdout (chunk is 0 for non-threaded kernels) */
        printf("%s,%zu,%d,%zu,%d,%llu\n", name, len, nthreads, chunk, r,
               (unsigned long long)ns);

        /* Progress line to stderr (won't pollute the redirected CSV file) */
        fprintf(stderr,
                "[BENCH] %-13s len=%-9zu nt=%-2d chunk=%-7zu run %2d/%d  %10llu ns\n",
                name, len, nthreads, chunk, r + 1, RUNS, (unsigned long long)ns);
    }
    return 0;
}

int main(void) {
    /* Write CSV header to stdout */
    printf("kernel,len,nthreads,chunk,run,time_ns\n");

    for (size_t l = 0; l < num_lens; l++) {
        size_t len = lens[l];

        /* Allocate aligned memory buffers */
        float *a = alloc_aligned(len);
        float *b = alloc_aligned(len);
        float *c = alloc_aligned(len);
        float *ref = alloc_aligned(len);

        fill_inputs(a, b, len);
        compute_ref(a, b, ref, len);

        int rc = 0;

        /* Single-threaded kernels: scalar, neon */
        for (size_t k = 0; k < num_kernels && !rc; k++) {
            rc = run_config(kernels[k].name, kernels[k].fn, 1, 0,
                            a, b, c, ref, len);
        }

        /* Threaded NEON: sweep chunk size and thread count */
        for (size_t ch = 0; ch < num_chunks && !rc; ch++) {
            for (int nt = 1; nt <= MAX_THREADS && !rc; nt++) {
                g_nthreads = nt;
                g_chunk = chunks[ch];
                rc = run_config("neon_threads", vadd_threads_adapter,
                                nt, chunks[ch], a, b, c, ref, len);
            }
        }

        /* Free memory for this vector size */
        free(a); free(b); free(c); free(ref);

        if (rc) return EXIT_FAILURE;
    }

    fprintf(stderr, "\nBenchmark complete.\n");
    return EXIT_SUCCESS;
}