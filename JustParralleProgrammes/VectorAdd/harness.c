#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "common.h"

float *alloc_aligned(size_t count) {
    size_t bytes = count * sizeof(float);
    bytes = (bytes + ALIGNMENT - 1) / ALIGNMENT * ALIGNMENT; //ensure multiple of ALIGNMENT
    if (bytes == 0) {
        bytes = ALIGNMENT;
    }

    float *p = aligned_alloc(ALIGNMENT, bytes); //Ensure starting address is multiple of ALIGNMENT and allocated size bytes
    if (!p) {
        fprintf(stderr, "alloc of %zu bytes failure\n", bytes);
        exit(EXIT_FAILURE); //
    }

    return p;
}

uint64_t now_ns(void) {
    return clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW); // fn specific to macos; If linux use clock_gettime
}

int verify(const float *c, const float *ref, size_t len) { // element by element comparison of an array
    for(size_t i=0; i<len; i++) {
        if (c[i] != ref[i]) {
            fprintf(stderr, "mismatch at %zu: got %f, expected %f\n", 
                i, c[i], ref[i]);
            return 1;
        }
    }
    return 0;
}