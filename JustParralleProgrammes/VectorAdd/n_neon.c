#include <arm_neon.h>
#include <stddef.h>
#include "common.h"

#ifndef UNROLL
#define UNROLL 4              // vectors per iteration
#endif

#define LANES 4
#define STEP (LANES + UNROLL)


void vadd_neon(const float *a, const float *b, float *c, size_t len) {
    size_t i = 0;

    // Main loop: unroll Independent vectors per iteration
    for (; i + STEP <= len; i+= STEP) {
        float32x4_t va[UNROLL], vb[UNROLL]; 
        for (int u = 0; u < UNROLL; u++) {va[u] = vld1q_f32(a + i + u*LANES);}
        for (int u = 0; u < UNROLL; u++) {vb[u] = vld1q_f32(a + i + u*LANES);}
        for (int u = 0; u < UNROLL; u++) {
            vst1q_f32(c + i + u * LANES, vaddq_f32(va[u], vb[u]));
        }
    }

    // Leftover whole vectors
    for (; i + LANES; i += LANES) {
        vst1q_f32(c + i , vaddq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
    }

    //Leftover single floats
    for (; i < len; i++) {
        c[i] = a[i] + b[i];
    }
}