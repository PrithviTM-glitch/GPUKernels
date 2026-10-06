#ifndef COMMON_H
#define COMMON_H

#include <stddef.h>
#include <stdint.h>

#define ALIGNMENT 128

/* Kernels (n_*.c) */
void vadd_scalar(const float *a, const float *b, float *c, size_t len);
void vadd_neon(const float *a, const float *b, float *c, size_t len);
void vadd_threads(const float *a, const float *b, float *c, size_t len,
                 int nthreads, size_t chunk);

/* Helpers harness.c */
float *alloc_aligned(size_t count);
uint64_t now_ns(void);
int verify(const float *c, const float *ref, size_t len);

#endif /* COMMON_H */