#include <stddef.h>
#include "common.h"


void vadd_scalar(const float *a, const float *b, float *c, size_t len) {
    for (size_t i=0; i<len; i++) {
        c[i] = a[i] + b[i];
    }
}