#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include "common.h"

typedef struct {
    const float *a, *b;
    float *c;
    size_t len, chunk;
    _Alignas(ALIGNMENT) _Atomic size_t next; // own cache line
} shared_t;

static void *worker(void *arg) {
    shared_t *s = arg;
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);

    for (;;) {
        size_t start = atomic_fetch_add_explicit(&s->next, s->chunk,
                                                 memory_order_relaxed);
        if (start >= s->len) break;
        size_t n = s->len - start;
        if (n > s->chunk) n = s->chunk;
        vadd_neon(s->a + start, s->b + start, s->c + start, n);
    }
    return NULL;
}

void vadd_threads(const float *a, const float *b, float *c, size_t len, int nthreads, size_t chunk) {
    if (nthreads < 1) nthreads = 1;
    if (chunk==0) chunk = len ? len : 1; // NO zero step counter

    shared_t sh = { .a = a, .b = b, .c = c, .len = len, .chunk = chunk };
    atomic_init(&sh.next, 0);

    pthread_t *tid = malloc((size_t)nthreads * sizeof *tid);
    if (!tid) { fprintf(stderr, "malloc failed\n"); exit(EXIT_FAILURE); }

    for (int t = 0; t < nthreads; t++) {
        if ( pthread_create(&tid[t], NULL, worker, &sh) != 0 ) {
            fprintf(stderr, "pthread creat failed\n");
            exit(EXIT_FAILURE);
        }
    }
    for (int t = 0; t<nthreads; t++) {
        pthread_join(tid[t], NULL);
    }
    
    free(tid);
}