#include <pthread.h>
#include <stdio.h>

#define NUM_THREADS 8

static void *thread_func(void *arg) {
    int id = (int)(long)arg;
    printf("Hello thread = %d\n", id);
    return NULL;
}

int main() {
    pthread_t threads[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_create(&threads[i], NULL, thread_func, (void *)(long)i);
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    return 0;
}
