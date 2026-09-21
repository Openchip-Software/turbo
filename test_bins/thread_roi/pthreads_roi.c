// Multi-threaded ROI test binary: each pthread runs its own uniquely named
// RAVE region with a distinct, deterministic instruction count, so the
// enricher's per-hart (per-vCPU) PerformanceData can be validated in
// isolation instead of only as a summed total across threads.
#include <pthread.h>
#include "rave_user_events_v2.h"

#define NUM_THREADS 3

static void *thread_func(void *arg) {
  int id = (int)(long)arg;

  switch (id) {
    case 0:
      rave_begin_region("thread_0_work");
      __asm__ volatile("xori a6, a0, 1" ::: "a6");
      __asm__ volatile("xori a6, a0, 1" ::: "a6");
      rave_end_region("thread_0_work");
      break;
    case 1:
      rave_begin_region("thread_1_work");
      __asm__ volatile("xori a6, a0, 1" ::: "a6");
      __asm__ volatile("xori a6, a0, 1" ::: "a6");
      __asm__ volatile("xori a6, a0, 1" ::: "a6");
      __asm__ volatile("xori a6, a0, 1" ::: "a6");
      rave_end_region("thread_1_work");
      break;
    default:
      rave_begin_region("thread_2_work");
      __asm__ volatile("xori a6, a0, 1" ::: "a6");
      __asm__ volatile("xori a6, a0, 1" ::: "a6");
      __asm__ volatile("xori a6, a0, 1" ::: "a6");
      __asm__ volatile("xori a6, a0, 1" ::: "a6");
      __asm__ volatile("xori a6, a0, 1" ::: "a6");
      __asm__ volatile("xori a6, a0, 1" ::: "a6");
      rave_end_region("thread_2_work");
      break;
  }

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
