#include "rave_user_events_v2.h"

int main() {
  __asm__ volatile("xori a6, a0, 1" ::: "a6");

  rave_begin_region("region_1");
  __asm__ volatile("xori a6, a0, 1" ::: "a6");
  rave_end_region("region_1");

  __asm__ volatile("xori a6, a0, 1" ::: "a6");

  // Counters from above regions are discarded
  rave_restart_trace();

  __asm__ volatile("xori a6, a0, 1" ::: "a6");

  rave_begin_region("region_2");
  __asm__ volatile("xori a6, a0, 1" ::: "a6");
  __asm__ volatile("xori a6, a0, 1" ::: "a6");
  rave_end_region("region_2");

  __asm__ volatile("xori a6, a0, 1" ::: "a6");
}
