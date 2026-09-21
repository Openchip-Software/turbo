#include "rave_user_events_v2.h"

int main() {
  __asm__ volatile("xori a6, a0, 1" ::: "a6");

  rave_begin_region("region_1");
  __asm__ volatile("xori a6, a0, 1" ::: "a6");
  rave_end_region("region_1");

  __asm__ volatile("xori a6, a0, 1" ::: "a6");

  // Ignored by v2
  rave_stop_trace();

  __asm__ volatile("xori a6, a0, 1" ::: "a6");

  rave_begin_region("region_2");
  __asm__ volatile("xori a6, a0, 1" ::: "a6");
  __asm__ volatile("xori a6, a0, 1" ::: "a6");
  rave_end_region("region_2");

  __asm__ volatile("xori a6, a0, 1" ::: "a6");

  // Ignored by v2
  rave_start_trace();

  __asm__ volatile("xori a6, a0, 1" ::: "a6");

  rave_begin_region("region_3");
  __asm__ volatile("xori a6, a0, 1" ::: "a6");
  __asm__ volatile("xori a6, a0, 1" ::: "a6");
  __asm__ volatile("xori a6, a0, 1" ::: "a6");
  rave_end_region("region_3");

  __asm__ volatile("xori a6, a0, 1" ::: "a6");
}
