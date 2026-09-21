#include "rave_user_events_v2.h"

int main() {
  __asm__ volatile("xori a6, a0, 1" ::: "a6");

  rave_begin_region("region_1");
  __asm__ volatile("xori a6, a0, 1" ::: "a6");
  rave_end_region("region_1");

  __asm__ volatile("xori a6, a0, 1" ::: "a6");

  rave_disable_regions();

  __asm__ volatile("xori a6, a0, 1" ::: "a6");

  rave_begin_region("region_2");
  __asm__ volatile("xori a6, a0, 1" ::: "a6");
  __asm__ volatile("xori a6, a0, 1" ::: "a6");
  rave_end_region("region_2");

  __asm__ volatile("xori a6, a0, 1" ::: "a6");

  rave_enable_regions();

  __asm__ volatile("xori a6, a0, 1" ::: "a6");

  rave_begin_region("region_3");
  __asm__ volatile("xori a6, a0, 1" ::: "a6");
  __asm__ volatile("xori a6, a0, 1" ::: "a6");
  __asm__ volatile("xori a6, a0, 1" ::: "a6");
  rave_end_region("region_3");

  __asm__ volatile("xori a6, a0, 1" ::: "a6");

  // Identical to rave_disable_regions
  rave_disable();

  __asm__ volatile("xori a6, a0, 1" ::: "a6");

  rave_begin_region("region_4");
  __asm__ volatile("xori a6, a0, 1" ::: "a6");
  rave_end_region("region_4");

  __asm__ volatile("xori a6, a0, 1" ::: "a6");

  // Identical to rave_enable_regions
  rave_enable();

  __asm__ volatile("xori a6, a0, 1" ::: "a6");

  rave_begin_region("region_5");
  __asm__ volatile("xori a6, a0, 1" ::: "a6");
  __asm__ volatile("xori a6, a0, 1" ::: "a6");
  rave_end_region("region_5");

  __asm__ volatile("xori a6, a0, 1" ::: "a6");
}
