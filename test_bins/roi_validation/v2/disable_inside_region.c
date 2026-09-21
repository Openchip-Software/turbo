#include "rave_user_events_v2.h"

// Region counting toggled *inside* an open region.
//
// The sibling enable_disable test only ever toggles between regions, so the
// stack is empty at every toggle. Here the toggles land while a region is
// open, which is the case that distinguishes "stop counting" from "stop
// noticing markers":
//
//   region_1  counting is disabled for part of the region's extent, then
//             re-enabled before the close. Instructions retired while
//             counting was off must not be attributed to the region.
//
//   region_2  counting is disabled and never re-enabled, so the close marker
//             arrives while disabled. The close must still take effect: if it
//             does not, the region stays open and keeps accumulating every
//             instruction to the end of the program.
int main() {
  __asm__ volatile("xori a6, a0, 1" ::: "a6");

  rave_begin_region("region_1");
  __asm__ volatile("xori a6, a0, 1" ::: "a6");

  rave_disable_regions();
  __asm__ volatile("xori a6, a0, 1" ::: "a6");
  __asm__ volatile("xori a6, a0, 1" ::: "a6");
  __asm__ volatile("xori a6, a0, 1" ::: "a6");
  rave_enable_regions();

  __asm__ volatile("xori a6, a0, 1" ::: "a6");
  rave_end_region("region_1");

  __asm__ volatile("xori a6, a0, 1" ::: "a6");

  rave_begin_region("region_2");
  __asm__ volatile("xori a6, a0, 1" ::: "a6");
  __asm__ volatile("xori a6, a0, 1" ::: "a6");
  rave_disable_regions();
  rave_end_region("region_2");

  __asm__ volatile("xori a6, a0, 1" ::: "a6");
  __asm__ volatile("xori a6, a0, 1" ::: "a6");
  __asm__ volatile("xori a6, a0, 1" ::: "a6");
}
