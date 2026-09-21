#include "rave_user_events_v2.h"

void foo();

int main() {

  __asm__ volatile("xori a6, a0, 1" ::: "a6");

  rave_begin_region("region_1");
  __asm__ volatile("xori a6, a0, 1" ::: "a6");
  foo();
  __asm__ volatile("xori a6, a0, 1" ::: "a6");
  rave_end_region("region_1");

  __asm__ volatile("xori a6, a0, 1" ::: "a6");
}

__attribute__((noinline)) void foo() {
  __asm__ volatile("xori a6, a0, 1" ::: "a6");
}
