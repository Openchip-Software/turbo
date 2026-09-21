#include "rave_user_events_v1.h"

int main() {
  rave_name_event(1000, "code_block");
  rave_name_value(1000, 0, "end");
  rave_name_value(1000, 1, "region_1");
  rave_name_value(1000, 2, "region_2");
  rave_name_value(1000, 3, "region_3");

  __asm__ volatile("xori a6, a0, 1" ::: "a6");

  rave_event_and_value(1000, 1);
  __asm__ volatile("xori a6, a0, 1" ::: "a6");
  rave_event_and_value(1000, 0);

  __asm__ volatile("xori a6, a0, 1" ::: "a6");

  // Following regions are not counted
  rave_stop_trace();

  __asm__ volatile("xori a6, a0, 1" ::: "a6");

  rave_event_and_value(1000, 2);
  __asm__ volatile("xori a6, a0, 1" ::: "a6");
  __asm__ volatile("xori a6, a0, 1" ::: "a6");
  rave_event_and_value(1000, 0);

  __asm__ volatile("xori a6, a0, 1" ::: "a6");

  // Following regions are counted
  rave_start_trace();

  __asm__ volatile("xori a6, a0, 1" ::: "a6");

  rave_event_and_value(1000, 3);
  __asm__ volatile("xori a6, a0, 1" ::: "a6");
  __asm__ volatile("xori a6, a0, 1" ::: "a6");
  __asm__ volatile("xori a6, a0, 1" ::: "a6");
  rave_event_and_value(1000, 0);

  __asm__ volatile("xori a6, a0, 1" ::: "a6");
}
