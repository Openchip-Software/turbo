#include "rave_user_events_v2.h"
#include <stdio.h>

static void spin(int n) {
  for (int i = 0; i < n; ++i) {
    __asm__ volatile("xori a6, a0, 1" ::: "a6");
  }
}

static void chain_region(int depth, int max_depth) {
  char name[32];
  snprintf(name, sizeof(name), "chain_%02d", depth);

  rave_begin_region(name);
  spin(depth);

  if (depth < max_depth) {
    chain_region(depth + 1, max_depth);
  } else {
    for (int i = 0; i < 3; ++i) {
      rave_begin_region("leaf_loop");
      spin(2 + i);
      rave_end_region("leaf_loop");
    }
  }

  rave_end_region(name);
}

int main(void) {
  spin(2);

  rave_begin_region("outer_driver");
  for (int pass = 0; pass < 3; ++pass) {
    rave_begin_region("fan_out");
    chain_region(1, 20);
    rave_end_region("fan_out");
  }

  rave_begin_region("tail_sibling");
  spin(4);
  rave_end_region("tail_sibling");
  rave_end_region("outer_driver");

  spin(2);
  return 0;
}
