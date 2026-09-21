#include "rave_user_events_v2.h"
#include <stdio.h>

void iterative(int n);
void recursive(int n);

int main() {
  int n = 10;
  iterative(n);

  n = 5;
  recursive(n);
}

void iterative(int n) {
  for (int i = 0; i < n; i++) {
    rave_begin_region("iterative loop");
    __asm__ volatile("xori a6, a0, 1" ::: "a6");
    rave_end_region("iterative loop");
  }
}

void recursive(int n) {
  if (n <= 0) {
    return;
  }
  const size_t NAME_LEN = 64;
  char name[NAME_LEN];
  snprintf(name, NAME_LEN, "recursive_region_%d", n);
  printf("%s\n", name);
  rave_begin_region(name);

  // just blurt in a few iterations :)
  iterative(1);

  rave_begin_region("recursion-work");
  __asm__ volatile("xori a6, a0, 1" ::: "a6");
  rave_end_region("recursion-work");

  recursive(n - 1);
  rave_end_region(name);
}
