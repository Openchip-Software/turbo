#include "rave_user_events_v1.h"

void iterative(int n);
// void recursive(int n);

int main() {
  rave_name_event(1000, "code_block");
  rave_name_value(1000, 0, "end");
  rave_name_value(1000, 1, "iterative loop");

  int n = 10;
  iterative(n);
  // recursive(n);
}

void iterative(int n) {
  for (int i = 0; i < n; i++) {
    rave_event_and_value(1000, 1);
    __asm__ volatile("xori a6, a0, 1" ::: "a6");
    rave_event_and_value(1000, 0);
  }
}

// Recursion is not supported for now

// void recursive(int n) {
//   if (n <= 0) {
//     return;
//   }
//   rave_event_and_value(1000, 1);
//   // Do work
//   rave_event_and_value(1000, 0);
//   recursive(n - 1);
// }
