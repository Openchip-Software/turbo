#include "rave_user_events_v2.h"
#include <string.h>

// Regression test for long region names passed by pointer+length from the
// stack -- the shape Fortran produces when calling into the C marker API.
// The name is deliberately longer than 64 bytes; names at or above that used
// to be dropped, leaving the region named "roi_region_0x<pc>".
#define LONG_NAME                                                              \
  "This is a very very very very very very very very very very very very"      \
  " very very long region name"

int main() {
  // Stack copy, so the pointer handed to the marker is not an ELF address.
  char name[sizeof(LONG_NAME)];
  memcpy(name, LONG_NAME, sizeof(LONG_NAME));

  rave_begin_region_len(name, sizeof(LONG_NAME) - 1);
  __asm__ volatile("xori a6, a0, 1" ::: "a6");
  rave_end_region_len(name, sizeof(LONG_NAME) - 1);
}
