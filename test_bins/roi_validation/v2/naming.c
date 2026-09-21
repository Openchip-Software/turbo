#include "rave_user_events_v2.h"

int main() {
  const char *region_name = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                            "abcdefghijklmnopqrstuvwxyz"
                            "0123456789"
                            "! #$%&'()*+,-./:;<=>?@[\\]^_`{|}~'\"?";
  rave_begin_region(region_name);
  __asm__ volatile("xori a6, a0, 1" ::: "a6");
  rave_end_region(region_name);
}
