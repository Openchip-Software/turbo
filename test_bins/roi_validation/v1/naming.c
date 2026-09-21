#include "rave_user_events_v1.h"

int main() {
  char *region_name = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                      "abcdefghijklmnopqrstuvwxyz"
                      "0123456789"
                      "! #$%&'()*+,-./:;<=>?@[\\]^_`{|}~'\"?";
  rave_name_event(1000, "code_block");
  rave_name_value(1000, 0, "end");
  rave_name_value(1000, 1, region_name);

  rave_event_and_value(1000, 1);
  __asm__ volatile("xori a6, a0, 1" ::: "a6");
  rave_event_and_value(1000, 0);
}
