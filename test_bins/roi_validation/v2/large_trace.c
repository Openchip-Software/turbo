#include "rave_user_events_v2.h"

// Volatile to prevent optimization
volatile long result = 0;

// Helper function to generate work (instructions)
void do_work(int iterations) {
  volatile long temp = 0;
  for (int i = 0; i < iterations; i++) {
    temp += i * 2;
    temp -= i / 2;
    temp ^= i;
    temp |= i << 1;
    temp &= ~(i >> 1);
  }
  result = temp;
}

int main() {
  // Enable V2 regions
  rave_enable_regions();

  // Loop enough times to generate >2M instructions
  // This should cross many PC batch boundaries (batches are 100K)
  // Increased from 1000 to 10000 to generate much more work
  for (int iteration = 0; iteration < 1000; iteration++) {
    int n = 50;
    // Region 1: compute phase
    rave_begin_region("region_1");
    do_work(n); // Increased from n
    rave_end_region("region_1");

    // Region 2: transform phase
    rave_begin_region("region_2");
    do_work(n); // Increased from n
    rave_end_region("region_2");

    // Region 3: finalize phase
    rave_begin_region("region_3");
    do_work(n); // Increased from n
    rave_end_region("region_3");

    rave_begin_region("region_4");
    do_work(n); // Increased from n
    rave_end_region("region_4");

    rave_begin_region("region_5");
    do_work(n); // Increased from n
    rave_end_region("region_5");

    rave_begin_region("region_6");
    do_work(n); // Increased from n
    rave_end_region("region_6");
  }

  return 0;
}
