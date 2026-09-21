#include <stdio.h>

__attribute__((noinline))
int other_function(int x) {
    return x * x;
}

int main() {
    volatile int sum = 0;  // volatile to prevent optimization
    for (int i = 0; i < 10; i++) {
        sum += i;

        // random function calls
        if ((i % 3) == 0) {
            sum += other_function(sum);
        }
    }

    // printf as a jump to somewhere else?
    printf("sum = %d, exiting now.\n", sum);
    return 0;
}
