#include <omp.h>
#include <stdio.h>

int main() {
   #pragma omp parallel
   {
       printf("Hello thread = %d\n", omp_get_thread_num());
   }
   return 0;
}
