/*
 * linker_flavours - add.c
 *
 * Compiled into a shared object (libadd.so / libadd_gtod.so). The same source
 * yields a plain version and one that itself calls gettimeofday(), selected by:
 *
 *   ADD_USE_GETTIMEOFDAY - add() calls gettimeofday() internally, so the vDSO
 *                          is reached from inside the shared library as well as
 *                          (potentially) from the main executable.
 */

#ifdef ADD_USE_GETTIMEOFDAY
#include <stdio.h>
#include <sys/time.h>
#endif

int add(int a, int b)
{
#ifdef ADD_USE_GETTIMEOFDAY
	struct timeval tv;
	gettimeofday(&tv, NULL);
	printf("add.so: gettimeofday -> %ld.%06ld\n",
	       (long)tv.tv_sec, (long)tv.tv_usec);
#endif
	return a + b;
}
