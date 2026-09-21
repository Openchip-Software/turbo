/*
 * linker_flavours - main.c
 *
 * A single source file that compiles into several "flavours" of the same
 * program, selected entirely by -D defines.  This lets us exercise every
 * combination of static/dynamic linking and vDSO usage with one body of code.
 *
 *   USE_ADD_SO        - the addition is performed by add() from a shared
 *                       library (libadd*.so) instead of inline. Forces the
 *                       program to be dynamically linked against our own .so.
 *   USE_GETTIMEOFDAY  - call gettimeofday(), which on RISC-V Linux is serviced
 *                       through the kernel-provided vDSO. Pulls the vDSO into
 *                       the picture for both static and dynamic binaries.
 *   USE_DLOPEN        - load libadd.so at *run time* via dlopen()/dlsym()
 *                       instead of as a NEEDED dependency. This is the only
 *                       flavour that maps a shared object *after* startup, so
 *                       it exercises the plugin's openat/mmap syscall discovery
 *                       channel on the live path rather than at loader init.
 *
 * See the Makefile for the concrete flavours that get built.
 */

#include <stdio.h>

#ifdef USE_GETTIMEOFDAY
#include <sys/time.h>
#endif

#ifdef USE_DLOPEN
#include <dlfcn.h>
#elif defined(USE_ADD_SO)
/* Provided by libadd.so / libadd_gtod.so at link/run time. */
extern int add(int a, int b);
#endif

int main(void)
{
	int result;

#ifdef USE_DLOPEN
	void *h = dlopen("libadd.so", RTLD_NOW);
	if (!h) {
		fprintf(stderr, "dlopen(libadd.so) failed: %s\n", dlerror());
		return 1;
	}
	int (*add)(int, int) = (int (*)(int, int))dlsym(h, "add");
	if (!add) {
		fprintf(stderr, "dlsym(add) failed: %s\n", dlerror());
		return 1;
	}
	result = add(2, 3);
#elif defined(USE_ADD_SO)
	result = add(2, 3);
#else
	result = 2 + 3;
#endif

#ifdef USE_GETTIMEOFDAY
	struct timeval tv;
	gettimeofday(&tv, NULL);
	printf("main: gettimeofday -> %ld.%06ld\n",
	       (long)tv.tv_sec, (long)tv.tv_usec);
#endif

	printf("result = %d\n", result);
	return 0;
}
