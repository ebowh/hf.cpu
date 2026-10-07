# hfcpu - POSIX make; works with GNU make and BSD make.
CC      ?= cc
CFLAGS  ?= -O2
WARN     = -std=c99 -Wall -Wextra -pedantic
LDLIBS   = -pthread -lm

LIBOBJ = src/mem.o src/pal.o src/probe.o src/kern_generic.o src/kern_avx2.o src/opts.o
HDR    = src/hfc.h src/pal.h src/probe.h src/kern.h src/opts.h
TESTS  = tests/test_opts

.SUFFIXES: .c .o
.c.o:
	$(CC) $(WARN) $(CFLAGS) -c -o $@ $<

all: $(LIBOBJ)

# The AVX2 kernels need ISA flags; the file is empty on other architectures.
src/kern_avx2.o: src/kern_avx2.c src/kern.h
	@case `uname -m` in x86_64|amd64) F="-mavx2 -mfma";; *) F="";; esac; \
	echo "$(CC) $(WARN) $(CFLAGS) $$F -c -o $@ src/kern_avx2.c"; \
	$(CC) $(WARN) $(CFLAGS) $$F -c -o $@ src/kern_avx2.c

$(LIBOBJ): $(HDR)

tests/test_opts: tests/test_opts.c tests/t.h $(LIBOBJ)
	$(CC) $(WARN) $(CFLAGS) -o $@ tests/test_opts.c $(LIBOBJ) $(LDLIBS)

test: $(TESTS)
	@for t in $(TESTS); do ./$$t || exit 1; done

clean:
	rm -f src/*.o $(TESTS) hfcpu
