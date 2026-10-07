# hfcpu - POSIX make; works with GNU make and BSD make.
CC      ?= cc
CFLAGS  ?= -O2
WARN     = -std=c99 -Wall -Wextra -pedantic -ffp-contract=off
LDLIBS   = -pthread -lm

LIBOBJ = src/mem.o src/pal.o src/probe.o src/kern_generic.o src/kern_avx2.o src/opts.o src/ggtype.o src/gguf.o src/proto.o src/ops.o src/tok.o src/unitab.o src/mathx.o src/model.o src/sample.o src/pool.o
HDR    = src/hfc.h src/pal.h src/probe.h src/kern.h src/opts.h src/ggtype.h src/gguf.h src/proto.h src/ops.h src/tok.h src/unitab.h src/mathx.h src/model.h src/sample.h src/pool.h
TESTS  = tests/test_opts tests/test_gguf tests/test_tok tests/test_kern

.SUFFIXES: .c .o
.c.o:
	$(CC) $(WARN) $(CFLAGS) -c -o $@ $<

all: hfcpu

hfcpu: src/main.o $(LIBOBJ)
	$(CC) $(WARN) $(CFLAGS) -o $@ src/main.o $(LIBOBJ) $(LDLIBS)

src/main.o: $(HDR)

# The AVX2 kernels need ISA flags; the file is empty on other architectures.
src/kern_avx2.o: src/kern_avx2.c src/kern.h
	@case `uname -m` in x86_64|amd64) F="-mavx2 -mfma -mf16c";; *) F="";; esac; \
	echo "$(CC) $(WARN) $(CFLAGS) $$F -c -o $@ src/kern_avx2.c"; \
	$(CC) $(WARN) $(CFLAGS) $$F -c -o $@ src/kern_avx2.c

$(LIBOBJ): $(HDR)

tests/test_opts: tests/test_opts.c tests/t.h $(LIBOBJ)
	$(CC) $(WARN) $(CFLAGS) -o $@ tests/test_opts.c $(LIBOBJ) $(LDLIBS)

tests/test_gguf: tests/test_gguf.c tests/t.h tests/gguf_builder.h $(LIBOBJ)
	$(CC) $(WARN) $(CFLAGS) -o $@ tests/test_gguf.c $(LIBOBJ) $(LDLIBS)

tests/test_tok: tests/test_tok.c tests/t.h $(LIBOBJ)
	$(CC) $(WARN) $(CFLAGS) -o $@ tests/test_tok.c $(LIBOBJ) $(LDLIBS)

tests/test_kern: tests/test_kern.c tests/t.h $(LIBOBJ)
	$(CC) $(WARN) $(CFLAGS) -o $@ tests/test_kern.c $(LIBOBJ) $(LDLIBS)

test: hfcpu $(TESTS)
	@for t in $(TESTS); do ./$$t || exit 1; done
	perl tests/e2e.pl ./hfcpu
	perl tests/oracle.pl ./hfcpu

# Compare dequantizers with upstream ggml-quants.c: make diff-dequant REF=/path/to/ggml/src
diff-dequant:
	perl tests/diff_dequant.pl $(REF)

clean:
	rm -f src/*.o $(TESTS) hfcpu

# Rebuild everything with AddressSanitizer + UBSan and run the tests.
asan:
	$(MAKE) clean
	$(MAKE) test CFLAGS="-O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=undefined"
	$(MAKE) clean
