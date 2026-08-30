# PharmCast fingerprint decoders. Build and verify.
#
# Scoped deliberately to the four decoder binaries. This Makefile must never
# build, link or reference the reference generator -- it is not part of this tree.
#
#   make            build for the host architecture
#   make all-archs  build arm64 and x86_64 (macOS cross-compile)
#   make test       run the bit-order regression suite
#   make clean      remove build artifacts

CC      ?= clang
CFLAGS  ?= -std=gnu89 -O2 -w
LDLIBS  ?= -lm
PROGS    = pfp2bits pharmstat
ARCHS    = arm64 x86_64

.PHONY: all all-archs test clean

all: $(PROGS)

# K&R era sources: main is declared with no return type, which modern clang
# rejects without -std=gnu89.
pfp2bits: pfp2bits.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

pharmstat: pharmstat.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

all-archs:
	@for arch in $(ARCHS); do \
	  for prog in $(PROGS); do \
	    echo "  $$prog ($$arch)"; \
	    $(CC) $(CFLAGS) -arch $$arch -o $${prog}_$${arch} $${prog}.c $(LDLIBS) || exit 1; \
	  done; \
	done

test: all
	@python3 run_tests.py

clean:
	rm -f $(PROGS) $(foreach a,$(ARCHS),$(foreach p,$(PROGS),$(p)_$(a)))
