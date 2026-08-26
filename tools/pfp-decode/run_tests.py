#!/usr/bin/env python3
"""Bit-order regression suite for the PharmCast fingerprint decoders.

Bit order is the whole contract. A silent change here corrupts every downstream
consumer, and nothing else in the pipeline would notice. Every assertion below
exists to pin one edge of that contract against the committed fixture.

    make test          # or: python3 run_tests.py

Skips the cross-architecture check when only one architecture is built, rather
than failing -- CI runners are frequently single-arch. It says so out loud.
"""
import html
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PFP = os.path.join(HERE, "examples", "CHEMBL163631.pfp")
EXPECT_BITS = 173
EXPECT_WIDTH = 10549

fails, skips = [], []

def check(name, cond, detail=""):
    print(("  ok   " if cond else "  FAIL ") + name + (("  -- " + detail) if detail and not cond else ""))
    if not cond:
        fails.append(name)

def skip(name, why):
    print("  skip " + name + "  -- " + why)
    skips.append(name)

def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=HERE)
    return r.stdout

def build(prog):
    """Prefer an arch-suffixed binary, else the plain one, else build it."""
    plain = os.path.join(HERE, prog)
    if not os.path.exists(plain):
        subprocess.run(["clang", "-std=gnu89", "-O2", "-w", "-o", plain,
                        os.path.join(HERE, prog + ".c"), "-lm"], check=True)
    return plain

print("PharmCast decoder regression suite")
print("fixture:", os.path.relpath(PFP, HERE))
print()

# --- 1. pfp2bits, the reference bit string ---------------------------------
bits_out = run([build("pfp2bits"), PFP])
parts = bits_out.split()
check("pfp2bits emits name plus one bit string", len(parts) == 2,
      "got %d fields" % len(parts))
name, bits = (parts + ["", ""])[:2]
check("bit string is exactly %d characters" % EXPECT_WIDTH,
      len(bits) == EXPECT_WIDTH, "got %d" % len(bits))
check("exactly %d bits set" % EXPECT_BITS,
      bits.count("1") == EXPECT_BITS, "got %d" % bits.count("1"))
check("bit string is 0/1 only", set(bits) <= {"0", "1"})

# --- 2. cross-architecture agreement ---------------------------------------
arm = os.path.join(HERE, "pfp2bits_arm64")
x86 = os.path.join(HERE, "pfp2bits_x86_64")
if os.path.exists(arm) and os.path.exists(x86):
    check("arm64 output byte identical to x86_64", run([arm, PFP]) == run([x86, PFP]))
else:
    skip("arm64 vs x86_64", "both arch binaries not built (make all-archs)")

# --- 3. the Python decoder must agree with the C one -----------------------
py = run([sys.executable, os.path.join(HERE, "pfp_report.py"), PFP, "--bits"])
check("pfp_report.py --bits byte identical to pfp2bits", py == bits_out)

# --- 4. pharmstat summary ---------------------------------------------------
ps = run([build("pharmstat"), PFP, os.path.join(HERE, "pharm10549.list")])
check("pharmstat reports 1 bitstring processed",
      re.search(r"^\s*1 = number of bitstrings processed", ps, re.M) is not None)
check("pharmstat reports %.1f average" % EXPECT_BITS,
      re.search(r"^\s*%.1f = average number" % EXPECT_BITS, ps, re.M) is not None)
check("pharmstat hit list has %d rows" % EXPECT_BITS,
      len(re.findall(r"^\s*\d+ \|", ps, re.M)) == EXPECT_BITS,
      "got %d" % len(re.findall(r"^\s*\d+ \|", ps, re.M)))

# --- 5. the HTML copy blocks must carry the real thing ----------------------
rep = os.path.join(HERE, "examples", "CHEMBL163631_report.html")
doc = open(rep).read()

def block(bid):
    m = re.search(r'<pre id="%s"[^>]*>(.*?)</pre>' % bid, doc, re.S)
    return html.unescape(m.group(1)).strip() if m else None

check("rawblock unescapes to the exact .pfp contents",
      block("rawblock") == open(PFP).read().strip())
check("bitblock unescapes to exact pfp2bits output",
      block("bitblock") == bits_out.strip())

print()
if fails:
    print("FAILED: " + ", ".join(fails))
    sys.exit(1)
print("all checks passed" + (" (%d skipped)" % len(skips) if skips else ""))
