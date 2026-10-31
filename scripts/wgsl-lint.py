#!/usr/bin/env python3
r"""Lint generated GPU shaders for a division that Apple's Metal compiler
miscompiles.

Apple's GPU compiler (seen on an M4 under macOS 26.3, through both naga's and
Chrome's WGSL-to-MSL translations) returns wrong quotients and remainders for
an unsigned division or remainder by a constant that is not a power of two
when the dividend is a right shift by half the word:

    naga_div(x >> 16u, K)          what naga writes for WGSL (x >> 16) / K
    (x >> 16u) % K
    uint(p >> 32) / K              p a 64-bit value: its high word

#225 (the finite-difference seed) and #228 (the overlap join's top
certificates) removed every such site they found. This script keeps new ones
out. It reads the WGSL CubeCL generates (the `CUBECL_DEBUG_LOG` dump, or
plain .wgsl files) and also naga's MSL output, and fails on any division or
remainder by a literal non-power-of-two constant whose dividend is:

    HALF   x >> 16 (the bare shift, after the folding below)
    HI32   a 64-bit value shifted right by 32, cast to u32 or not
    HI32 >> 16, which is a HALF of a HI32

usage:
    wgsl-lint.py [--require KERNEL[@BASE]]... [--verbose] [FILE...]

A FILE holding `[START_KERNEL_COMPILATION]` blocks is a CubeCL debug log
(one kernel per block, named and based from the block's header); any other
file is one shader. `--require` fails the run unless that kernel (at that
base) was found and had at least one constant division recognised, which
catches a dump that quietly stopped covering a kernel, and a change in
CubeCL's output format that these patterns no longer match. The built-in
cases (`--self-test`) run before every scan.

Exit status: 0 clean, 1 flagged sites, 2 a requirement or self-test failed.

How the matching works, and its limits
--------------------------------------
CubeCL's WGSL is single-assignment: one operation per `let l_N = ...;`, with
`l_mut_N = ...;` for mutable variables. naga's MSL is the same shape
(`uint l_N_ = ...;`, `l_mut_N_ = ...;`, `uint _eN = l_mut_N_;`). Each value
gets a shape from these line patterns (WGSL / MSL spellings):

    LET     ^let (\w+) = (.*);$   or, in MSL,   ^u?(int|long)\w* (\w+) = (.*);$
    ASSIGN  ^(\w+) = (.*);$                      (a mutable variable)
    SHIFT   ^(\w+) >> NUM$        16 is HALF, unless NUM is spelled 64-bit; 32 is HI32
    LSHIFT  ^(\w+) << NUM$        of a zero is zero
    CAST    ^(CAST\()+(\w+)\)+$   keeps the shape (u32(..), bitcast<u64>(..),
                                  static_cast<uint>(..), as_type<ulong>(..), ...)
    OR      ^(\w+) \| (\w+)$      0 | y is y
    MASK    ^(\w+) & NUM$         HALF & 65535 is HALF
    DIV     ^(\w+) [/%] NUM$   or   ^naga_(div|mod)\((\w+), NUM\)$

where NUM is a literal, bare or cast: 16, 16u, 32uL, u32(16),
u32(bitcast<u64>(u64(32))), static_cast<uint>(as_type<ulong>(32uL)).

Literal zeros are folded: a mutable assigned `u32(0)` / `0u` is zero until
its next assignment or the next line holding `{` or `}`; `0 << k` is 0 and
`0 | y` is y. A remainder carried through a run-time loop is therefore never
treated as zero: any control flow clears what is known, which is
conservative in the right direction for loops (no false alarm on the
per-number kernel's digit loop) and the wrong one for straight-line code
inside a block (a zeroed variable read after an `if` is not tracked). Also
`HALF & 65535` stays HALF, and a copy keeps its source's shape.

Not seen, by construction: dividends built by other operations (`a * 65536
+ (x >> 16)`, XOR, select), values passed through arrays or memory, shifts
other than exactly half the width (a `>> 15` or `>> 24` is not flagged, and
was exact on the M4), division by a run-time divisor, and anything a
downstream compiler folds into the bad form after this text (for instance a
loop it peels or unrolls so that a carried remainder becomes a literal 0).
The lint narrows the search; only a run on Apple hardware settles it.
"""
import gzip
import re
import sys

ID = r"(\w+)"
# A cast or reinterpretation, as either language spells it.
CASTFN = r"(?:u32|u64|i32|uint|ulong|int|metal::u?int|metal::u?long|bitcast<\w+>|static_cast<[\w:]+>|as_type<[\w:]+>)"
# A literal integer, bare (16, 16u, 32uL) or wrapped in casts
# (u32(16), u32(bitcast<u64>(u64(32))), static_cast<uint>(as_type<ulong>(32uL))).
NUM = r"((?:" + CASTFN + r"\()*(\d+)(?:[uU][lL]?|[lL])?\)*)"

LET = [
    re.compile(r"^\s*let " + ID + r" = (.*);\s*$"),
    re.compile(r"^\s*(?:const\s+)?(?:metal::)?u?(?:int|long)\d*\s+" + ID + r" = (.*);\s*$"),
]
ASSIGN = re.compile(r"^\s*" + ID + r" = (.*);\s*$")
SHR = re.compile(r"^" + ID + r" >> " + NUM + r"$")
SHL = re.compile(r"^" + ID + r" << " + NUM + r"$")
CAST = re.compile(r"^(?:" + CASTFN + r"\()+" + ID + r"\)+$")
OR = re.compile(r"^" + ID + r" \| " + ID + r"$")
MASK = re.compile(r"^" + ID + r" & " + NUM + r"$")
DIV = re.compile(r"^" + ID + r" ([/%]) " + NUM + r"$")
NAGA = re.compile(r"^naga_(div|mod)\(" + ID + r", " + NUM + r"\)$")
ZERO_LIT = re.compile(r"^" + NUM + r"$")
WIDE = re.compile(r"64|long|[uU][lL]")

BAD = ("HALF", "HI32")


def analyse(src):
    """Flagged sites and the count of constant divisions in one shader.

    Returns (sites, ndiv): sites are (line, text, dividend shape, divisor).
    """
    shape = {}  # SSA value -> shape
    mut = {}  # mutable -> shape, valid until the next brace
    sites = []
    ndiv = 0

    def sh(name):
        if (z := ZERO_LIT.match(name)) and int(z.group(2)) == 0:
            return "ZERO"
        if name in mut:
            return mut[name]
        return shape.get(name, "")

    def classify(e):
        if (z := ZERO_LIT.match(e)) and int(z.group(2)) == 0:
            return "ZERO"
        if m := SHR.match(e):
            # A shift by 32 can only be of a 64-bit value (WGSL rejects it on
            # a u32, MSL leaves it undefined), so it takes the high word. A
            # shift by 16 is half the width unless its amount is spelled as
            # a 64-bit literal, as CubeCL and naga spell a u64 shift's.
            k, wide = int(m.group(3)), bool(WIDE.search(m.group(2)))
            if k == 32:
                return "HI32"
            return "HALF" if k == 16 and not wide else ""
        if m := SHL.match(e):
            return "ZERO" if sh(m.group(1)) == "ZERO" else ""
        if m := CAST.match(e):
            return sh(m.group(1)) if sh(m.group(1)) in BAD + ("ZERO",) else ""
        if m := OR.match(e):
            a, b = sh(m.group(1)), sh(m.group(2))
            if a == "ZERO":
                return b
            if b == "ZERO":
                return a
            return ""
        if m := MASK.match(e):
            # x >> 16 is below 2^16 already, so the mask changes nothing.
            return "HALF" if sh(m.group(1)) == "HALF" and int(m.group(3)) == 65535 else ""
        if re.fullmatch(ID, e):
            return sh(e)
        return ""

    for ln, line in enumerate(src.splitlines(), 1):
        if "{" in line or "}" in line:
            mut.clear()
        name = expr = None
        is_let = False
        for pat in LET:
            if m := pat.match(line):
                name, expr, is_let = m.group(1), m.group(2).strip(), True
                break
        if name is None and (m := ASSIGN.match(line)):
            name, expr = m.group(1), m.group(2).strip()
        if name is None:
            continue
        div = DIV.match(expr)
        naga = NAGA.match(expr)
        if div or naga:
            ndiv += 1
            if div:
                dividend, k = div.group(1), int(div.group(4))
            else:
                dividend, k = naga.group(2), int(naga.group(4))
            ds = sh(dividend)
            if ds in BAD and k & (k - 1) != 0:
                sites.append((ln, line.strip(), ds, k))
            s = ""
        else:
            s = classify(expr)
        if is_let:
            shape[name] = s
        else:
            mut[name] = s
    return sites, ndiv


def shaders(path):
    """(kernel, base, source) for each shader in the file."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", errors="replace") as f:
        text = f.read()
    if "[START_KERNEL_COMPILATION]" not in text:
        yield path, None, text
        return
    for blk in re.findall(r"\[START_KERNEL_COMPILATION\](.*?)\[END_KERNEL_COMPILATION\]", text, re.S):
        name = re.search(r"^name: \S+?::(\w+)<", blk, re.M)
        base = re.search(r"^\s+base: (\d+),", blk, re.M)
        src = blk.split("\nsource:", 1)[-1]
        yield (name.group(1) if name else "?"), (int(base.group(1)) if base else None), src


SELF_TEST = [
    # (expect flagged, source)
    (True, "l_mut_1 = u32(0);\nlet l_2 = l_mut_1 << u32(16);\nlet l_3 = l_0 >> u32(16);\n"
           "let l_4 = l_2 | l_3;\nlet l_5 = l_4 / u32(2500);"),
    (True, "let l_3 = l_0 >> u32(16);\nlet l_4 = l_3 % u32(12167);"),
    (True, "let l_3 = l_0 >> u32(16);\nlet l_4 = l_3 & u32(65535);\nlet l_5 = l_4 / u32(3);"),
    (True, "let l_1 = l_0 >> u32(bitcast<u64>(u64(32)));\nlet l_2 = u32(bitcast<u64>(l_1));\n"
           "let l_3 = l_2 / u32(3);"),
    (True, "let l_1 = l_0 >> u32(bitcast<u64>(u64(32)));\nlet l_2 = u32(bitcast<u64>(l_1));\n"
           "let l_3 = l_2 >> u32(16);\nlet l_4 = l_3 / u32(64000);"),
    (True, "let l_1 = l_0 >> u32(bitcast<u64>(u64(32)));\nlet l_2 = l_1 / u64(1000);"),
    (True, "uint l_6_ = l_2_ >> 16u;\nuint l_7_ = naga_div(l_6_, 3u);"),
    (True, "l_mut_113_ = 0u;\nuint _e344 = l_mut_113_;\nuint l_115_ = _e344 << 16u;\n"
           "uint l_116_ = l_114_ >> 16u;\nuint l_117_ = l_115_ | l_116_;\n"
           "uint l_118_ = naga_mod(l_117_, 12167u);"),
    (True, "ulong l_9_ = l_8_ >> static_cast<uint>(as_type<ulong>(32uL));\n"
           "uint l_10_ = static_cast<uint>(as_type<ulong>(l_9_));\nuint l_11_ = naga_div(l_10_, 3u);"),
    # A u64 shifted by 16 is a quarter of the width, not half.
    (False, "let l_1 = l_0 >> u32(bitcast<u64>(u64(16)));\nlet l_2 = u32(bitcast<u64>(l_1));\n"
            "let l_3 = l_2 / u32(3);"),
    (False, "let l_3 = l_0 >> u32(16);\nlet l_4 = countOneBits(l_3);\nlet l_5 = l_4 / u32(3);"),
    # The per-number kernel: the remainder is carried through a run-time loop.
    (False, "l_mut_1 = u32(0);\nloop {\nlet l_2 = l_mut_1 << u32(16);\nlet l_3 = l_0 >> u32(16);\n"
            "let l_4 = l_2 | l_3;\nlet l_5 = l_4 / u32(2500);\nl_mut_1 = l_5;\n}"),
    # #225/#228's fix: the top two limbs as one word, never a bare shift.
    (False, "let l_2 = l_1 << u32(16);\nlet l_3 = l_2 | l_0;\nlet l_4 = l_3 / u32(2500);"),
    (False, "let l_3 = l_0 >> u32(16);\nlet l_4 = l_3 / u32(4096);"),
    (False, "let l_3 = l_0 >> u32(15);\nlet l_4 = l_3 / u32(3);"),
    (False, "let l_3 = l_0 & u32(65535);\nlet l_4 = l_3 / u32(3);"),
    (False, "let l_3 = l_0 >> u32(16);\nlet l_4 = l_3 / l_9;"),
    (False, "let l_1 = l_0 >> u32(bitcast<u64>(u64(8)));\nlet l_2 = u32(bitcast<u64>(l_1));\n"
            "let l_3 = l_2 / u32(3);"),
]


def self_test():
    bad = []
    for want, src in SELF_TEST:
        sites, ndiv = analyse(src)
        if bool(sites) != want or (want and ndiv == 0):
            bad.append(f"expected {'a flag' if want else 'no flag'}, got {sites} ({ndiv} divisions) in:\n{src}")
    return bad


def main(argv):
    require = []
    verbose = False
    files = []
    it = iter(argv)
    for a in it:
        if a == "--require":
            require.append(next(it))
        elif a.startswith("--require="):
            require.append(a.split("=", 1)[1])
        elif a == "--verbose":
            verbose = True
        elif a == "--self-test":
            pass  # it always runs; with no files, it is all that runs
        else:
            files.append(a)

    problems = self_test()
    if problems:
        print("wgsl-lint self-test FAILED:\n" + "\n".join(problems))
        return 2
    print(f"wgsl-lint: self-test passed ({len(SELF_TEST)} cases)")
    if not files:
        return 0

    seen = {}  # (kernel, base) -> constant divisions recognised
    flagged = 0
    for path in files:
        for kernel, base, src in shaders(path):
            sites, ndiv = analyse(src)
            key = (kernel, base)
            seen[key] = max(seen.get(key, 0), ndiv)
            tag = kernel.rsplit("/", 1)[-1] + (f" b{base}" if base is not None else "")
            if sites or verbose:
                print(f"{tag}: {ndiv} constant divisions, {len(sites)} flagged")
            for ln, text, ds, k in sites:
                print(f"  {path}:{ln} [{tag}] divides a {ds} dividend by {k}: {text}")
            flagged += len(sites)

    kernels = sorted({k.rsplit("/", 1)[-1] for k, _ in seen})
    print(f"wgsl-lint: {len(seen)} shaders ({', '.join(kernels)}), {flagged} flagged sites")
    missing = []
    for r in require:
        name, _, b = r.partition("@")
        hits = [n for (k, kb), n in seen.items() if k == name and (not b or kb == int(b))]
        if not hits:
            missing.append(f"{r}: not in the dump")
        elif max(hits) == 0:
            missing.append(f"{r}: no constant division recognised (has the shader format changed?)")
    for m in missing:
        print(f"REQUIRED SHADER MISSING: {m}")
    if flagged:
        print("FAILED: a constant division of a half-width shift; Apple's Metal compiler miscompiles it "
              "(see the notes at the top of scripts/wgsl-lint.py)")
        return 1
    return 2 if missing else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
