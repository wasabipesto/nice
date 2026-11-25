#!/usr/bin/env python3
r"""Check that CubeCL's SPIR-V compiler kept every atomic operation of every
kernel, against the same kernels compiled to WGSL.

CubeCL 0.10's direct SPIR-V compiler (the `cubecl-spirv` feature, which the
`-gpu` docker image ships) runs an optimizer that merges equal instructions
within a basic block. An atomic store keeps its target outside what that pass
compares, so a second atomic store of the same value in the same block is
taken for a repeat of the first and removed, whatever it points at. The kernel
then reads stale memory on SPIR-V while staying correct through WGSL (which
skips that optimizer), CUDA and Metal; #263 lost a queue account to it.
Stores in separate blocks, plain (non-atomic) stores, and atomic
read-modify-write operations were seen to survive.

This script reads two `CUBECL_DEBUG_LOG` dumps of the same kernels, one from
a `cubecl` build (WGSL) and one from a `cubecl-spirv` build, pairs each kernel
variant (its name and compile-time arguments), and counts its atomic stores
and read-modify-write operations in each. A variant with fewer of any of them
in SPIR-V than in WGSL is flagged. Loads are not compared.

usage:
    spirv-atomic-check.py WGSL_LOG SPIRV_LOG [--require KERNEL]...
                          [--expect-dropped KERNEL]... [--verbose]

`--require` fails the run unless KERNEL was paired in both dumps with at
least one atomic operation, which catches a dump that quietly stopped
covering it and a change in either output format that the counts no longer
match. `--expect-dropped` names a kernel that must be flagged: the test dump
compiles one on purpose (`atomic_store_trap_kernel` in
common/src/cubecl_backend.rs), so a run that does not flag it is a check that
cannot see the drop, or a CubeCL whose optimizer no longer makes it (then
remove the kernel and, if the fix is confirmed, this check).

Exit status: 0 clean, 1 a kernel lost an atomic operation, 2 a requirement
or expectation failed.

Limits: it compares counts per kernel, so a lost operation hidden by an extra
one of the same kind elsewhere in that kernel is not seen, and it covers only
the kernel variants the dump compiles (see `wgsl_dump_for_lint`).
"""
import re
import sys

OPS = {
    # name: (SPIR-V instruction, WGSL builtin)
    "store": (r"\bOpAtomicStore\b", r"\batomicStore\("),
    "add": (r"\bOpAtomicIAdd\b", r"\batomicAdd\("),
    "sub": (r"\bOpAtomicISub\b", r"\batomicSub\("),
    "or": (r"\bOpAtomicOr\b", r"\batomicOr\("),
    "and": (r"\bOpAtomicAnd\b", r"\batomicAnd\("),
    "xor": (r"\bOpAtomicXor\b", r"\batomicXor\("),
    "max": (r"\bOpAtomic[US]Max\b", r"\batomicMax\("),
    "min": (r"\bOpAtomic[US]Min\b", r"\batomicMin\("),
    "exchange": (r"\bOpAtomicExchange\b", r"\batomicExchange\("),
    "compare-exchange": (r"\bOpAtomicCompareExchange\b", r"\batomicCompareExchangeWeak\("),
}


# Compile-time arguments the code sets from the compiler, not the work, so
# the two dumps hold different values: left out when pairing variants. Each
# must not change a kernel's atomic operations.
#   pf_count  the overlap join's prefilter tests its digits with a population
#             count on SPIR-V (cubecl_join.rs, `R::name(client)`); a pure test
#             on registers either way.
COMPILER_CHOSEN = ("pf_count",)


def fail(msg):
    """A dump this script cannot read: exit status 2."""
    print(f"spirv-atomic-check: {msg}")
    sys.exit(2)


def kernels(path, lang):
    """{(kernel name, compile-time arguments): {op: count}} for one dump."""
    col = 0 if lang == "spirv" else 1
    try:
        with open(path, errors="replace") as f:
            text = f.read()
    except OSError as e:
        fail(f"{path}: cannot be read ({e.strerror})")
    out = {}
    blocks = re.findall(r"\[START_KERNEL_COMPILATION\](.*?)\[END_KERNEL_COMPILATION\]", text, re.S)
    for blk in blocks:
        head, sep, src = blk.partition("\nsource:")
        name = re.search(r"^name: \S+?::(\w+)<", head, re.M)
        if not sep or not name:
            continue
        fence = re.match(r"\s*```(\w+)", src)
        if not fence or fence.group(1) != lang:
            fail(f"{path}: a {name.group(1)} block holds "
                             f"{fence.group(1) if fence else 'unfenced'} source, expected {lang}")
        # The kernel's identity: everything in the header but the name line,
        # whitespace-insensitive.
        ident = re.sub(r"\s+", " ", head.split("\nid:", 1)[-1]).strip()
        for arg in COMPILER_CHOSEN:
            ident = re.sub(rf"\b{arg}: \w+, ", "", ident)
        counts = {op: len(re.findall(pats[col], src)) for op, pats in OPS.items()}
        key = (name.group(1), ident)
        if key in out and out[key] != counts:
            fail(f"{path}: {name.group(1)} compiled twice with different atomic counts")
        out[key] = counts
    return out


def short(counts):
    return ", ".join(f"{op} {n}" for op, n in counts.items() if n) or "none"


def main(argv):
    require, expect, files = [], [], []
    verbose = False
    it = iter(argv)
    for a in it:
        if a in ("--require", "--expect-dropped"):
            name = next(it, None)
            if name is None:
                fail(f"{a} needs a kernel name")
            (require if a == "--require" else expect).append(name)
        elif a == "--verbose":
            verbose = True
        else:
            files.append(a)
    if len(files) != 2:
        print(__doc__)
        return 2
    wgsl, spirv = kernels(files[0], "wgsl"), kernels(files[1], "spirv")

    paired = sorted(set(wgsl) & set(spirv))
    flagged = {}  # kernel name -> variants flagged
    for key in paired:
        w, s = wgsl[key], spirv[key]
        lost = {op: (w[op], s[op]) for op in OPS if s[op] < w[op]}
        name = key[0]
        if lost:
            flagged[name] = flagged.get(name, 0) + 1
            what = ", ".join(f"{op} {a} in WGSL, {b} in SPIR-V" for op, (a, b) in lost.items())
            print(f"{name}: lost in SPIR-V: {what}")
        elif verbose:
            print(f"{name}: {short(w)}")

    names = sorted({k[0] for k in paired})
    print(f"spirv-atomic-check: {len(wgsl)} WGSL and {len(spirv)} SPIR-V kernel variants, "
          f"{len(paired)} paired ({', '.join(names)})")
    for side, mine, other in (("WGSL", wgsl, spirv), ("SPIR-V", spirv, wgsl)):
        alone = sorted({k[0] for k in set(mine) - set(other)})
        if alone:
            print(f"  only in the {side} dump, not compared: {', '.join(alone)}")

    problems = []
    for r in require:
        ops = [sum(wgsl[k].values()) for k in paired if k[0] == r]
        if not ops:
            problems.append(f"{r}: not paired in both dumps")
        elif max(ops) == 0:
            problems.append(f"{r}: no atomic operation counted (has a shader format changed?)")
    for e in expect:
        if e not in flagged:
            problems.append(f"{e}: expected to lose an atomic store in SPIR-V and did not "
                            "(see --expect-dropped in this script's notes)")
    for p in problems:
        print(f"REQUIREMENT FAILED: {p}")
    unexpected = sorted(set(flagged) - set(expect))
    if unexpected:
        print(f"FAILED: {', '.join(unexpected)} lost an atomic operation in SPIR-V; most likely two "
              "atomic stores of one value in one block (see the notes at the top of "
              "scripts/spirv-atomic-check.py)")
        return 1
    if problems:
        return 2
    print("spirv-atomic-check: no kernel lost an atomic operation"
          + (f" (expected: {', '.join(expect)})" if expect else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
