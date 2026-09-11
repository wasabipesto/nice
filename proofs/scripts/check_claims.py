#!/usr/bin/env python3
"""Validate CLAIMS.md against the built Lean library and the Rust `Lean:` tags.

Checks, in order:
1. Every row parses and has a known status.
2. Every `Lean:` tag in the Rust sources (`/// Lean: `Nice.Foo.bar` (ID)`) names a
   registry row whose `lean` column is that declaration.
3. Every declaration with status `def`, `proved`, `stated`, `conjecture` or
   `refuted` exists in the built library (`#check`).
4. Every `proved` (and `def`) declaration depends on no axiom beyond
   `propext`, `Classical.choice`, `Quot.sound` (so no `sorryAx`, no
   `Lean.ofReduceBool` from `native_decide`).
5. A `stated`/`planned` declaration that exists and is sorry-free is reported as
   promotable (warning, not failure).

Run from `proofs/` after `lake build`, or via `just lean-claims`.
Exit code 1 on any failure.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent  # proofs/
REPO = HERE.parent
CLAIMS = HERE / "CLAIMS.md"
RUST_ROOTS = [REPO / "common" / "src", REPO / "client" / "src", REPO / "api" / "src",
              REPO / "jobs" / "src", REPO / "scripts"]
STATUSES = {"def", "proved", "stated", "planned", "conjecture", "refuted",
            "rust-test", "device-test", "sql-audit"}
LEAN_STATUSES = {"def", "proved", "stated", "conjecture", "refuted"}
AXIOM_OK = {"propext", "Classical.choice", "Quot.sound"}
TAG_RE = re.compile(r"Lean:\s*`?([A-Za-z0-9_.'«»]+)`?\s*\(([A-Z]+-[0-9A-Za-z]+)\)")


def parse_claims() -> list[dict]:
    text = CLAIMS.read_text()
    body = text.split("<!-- claims:begin -->")[1].split("<!-- claims:end -->")[0]
    rows = []
    for line in body.strip().splitlines():
        if not line.startswith("|") or line.startswith("|---") or line.startswith("| id"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 7:
            sys.exit(f"CLAIMS.md: bad row ({len(cells)} cells): {line}")
        cid, stmt, lean, rust, evidence, status, phase = cells
        lean = lean.strip("`")
        if lean in {"—", "-", ""}:
            lean = None
        rows.append(dict(id=cid, statement=stmt, lean=lean, rust=rust,
                         evidence=evidence, status=status, phase=phase))
    return rows


def rust_tags() -> list[tuple[Path, int, str, str]]:
    out = []
    for root in RUST_ROOTS:
        if not root.exists():
            continue
        for path in list(root.rglob("*.rs")) + list(root.rglob("*.cu")):
            for lineno, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
                for m in TAG_RE.finditer(line):
                    out.append((path, lineno, m.group(1), m.group(2)))
    return out


def lean_query(decls: list[str], axioms_for: list[str]) -> dict[str, dict]:
    """One `lake env lean` run: `#check` every decl, `#print axioms` for some."""
    lines = ["import Nice", ""]
    for d in decls:
        lines.append(f'#check @{d}')
    for d in axioms_for:
        lines.append(f"#print axioms {d}")
    with tempfile.NamedTemporaryFile("w", suffix=".lean", delete=False, dir=HERE) as f:
        f.write("\n".join(lines) + "\n")
        tmp = Path(f.name)
    try:
        proc = subprocess.run(["lake", "env", "lean", str(tmp)], cwd=HERE,
                              capture_output=True, text=True)
    finally:
        tmp.unlink()
    result = {d: {"exists": True, "axioms": None} for d in decls}
    for line in proc.stdout.splitlines() + proc.stderr.splitlines():
        m = re.search(r"unknown (?:constant|identifier) [`']?([^`'\s]+)[`']?", line, re.I)
        if m:
            name = m.group(1).lstrip("@")
            if name in result:
                result[name]["exists"] = False
        m = re.match(r"'([^']+)' depends on axioms: \[(.*)\]", line.strip())
        if m:
            result.setdefault(m.group(1), {"exists": True})["axioms"] = \
                [a.strip() for a in m.group(2).split(",") if a.strip()]
        m = re.match(r"'([^']+)' does not depend on any axioms", line.strip())
        if m:
            result.setdefault(m.group(1), {"exists": True})["axioms"] = []
    # Only "unknown constant/identifier" errors are expected (missing
    # declarations); anything else means the library did not build.
    for line in proc.stdout.splitlines() + proc.stderr.splitlines():
        if "error" in line and not re.search(r"unknown (?:constant|identifier)", line, re.I):
            print(proc.stdout, proc.stderr, file=sys.stderr)
            sys.exit("lake env lean failed; run `lake build` first")
    return result


def main() -> int:
    rows = parse_claims()
    failures: list[str] = []
    warnings: list[str] = []
    by_id = {r["id"]: r for r in rows}
    by_lean = {r["lean"]: r for r in rows if r["lean"]}

    for r in rows:
        if r["status"] not in STATUSES:
            failures.append(f"{r['id']}: unknown status {r['status']!r}")
        if r["status"] in LEAN_STATUSES and not r["lean"]:
            failures.append(f"{r['id']}: status {r['status']} needs a lean declaration")

    for path, lineno, decl, cid in rust_tags():
        where = f"{path.relative_to(REPO)}:{lineno}"
        row = by_id.get(cid)
        if row is None:
            failures.append(f"{where}: tag names unknown claim {cid}")
        elif row["lean"] != decl:
            failures.append(f"{where}: tag {decl} but {cid} is {row['lean']}")

    to_check = [r["lean"] for r in rows if r["lean"] and r["status"] in LEAN_STATUSES | {"planned"}]
    axioms_for = [r["lean"] for r in rows if r["lean"] and r["status"] in LEAN_STATUSES | {"planned"}]
    info = lean_query(to_check, axioms_for)

    for r in rows:
        d = r["lean"]
        if not d or d not in info:
            continue
        exists = info[d]["exists"]
        axioms = info[d]["axioms"]
        if r["status"] in LEAN_STATUSES and not exists:
            failures.append(f"{r['id']}: {d} is {r['status']} but does not exist")
            continue
        if not exists:
            continue
        bad = set(axioms or []) - AXIOM_OK
        if r["status"] in {"proved", "def"} and bad:
            failures.append(f"{r['id']}: {d} is {r['status']} but depends on {sorted(bad)}")
        if r["status"] in {"stated", "planned"} and axioms is not None and not bad:
            warnings.append(f"{r['id']}: {d} is sorry-free; promote to proved")

    for w in warnings:
        print(f"warning: {w}")
    for f in failures:
        print(f"FAIL: {f}")
    counts = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print("status counts:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    update_readme(rows)
    return 1 if failures else 0


def update_readme(rows: list[dict]) -> None:
    """Rewrite the README's status section: counts per phase and the proved rows."""
    readme = HERE / "README.md"
    text = readme.read_text()
    phases = sorted({r["phase"] for r in rows}, key=lambda p: (p == "—", p))
    lines = ["| phase | def | proved | stated | planned | other |", "|---|---|---|---|---|---|"]
    for ph in phases:
        rs = [r for r in rows if r["phase"] == ph]
        c = lambda st: sum(1 for r in rs if r["status"] == st)  # noqa: E731
        other = sum(1 for r in rs if r["status"] not in {"def", "proved", "stated", "planned"})
        lines.append(f"| {ph} | {c('def')} | {c('proved')} | {c('stated')} | {c('planned')} | {other} |")
    lines.append("")
    lines.append("Proved or defined so far:")
    lines.append("")
    for r in rows:
        if r["status"] in {"def", "proved", "refuted"}:
            lines.append(f"- **{r['id']}** `{r['lean']}`: {r['statement']}")
    section = "<!-- status:begin -->\n" + "\n".join(lines) + "\n<!-- status:end -->"
    new = re.sub(r"<!-- status:begin -->.*?<!-- status:end -->", section, text, flags=re.S)
    if new != text:
        readme.write_text(new)
        print("README.md status section updated")


if __name__ == "__main__":
    sys.exit(main())
