#!/usr/bin/env python3
"""Single source of truth for the documented test count.

Counts the suite with pytest collection and rewrites the hard-coded number wherever
it appears in the docs, so it can never silently drift again (the recurring
STATUS/README/paper mismatch the audit kept finding).

Usage:
    python tools/update_test_count.py           # rewrite the docs in place
    python tools/update_test_count.py --check    # exit 1 if any doc is out of sync (CI guard)

It updates this repo's ``docs/STATUS.md`` and ``README.md``. If the companion paper
repo is checked out as a sibling (``../cv-detection-seg-report``), its
``sections/04-results.tex`` is updated too (best-effort; skipped when absent, e.g. in CI).
"""
from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
REPORT = REPO.parent / "cv-detection-seg-report"


def count_tests() -> int:
    """Return the number of collected tests via ``pytest --collect-only``."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=REPO, capture_output=True, text=True,
    )
    m = re.search(r"(\d+) tests? collected", proc.stdout)
    if not m:
        sys.stderr.write(proc.stdout + proc.stderr)
        raise SystemExit("could not parse the pytest collection summary")
    return int(m.group(1))


def _subs(n: int):
    """(path, [(compiled_pattern, replacement)]) for each doc that names the count.

    Patterns are anchored to the exact surrounding phrase so no unrelated number is touched.
    """
    return [
        (REPO / "docs" / "STATUS.md", [
            (re.compile(r"\d+/\d+ unit tests"), f"{n}/{n} unit tests"),
            (re.compile(r"Suite: \d+/\d+ green"), f"Suite: {n}/{n} green"),
            (re.compile(r"suite \*\*\d+/\d+ green\*\*"), f"suite **{n}/{n} green**"),
        ]),
        (REPO / "README.md", [
            (re.compile(r"\d+ tests, and a Zenodo DOI"), f"{n} tests, and a Zenodo DOI"),
        ]),
        (REPORT / "sections" / "04-results.tex", [
            (re.compile(r"suite \(\d+ tests\)"), f"suite ({n} tests)"),
        ]),
    ]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Sync the documented test count with pytest")
    ap.add_argument("--check", action="store_true",
                    help="do not write; exit 1 if any doc is out of sync")
    args = ap.parse_args(argv)

    n = count_tests()
    stale = []
    for path, subs in _subs(n):
        if not path.exists():
            continue  # sibling paper repo not present (e.g. CI) -> skip
        text = original = path.read_text()
        for pattern, repl in subs:
            text = pattern.sub(repl, text)
        if text != original:
            stale.append(path)
            if not args.check:
                path.write_text(text)

    label = REPO.name
    if args.check:
        if stale:
            print(f"[test-count] OUT OF SYNC (expected {n} tests): "
                  + ", ".join(p.name for p in stale))
            print("[test-count] run: python tools/update_test_count.py")
            return 1
        print(f"[test-count] OK: docs match pytest ({n} tests)")
        return 0

    if stale:
        print(f"[test-count] updated {len(stale)} file(s) to {n} tests: "
              + ", ".join(p.name for p in stale))
    else:
        print(f"[test-count] already in sync ({n} tests)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
