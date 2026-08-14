"""Compare regenerated results/ against the committed ones, numerically.

This replaces `git diff --exit-code -- results/`.

That check asserted the CSVs were byte-identical, which is a claim about
floating point, not about this code. It held for a while and then stopped: the
numbers move in their last digits when numpy and scipy change, and -- as CI
proved -- they also move between machines on the *same* pinned versions,
because the OpenBLAS shipped inside those wheels dispatches to a different
kernel depending on the CPU it finds. Pinning versions cannot pin a CPU.

So byte-identity was never a property anyone could hold; it was luck that
expired. What is true, checkable anywhere, and actually worth defending is
that the committed numbers are reproducible to a tolerance far tighter than
the four significant figures the README prints.

Non-numeric columns must still match exactly. A renamed battery or a moved
split point is a real difference, not a rounding one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

#: One part in a million. The same tolerance tests/test_reproducibility.py
#: uses, for the same reason: below roughly 1e-8 these fits report the LAPACK
#: underneath them rather than anything about this repository.
RTOL = 1e-6


def compare(committed: Path, fresh: Path) -> list[str]:
    problems: list[str] = []
    for path in sorted(committed.glob("*.csv")):
        other = fresh / path.name
        if not other.is_file():
            problems.append(f"{path.name}: regenerated file is missing")
            continue
        a, b = pd.read_csv(path), pd.read_csv(other)
        if list(a.columns) != list(b.columns):
            problems.append(f"{path.name}: columns changed")
            continue
        if len(a) != len(b):
            problems.append(f"{path.name}: {len(a)} rows committed, {len(b)} fresh")
            continue
        for col in a.columns:
            if pd.api.types.is_numeric_dtype(a[col]):
                x, y = a[col].to_numpy(float), b[col].to_numpy(float)
                both_nan = pd.isna(x) & pd.isna(y)
                if not (pd.isna(x) == pd.isna(y)).all():
                    problems.append(f"{path.name}:{col}: NaN pattern changed")
                    continue
                mask = ~both_nan
                if not mask.any():
                    continue
                denom = abs(x[mask])
                denom[denom == 0] = 1.0
                worst = float((abs(x[mask] - y[mask]) / denom).max())
                if worst > RTOL:
                    problems.append(
                        f"{path.name}:{col}: worst relative change {worst:.3e} "
                        f"exceeds {RTOL:.0e}"
                    )
                else:
                    print(f"  {path.name}:{col:<12} worst {worst:.3e}")
            elif not a[col].equals(b[col]):
                problems.append(f"{path.name}:{col}: non-numeric values changed")
    return problems


def main() -> int:
    committed, fresh = Path(sys.argv[1]), Path(sys.argv[2])
    print(f"comparing {committed} (committed) against {fresh} (regenerated)")
    problems = compare(committed, fresh)
    if problems:
        print("\nthe regenerated results do not match what is committed:")
        for p in problems:
            print(f"  {p}")
        return 1
    print(f"\nevery number reproduces within {RTOL:.0e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
