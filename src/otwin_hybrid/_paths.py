"""Where the data, results and figures live.

This module exists because of a real failure. Both scripts originally used
``Path(__file__).parent`` while they sat at the repository root. Moving them
into ``src/otwin_hybrid/`` silently repointed every path *inside the package* —
so ``python -m otwin_hybrid.figures``, the exact command the README tells people
to run, looked for ``src/otwin_hybrid/results/curves.csv`` and crashed. The
files were sitting one directory up the whole time.

Nothing about that is exotic; it is what happens whenever a script becomes a
package. The fix is to anchor on a file that only ever exists at the repository
root, and to say so out loud when it cannot be found.
"""

from __future__ import annotations

import os
from pathlib import Path

# DATA_FILE / RESULTS_DIR / FIGURES_DIR are also importable -- see __getattr__
# at the bottom. They are deliberately out of __all__ because they are not
# module attributes until something asks for them.
__all__ = ["repo_root", "data_file", "results_dir", "figures_dir"]

# The anchor. Present at the root of a checkout and nowhere else.
_ANCHOR = Path("data") / "battery_soh.csv"


def repo_root() -> Path:
    """Locate the repository root.

    Order of preference:

    1. ``OTWIN_HYBRID_ROOT`` — for anyone vendoring this into a different layout.
    2. Walk up from this file until the anchor appears. Covers an editable
       install, a plain checkout, and being run from any subdirectory.
    3. The current working directory, if the anchor is there. Covers a Colab
       runtime that cloned the repo and ``cd``-ed into it.

    Raises:
        FileNotFoundError: with the paths that were tried, rather than a
            ``FileNotFoundError`` on some CSV three frames down that gives no
            hint about which of these went wrong.
    """
    if env := os.environ.get("OTWIN_HYBRID_ROOT"):
        root = Path(env).expanduser().resolve()
        if (root / _ANCHOR).is_file():
            return root
        raise FileNotFoundError(f"OTWIN_HYBRID_ROOT={root} does not contain {_ANCHOR}")

    here = Path(__file__).resolve()
    tried = []
    for candidate in (*here.parents, Path.cwd().resolve()):
        tried.append(candidate)
        if (candidate / _ANCHOR).is_file():
            return candidate

    raise FileNotFoundError(
        f"Could not locate the otwin-hybrid repository root: no directory "
        f"containing {_ANCHOR} was found in {tried[0]} .. {tried[-1]}. "
        f"Set OTWIN_HYBRID_ROOT, or run from a checkout."
    )


# Resolved lazily. Importing this module must never raise: the wheel ships
# `data/battery_soh.csv` as package data, so `import otwin_hybrid.comparison`
# has to work in a site-packages install where there is no repository at all.
def _packaged_data() -> Path | None:
    """The copy of the dataset that ships inside the wheel, if present."""
    p = Path(__file__).resolve().parent / "data" / "battery_soh.csv"
    return p if p.is_file() else None


def data_file() -> Path:
    """The dataset. Prefers a repository checkout, falls back to the wheel copy.

    The checkout wins so that editing `data/battery_soh.csv` and re-running has
    the effect you expect; the packaged copy is what makes `pip install
    otwin-hybrid && python -m otwin_hybrid.comparison` work at all.
    """
    try:
        return repo_root() / _ANCHOR
    except FileNotFoundError:
        packaged = _packaged_data()
        if packaged is not None:
            return packaged
        raise


def results_dir() -> Path:
    """Where `comparison` writes. Falls back to the working directory."""
    return _output_root() / "results"


def figures_dir() -> Path:
    """Where `figures` writes. Falls back to the working directory."""
    return _output_root() / "figures"


def _output_root() -> Path:
    try:
        return repo_root()
    except FileNotFoundError:
        # Installed, not checked out. Writing into site-packages would be
        # wrong, so outputs land beside the user instead.
        return Path.cwd()


def __getattr__(name: str):
    """Keep `DATA_FILE` / `RESULTS_DIR` / `FIGURES_DIR` working as names.

    Module-level `__getattr__` (PEP 562) makes these behave like constants at
    the call site while resolving on first access rather than at import. The
    earlier eager version made the whole package unimportable outside a
    checkout, which is exactly the failure this file was written to prevent.
    """
    if name == "DATA_FILE":
        return data_file()
    if name == "RESULTS_DIR":
        return results_dir()
    if name == "FIGURES_DIR":
        return figures_dir()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
