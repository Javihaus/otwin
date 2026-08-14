"""The numbers in the README must be the numbers the code produces.

This file exists because a tutorial repository fails in a specific way: the code
gets improved, the prose does not, and six months later the headline table is
describing a run nobody can reproduce. Every claim in the README that is a
number is pinned here.

The tolerances match the precision the README prints to, because ``SEED = 0``
is fixed and these are deterministic fits. A failure here means either a real regression or an
intentional change that the README has not caught up with — in both cases the
right response is to look, not to widen the tolerance.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from otwin_hybrid.comparison import (
    EOL,
    SEED,
    TRAIN_FRACTION,
    eol_cycle,
    fit_physics,
    load_soh,
    wang_power_law,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "battery_soh.csv"
SUMMARY = ROOT / "results" / "summary.json"

# From the README table. Written out longhand rather than loaded from the
# results file, so that regenerating results/ cannot silently update the
# expectation it is being checked against.
README_RMSE = {
    "hybrid": 0.0398,
    "drift": 0.0490,
    "physics": 0.0544,
    "persistence": 0.1109,
    "gp": 0.1791,
}
README_SKILL = {
    "hybrid": 0.36,
    "drift": 0.44,
    "physics": 0.50,
    "persistence": 1.00,
    "gp": 1.62,
}
README_EOL_ABS_ERROR = {"physics": 13.0, "hybrid": 21.9, "drift": 25.7}


@pytest.fixture(scope="module")
def summary() -> dict:
    if not SUMMARY.exists():  # pragma: no cover - guards a broken checkout
        pytest.skip("results/summary.json missing; run `python -m otwin_hybrid.comparison`")
    return json.loads(SUMMARY.read_text())


# ---------------------------------------------------------------------------
# The protocol itself
# ---------------------------------------------------------------------------


def test_protocol_constants_are_what_the_readme_says(summary: dict) -> None:
    assert SEED == 0
    assert TRAIN_FRACTION == 0.40
    assert EOL == 0.80
    assert summary["seed"] == SEED
    assert summary["train_fraction"] == TRAIN_FRACTION
    assert summary["eol_threshold"] == EOL


def test_split_is_temporal_not_random(summary: dict) -> None:
    """Train must be a strict prefix: n_train + n_test == total, no overlap."""
    df = load_soh(DATA)
    for row in summary["params"]:
        cell = df[df["Battery"] == row["battery"]]
        assert row["n_train"] + row["n_test"] == len(cell)
        assert row["n_train"] == int(len(cell) * TRAIN_FRACTION)  # floor, not round


# ---------------------------------------------------------------------------
# The headline table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model,expected", README_RMSE.items())
def test_rmse_matches_readme(summary: dict, model: str, expected: float) -> None:
    assert summary["summary"]["rmse"][model] == pytest.approx(expected, abs=5e-5)


@pytest.mark.parametrize("model,expected", README_SKILL.items())
def test_skill_matches_readme(summary: dict, model: str, expected: float) -> None:
    assert summary["summary"]["skill"][model] == pytest.approx(expected, abs=5e-3)


@pytest.mark.parametrize("model,expected", README_EOL_ABS_ERROR.items())
def test_eol_error_matches_readme(summary: dict, model: str, expected: float) -> None:
    got = summary["eol_error_cycles"]["mean_abs_error"][model]
    assert got == pytest.approx(expected, abs=5e-2)


def test_the_claim_that_the_ranking_inverts(summary: dict) -> None:
    """The README's central argument. If this ever stops holding, the prose is wrong.

    Hybrid wins on RMSE; physics wins on end-of-life. That inversion *is* the
    point of the page, so it gets its own test rather than being implied by the
    two tables above.
    """
    rmse = summary["summary"]["rmse"]
    eol = summary["eol_error_cycles"]["mean_abs_error"]
    assert rmse["hybrid"] < rmse["physics"], "hybrid should win on RMSE"
    assert eol["physics"] < eol["hybrid"], "physics should win on end-of-life"


def test_gp_is_worse_than_persistence(summary: dict) -> None:
    """Skill above 1.0 means the model lost to 'assume nothing changes'.

    Reported as a finding, not hidden. Pinned so nobody 'fixes' the GP into
    looking respectable without noticing the narrative depends on it.
    """
    assert summary["summary"]["skill"]["gp"] > 1.0


def test_drift_baseline_beats_physics_on_rmse(summary: dict) -> None:
    """The uncomfortable row. A straight line has lower RMSE than the physics."""
    assert summary["summary"]["rmse"]["drift"] < summary["summary"]["rmse"]["physics"]


# ---------------------------------------------------------------------------
# The physics
# ---------------------------------------------------------------------------


def test_power_law_is_one_at_zero_wear() -> None:
    assert wang_power_law(np.array([1.0]), c=0.0, z=1.0)[0] == pytest.approx(1.0)


def test_power_law_is_monotone_decreasing() -> None:
    n = np.arange(1, 500, dtype=float)
    soh = wang_power_law(n, c=1e-3, z=1.1)
    assert np.all(np.diff(soh) < 0)


def test_log_space_seeding_recovers_known_parameters() -> None:
    """Generate from the law, fit it back. Guards the conditioning fix itself."""
    n = np.arange(1, 200, dtype=float)
    c_true, z_true = 2.0e-4, 1.2
    soh = wang_power_law(n, c_true, z_true)
    c, z = fit_physics(n, soh)
    assert z == pytest.approx(z_true, rel=1e-3)
    assert c == pytest.approx(c_true, rel=1e-2)


def test_exponent_bound_is_disclosed_where_it_binds(summary: dict) -> None:
    """Two cells sit on z = 1.5. The README says so; keep that true.

    If a future change moves them off the bound, this test fails and the README
    paragraph about the bound must be rewritten rather than left stale.
    """
    on_bound = [p["battery"] for p in summary["params"] if p["z"] > 1.4999]
    assert sorted(on_bound) == ["B0005", "B0007"]


def test_eol_returns_none_when_threshold_is_never_crossed() -> None:
    n = np.arange(1, 50, dtype=float)
    soh = np.linspace(1.0, 0.95, n.size)
    assert eol_cycle(n, soh, threshold=0.80) is None


def test_b0006_is_excluded_from_eol_because_it_crosses_during_training(
    summary: dict,
) -> None:
    """Leaving it in makes every model look perfect. That is the trap."""
    df = load_soh(DATA)
    cell = df[df["Battery"] == "B0006"]
    n_train = int(len(cell) * TRAIN_FRACTION)
    train = cell.iloc[:n_train]
    crossing = eol_cycle(train["id_cycle"].to_numpy(), train["SoH"].to_numpy(), EOL)
    assert crossing is not None, "B0006 should cross 80% within the training window"


# ---------------------------------------------------------------------------
# Data integrity
# ---------------------------------------------------------------------------


def test_data_is_the_four_documented_cells() -> None:
    df = load_soh(DATA)
    assert sorted(df["Battery"].unique()) == ["B0005", "B0006", "B0007", "B0018"]


def test_soh_is_finite_and_plausible() -> None:
    df = load_soh(DATA)
    soh = df["SoH"].to_numpy()
    assert np.all(np.isfinite(soh))
    assert soh.min() > 0.0
    assert soh.max() <= 1.05


# ---------------------------------------------------------------------------
# End to end
#
# Everything above reads results/summary.json. That is the right check for
# "does the README match what was committed", and the wrong check for "does the
# code still work" -- a fresh reviewer showed that replacing rmse() with
# `return 0.0` left all of it green, because nothing recomputed. CI catches it
# via `git diff --exit-code -- results/`, but a contributor running pytest
# locally saw a false pass.
#
# This test closes that gap: it runs main() into a temporary directory and
# checks the numbers it actually produces.
# ---------------------------------------------------------------------------


# On the tolerance below, because the module docstring says to look rather
# than widen, and this is the record of having looked.
#
# It was rel=REPRODUCTION_TOLERANCE, and that failed on the newer half of the CI matrix. Measured
# values, same seed, same code, same machine:
#
#   numpy 2.2.6 / scipy 1.15.3   hybrid RMSE = 0.03983393462630047
#   numpy 2.4.6 / scipy 1.17.1   hybrid RMSE = 0.03983393462630047
#   numpy 2.5.2 / scipy 1.18.0   hybrid RMSE = 0.039833934781460995
#
# A relative difference of 3.9e-9, and it splits by library version, not by
# Python version. fit_physics is a nonlinear least-squares fit; the newer stack
# takes a different path through LAPACK and converges to a marginally different
# point. No seed fixes that, because it is not randomness -- it is
# floating-point associativity in someone else's library.
#
# So rel=REPRODUCTION_TOLERANCE was not testing this repository. It was asserting bit-identical
# LAPACK across releases, which is not a property anyone can hold, and it would
# have gone red on a dependency bump with nothing wrong here.
#
# rel=1e-6 is one part in a million: still four orders of magnitude tighter
# than the four significant figures the README prints, and far tighter than
# anything that could change a maintenance decision. The regressions this test
# exists to catch -- a wrong split, a broken residual model, rmse() returning
# zero -- move these numbers by percent, and are still caught.
#
# Bit-exactness has not been given up. It lives in the `reproduce` CI job,
# which regenerates results/ in one pinned environment and fails on
# `git diff --exit-code`. Exactness is a claim about one toolchain, not four.

#: Tolerance for "the code still produces the committed result". See above.
REPRODUCTION_TOLERANCE = 1e-6


def test_recomputing_from_scratch_reproduces_the_committed_summary(
    tmp_path, monkeypatch, summary: dict
) -> None:
    from otwin_hybrid import comparison

    monkeypatch.setattr(comparison, "results_dir", lambda: tmp_path)
    comparison.main()

    fresh = json.loads((tmp_path / "summary.json").read_text())

    for model, expected in summary["summary"]["rmse"].items():
        assert fresh["summary"]["rmse"][model] == pytest.approx(
            expected, rel=REPRODUCTION_TOLERANCE
        ), f"{model} RMSE moved: the code no longer produces the committed result"
    for model, expected in summary["eol_error_cycles"]["mean_abs_error"].items():
        assert fresh["eol_error_cycles"]["mean_abs_error"][model] == pytest.approx(
            expected, rel=REPRODUCTION_TOLERANCE
        )
    for a, b in zip(fresh["params"], summary["params"], strict=True):
        assert a["battery"] == b["battery"]
        assert a["z"] == pytest.approx(b["z"], rel=REPRODUCTION_TOLERANCE)
        assert a["c"] == pytest.approx(b["c"], rel=REPRODUCTION_TOLERANCE)
