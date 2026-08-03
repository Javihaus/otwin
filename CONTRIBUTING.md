# Contributing to otwin-hybrid

This repository is a **tutorial**, and that shapes what a good contribution is.
The engineered library lives elsewhere in [the Otwin
ecosystem](https://github.com/otwin-core); here the deliverable is
*understanding*, and a change that makes the code better while making the story
harder to follow is a change in the wrong direction.

---

## The four things most wanted

### 1. Get the Julia or R notebook actually running

**This is the highest-value contribution and it needs nothing but a working
interpreter.**

The Python notebook is executed end to end in CI on every commit. The Julia and
R notebooks were written against their libraries but **could not be executed
where they were built** — no Julia or R runtime was available. The README says
so plainly rather than implying all three are verified.

If you open one in Colab and it fails, that is a bug and an issue is welcome.
If you fix it, that is a merged PR and your name on the citation.

What "running" means: the notebook reproduces the headline table, and the
ranking (hybrid < drift < physics < persistence < gp on RMSE) is unchanged.
Last-digit differences between languages are expected — different optimisers,
different RNGs — and are fine. A different *ranking* is a real finding and
should be reported as one rather than tuned away.

### 2. Add uncertainty

Every forecast here is a single line, and a single line is not a forecast. This
is the tutorial's biggest gap and it is stated as such in the README.

A contribution that produces a calibrated interval — and then *measures* whether
the stated 90 % band contains the truth 90 % of the time — would be the single
biggest improvement to the page. See
[`otwin-uq`](https://github.com/otwin-core/otwin-uq).

### 3. Break a result

If you re-run this and get different numbers, that is important and it is
welcome. Open an issue with your platform, Python version and what you got.

The most useful version of this is a **negative** result: a cell, a split
fraction or a metric under which the hybrid loses. The README already reports
three of these (B0018, the drift baseline beating the physics on RMSE, the GP
scoring worse than persistence). A fourth would be an improvement, not an
embarrassment.

### 4. Another chemistry or another temperature

All four cells are at 24 °C, and temperature is the single biggest driver of
degradation. Anything this repository says about a hotter or colder cell would
be invention.

A second dataset at a different temperature would let the power-law exponent
mean something across conditions rather than within one.

---

## What will not be merged

- **A bigger network.** `(16, 16)` forecasts better than `(256, 256)` and that
  is the point of the section. If you can show otherwise on held-out data, that
  is a finding — open an issue first.
- **Random splits.** `train_test_split` on a degradation series leaks, and the
  scores it produces are spectacular. That is the trap the whole page is about.
- **Removing a losing row from the table.** The GP's skill of 1.62 and the drift
  baseline beating the physics both stay.
- **A tuned result with no protocol change.** If a number improves, the commit
  must say what changed about the *method*, not just that the number is nicer.

---

## Working on it

```bash
git clone https://github.com/otwin-core/otwin-hybrid
cd otwin-hybrid
pip install -e ".[dev,examples]"

pytest -ra                        # the README's numbers are pinned here
ruff check . && black --check .
python -m otwin_hybrid.comparison # regenerates results/
python -m otwin_hybrid.figures    # regenerates figures/
```

**Commit `results/` when it changes.** CI runs
`git diff --exit-code -- results/` after regenerating, so a stale committed
result is a red build rather than a quiet inconsistency.

**If you change a number, change the README in the same commit.**
`tests/test_reproducibility.py` holds the README's table as literal constants
and will fail until you do. That is deliberate: the failure mode this repository
is guarding against is prose that describes a run nobody can reproduce.

---

## Style

The notebooks are excluded from `ruff` and `black` on purpose. They are written
to be read top to bottom on a Colab pane, which means compact idioms a linter
dislikes. Keep them that way.

The library code under `src/` is linted and formatted normally.

---

## The bar for prose

Every claim needs a number behind it, and every number needs a script that
produces it. If you write "the hybrid performs better", say better at *what*,
by *how much*, against *which baseline*, on *which split*.

Where something did not work, say so in the same voice as where it did.
