# Getting Started with otwin v2

> **Port-Hamiltonian Digital Twins with Structure by Construction**

---

## Quick Install

### Core (numpy + scipy only)
```bash
cd otwin
pip install -e .
```

### With all extras (recommended for development)
```bash
pip install -e ".[dev,torch,viz]"
```

### Individual extras
```bash
pip install -e ".[torch]"  # For learned PHS (Phase 4)
pip install -e ".[gp]"     # For GP-PHS (Phase 4)
pip install -e ".[viz]"    # For plotting
pip install -e ".[dev]"    # For testing/linting/typing
```

---

## First Steps

### 1. Verify Installation

```bash
python -c "import otwin; print(f'otwin version: {otwin.__version__}')"
```

Expected output: `otwin version: 2.0.0-alpha`

### 2. Run Water Tank Example

```bash
cd examples/water_tank_phs
python water_tank_demo.py
```

This demonstrates:
- Port-Hamiltonian structure
- Energy decay (passivity)
- Power balance identity
- Structure preservation

### 3. Run Tests

```bash
# All tests
make test

# Fast tests only (skip slow property tests)
make test-fast

# With coverage
make test-coverage
```

### 4. Check Code Quality

```bash
# Run all checks (lint + type + test)
make check

# Individual checks
make lint    # Ruff linter
make type    # mypy --strict
make format  # Black formatter
```

---

## Your First Digital Twin

### Example 1: Water Tank (Analytic PHS)

```python
from otwin import DigitalTwin, water_tank
import numpy as np

# Create port-Hamiltonian system
tank = water_tank(A=1.0, a=0.1, g=9.81)

# Create digital twin
twin = DigitalTwin(model=tank)

# Initial state: 2m water height
x0 = np.array([2.0])

# Time grid
t = np.linspace(0, 20, 200)

# No inflow
u = np.zeros((200, 1))

# Forecast
forecast = twin.forecast(x0, t, u)

# Check passivity: energy should decrease
energies = [tank.energy(x) for x in forecast['x']]
print(f"Initial energy: {energies[0]:.2f} J")
print(f"Final energy: {energies[-1]:.2f} J")
print(f"Energy decreased: {energies[-1] < energies[0]}")  # True

# Verify power balance
pb = tank.power_balance(forecast['x'][0], np.array([0.0]))
print(f"Power balance: dH/dt = {pb['dH_dt']:.6f}")
print(f"Dissipated: {pb['dissipated']:.6f}")
print(f"Supplied: {pb['supplied']:.6f}")
```

### Example 2: Mass-Spring-Damper

```python
from otwin import DigitalTwin, mass_spring_damper
import numpy as np

# Create system
sys = mass_spring_damper(m=1.0, k=1.0, c=0.5)

# Initial state: displaced, at rest
x0 = np.array([1.0, 0.0])  # [position, momentum]

# Time and inputs
t = np.linspace(0, 10, 100)
u = np.zeros((100, 1))  # No external force

# Create twin and forecast
twin = DigitalTwin(model=sys)
result = twin.forecast(x0, t, u)

# Energy should decay exponentially with damping
import matplotlib.pyplot as plt   # requires: pip install "otwin[viz]"
energies = [sys.energy(x) for x in result['x']]
plt.plot(result['t'], energies)
plt.xlabel('Time (s)')
plt.ylabel('Energy (J)')
plt.title('Energy Decay in Damped System')
plt.grid(True)
plt.show()
```

### Example 3: Rigorous Evaluation

```python
from otwin import evaluate
from otwin.evaluation import temporal_holdout
import numpy as np

# Generate synthetic data
# (In real use: load your time-series data)
data = np.cumsum(np.random.randn(200, 1) * 0.1) + 10

# Create a simple model for demonstration
class SimpleModel:
    def predict(self, X):
        # Naive: predict mean
        return np.full_like(X, X.mean())

model = SimpleModel()

# Evaluate with temporal split + baselines
report = evaluate(
    model,
    data,
    protocol='temporal_holdout',  # Default: temporal split
    test_frac=0.2
)

print(report)
# Output shows:
# - Skill score (vs best baseline) FIRST
# - Baseline name (e.g., "persistence")
# - Point metrics: RMSE, MAE, MASE, Theil_U
# - All compared to baseline
```

---

## Understanding the Structure

### Port-Hamiltonian Systems

A PHS has the form:
```
ẋ = (J(x) − R(x)) ∇H(x) + g(x) u
y = g(x)ᵀ ∇H(x)
```

where:
- `H(x)`: Energy/storage function
- `J(x) = −J(x)ᵀ`: Skew-symmetric (lossless interconnection)
- `R(x) ⪰ 0`: Positive semidefinite (dissipation)
- `g(x)`: Input/output map

**Power Balance (guaranteed by construction):**
```
dH/dt = −∇Hᵀ R ∇H + yᵀu ≤ yᵀu
```

With `u = 0`, energy is non-increasing → **passive system**.

### Why This Matters

**Traditional neural ODEs:**
```text
dx = neural_net(x, u)          # no structure
```
→ Can violate physics, create energy, drift on long horizons

**Port-Hamiltonian neural networks (Phase 4):**
```text
dx = (J_θ(x) − R_θ(x)) ∇H_θ(x) + g_θ(x) u
J_θ = A_θ − A_θᵀ   (skew by construction)
R_θ = L_θ L_θᵀ   (PSD by construction)
```
→ **Structure enforced regardless of learned weights**
→ No energy-creating drift, by algebra

---

## Rigorous Evaluation

### Default: Temporal Split
```python
from otwin.evaluation import temporal_holdout

train, test = temporal_holdout(data, test_frac=0.2)
# Last 20% is test (chronologically)
# This measures FORECASTING
```

### Recommended: Rolling-Origin
```python
report = evaluate(
    model,
    data,
    protocol='rolling_origin',
    n_folds=5,
    horizon=10
)
# Multiple train/test windows
# Most robust for forecasting evaluation
```

### Warning: Random Split
```python
from otwin.evaluation import random_split

train, test = random_split(data, test_frac=0.2)
# ⚠️  Emits LOUD WARNING:
# "Random split measures INTERPOLATION, not FORECASTING"
```

### Baselines (Automatic)
The evaluation always computes:
- `persistence`: Last value repeated
- `drift`: Linear extrapolation
- `mean`: Training mean
- `seasonal_naive`: Last season (if period given)

**Best baseline** is selected by RMSE and used for skill score.

### Skill Score (Headline Metric)
```
Skill Score = (baseline_error - model_error) / baseline_error

> 0: Better than baseline
= 0: Same as baseline
< 0: Worse than baseline
```

This is shown **FIRST** in every evaluation report.

---

## File Structure

```
otwin/
├── src/otwin/              # Main package
│   ├── systems/          # PHS, IPHS, library
│   ├── learn/            # Learned PHS (Phase 4)
│   ├── integrate/        # Time integration
│   ├── uq/               # Uncertainty quantification
│   ├── twin/             # DigitalTwin interface
│   ├── evaluation/       # Evaluation harness
│   ├── data/             # Data loaders
│   └── utils/            # Utilities
├── tests/                # Comprehensive tests
├── examples/             # Runnable examples
├── benchmarks/           # Reproducible benchmarks
└── docs/                 # Documentation

# v2-specific files
├── pyproject_v2.toml     # New dependencies/config
├── CITATIONS.md          # Reference tracking
├── CHANGELOG.md          # v2 changelog
├── NOTICE_v1_to_v2.md    # Migration story
└── PHASE_*_REPORT.md     # Implementation reports
```

---

## Common Tasks

### Verify Structure is Preserved
```python
from otwin import water_tank

tank = water_tank()
x = np.array([1.5])

# Check structural properties
structure = tank.check_structure(x)

is_skew, violation = structure['J_skew']
print(f"J skew-symmetric: {is_skew} (violation: {violation})")

is_psd, min_eig = structure['R_psd']
print(f"R positive semidefinite: {is_psd} (min eigenvalue: {min_eig})")
```

### Check Power Balance
```python
x = np.array([2.0])
u = np.array([0.0])

pb = tank.power_balance(x, u)

print(f"dH/dt = {pb['dH_dt']:.6f}")
print(f"Dissipated = {pb['dissipated']:.6f}")
print(f"Supplied = {pb['supplied']:.6f}")

# Identity must hold:
assert abs(pb['dH_dt'] - (pb['dissipated'] + pb['supplied'])) < 1e-8
```

### Generate Evaluation Report
```python
from otwin import evaluate

report = evaluate(
    model,
    data,
    protocol='rolling_origin',
    n_folds=5,
    horizon=10,
    seed=42
)

# Print to console
print(report)

# Save to JSON (reproducible)
report.to_json('results/my_model.json')

# Generate markdown table
print(report.to_markdown())
```

---

## Development Workflow

### 1. Create Feature Branch
```bash
git checkout -b feature/my-feature
```

### 2. Make Changes
```bash
# Edit code in src/otwin/
# Add tests in tests/
```

### 3. Run Checks
```bash
make format  # Format code
make check   # Lint + type + test
```

### 4. Run Full Test Suite
```bash
make test-coverage
# Check htmlcov/index.html for coverage report
```

### 5. Commit with Hooks
```bash
# CI will run automatically on push
# All checks must pass (no continue-on-error)
```

---

## What's Implemented vs Planned

### ✅ **Fully Functional** (Use Now)
- Port-Hamiltonian systems (analytic)
- Irreversible PHS with entropy production
- Time integration (scipy RK45)
- Evaluation harness (temporal splits, baselines, metrics)
- DigitalTwin interface (analytic models)
- Water tank & mass-spring-damper examples
- Property-based tests (structural guarantees)

### 🔧 **Structured** (Ready for Implementation)
- PortHamiltonianNN (learned, requires torch)
- Deep ensembles (UQ)
- GP-PHS (requires gpytorch)
- Port interconnection (composition)
- Battery NASA example (learned model)

### 📋 **Planned** (Future)
- More reference systems (electrical, thermal)
- Extended Kalman Filter (data assimilation)
- Structure-preserving integrators (implicit-midpoint)
- Full documentation website

---

## Getting Help

### Documentation
- API docs: See docstrings in source
- Theory: See phase reports (PHASE_1_REPORT.md, etc.)
- Examples: `examples/` directory

### Debugging

**Import error:**
```bash
# Make sure you're in the repo root
cd otwin

# Install in development mode
pip install -e .

# Or set PYTHONPATH
export PYTHONPATH=src:$PYTHONPATH
```

**Test failures:**
```bash
# Run specific test
pytest tests/systems/test_phs.py::test_water_tank_structure -v

# Run with debugging
pytest tests/systems/test_phs.py -v -s
```

**Type errors:**
```bash
# Check specific file
mypy --strict src/otwin/systems/phs.py
```

---

## Next Steps

1. **Run the examples** to see the library in action
2. **Read the phase reports** to understand the design
3. **Explore the tests** to see usage patterns
4. **Try your own PHS** using the library functions
5. **Evaluate** with the evaluation harness (temporal split + baselines)

---

**Welcome to otwin v2 — where structure is guaranteed, not hoped for! 🎉**
