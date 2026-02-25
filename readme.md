# Two-Phase Simplex (MATLAB + Python)

Two implementations of a two-phase simplex solver for LPs of the form:

- Maximize: `c^T x`
- Subject to: `A x (<=, >=, =) b`
- With non-negativity: `x >= 0`

Both versions solve the same model.  
The Python version has a more advanced teaching/visualization interface.

## Project Structure

- `two_phase_/`  
  MATLAB implementation:
  - `two_phase_simplex` (solver)
  - `pivot.m` (required helper)
  - `run.m` (example runner)

- `python simplex/`  
  Python implementation:
  - `two_phase_simplex.py` (solver + interactive viewer)
  - `big_m_simplex.py` (Big-M solver)
  - `simplex_utils.py` (shared simplex/viewer/report utilities)
  - `usage.py` (example usage)

- `requirements.txt`  
  Python dependencies used in this environment.

## Versions and Capabilities

- Python version: `3.13`
- MATLAB version: included in `two_phase_/`

Viewer capabilities:

- Python:
  - 2D and 3D feasible-region visualization
  - Step-by-step simplex path
  - Teaching mode with pivot rationale
  - Objective-improvement plot under the tableau
- MATLAB:
  - Tableau viewer + 2D visualization support
  - 2-variable plotting focus

## Python Setup

From repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install numpy matplotlib scipy
```

`scipy` is optional but recommended for more robust convex hull behavior.

## Python Quick Start

Run the provided example:

```bash
cd "python simplex"
python usage.py
```

Or call directly:

```python
from two_phase_simplex import two_phase_simplex

c = [70, 130]
A = [[12, 6],
     [0, 15],
     [2, 8],
     [0, 1]]
b = [600, 300, 220, 10]
sense = ["<=", "<=", "<=", ">="]

res = two_phase_simplex(
    c, A, b, sense,
    opts={
        "launch_viewer": True,
        "teaching_mode": True,
        "pivot_rule": "dantzig",  # or "bland"
    },
)

print(res["x"], res["z"])
```

## Python API

### Function

```python
two_phase_simplex(c, A, b, sense, opts=None)
```

Inputs:

- `c`: objective coefficients (`list`/`array`, length `n`)
- `A`: constraint matrix (`m x n`)
- `b`: RHS vector (length `m`)
- `sense`: constraint types (length `m`, each one `"<="`, `">="`, or `"="`)
- `opts`:
  - `launch_viewer` (`bool`, default `True`)
  - `teaching_mode` (`bool`, default `True`)
  - `pivot_rule` (`"dantzig"` or `"bland"`, default `"dantzig"`)
  - `report_pdf_path` (`None`, `True/False`, or file path string, default `None`)
  - `tol` (`float`, default `1e-10`)

Returns (`dict`):

- `x`: optimal decision vector
- `z`: optimal objective value
- `tableau`: final tableau
- `basis`: final basis indices
- `var_names`: variable names in tableau
- `states`: full iteration history (for viewer/teaching)

### `states` (teaching/visualization trace)

Each state stores:

- tableau snapshot
- basis
- phase and step number
- entering/leaving variable names
- ratio-test values
- current plotted point (`x1..x3` when available)
- Phase II objective value `z`
- teaching metadata (`info`) including:
  - pivot rule used
  - reduced-cost and ratio tie diagnostics
  - degeneracy flags
  - phase transition diagnostics (artificial variable cleanup)

## Python Viewer Controls

- `Prev` / `Next`: move state-by-state
- `State` slider: jump to any tableau state
- `Teaching: ON/OFF` button: toggle explanations in the text panel

Displayed panels:

- Left: feasible region + simplex path (2D/3D)
- Right top: state header + tableau text
- Right bottom: objective progress chart for Phase II

## PDF Report Export (Python)

You can export all simplex states into a single PDF, including:

- state header (`phase`, `step`, entering/leaving variables, `Z`)
- teaching comments for each state
- full tableau text for each state
- geometric simplex plot (2D/3D when available)
- objective-progress chart on each page

Example:

```python
res = two_phase_simplex(
    c, A, b, sense,
    opts={
        "launch_viewer": False,
        "teaching_mode": True,
        "report_pdf_path": "simplex_report.pdf",
    },
)
```

Notes:

- `report_pdf_path=True` uses default filename `simplex_report.pdf`.
- `report_pdf_path=None` (or `False`) disables report generation.

## MATLAB Usage

In MATLAB:

1. Open `two_phase_/` (or `cd` into it).
2. Run:

```matlab
run
```

Direct call pattern:

```matlab
c = [...];
A = [...];
b = [...];
sense = ["<="; ">="; "="];

out = two_phase_simplex(c, A, b, sense);
disp(out.x);
disp(out.z);
```

MATLAB return struct fields:

- `out.x`
- `out.z`
- `out.tableau`
- `out.basis`
- `out.varNames`
- `out.states`

## Notes and Limitations

- LP model assumes non-negativity (`x >= 0`).
- Python plotting is available for `n = 2` and `n = 3`.
- For higher dimensions, solver still runs, but geometric plotting is not shown.
- MATLAB environment may use symbolic arithmetic in its solver path.

## Troubleshooting

- If Python import fails, run from inside `python simplex/` or add that folder to `PYTHONPATH`.
- If Matplotlib warns about cache directory permissions, set:

```bash
export MPLCONFIGDIR=/tmp/matplotlib-cache
```

before running Python scripts.
