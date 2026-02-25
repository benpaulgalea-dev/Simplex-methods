# Simplex Methods Project (MATLAB + Python)

This repository contains teaching-oriented simplex implementations for linear programs of the form:

- Maximize: `c^T x`
- Subject to: `A x (<=, >=, =) b`
- Non-negativity: `x >= 0`

The Python side includes:
- Two-Phase Simplex
- Big-M Simplex
- Interactive 2D/3D visualization
- Step-by-step tableau states and explanations
- PDF report export for every tableau state

## Project Structure

- `two_phase_/`
  - `two_phase_simplex` (MATLAB solver)
  - `pivot.m` (MATLAB helper)
  - `run.m` (MATLAB example)

- `python simplex/`
  - `two_phase_simplex.py` (Two-Phase solver)
  - `big_m_simplex.py` (Big-M solver)
  - `simplex_utils.py` (shared simplex/viewer/report utilities)
  - `usage.py` (Python usage example)
  - `small.py` (small 2D example)
  - `showcase.py` (interactive chooser: 2D/3D + Two-Phase/Big-M)
  - `super_showcase.py` (complex 3D full-capability demo)
  - `reports/` (generated PDF reports)

- `requirements.txt`
- `readme.md`

## Python Setup

From repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional (recommended for more robust hull/geometry handling):

```bash
pip install scipy
```

## Quick Start

### 1) Basic example

```bash
cd "python simplex"
python usage.py
```

### 2) Interactive showcase

```bash
cd "python simplex"
python showcase.py
```

`showcase.py` will ask for:
- dimension (`2D` or `3D`)
- method (`Two-Phase` or `Big-M`)

Then it runs a prebuilt example, launches the viewer, and generates a named report.

### 3) Super showcase (complex 3D)

```bash
cd "python simplex"
python super_showcase.py
```

This runs a larger mixed-constraint 3D model, executes both Two-Phase and Big-M, compares results, and writes two reports.

## Python API

### `two_phase_simplex`

```python
two_phase_simplex(c, A, b, sense, opts=None)
```

### `big_m_simplex`

```python
big_m_simplex(c, A, b, sense, opts=None)
```

### Shared input contract

- `c`: objective coefficients, length `n`
- `A`: constraint matrix, shape `m x n`
- `b`: RHS vector, length `m`
- `sense`: list/array of length `m`, each entry one of `"<="`, `">="`, `"="`

### Options (`opts`)

Common options:
- `launch_viewer` (`bool`, default `True`)
- `teaching_mode` (`bool`, default `True`)
- `pivot_rule` (`"dantzig"` or `"bland"`, default `"dantzig"`)
- `report_pdf_path` (`None`, `True/False`, or non-empty path string)
- `tol` (`float`, default `1e-10`)

Big-M specific:
- `big_m` (`float`, optional): penalty size for artificial variables

### Return dictionary

Both solvers return:
- `x`: optimal decision vector
- `z`: objective value
- `tableau`: final tableau
- `basis`: final basis indices
- `var_names`: tableau variable names
- `states`: full state history

`big_m_simplex` also returns:
- `big_m`: penalty value used

## Viewer + Teaching Features

The Python viewer supports:
- 2D/3D feasible region and extreme points
- simplex path through states
- state slider + prev/next controls
- teaching-mode explanations for pivot selection and transitions
- objective-progress plot for Phase II

For each tableau state, text output includes:
- current solution (`x1, x2, ...`)
- current objective value (`z` in Phase II)
- full tableau with ratio column

## Report Generation

Reports are multi-page PDFs containing each state:
- header (`phase`, `step`, entering/leaving vars)
- teaching notes
- current solution and objective summary
- tableau text
- geometry/path plot (2D/3D when applicable)
- objective progress plot

### Where reports are saved

- `showcase.py` and `super_showcase.py` save into:
  - `python simplex/reports/`
- direct solver calls save wherever `report_pdf_path` points.

### Report naming

Examples:
- `showcase_3d_big-m_3d-mixed-constraints-big-m_YYYYMMDD_HHMMSS.pdf`
- `super_showcase_two_phase_YYYYMMDD_HHMMSS.pdf`

## Caching Behavior

There is currently **no solver-state cache** between runs.

What persists:
- generated reports in `python simplex/reports/`
- Python bytecode in `python simplex/__pycache__/`
- Matplotlib/font cache files (environment dependent)

Each solve run recomputes simplex iterations from scratch.

## MATLAB Usage

In MATLAB:

1. Open `two_phase_/` (or `cd` into it)
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
```

## Notes and Limitations

- Model assumes non-negativity (`x >= 0`).
- Plot rendering is supported for `n = 2` and `n = 3`.
- For higher dimensions, solving still works but geometric plotting is not shown.
- If constraints produce infeasible/unbounded models, solver raises `RuntimeError`.

## Troubleshooting

If Python imports fail, run from inside `python simplex/`:

```bash
cd "python simplex"
python usage.py
```

If Matplotlib reports cache permission warnings, set:

```bash
export MPLCONFIGDIR=/tmp/matplotlib-cache
```

before running scripts.
