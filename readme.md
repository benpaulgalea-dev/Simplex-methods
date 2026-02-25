# Two-Phase Simplex — MATLAB and Python Versions

This project contains two implementations with similar core functionality:
- MATLAB version in `two_phase_/`
- Python version in `python simplex/`

## Version and Capability Notes
- Python version: `Python 3.13`
- Both versions solve the same two-phase simplex problem setup.
- The Python implementation is more capable in visualization and interaction.
- Python viewer supports **2D and 3D** exploration of feasible regions and simplex path.
- MATLAB viewer is limited to **2D** visualization.

## MATLAB Files
- `two_phase_/two_phase_simplex.m` – Two-phase simplex solver with optional GUI tableau viewer.
- `two_phase_/run.m` – Script where you define and run your linear program.
- `two_phase_/pivot.m` – **Required** helper function.

## Python Files
- `python simplex/two_phase_simplex.py` – Solver + interactive viewer.
- `python simplex/usage.py` – Example usage script.

## MATLAB: How to Run
1. Open `two_phase_/` in MATLAB (or `cd` into it).
2. Run:
   ```matlab
   run
   ```

## Problem Definition
- `c` – Objective function coefficients (maximization).
- `A` – Left-hand side matrix of constraints.
- `b` – Right-hand side vector of constraints.
- `sense` – Type of each constraint (`"<="`, `">="`, or `"="`).

## Output
- `x` / `out.x` – Optimal solution vector.
- `z` / `out.z` – Optimal objective value.
- Additional fields include final tableau, basis, and iteration states.
