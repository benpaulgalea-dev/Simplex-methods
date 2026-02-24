# Two-Phase Simplex (MATLAB) — Quick README

## Files
- `two_phase_simplex.m` – Two-phase simplex solver with optional GUI tableau viewer.
- `run.m` – Script where you define and run your linear program.
- `pivot.m` – **Required**. Must be in the same folder as the file you run.

## How to Run
1. Place `run.m`, `two_phase_simplex.m`, and `pivot.m` in the same directory.
2. Open that folder in MATLAB (or `cd` into it).
3. Run:
   ```matlab
   run
   ```

## Problem Definition (in `run.m`)
- `c` – Objective function coefficients (maximization).
- `A` – Left-hand side matrix of constraints.
- `b` – Right-hand side vector of constraints.
- `sense` – Type of each constraint (`"<="`, `">="`, or `"="`).

Example:
```matlab
c = [...];
A = [...];
b = [...];
sense = ["<="; ">="; "="];

out = two_phase_simplex(c, A, b, sense);
disp(out.x);
disp(out.z);
```

## Output
- `out.x` – Optimal solution vector.
- `out.z` – Optimal objective value.
- Additional fields include the final tableau, basis, and iteration states.

## Limitation
The GUI plot works **only for 2-variable (2D) problems**.  
For higher dimensions, the algorithm runs correctly, but visualization is disabled.