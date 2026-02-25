from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np

from big_m_simplex import big_m_simplex
from two_phase_simplex import two_phase_simplex


def build_complex_3d_problem():
    c = [15, 10, 12]
    A = [
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [4, 4, 0], [3, 0, 6], [0, 0, 2], [2, 1, 0],
        [5, 5, 3], [5, 4, 3], [5, 6, 7], [6, 5, 5], [2, 2, 6],
        [4, 7, 3], [3, 6, 1], [5, 7, 2], [5, 0, 3], [3, 1, 6],
        [3, 2, 7], [1, 1, 0], [4, 6, 1], [2, 3, 0], [4, 6, 3],
        [3, 4, 4], [5, 0, 3], [6, 6, 5], [0, 1, 3], [3, 6, 4], [5, 5, 4],
    ]
    b = [
        12, 12, 12,
        27.380577, 25.312655, 3.843491, 7.770306,
        41.709889, 39.319691, 55.944707, 50.365908, 27.993998,
        47.023606, 35.81098, 46.557903, 26.245753, 26.190921,
        37.484833, 6.902326, 41.590038, 20.459915, 42.700085,
        33.500766, 20.571865, 58.306897, 9.629378, 41.00096, 43.546662,
    ]
    sense = [
        "<=", "<=", "<=",
        ">=", "<=", ">=", ">=",
        ">=", "<=", ">=", ">=", ">=",
        ">=", "<=", ">=", "<=", ">=",
        "<=", ">=", "<=", "<=", ">=",
        ">=", ">=", "<=", ">=", ">=", ">=",
    ]
    return c, A, b, sense


def _report_path(tag):
    out_dir = Path(__file__).resolve().parent / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(out_dir / f"super_showcase_{tag}_{ts}.pdf")


def _run_solver(label, solver, c, A, b, sense, launch_viewer):
    report = _report_path(label.replace(" ", "_").lower())
    t0 = perf_counter()
    res = solver(
        c,
        A,
        b,
        sense,
        opts={
            "launch_viewer": launch_viewer,
            "teaching_mode": True,
            "pivot_rule": "dantzig",
            "report_pdf_path": report,
        },
    )
    dt = perf_counter() - t0
    return res, report, dt


def super_showcase(launch_viewer=True):
    c, A, b, sense = build_complex_3d_problem()

    print("Super Showcase: Complex 3D LP")
    print("-----------------------------")
    print(f"Variables: {len(c)} | Constraints: {len(b)}")
    print("Running Two-Phase and Big-M with teaching states + PDF reports.")

    tp_res, tp_report, tp_time = _run_solver(
        "two_phase", two_phase_simplex, c, A, b, sense, launch_viewer=launch_viewer
    )
    bm_res, bm_report, bm_time = _run_solver(
        "big_m", big_m_simplex, c, A, b, sense, launch_viewer=False
    )

    x_delta = np.max(np.abs(tp_res["x"] - bm_res["x"]))
    z_delta = abs(tp_res["z"] - bm_res["z"])

    print("\nResults")
    print(f"Two-Phase: z*={tp_res['z']:.8g}, states={len(tp_res['states'])}, time={tp_time:.3f}s")
    print(f"Big-M:     z*={bm_res['z']:.8g}, states={len(bm_res['states'])}, time={bm_time:.3f}s")
    print(f"Agreement: max|x_tp-x_bm|={x_delta:.3e}, |z_tp-z_bm|={z_delta:.3e}")
    print(f"Two-Phase report: {tp_report}")
    print(f"Big-M report:     {bm_report}")

    return {
        "two_phase": tp_res,
        "big_m": bm_res,
        "two_phase_report": tp_report,
        "big_m_report": bm_report,
    }


if __name__ == "__main__":
    super_showcase(launch_viewer=True)
