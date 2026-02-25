from two_phase_simplex import two_phase_simplex

c = [3, 2]  # maximize 3x1 + 2x2

A = [
    [1, 1],  # x1 + x2 <= 4
    [1, 0],  # x1 <= 2
    [0, 1],  # x2 <= 3
]

b = [4, 2, 3]
sense = ["<=", "<=", "<="]

res = two_phase_simplex(
    c, A, b, sense,
    opts={
        "launch_viewer": True,
        "teaching_mode": True,
        "pivot_rule": "dantzig",
        "report_pdf_path": "simplex_report.pdf",  # or True for default filename
    },
)

print(res["x"], res["z"], "states:", len(res["states"]))