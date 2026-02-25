EXAMPLES = {
    "2d": {
        "two_phase": {
            "name": "2D Production Mix (Two-Phase)",
            "c": [70, 130],
            "A": [
                [12, 6],
                [0, 15],
                [2, 8],
                [0, 1],
            ],
            "b": [600, 300, 220, 10],
            "sense": ["<=", "<=", "<=", ">="],
        },
        "big_m": {
            "name": "2D Mixed Constraints (Big-M)",
            "c": [5, 4],
            "A": [
                [1, 1],
                [1, 0],
                [0, 1],
            ],
            "b": [4, 3, 1],
            "sense": [">=", "<=", "="],
        },
    },
    "3d": {
        "two_phase": {
            "name": "3D Bound-Constrained Model (Two-Phase)",
            "c": [15, 10, 12],
            "A": [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [4, 4, 0],
                [3, 0, 6],
                [0, 0, 2],
                [2, 1, 0],
            ],
            "b": [12, 12, 12, 27.380577, 25.312655, 3.843491, 7.770306],
            "sense": ["<=", "<=", "<=", ">=", "<=", ">=", ">="],
        },
        "big_m": {
            "name": "3D Mixed Constraints (Big-M)",
            "c": [11, 9, 7],
            "A": [
                [3, 2, 1],
                [2, 5, 3],
                [4, 1, 2],
                [1, 3, 4],
                [2, 2, 5],
                [1, 1, 1],
                [1, 2, 1],
            ],
            "b": [24, 33, 28, 30, 32, 4, 6],
            "sense": ["<=", "<=", "<=", "<=", "<=", ">=", ">="],
        },
    },
}


def _normalize(raw):
    return raw.strip().lower()


def _pick_from_menu(prompt, options, default_key):
    while True:
        print(prompt)
        for i, opt in enumerate(options, start=1):
            marker = " (default)" if opt["key"] == default_key else ""
            print(f"  {i}) {opt['label']}{marker}")
        raw = _normalize(input("> "))

        if raw in {"q", "quit", "exit"}:
            raise KeyboardInterrupt
        if raw == "":
            return default_key
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return options[idx]["key"]

        for opt in options:
            if raw in opt["aliases"]:
                return opt["key"]

        valid = ", ".join(opt["label"] for opt in options)
        print(f"Invalid choice. Enter a number or one of: {valid}.")
        print("Type q to quit.")


def _pick_dimension():
    options = [
        {"key": "2d", "label": "2D", "aliases": {"2", "2d"}},
        {"key": "3d", "label": "3D", "aliases": {"3", "3d"}},
    ]
    return _pick_from_menu("Choose problem dimension:", options, default_key="2d")


def _pick_method():
    options = [
        {
            "key": "two_phase",
            "label": "Two-Phase",
            "aliases": {"two-phase", "twophase", "two phase", "phase2", "2phase", "ii"},
        },
        {
            "key": "big_m",
            "label": "Big-M",
            "aliases": {"big-m", "bigm", "big m", "m"},
        },
    ]
    return _pick_from_menu("Choose simplex method:", options, default_key="two_phase")


def _confirm_run(dim, method):
    dim_label = dim.upper()
    method_label = "Two-Phase" if method == "two_phase" else "Big-M"
    while True:
        raw = _normalize(input(f"Run {method_label} on {dim_label}? [Y/n]: "))
        if raw in {"", "y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        if raw in {"q", "quit", "exit"}:
            raise KeyboardInterrupt
        print("Please answer y or n. Type q to quit.")


def run_showcase():
    print("Simplex Showcase")
    print("----------------")
    print("Tip: choose with number keys. Press Enter to accept defaults. Type q to quit.")
    try:
        dim = _pick_dimension()
        method = _pick_method()
    except KeyboardInterrupt:
        print("\nCancelled.")
        return

    if not _confirm_run(dim, method):
        print("Cancelled.")
        return

    example = EXAMPLES[dim][method]
    if method == "two_phase":
        from two_phase_simplex import two_phase_simplex
        solver = two_phase_simplex
    else:
        from big_m_simplex import big_m_simplex
        solver = big_m_simplex
    method_label = "Two-Phase" if method == "two_phase" else "Big-M"

    print(f"\nRunning: {example['name']}")
    print(f"Method: {method_label} | Dimension: {dim.upper()}")
    print("Launching interactive viewer...")

    res = solver(
        example["c"],
        example["A"],
        example["b"],
        example["sense"],
        opts={
            "launch_viewer": True,
            "teaching_mode": True,
            "pivot_rule": "dantzig",
            "report_pdf_path": None,
        },
    )

    print("\nDone.")
    print("x* =", res["x"])
    print("z* =", res["z"])
    print("states =", len(res["states"]))


if __name__ == "__main__":
    run_showcase()
