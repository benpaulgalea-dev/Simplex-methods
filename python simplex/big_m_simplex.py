import numpy as np

from simplex_utils import (
    _add_state,
    _defaults,
    _extract_solution,
    _make_objective_consistent,
    _pivot_out_artificial_basics,
    _remap_basis,
    _simplex_core,
    export_states_pdf_report,
    simplex_viewer,
)


def big_m_simplex(c, A, b, sense, opts=None):
    if opts is None:
        opts = {}
    opts = _defaults(opts)

    c = np.asarray(c, dtype=float).reshape(-1)
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    sense = np.asarray(sense).reshape(-1)

    m, n = A.shape
    A0, b0, sense0 = A.copy(), b.copy(), sense.copy()

    for i in range(m):
        if b[i] < 0:
            A[i, :] *= -1.0
            b[i] *= -1.0
            if sense[i] == "<=":
                sense[i] = ">="
            elif sense[i] == ">=":
                sense[i] = "<="

    big_m = opts.get("big_m", None)
    if big_m is None:
        big_m = 1e6 * max(1.0, float(np.max(np.abs(c))) if c.size else 1.0)
    else:
        big_m = float(big_m)
        if (not np.isfinite(big_m)) or big_m <= 0:
            raise ValueError("opts['big_m'] must be a positive finite number.")

    var_names = [f"x{k+1}" for k in range(n)]
    S_cols = []
    extra_names = []
    basis = np.zeros(m, dtype=int)
    art = []

    for i in range(m):
        e = np.zeros(m)
        e[i] = 1.0

        if sense[i] == "<=":
            S_cols.append(e.copy())
            extra_names.append(f"s{i+1}")
            basis[i] = n + len(S_cols) - 1
        elif sense[i] == ">=":
            S_cols.append(-e.copy())
            extra_names.append(f"u{i+1}")
            S_cols.append(e.copy())
            extra_names.append(f"a{i+1}")
            basis[i] = n + len(S_cols) - 1
            art.append(basis[i])
        elif sense[i] == "=":
            S_cols.append(e.copy())
            extra_names.append(f"a{i+1}")
            basis[i] = n + len(S_cols) - 1
            art.append(basis[i])
        else:
            raise ValueError("sense entries must be <=, >=, =")

    S = np.column_stack(S_cols) if S_cols else np.zeros((m, 0))
    Astd = np.hstack([A, S])
    names = var_names + extra_names

    T = np.hstack([Astd, b[:, None]])
    states = []

    c_big = np.zeros(Astd.shape[1])
    c_big[:n] = c
    if art:
        c_big[art] = -big_m
    T = np.vstack([T, np.hstack([-c_big, [0.0]])])
    T = _make_objective_consistent(T, c_big, basis)
    states = _add_state(
        states,
        T,
        names,
        basis,
        "BIG-M",
        0,
        "",
        "",
        [],
        n,
        c,
        info={
            "event": "phase_start",
            "reason": f"Big-M initialized with artificial penalty M={big_m:.6g}.",
        },
    )

    T, basis, states = _simplex_core(
        T, basis, names, states, n, c, "BIG-M", opts["tol"], opts["pivot_rule"]
    )

    x_big = _extract_solution(T, basis)
    if art and np.any(x_big[art] > opts["tol"]):
        raise RuntimeError("LP is infeasible (positive artificial variable remains after Big-M optimization).")

    big_m_obj = float(T[-1, -1])
    art_set = set(art)
    basic_art_before = [names[basis[i]] for i in range(len(basis)) if basis[i] in art_set]

    T, basis, pivot_out_actions = _pivot_out_artificial_basics(T, basis, art, names, opts["tol"])
    basic_art_after_pivot = [names[basis[i]] for i in range(len(basis)) if basis[i] in art_set]

    if art:
        keep = np.ones(T.shape[1] - 1, dtype=bool)
        keep[art] = False
        coeff = T[:, :-1]
        rhs = T[:, -1][:, None]
        T = np.hstack([coeff[:, keep], rhs])

        removed_art_names = [names[j] for j in art if j < len(names)]
        names = [nm for j, nm in enumerate(names) if keep[j]]
        basis = _remap_basis(basis, keep)
    else:
        removed_art_names = []

    states = _add_state(
        states,
        T,
        names,
        basis,
        "BIG-M -> PHASE II",
        0,
        "",
        "",
        [],
        n,
        c,
        info={
            "event": "phase_transition",
            "phase1_objective": big_m_obj,
            "artificial_variables": removed_art_names,
            "basic_art_before": basic_art_before,
            "pivot_out_actions": pivot_out_actions,
            "basic_art_after_pivot": basic_art_after_pivot,
            "reason": "Big-M phase complete. Artificial variables removed before restoring original objective.",
        },
    )

    c2 = np.zeros(len(names))
    c2[:n] = c
    T[-1, :] = np.hstack([-c2, [0.0]])
    T = _make_objective_consistent(T, c2, basis)
    states = _add_state(
        states,
        T,
        names,
        basis,
        "PHASE II",
        0,
        "",
        "",
        [],
        n,
        c,
        info={
            "event": "phase_start",
            "reason": "Original objective restored after Big-M cleanup.",
        },
    )

    T, basis, states = _simplex_core(
        T, basis, names, states, n, c, "PHASE II", opts["tol"], opts["pivot_rule"]
    )

    x_all = _extract_solution(T, basis)
    out = {
        "x": x_all[:n].copy(),
        "z": float(T[-1, -1]),
        "tableau": T.copy(),
        "basis": basis.copy(),
        "var_names": names[:],
        "states": states,
        "big_m": big_m,
    }

    model = {
        "A": A0,
        "b": b0,
        "sense": sense0,
        "c": c,
        "n": n,
        "tol": opts["tol"],
        "teaching_mode": opts["teaching_mode"],
    }

    if opts["report_pdf_path"] is not None:
        export_states_pdf_report(states, model, opts["report_pdf_path"])

    if opts["launch_viewer"]:
        simplex_viewer(states, model)

    return out


def demo():
    c = [11, 9, 7]
    A = [
        [3, 2, 1],
        [2, 5, 3],
        [4, 1, 2],
        [1, 3, 4],
        [2, 2, 5],
        [1, 1, 1],
        [1, 2, 1],
    ]
    b = [24, 33, 28, 30, 32, 4, 6]
    sense = ["<=", "<=", "<=", "<=", "<=", ">=", ">="]

    res = big_m_simplex(c, A, b, sense, opts={"launch_viewer": True})
    print("x* =", res["x"], "z* =", res["z"], "M =", res["big_m"])
    return res


if __name__ == "__main__":
    demo()
