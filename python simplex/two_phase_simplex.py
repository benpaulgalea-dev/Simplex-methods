import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from itertools import combinations
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from mpl_toolkits.mplot3d import proj3d

try:
    from scipy.spatial import ConvexHull
    _HAS_SCIPY = True
except Exception:
    ConvexHull = None
    _HAS_SCIPY = False


def two_phase_simplex(c, A, b, sense, opts=None):
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

    c1 = np.zeros(Astd.shape[1])
    if art:
        c1[art] = -1.0
    T = np.vstack([T, np.hstack([-c1, [0.0]])])
    T = _make_objective_consistent(T, c1, basis)
    states = _add_state(states, T, names, basis, "PHASE I", 0, "", "", [], n, c)

    T, basis, states = _simplex_core(T, basis, names, states, n, c, "PHASE I", opts["tol"])

    if abs(T[-1, -1]) > opts["tol"]:
        raise RuntimeError("LP is infeasible (Phase I optimum not zero).")

    T, basis = _pivot_out_artificial_basics(T, basis, art, opts["tol"])
    if art:
        keep = np.ones(T.shape[1] - 1, dtype=bool)
        keep[art] = False
        coeff = T[:, :-1]
        rhs = T[:, -1][:, None]
        T = np.hstack([coeff[:, keep], rhs])

        names = [nm for j, nm in enumerate(names) if keep[j]]
        basis = _remap_basis(basis, keep)

    c2 = np.zeros(len(names))
    c2[:n] = c
    T[-1, :] = np.hstack([-c2, [0.0]])
    T = _make_objective_consistent(T, c2, basis)
    states = _add_state(states, T, names, basis, "PHASE II", 0, "", "", [], n, c)

    T, basis, states = _simplex_core(T, basis, names, states, n, c, "PHASE II", opts["tol"])

    x_all = _extract_solution(T, basis)
    out = {
        "x": x_all[:n].copy(),
        "z": float(T[-1, -1]),
        "tableau": T.copy(),
        "basis": basis.copy(),
        "var_names": names[:],
        "states": states,
    }

    if opts["launch_viewer"]:
        model = {
            "A": A0,
            "b": b0,
            "sense": sense0,
            "c": c,
            "n": n,
            "tol": opts["tol"],
        }
        simplex_viewer(states, model)

    return out


def _simplex_core(T, basis, names, states, n_orig, c_orig, phase, tol):
    m = T.shape[0] - 1
    n = T.shape[1] - 1
    step = 0

    while True:
        enter_col = int(np.argmin(T[-1, :n]))
        if T[-1, enter_col] >= -tol:
            break

        ratios = np.full(m, np.inf)
        col = T[:m, enter_col]
        rhs = T[:m, -1]
        mask = col > tol
        ratios[mask] = rhs[mask] / col[mask]

        leave_row = int(np.argmin(ratios))
        if not np.isfinite(ratios[leave_row]):
            raise RuntimeError("Unbounded LP.")

        entering = names[enter_col]
        leaving = names[basis[leave_row]]

        piv = T[leave_row, enter_col]
        T[leave_row, :] /= piv
        for r in range(T.shape[0]):
            if r != leave_row:
                T[r, :] -= T[r, enter_col] * T[leave_row, :]

        basis[leave_row] = enter_col
        step += 1
        states = _add_state(states, T, names, basis, phase, step, entering, leaving, ratios, n_orig, c_orig)

    return T, basis, states


def _add_state(states, T, names, basis, phase, step, entering, leaving, ratios, n_orig, c_orig):
    x = _extract_solution(T, basis)
    k = min(3, n_orig)
    xx = np.full(k, np.nan)
    xx[:k] = x[:k]
    z = 0.0
    if phase == "PHASE II":
        z = float(np.dot(c_orig[:n_orig], x[:n_orig]))

    states.append({
        "T": T.copy(),
        "names": names[:],
        "basis": basis.copy(),
        "phase": phase,
        "step": step,
        "entering": entering,
        "leaving": leaving,
        "ratios": np.array(ratios, dtype=float) if len(ratios) else np.array([]),
        "x": xx,
        "z": z,
    })
    return states


def simplex_viewer(states, model):
    n = model["n"]
    E, hull_data = extreme_points_nd(model)
    hover = {"scatter": None, "label": None}

    fig = plt.figure(figsize=(16, 9))
    ax_plot = fig.add_axes([0.05, 0.18, 0.52, 0.75], projection="3d" if n == 3 else None)
    ax_txt = fig.add_axes([0.60, 0.18, 0.38, 0.75])
    ax_txt.axis("off")
    ax_slider = fig.add_axes([0.10, 0.08, 0.35, 0.03])
    ax_prev = fig.add_axes([0.48, 0.07, 0.05, 0.05])
    ax_next = fig.add_axes([0.54, 0.07, 0.05, 0.05])

    slider = Slider(ax_slider, "State", 1, len(states), valinit=1, valstep=1)
    btn_prev = Button(ax_prev, "Prev")
    btn_next = Button(ax_next, "Next")

    def render(k):
        nonlocal hover
        idx = int(k) - 1
        s = states[idx]
        hover["scatter"] = None
        hover["label"] = None
        ax_txt.clear()
        ax_txt.axis("off")

        header = f'{idx+1}/{len(states)} | {s["phase"]} step {s["step"]}'
        if s["entering"]:
            header += f' | ENTER: {s["entering"]} | LEAVE: {s["leaving"]}'
        if s["phase"] == "PHASE II":
            header += f' | Z={s["z"]:.6g}'

        ax_txt.text(0.0, 1.0, header, va="top", ha="left", fontsize=11, fontweight="bold")
        ax_txt.text(0.0, 0.96, _tableau_to_text(s["T"], s["names"], s["basis"], s["ratios"]),
                    va="top", ha="left", family="monospace", fontsize=8)

        path = np.array([st["x"] for st in states[:idx + 1]], dtype=float)
        ax_plot.clear()

        if n == 2:
            ax_plot.set_xlabel("x1")
            ax_plot.set_ylabel("x2")
            ax_plot.set_title("Feasible region + extreme points + simplex path")
            ax_plot.grid(True)

            if E.size:
                hover["scatter"] = ax_plot.scatter(E[:, 0], E[:, 1], c="k", s=25, zorder=3)
            if hull_data is not None and len(hull_data) >= 3:
                poly = E[hull_data]
                ax_plot.fill(poly[:, 0], poly[:, 1], color="#8ecae6", alpha=0.35, edgecolor="#1f5f8b", zorder=2)

            p = path[np.all(np.isfinite(path[:, :2]), axis=1), :2]
            if len(p):
                ax_plot.plot(p[:, 0], p[:, 1], "-o", color="#d6451d", linewidth=2, markersize=5, zorder=4)
                ax_plot.plot(p[-1, 0], p[-1, 1], "ro", markersize=8, zorder=5)

            if s["phase"] == "PHASE II":
                _draw_objective_2d(ax_plot, model["c"][:2], s["z"], E)

            hover["label"] = ax_plot.annotate(
                "",
                xy=(0.0, 0.0),
                xytext=(8, 10),
                textcoords="offset points",
                bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.5", "alpha": 0.95},
                fontsize=8,
            )
            hover["label"].set_visible(False)

        elif n == 3:
            ax_plot.set_xlabel("x1")
            ax_plot.set_ylabel("x2")
            ax_plot.set_zlabel("x3")
            ax_plot.set_title("Feasible polytope + extreme points + simplex path")

            # draw objective first (background)
            if s["phase"] == "PHASE II":
                _draw_objective_3d(ax_plot, model["c"][:3], s["z"], E)

            if E.size:
                hover["scatter"] = ax_plot.scatter(E[:, 0], E[:, 1], E[:, 2], c="k", s=36, depthshade=False)

            if hull_data:
                polys = hull_data.get("polys", [])
                if polys:
                    poly3d = [E[list(poly)] for poly in polys]
                    mesh = Poly3DCollection(poly3d, facecolor="#8ecae6", edgecolor="#1f5f8b",
                                            linewidths=1.2, alpha=0.50)
                    ax_plot.add_collection3d(mesh)

                edges = hull_data.get("edges", [])
                if edges:
                    segs = [(E[i], E[j]) for (i, j) in edges]
                    edge_lines = Line3DCollection(segs, colors="#0b4f6c", linewidths=2.2, alpha=1.0)
                    ax_plot.add_collection3d(edge_lines)

            p = path[np.all(np.isfinite(path[:, :3]), axis=1), :3]
            if len(p):
                ax_plot.plot(p[:, 0], p[:, 1], p[:, 2], "-o", color="#d6451d", linewidth=2.2, markersize=5)
                ax_plot.scatter([p[-1, 0]], [p[-1, 1]], [p[-1, 2]], c="r", s=65, depthshade=False)

            if E.size:
                pad = 0.6
                ax_plot.set_xlim(np.min(E[:, 0]) - pad, np.max(E[:, 0]) + pad)
                ax_plot.set_ylim(np.min(E[:, 1]) - pad, np.max(E[:, 1]) + pad)
                ax_plot.set_zlim(np.min(E[:, 2]) - pad, np.max(E[:, 2]) + pad)

            hover["label"] = ax_plot.text2D(
                0.02, 0.88, "", transform=ax_plot.transAxes, fontsize=8,
                bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.5", "alpha": 0.95}
            )
            hover["label"].set_visible(False)

        else:
            ax_plot.text(0.1, 0.5, "Plot is available for 2D or 3D LP only", transform=ax_plot.transAxes)

        fig.canvas.draw_idle()

    def on_prev(_):
        slider.set_val(max(1, int(slider.val) - 1))

    def on_next(_):
        slider.set_val(min(len(states), int(slider.val) + 1))

    def on_hover(event):
        if event.inaxes != ax_plot or not E.size or hover["label"] is None:
            if hover["label"] is not None and hover["label"].get_visible():
                hover["label"].set_visible(False)
                fig.canvas.draw_idle()
            return

        if n == 2 and hover["scatter"] is not None:
            contains, info = hover["scatter"].contains(event)
            if contains and info.get("ind"):
                i = int(info["ind"][0])
                p = E[i, :2]
                hover["label"].xy = (p[0], p[1])
                hover["label"].set_text(f"E{i+1}: ({p[0]:.6g}, {p[1]:.6g})")
                hover["label"].set_visible(True)
                fig.canvas.draw_idle()
            elif hover["label"].get_visible():
                hover["label"].set_visible(False)
                fig.canvas.draw_idle()
            return

        if n == 3:
            x2, y2, _ = proj3d.proj_transform(E[:, 0], E[:, 1], E[:, 2], ax_plot.get_proj())
            xy_disp = ax_plot.transData.transform(np.column_stack([x2, y2]))
            d2 = (xy_disp[:, 0] - event.x) ** 2 + (xy_disp[:, 1] - event.y) ** 2
            i = int(np.argmin(d2))
            if d2[i] <= 12.0 ** 2:
                p = E[i, :3]
                hover["label"].set_text(f"E{i+1}: ({p[0]:.6g}, {p[1]:.6g}, {p[2]:.6g})")
                hover["label"].set_visible(True)
                fig.canvas.draw_idle()
            elif hover["label"].get_visible():
                hover["label"].set_visible(False)
                fig.canvas.draw_idle()

    slider.on_changed(render)
    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)
    fig.canvas.mpl_connect("motion_notify_event", on_hover)

    render(1)
    plt.show()


def _draw_objective_2d(ax, c2, z, E):
    c2 = np.asarray(c2, dtype=float)
    if c2.size < 2:
        return
    c1, c2y = c2[0], c2[1]
    if abs(c1) < 1e-12 and abs(c2y) < 1e-12:
        return

    x_min = min(0.0, np.min(E[:, 0]) if E.size else 0.0) - 1.0
    x_max = max(1.0, np.max(E[:, 0]) if E.size else 1.0) + 1.0
    y_min = min(0.0, np.min(E[:, 1]) if E.size else 0.0) - 1.0
    y_max = max(1.0, np.max(E[:, 1]) if E.size else 1.0) + 1.0

    if abs(c2y) >= abs(c1) and abs(c2y) > 1e-12:
        xx = np.linspace(x_min, x_max, 300)
        yy = (z - c1 * xx) / c2y
        ax.plot(xx, yy, "k--", linewidth=1.6, label=f"{c1:.3g}x1 + {c2y:.3g}x2 = {z:.3g}", zorder=1)
    elif abs(c1) > 1e-12:
        yy = np.linspace(y_min, y_max, 300)
        xx = (z - c2y * yy) / c1
        ax.plot(xx, yy, "k--", linewidth=1.6, label=f"{c1:.3g}x1 + {c2y:.3g}x2 = {z:.3g}", zorder=1)

    ax.legend(loc="upper right", fontsize=8, frameon=True)


def _draw_objective_3d(ax, c3, z, E):
    c3 = np.asarray(c3, dtype=float)
    if c3.size < 3:
        return
    c1, c2, c3z = c3[0], c3[1], c3[2]
    if abs(c1) < 1e-12 and abs(c2) < 1e-12 and abs(c3z) < 1e-12:
        return

    x_min = min(0.0, np.min(E[:, 0]) if E.size else 0.0) - 0.2
    x_max = max(1.0, np.max(E[:, 0]) if E.size else 1.0) + 0.2
    y_min = min(0.0, np.min(E[:, 1]) if E.size else 0.0) - 0.2
    y_max = max(1.0, np.max(E[:, 1]) if E.size else 1.0) + 0.2
    z_min = min(0.0, np.min(E[:, 2]) if E.size else 0.0) - 0.2
    z_max = max(1.0, np.max(E[:, 2]) if E.size else 1.0) + 0.2

    dom = int(np.argmax(np.abs([c1, c2, c3z])))
    grid_n = 20

    if dom == 2 and abs(c3z) > 1e-12:
        X, Y = np.meshgrid(np.linspace(x_min, x_max, grid_n), np.linspace(y_min, y_max, grid_n))
        Z = (z - c1 * X - c2 * Y) / c3z
        M = np.isfinite(Z) & (Z >= z_min) & (Z <= z_max)
        X, Y, Z = np.where(M, X, np.nan), np.where(M, Y, np.nan), np.where(M, Z, np.nan)
    elif dom == 1 and abs(c2) > 1e-12:
        X, Z = np.meshgrid(np.linspace(x_min, x_max, grid_n), np.linspace(z_min, z_max, grid_n))
        Y = (z - c1 * X - c3z * Z) / c2
        M = np.isfinite(Y) & (Y >= y_min) & (Y <= y_max)
        X, Y, Z = np.where(M, X, np.nan), np.where(M, Y, np.nan), np.where(M, Z, np.nan)
    elif abs(c1) > 1e-12:
        Y, Z = np.meshgrid(np.linspace(y_min, y_max, grid_n), np.linspace(z_min, z_max, grid_n))
        X = (z - c2 * Y - c3z * Z) / c1
        M = np.isfinite(X) & (X >= x_min) & (X <= x_max)
        X, Y, Z = np.where(M, X, np.nan), np.where(M, Y, np.nan), np.where(M, Z, np.nan)
    else:
        return

    ax.plot_surface(X, Y, Z, alpha=0.10, color="#f2b134", edgecolor="none")
    ax.text2D(0.02, 0.95, f"{c1:.3g}x1 + {c2:.3g}x2 + {c3z:.3g}x3 = {z:.3g}",
              transform=ax.transAxes, fontsize=8)


def extreme_points_nd(model):
    n = model["n"]
    if n not in (2, 3):
        return np.empty((0, n)), None

    A = np.asarray(model["A"], dtype=float)
    b = np.asarray(model["b"], dtype=float).reshape(-1)
    sense = np.asarray(model["sense"]).reshape(-1)

    G, h = [], []
    for i in range(A.shape[0]):
        ai, bi = A[i], b[i]
        if sense[i] == "<=":
            G.append(ai)
            h.append(bi)
        elif sense[i] == ">=":
            G.append(-ai)
            h.append(-bi)
        elif sense[i] == "=":
            G.append(ai)
            h.append(bi)
            G.append(-ai)
            h.append(-bi)
        else:
            raise ValueError("sense entries must be <=, >=, =")

    for j in range(n):
        ej = np.zeros(n)
        ej[j] = -1.0
        G.append(ej)
        h.append(0.0)

    G = np.asarray(G, dtype=float)
    h = np.asarray(h, dtype=float)

    pts = []
    for idxs in combinations(range(len(h)), n):
        M = G[list(idxs), :]
        d = h[list(idxs)]
        if np.linalg.matrix_rank(M) < n:
            continue
        try:
            x = np.linalg.solve(M, d)
        except np.linalg.LinAlgError:
            continue
        if np.all(G @ x <= h + 1e-8):
            pts.append(x)

    if not pts:
        return np.empty((0, n)), None

    E = np.unique(np.round(np.vstack(pts), 10), axis=0)

    if n == 2:
        if E.shape[0] >= 3:
            if _HAS_SCIPY:
                hull = ConvexHull(E).vertices
            else:
                c = E.mean(axis=0)
                ang = np.arctan2(E[:, 1] - c[1], E[:, 0] - c[0])
                hull = np.argsort(ang)
            return E, hull
        return E, np.arange(E.shape[0])

    return E, _convex_hull_3d(E)


def _convex_hull_3d(E, tol=1e-10):
    out = {"polys": [], "edges": [], "faces": []}
    n = E.shape[0]

    if n < 2:
        return out

    # Special case: 4 nearly coplanar points should render as one quadrilateral face.
    quad_order = _ordered_coplanar_polygon(E, plane_tol=max(tol * 100.0, 1e-8)) if n == 4 else None
    if quad_order is not None and len(quad_order) == 4:
        out["polys"] = [tuple(quad_order.tolist())]
        out["edges"] = sorted(
            tuple(sorted((int(quad_order[i]), int(quad_order[(i + 1) % 4]))))
            for i in range(4)
        )
        o0 = int(quad_order[0])
        out["faces"] = [(o0, int(quad_order[1]), int(quad_order[2])),
                        (o0, int(quad_order[2]), int(quad_order[3]))]
        return out

    centered = E - E.mean(axis=0, keepdims=True)
    _, s, vh = np.linalg.svd(centered, full_matrices=False)
    if s.size == 0:
        dim = 0
    else:
        dim = int(np.sum(s > (tol * max(s[0], 1.0))))

    if dim <= 1:
        d = centered @ vh[0]
        i_min, i_max = int(np.argmin(d)), int(np.argmax(d))
        if i_min != i_max:
            out["edges"] = [(min(i_min, i_max), max(i_min, i_max))]
        return out

    if dim == 2:
        B = vh[:2].T
        UV = centered @ B

        if n >= 3:
            if _HAS_SCIPY:
                try:
                    hull2 = ConvexHull(UV)
                    order = hull2.vertices
                except Exception:
                    c2 = UV.mean(axis=0)
                    ang = np.arctan2(UV[:, 1] - c2[1], UV[:, 0] - c2[0])
                    order = np.argsort(ang)
            else:
                c2 = UV.mean(axis=0)
                ang = np.arctan2(UV[:, 1] - c2[1], UV[:, 0] - c2[0])
                order = np.argsort(ang)

            order = np.array(order, dtype=int)
            out["polys"] = [tuple(order.tolist())]
            edges = set()
            for i in range(len(order)):
                a = int(order[i])
                b = int(order[(i + 1) % len(order)])
                edges.add(tuple(sorted((a, b))))
            out["edges"] = sorted(edges)
            if len(order) >= 3:
                o0 = int(order[0])
                out["faces"] = [(o0, int(order[i]), int(order[i + 1])) for i in range(1, len(order) - 1)]
        return out

    if _HAS_SCIPY:
        try:
            hull = ConvexHull(E, qhull_options="QJ")
            faces = [tuple(map(int, face)) for face in hull.simplices]
            edges = set()
            for a, b, c in faces:
                edges.add(tuple(sorted((a, b))))
                edges.add(tuple(sorted((b, c))))
                edges.add(tuple(sorted((a, c))))
            out["faces"] = faces
            out["polys"] = faces
            out["edges"] = sorted(edges)
            return out
        except Exception:
            pass

    faces = set()
    for i, j, k in combinations(range(n), 3):
        p1, p2, p3 = E[i], E[j], E[k]
        normal = np.cross(p2 - p1, p3 - p1)
        if np.linalg.norm(normal) < tol:
            continue
        side = (E - p1) @ normal
        pos = np.any(side > tol)
        neg = np.any(side < -tol)
        if not (pos and neg):
            faces.add(tuple(sorted((i, j, k))))

    faces = sorted(faces)
    edges = set()
    for a, b, c in faces:
        edges.add(tuple(sorted((a, b))))
        edges.add(tuple(sorted((b, c))))
        edges.add(tuple(sorted((a, c))))

    out["faces"] = faces
    out["polys"] = faces
    out["edges"] = sorted(edges)
    return out


def _ordered_coplanar_polygon(E, plane_tol=1e-8):
    if E.shape[0] < 3:
        return None

    centered = E - E.mean(axis=0, keepdims=True)
    _, s, vh = np.linalg.svd(centered, full_matrices=False)
    if s.size < 2:
        return None

    # If the third singular value is small, points are coplanar.
    if s.size >= 3 and s[2] > plane_tol * max(s[0], 1.0):
        return None

    B = vh[:2].T
    UV = centered @ B
    c2 = UV.mean(axis=0)
    ang = np.arctan2(UV[:, 1] - c2[1], UV[:, 0] - c2[0])
    order = np.argsort(ang).astype(int)
    return order


def _tableau_to_text(T, names, basis, ratios):
    m = T.shape[0] - 1
    n = T.shape[1] - 1

    def fnum(v):
        if not np.isfinite(v):
            return "inf"
        if abs(v) < 1e-12:
            v = 0.0
        return f"{v:.6g}"

    rows = []
    header = ["row"] + names + ["rhs", "ratio"]
    rows.append(header)

    for i in range(m):
        ratio_txt = "inf" if (len(ratios) == 0 or not np.isfinite(ratios[i])) else fnum(ratios[i])
        row = [f"R{i+1}({names[basis[i]]})"] + [fnum(T[i, j]) for j in range(n)] + [fnum(T[i, -1]), ratio_txt]
        rows.append(row)

    rows.append(["Rz"] + [fnum(T[-1, j]) for j in range(n)] + [fnum(T[-1, -1]), "-"])

    widths = [max(len(str(rows[r][c])) for r in range(len(rows))) for c in range(len(rows[0]))]

    out = []
    out.append(" | ".join(str(rows[0][c]).rjust(widths[c]) for c in range(len(widths))))
    out.append("-+-".join("-" * widths[c] for c in range(len(widths))))
    for r in range(1, len(rows)):
        out.append(" | ".join(str(rows[r][c]).rjust(widths[c]) for c in range(len(widths))))
    return "\n".join(out)


def _make_objective_consistent(T, c, basis):
    T = T.copy()
    for i, bi in enumerate(basis):
        cb = c[bi]
        if abs(cb) > 0:
            T[-1, :] += cb * T[i, :]
    return T


def _extract_solution(T, basis):
    n = T.shape[1] - 1
    x = np.zeros(n)
    for i, bi in enumerate(basis):
        x[bi] = T[i, -1]
    return x


def _pivot_out_artificial_basics(T, basis, art, tol):
    T = T.copy()
    m = T.shape[0] - 1
    n = T.shape[1] - 1
    art_set = set(art)

    for i in range(m):
        if basis[i] in art_set:
            for j in range(n):
                if j not in art_set and abs(T[i, j]) > tol:
                    T[i, :] /= T[i, j]
                    for r in range(T.shape[0]):
                        if r != i:
                            T[r, :] -= T[r, j] * T[i, :]
                    basis[i] = j
                    break
    return T, basis


def _remap_basis(basis, keep):
    mapping = np.full(len(keep), -1, dtype=int)
    mapping[np.where(keep)[0]] = np.arange(np.count_nonzero(keep))
    out = basis.copy()
    for i in range(len(out)):
        out[i] = mapping[out[i]]
    return out


def _defaults(opts):
    out = dict(opts)
    out.setdefault("launch_viewer", True)
    out.setdefault("tol", 1e-10)
    return out


def demo():
    """Small demo run. Safe to import this module from other files."""
    # Maximize:
    #   z = 11x1 + 9x2 + 7x3
    #
    # Subject to:
    #   3x1 + 2x2 + 1x3 <= 24
    #   2x1 + 5x2 + 3x3 <= 33
    #   4x1 + 1x2 + 2x3 <= 28
    #   1x1 + 3x2 + 4x3 <= 30
    #   2x1 + 2x2 + 5x3 <= 32
    #   x1 + x2 + x3 >= 4
    #   x1 + 2x2 + x3 >= 6
    #   x1, x2, x3 >= 0

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

    res = two_phase_simplex(c, A, b, sense, opts={"launch_viewer": True})
    print("x* =", res["x"], "z* =", res["z"])
    return res


if __name__ == "__main__":
    demo()
