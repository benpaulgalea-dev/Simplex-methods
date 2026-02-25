from two_phase_simplex import two_phase_simplex  # main solver
# or: from two_phase_simplex import demo          # runs the built-in example

c = [70, 130]
A = [[12, 6],
     [0, 15],
     [2, 8],
     [0, 1]]
b = [600, 300, 220, 10]
sense = ["<=", "<=", "<=", ">="]

res = two_phase_simplex(c, A, b, sense, opts={"launch_viewer": True})
print(res["x"], res["z"])