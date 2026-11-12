"""Tron light cycles: the shared simulator, observation and baseline bots (numba-compiled).

Board: 30x30 playable cells inside a 1-cell wall border (array 32x32). grid value 0 = empty, 1 = wall,
2 + b = trail of bike b. Each tick every live bike turns (0 left, 1 straight, 2 right) and moves one cell.
A bike dies if it moves into a wall or any trail; two bikes moving into the same cell both die.
Last bike alive wins; all remaining dying on the same tick (or the tick limit) is a draw.
"""
import numpy as np
from numba import njit

N = 30                     # playable size
S = N + 2                  # array size incl. border
MAX_TICKS = 900
R = 7                      # observation window radius -> 15x15
W = 2 * R + 1
NF = 23                    # number of scalar features

DY = np.array([-1, 0, 1, 0], dtype=np.int32)   # 0 up, 1 right, 2 down, 3 left
DX = np.array([0, 1, 0, -1], dtype=np.int32)


# ---------------------------------------------------------------- game state
def new_state(n_bikes):
    grid = np.zeros((S, S), dtype=np.int8)
    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = 1
    return {"grid": grid, "pos": np.zeros((n_bikes, 2), np.int32), "dirs": np.zeros(n_bikes, np.int32),
            "alive": np.ones(n_bikes, np.bool_), "t": 0}


def start_positions(rng, n_bikes):
    """1v1: a random start and its point mirror (180 degree rotation), so both sides are identical.
    3-5 bikes: random cells at least 8 apart (Manhattan) and 3 from the wall."""
    if n_bikes == 2:
        y, x = int(rng.integers(3, 29)), int(rng.integers(3, 13))
        d = int(rng.integers(4))
        return np.array([[y, x], [S - 1 - y, S - 1 - x]], np.int32), np.array([d, (d + 2) % 4], np.int32)
    while True:
        p = rng.integers(4, N - 2, size=(n_bikes, 2)).astype(np.int32)
        ok = all(abs(p[i] - p[j]).sum() >= 8 for i in range(n_bikes) for j in range(i))
        if ok:
            return p, rng.integers(4, size=n_bikes).astype(np.int32)


def reset(state, seed):
    rng = np.random.default_rng(seed)
    g = state["grid"]
    g[1:-1, 1:-1] = 0
    pos, dirs = start_positions(rng, len(state["pos"]))
    state["pos"][:] = pos
    state["dirs"][:] = dirs
    state["alive"][:] = True
    state["t"] = 0
    for b in range(len(pos)):
        g[pos[b, 0], pos[b, 1]] = 2 + b
    return state


def new_game(n_bikes, seed):
    return reset(new_state(n_bikes), seed)


@njit(cache=True)
def step_core(grid, pos, dirs, alive, actions):
    B = pos.shape[0]
    ny = np.zeros(B, np.int32)
    nx = np.zeros(B, np.int32)
    dead = np.zeros(B, np.bool_)
    for b in range(B):
        if alive[b]:
            d = (dirs[b] + actions[b] - 1) % 4
            dirs[b] = d
            ny[b] = pos[b, 0] + DY[d]
            nx[b] = pos[b, 1] + DX[d]
            if grid[ny[b], nx[b]] != 0:
                dead[b] = True
    for b in range(B):
        if alive[b]:
            for c in range(b + 1, B):
                if alive[c] and ny[b] == ny[c] and nx[b] == nx[c]:
                    dead[b] = True
                    dead[c] = True
    for b in range(B):
        if alive[b]:
            pos[b, 0] = ny[b]
            pos[b, 1] = nx[b]
            if dead[b]:
                alive[b] = False
            else:
                grid[ny[b], nx[b]] = 2 + b


def step(state, actions):
    """Advance one tick. Returns None while running, else the result: winner index, or -1 for a draw."""
    step_core(state["grid"], state["pos"], state["dirs"], state["alive"], np.asarray(actions, np.int32))
    state["t"] += 1
    return result(state)


def result(state):
    n = int(state["alive"].sum())
    if n == 0:
        return -1
    if n == 1 and len(state["alive"]) > 1:
        return int(np.argmax(state["alive"]))
    if state["t"] >= MAX_TICKS:
        return -1
    return None


# ---------------------------------------------------------------- search helpers
@njit(cache=True)
def _bfs(grid, sy, sx, n_src, dist, queue):
    """Multi-source BFS over empty cells. Sources (sy[i], sx[i]) get distance 0 even if occupied.
    dist must be pre-filled with -1. Returns the number of cells reached (sources included)."""
    head = 0
    tail = 0
    for i in range(n_src):
        if dist[sy[i], sx[i]] < 0:
            dist[sy[i], sx[i]] = 0
            queue[tail] = sy[i] * S + sx[i]
            tail += 1
    while head < tail:
        c = queue[head]
        head += 1
        y = c // S
        x = c % S
        for d in range(4):
            yy = y + DY[d]
            xx = x + DX[d]
            if grid[yy, xx] == 0 and dist[yy, xx] < 0:
                dist[yy, xx] = dist[y, x] + 1
                queue[tail] = yy * S + xx
                tail += 1
    return tail


@njit(cache=True)
def flood_after(grid, y, x, d, a, dist, queue):
    """Cells reachable after taking action a (0 if that move dies right away)."""
    nd = (d + a - 1) % 4
    yy = y + DY[nd]
    xx = x + DX[nd]
    if grid[yy, xx] != 0:
        return 0
    dist[:, :] = -1
    sy = np.empty(1, np.int32)
    sx = np.empty(1, np.int32)
    sy[0] = yy
    sx[0] = xx
    return _bfs(grid, sy, sx, 1, dist, queue)


@njit(cache=True)
def features(grid, pos, dirs, alive, b, out_win, out_feat):
    """Egocentric observation of bike b (rotated so it faces up).
    out_win (3, W, W): obstacles (walls/trails, outside the board = 1), own trail, other bikes' heads.
    out_feat (NF,): 8 ray distances, flood fill after each action (3), territory score after each action (3),
    nearest opponent position (forward, right) and distance (3), its heading relative to ours (4),
    live opponents (1), separated from all opponents (1)."""
    B = pos.shape[0]
    y = pos[b, 0]
    x = pos[b, 1]
    d = dirs[b]
    fy = DY[d]
    fx = DX[d]
    ry = DY[(d + 1) % 4]
    rx = DX[(d + 1) % 4]
    out_win[:, :, :] = 0.0
    for i in range(W):
        for j in range(W):
            wy = y + fy * (R - i) + ry * (j - R)
            wx = x + fx * (R - i) + rx * (j - R)
            if wy < 0 or wy >= S or wx < 0 or wx >= S:
                out_win[0, i, j] = 1.0
            else:
                v = grid[wy, wx]
                if v != 0:
                    out_win[0, i, j] = 1.0
                if v == 2 + b:
                    out_win[1, i, j] = 1.0
    for c in range(B):
        if c != b and alive[c]:
            oy = pos[c, 0] - y
            ox = pos[c, 1] - x
            fwd = oy * fy + ox * fx
            rgt = oy * ry + ox * rx
            i = R - fwd
            j = R + rgt
            if 0 <= i < W and 0 <= j < W:
                out_win[2, i, j] = 1.0
    # rays: forward, fwd-right, right, back-right, back, back-left, left, fwd-left
    k = 0
    for (a1, b1) in ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)):
        vy = fy * a1 + ry * b1
        vx = fx * a1 + rx * b1
        n = 0
        yy = y + vy
        xx = x + vx
        while 0 <= yy < S and 0 <= xx < S and grid[yy, xx] == 0:
            n += 1
            yy += vy
            xx += vx
        out_feat[k] = n / N
        k += 1
    # opponents' distance field (from their heads)
    dist_o = np.full((S, S), -1, np.int32)
    dist_m = np.full((S, S), -1, np.int32)
    queue = np.empty(S * S, np.int32)
    sy = np.empty(B, np.int32)
    sx = np.empty(B, np.int32)
    n_src = 0
    best = -1
    best_d = 10 ** 6
    for c in range(B):
        if c != b and alive[c]:
            sy[n_src] = pos[c, 0]
            sx[n_src] = pos[c, 1]
            n_src += 1
            md = abs(pos[c, 0] - y) + abs(pos[c, 1] - x)
            if md < best_d:
                best_d = md
                best = c
    if n_src > 0:
        _bfs(grid, sy, sx, n_src, dist_o, queue)
    separated = 1.0
    for a in range(3):
        nd = (d + a - 1) % 4
        yy = y + DY[nd]
        xx = x + DX[nd]
        if grid[yy, xx] != 0:
            out_feat[8 + a] = 0.0
            out_feat[11 + a] = -1.0
            continue
        dist_m[:, :] = -1
        sy[0] = yy
        sx[0] = xx
        cnt = _bfs(grid, sy, sx, 1, dist_m, queue)
        out_feat[8 + a] = cnt / (N * N)
        mine = 0
        theirs = 0
        for q in range(cnt):
            cy = queue[q] // S
            cx = queue[q] % S
            do = dist_o[cy, cx]
            dm = dist_m[cy, cx] + 1
            if do >= 0:
                separated = 0.0
            if do < 0 or dm < do:
                mine += 1
            elif do < dm:
                theirs += 1
        if n_src > 0:
            # cells only opponents reach count for them too
            for cy in range(1, S - 1):
                for cx in range(1, S - 1):
                    if dist_o[cy, cx] > 0 and dist_m[cy, cx] < 0:
                        theirs += 1
        out_feat[11 + a] = (mine - theirs) / (N * N)
    if best >= 0:
        oy = pos[best, 0] - y
        ox = pos[best, 1] - x
        out_feat[14] = (oy * fy + ox * fx) / N
        out_feat[15] = (oy * ry + ox * rx) / N
        out_feat[16] = best_d / (2 * N)
        rel = (dirs[best] - d) % 4
        for q in range(4):
            out_feat[17 + q] = 1.0 if q == rel else 0.0
    else:
        for q in range(14, 21):
            out_feat[q] = 0.0
    out_feat[21] = n_src / 4.0
    out_feat[22] = separated if n_src > 0 else 1.0


def observe(state, b):
    win = np.zeros((3, W, W), np.float32)
    feat = np.zeros(NF, np.float32)
    features(state["grid"], state["pos"], state["dirs"], state["alive"], b, win, feat)
    return win, feat


# ---------------------------------------------------------------- baseline bots
@njit(cache=True)
def safe_moves(grid, y, x, d):
    ok = np.zeros(3, np.bool_)
    for a in range(3):
        nd = (d + a - 1) % 4
        ok[a] = grid[y + DY[nd], x + DX[nd]] == 0
    return ok


def bot_random(state, b, rng):
    return int(rng.integers(3))


def bot_avoid_walls(state, b, rng):
    ok = safe_moves(state["grid"], state["pos"][b, 0], state["pos"][b, 1], state["dirs"][b])
    choices = np.flatnonzero(ok)
    if len(choices) == 0:
        return 1
    if ok[1] and rng.random() < 0.6:
        return 1
    return int(rng.choice(choices))


@njit(cache=True)
def space_filler_core(grid, y, x, d):
    dist = np.empty((S, S), np.int32)
    queue = np.empty(S * S, np.int32)
    best = 1
    best_n = -1
    for a in (1, 0, 2):           # ties go to straight
        n = flood_after(grid, y, x, d, a, dist, queue)
        if n > best_n:
            best_n = n
            best = a
    return best


def bot_space_filler(state, b, rng):
    return int(space_filler_core(state["grid"], state["pos"][b, 0], state["pos"][b, 1], state["dirs"][b]))


def bot_territory(state, b, rng):
    """The classic Tron heuristic: the move that leaves the most cells closer to you than to any opponent
    (territory, a Voronoi split by breadth-first search); ties go to the most reachable cells, then straight."""
    _, f = observe(state, b)
    key = [f[11 + a] * 10_000 + f[8 + a] + (1e-6 if a == 1 else 0) for a in range(3)]
    return int(np.argmax(key))


BOTS = {"territory": bot_territory, "random": bot_random, "avoid-walls": bot_avoid_walls, "space-filler": bot_space_filler}
