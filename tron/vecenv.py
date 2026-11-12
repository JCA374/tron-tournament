"""Many 1v1 Tron games side by side, for training. Player 0 is the learner, player 1 the opponent.
Finished games restart automatically with a fresh seed."""
import numpy as np
from numba import njit, prange

from tron import sim


@njit(parallel=True, cache=True)
def _obs_batch(grids, pos, dirs, alive, b, wins, feats):
    for i in prange(grids.shape[0]):
        if alive[i, b]:
            sim.features(grids[i], pos[i], dirs[i], alive[i], b, wins[i], feats[i])


@njit(cache=True)
def _step_batch(grids, pos, dirs, alive, acts):
    for i in range(grids.shape[0]):
        sim.step_core(grids[i], pos[i], dirs[i], alive[i], acts[i])


@njit(parallel=True, cache=True)
def _space_filler_batch(grids, pos, dirs, alive, b, out):
    for i in prange(grids.shape[0]):
        out[i] = sim.space_filler_core(grids[i], pos[i, b, 0], pos[i, b, 1], dirs[i, b]) if alive[i, b] else 1


@njit(cache=True)
def _avoid_walls_batch(grids, pos, dirs, alive, b, u, out):
    for i in range(grids.shape[0]):
        ok = sim.safe_moves(grids[i], pos[i, b, 0], pos[i, b, 1], dirs[i, b])
        n = ok.sum()
        if n == 0:
            out[i] = 1
        elif ok[1] and u[i, 0] < 0.6:
            out[i] = 1
        else:
            k = int(u[i, 1] * n)
            for a in range(3):
                if ok[a]:
                    if k == 0:
                        out[i] = a
                        break
                    k -= 1


class VecTron:
    def __init__(self, n, seed=0):
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.grids = np.zeros((n, sim.S, sim.S), np.int8)
        self.pos = np.zeros((n, 2, 2), np.int32)
        self.dirs = np.zeros((n, 2), np.int32)
        self.alive = np.ones((n, 2), np.bool_)
        self.t = np.zeros(n, np.int32)
        self.wins = np.zeros((n, 3, sim.W, sim.W), np.float32)
        self.feats = np.zeros((n, sim.NF), np.float32)
        for i in range(n):
            self._reset(i)

    def _reset(self, i):
        g = self.grids[i]
        g[:] = 0
        g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = 1
        p, d = sim.start_positions(self.rng, 2)
        self.pos[i] = p
        self.dirs[i] = d
        self.alive[i] = True
        self.t[i] = 0
        g[p[0, 0], p[0, 1]] = 2
        g[p[1, 0], p[1, 1]] = 3

    def obs(self, b=0):
        """(windows, features) for player b of every game (new arrays)."""
        wins = np.zeros_like(self.wins)
        feats = np.zeros_like(self.feats)
        _obs_batch(self.grids, self.pos, self.dirs, self.alive, b, wins, feats)
        return wins, feats

    def bot(self, kind, b=1):
        out = np.ones(self.n, np.int32)
        if kind == "territory":
            _, f = self.obs(b)
            key = f[:, 11:14] * 10_000 + f[:, 8:11] + np.array([0, 1e-6, 0], np.float32)
            out[:] = key.argmax(1)
            out[~self.alive[:, b]] = 1
        elif kind == "space-filler":
            _space_filler_batch(self.grids, self.pos, self.dirs, self.alive, b, out)
        elif kind == "avoid-walls":
            _avoid_walls_batch(self.grids, self.pos, self.dirs, self.alive, b, self.rng.random((self.n, 2)), out)
        else:
            out[:] = self.rng.integers(3, size=self.n)
        return out

    def step(self, a0, a1):
        """Returns reward for player 0 (+1 win, -1 loss, 0 otherwise), done flags, and episode lengths of the
        games that finished. Finished games are reset."""
        acts = np.stack([np.asarray(a0, np.int32), np.asarray(a1, np.int32)], 1)
        _step_batch(self.grids, self.pos, self.dirs, self.alive, acts)
        self.t += 1
        a = self.alive
        done = ~a[:, 0] | ~a[:, 1] | (self.t >= sim.MAX_TICKS)
        rew = np.where(a[:, 0] & ~a[:, 1], 1.0, np.where(~a[:, 0] & a[:, 1], -1.0, 0.0)).astype(np.float32)
        lens = self.t[done].copy()
        for i in np.flatnonzero(done):
            self._reset(i)
        return rew, done, lens
