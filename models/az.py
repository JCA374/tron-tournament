"""Mini-AlphaZero for a simultaneous-move game: decoupled PUCT tree search (every live bike keeps its own visit counts
and values in each node; the child is the joint move), guided by a small policy + value network, trained on self-play.
Several self-play games run side by side so the network evaluates their leaves in one batch.
    .venv/bin/python -m models.az --run az --cores 1 [--resume]"""
import argparse
import collections
import math
import multiprocessing as mp
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from models.common import Run, T, atomic_torch_save, make_net, setup_threads
from tron import sim

HP = dict(hidden=128, sims=48, eval_sims=32, c_puct=1.5, dir_alpha=0.6, dir_frac=0.25, temp_moves=12,
          parallel_games=16, buffer=80_000, batch=256, lr=1e-3, wd=1e-4, train_per_game=6, p_bot_game=0.0)


def copy_state(s):
    return {"grid": s["grid"].copy(), "pos": s["pos"].copy(), "dirs": s["dirs"].copy(), "alive": s["alive"].copy(), "t": s["t"]}


class Node:
    __slots__ = ("state", "P", "N", "W", "children", "value", "terminal", "expanded")

    def __init__(self, state):
        self.state = state
        B = len(state["alive"])
        self.P = np.full((B, 3), 1 / 3, np.float32)
        self.N = np.zeros((B, 3), np.float32)
        self.W = np.zeros((B, 3), np.float32)
        self.children = {}
        self.value = None
        self.terminal = False
        self.expanded = False


def terminal_values(state, prev_alive):
    r = sim.result(state)
    if r is None:
        return None
    B = len(state["alive"])
    v = np.full(B, -1.0, np.float32)
    if r >= 0:
        v[r] = 1.0
    else:
        v[prev_alive] = 0.0           # died together this tick (or the tick limit): a draw for them
    return v


def select(node, c):
    """Each live bike picks its move by PUCT on its own statistics."""
    st = node.state
    acts = np.ones(len(st["alive"]), np.int32)
    for b in np.flatnonzero(st["alive"]):
        n = node.N[b]
        q = np.where(n > 0, node.W[b] / np.maximum(n, 1), 0.0)
        u = c * node.P[b] * math.sqrt(n.sum() + 1) / (1 + n)
        acts[b] = int(np.argmax(q + u))
    return acts


def leaf_obs(state):
    alive = np.flatnonzero(state["alive"])
    ws, fs = [], []
    for b in alive:
        w, f = sim.observe(state, b)
        ws.append(w)
        fs.append(f)
    return alive, ws, fs


def run_search(roots, net, sims, c, noise=None):
    """Grow all trees `sims` times. Leaves of every tree are evaluated in one network batch per simulation."""
    pending = [r for r in roots if not r.expanded]
    if pending:
        _expand(pending, net)
    if noise:
        for r in roots:
            for b in np.flatnonzero(r.state["alive"]):
                r.P[b] = (1 - noise[1]) * r.P[b] + noise[1] * np.random.dirichlet([noise[0]] * 3)
    for _ in range(sims):
        paths, leaves = [], []
        for root in roots:
            node, path = root, []
            while True:
                if node.terminal:
                    break
                a = select(node, c)
                key = a.tobytes()
                path.append((node, a))
                child = node.children.get(key)
                if child is None:
                    s2 = copy_state(node.state)
                    prev = s2["alive"].copy()
                    sim.step_core(s2["grid"], s2["pos"], s2["dirs"], s2["alive"], a)
                    s2["t"] += 1
                    child = Node(s2)
                    tv = terminal_values(s2, prev)
                    if tv is not None:
                        child.terminal, child.value, child.expanded = True, tv, True
                    node.children[key] = child
                    node = child
                    break
                node = child
                if not node.expanded:
                    break
            paths.append(path)
            leaves.append(node)
        todo = [n for n in leaves if not n.expanded]
        if todo:
            _expand(todo, net)
        for path, leaf in zip(paths, leaves):
            v = leaf.value
            for node, a in path:
                for b in np.flatnonzero(node.state["alive"]):
                    node.N[b, a[b]] += 1
                    node.W[b, a[b]] += v[b]


def _expand(nodes, net):
    owners, ws, fs = [], [], []
    for i, n in enumerate(nodes):
        alive, w, f = leaf_obs(n.state)
        owners += [(i, b) for b in alive]
        ws += w
        fs += f
    with torch.no_grad():
        logits, v = net(T(np.stack(ws)), T(np.stack(fs)))
        p = torch.softmax(logits, 1).numpy()
        v = torch.tanh(v).numpy()
    for n in nodes:
        n.value = np.where(n.state["alive"], 0.0, -1.0).astype(np.float32)
        n.expanded = True
    for k, (i, b) in enumerate(owners):
        nodes[i].P[b] = p[k]
        nodes[i].value[b] = v[k]


class AZAgent:
    """Tournament agent: a fresh tree search of eval_sims simulations for every move."""

    def __init__(self, net, sims, c):
        self.net, self.sims, self.c, self.name = net.eval(), sims, c, "az"

    def reset(self):
        pass

    def act(self, state, b):
        root = Node(copy_state(state))
        run_search([root], self.net, self.sims, self.c)
        return int(np.argmax(root.N[b] + 1e-3 * root.P[b]))


def load_agent(path, sims=None):
    ck = torch.load(path, weights_only=False)
    net = make_net(3, ck["hp"]["hidden"], value_head=True)
    net.load_state_dict(ck["net"])
    return AZAgent(net, sims or ck["hp"]["eval_sims"], ck["hp"]["c_puct"])


def selfplay(net, hp, seed):
    """Play hp['parallel_games'] 1v1 self-play games at once. Returns samples (win, feat, pi, z) and game lengths."""
    K = hp["parallel_games"]
    rng = np.random.default_rng(seed)
    games = [sim.new_game(2, int(rng.integers(1 << 30))) for _ in range(K)]
    roots = [Node(copy_state(g)) for g in games]
    hist = [[] for _ in range(K)]
    done = [None] * K
    while any(d is None for d in done):
        live = [i for i in range(K) if done[i] is None]
        run_search([roots[i] for i in live], net, hp["sims"], hp["c_puct"], noise=(hp["dir_alpha"], hp["dir_frac"]))
        for i in live:
            g, root = games[i], roots[i]
            acts = np.ones(2, np.int32)
            for b in np.flatnonzero(g["alive"]):
                n = root.N[b]
                pi = n / max(1, n.sum())
                w, f = sim.observe(g, b)
                hist[i].append((w, f, pi.astype(np.float32), b))
                if g["t"] < hp["temp_moves"]:
                    acts[b] = int(rng.choice(3, p=pi)) if n.sum() > 0 else 1
                else:
                    acts[b] = int(np.argmax(n))
            r = sim.step(g, acts)
            child = root.children.get(acts.tobytes())
            roots[i] = child if child is not None and not child.terminal else Node(copy_state(g))
            if r is not None:
                done[i] = r
    out, lens = [], []
    for i in range(K):
        r = done[i]
        lens.append(games[i]["t"])
        for (w, f, pi, b) in hist[i]:
            z = 0.0 if r < 0 else (1.0 if r == b else -1.0)
            out.append((w, f, pi, z))
    return out, lens, done


def worker(wpath, q, hp, seed, stop):
    torch.set_num_threads(1)
    net = make_net(3, hp["hidden"], value_head=True).eval()
    mtime = 0
    k = 0
    while not stop.is_set():
        try:
            m = os.path.getmtime(wpath)
            if m != mtime:
                net.load_state_dict(torch.load(wpath, weights_only=False))
                mtime = m
        except Exception:
            time.sleep(1)
            continue
        q.put(selfplay(net, hp, seed * 1000003 + k))
        k += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="az")
    ap.add_argument("--cores", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--minutes", type=float, default=0)
    a = ap.parse_args()
    setup_threads(1)
    run = Run(a.run, "az")
    ck_path = run.ckpt_path("pt")
    net = make_net(3, HP["hidden"], value_head=True)
    opt = torch.optim.AdamW(net.parameters(), lr=HP["lr"], weight_decay=HP["wd"])
    games, positions, updates, batches = 0, 0, 0, 0
    if a.resume and os.path.exists(ck_path):
        ck = torch.load(ck_path, weights_only=False)
        HP.update(ck["hp"])
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["opt"])
        games, positions, updates = ck["games"], ck["positions"], ck["updates"]
        run.event("resume", note=f"{games:,} self-play games; the replay buffer starts empty again")
    else:
        run.event("run_start", note=f"Mini-AlphaZero: decoupled PUCT search, {HP['sims']} simulations per move in self-play, "
                                    f"{HP['eval_sims']} in the tournament. hp={HP}")
    buf = collections.deque(maxlen=HP["buffer"])
    wpath = os.path.join(run.dir, "ckpt", "weights_for_workers.pt")
    atomic_torch_save(net.state_dict(), wpath)
    workers, q, stop = [], None, None
    if a.cores > 1:
        ctx = mp.get_context("fork")
        q, stop = ctx.Queue(maxsize=4), ctx.Event()
        workers = [ctx.Process(target=worker, args=(wpath, q, dict(HP), int(time.time()) + i, stop), daemon=True)
                   for i in range(a.cores)]
        for w in workers:
            w.start()
    t_start, last_save, recent_len, g0 = time.time(), time.time(), collections.deque(maxlen=200), games
    loss_p = loss_v = 0.0

    def save():
        atomic_torch_save({"net": net.state_dict(), "opt": opt.state_dict(), "hp": HP, "games": games,
                           "positions": positions, "updates": updates}, ck_path)
        run.publish_latest(ck_path, games=games, positions=positions)

    while not run.stop:
        if run.paused():
            save()
            run.wait_if_paused()
        t0 = time.time()
        if workers:
            samples, lens, _ = q.get()
        else:
            net.eval()
            samples, lens, _ = selfplay(net, HP, int(time.time() * 1000) % (1 << 30))
        buf.extend(samples)
        games += len(lens)
        positions += len(samples)
        recent_len.extend(lens)
        batches += 1
        net.train()
        for _ in range(HP["train_per_game"] * len(lens)):
            if len(buf) < 2000:
                break
            idx = np.random.randint(0, len(buf), HP["batch"])
            w = T(np.stack([buf[i][0] for i in idx]))
            f = T(np.stack([buf[i][1] for i in idx]))
            pi = T(np.stack([buf[i][2] for i in idx]))
            z = torch.tensor([buf[i][3] for i in idx], dtype=torch.float32)
            logits, v = net(w, f)
            lp = -(pi * F.log_softmax(logits, 1)).sum(1).mean()
            lv = F.mse_loss(torch.tanh(v), z)
            opt.zero_grad()
            (lp + lv).backward()
            opt.step()
            updates += 1
            loss_p, loss_v = float(lp.detach()), float(lv.detach())
        net.eval()
        atomic_torch_save(net.state_dict(), wpath)
        run.stats({"games": games, "positions": positions, "updates": updates, "policy_loss": loss_p, "value_loss": loss_v,
                   "mean_len": round(float(np.mean(recent_len)), 1), "sec_per_batch": round(time.time() - t0, 1),
                   "games_per_hour": round((games - g0) / max(1, time.time() - t_start) * 3600)})
        if time.time() - last_save > 300 or batches == 1:
            save()
            last_save = time.time()
        if a.minutes and time.time() - t_start > a.minutes * 60:
            break
    if stop:
        stop.set()
    save()


if __name__ == "__main__":
    main()
