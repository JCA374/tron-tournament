"""Double + dueling DQN with self-play: 32 games at once against its own older snapshots and the fixed bots.
    .venv/bin/python -m models.dqn --run dqn --cores 1 [--resume]"""
import argparse
import collections
import copy
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from models.common import Run, NetAgent, T, atomic_torch_save, make_net, setup_threads
from tron import sim
from tron.vecenv import VecTron

HP = dict(n_envs=32, buffer=150_000, batch=256, lr=2.5e-4, gamma=0.99, tau=0.005, eps_start=1.0, eps_end=0.03,
          eps_steps=1_500_000, train_every=1, learn_start=10_000, snapshot_every=100_000, pool_size=12,
          p_bot_space=0.15, p_bot_avoid=0.05, opp_switch=2000, hidden=128, n_step=3)


def dueling_net(hidden):
    import torch.nn as nn
    base = make_net(4, hidden)            # head outputs [V, A0, A1, A2]

    class Duel(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = base

        def forward(self, w, f):
            o = self.base(w, f)
            v, adv = o[:, :1], o[:, 1:]
            return v + adv - adv.mean(1, keepdim=True)
    return Duel()


def load_agent(path):
    ck = torch.load(path, weights_only=False)
    net = dueling_net(ck["hp"]["hidden"])
    net.load_state_dict(ck["net"])
    return NetAgent(net, "dqn")


class Buffer:
    def __init__(self, n):
        self.n, self.i, self.full = n, 0, False
        self.w = np.zeros((n, 3, sim.W, sim.W), np.uint8)
        self.f = np.zeros((n, sim.NF), np.float32)
        self.a = np.zeros(n, np.int64)
        self.r = np.zeros(n, np.float32)
        self.w2 = np.zeros((n, 3, sim.W, sim.W), np.uint8)
        self.f2 = np.zeros((n, sim.NF), np.float32)
        self.d = np.zeros(n, np.float32)
        self.g = np.zeros(n, np.float32)       # discount for the bootstrap (gamma ** steps actually taken)

    def add(self, w, f, a, r, w2, f2, d, g):
        k = len(a)
        idx = (self.i + np.arange(k)) % self.n
        self.w[idx], self.f[idx], self.a[idx], self.r[idx] = w, f, a, r
        self.w2[idx], self.f2[idx], self.d[idx], self.g[idx] = w2, f2, d, g
        self.i = (self.i + k) % self.n
        self.full |= self.i < k

    def __len__(self):
        return self.n if self.full else self.i

    def sample(self, b):
        idx = np.random.randint(0, len(self), b)
        return (T(self.w[idx].astype(np.float32)), T(self.f[idx]), T(self.a[idx]), T(self.r[idx]),
                T(self.w2[idx].astype(np.float32)), T(self.f2[idx]), T(self.d[idx]), T(self.g[idx]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="dqn")
    ap.add_argument("--cores", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--minutes", type=float, default=0)
    a = ap.parse_args()
    setup_threads(a.cores)
    run = Run(a.run, "dqn")
    ck_path = run.ckpt_path("pt")
    net = dueling_net(HP["hidden"])
    opt = torch.optim.Adam(net.parameters(), lr=HP["lr"])
    steps, games, updates, pool = 0, 0, 0, []
    if a.resume and os.path.exists(ck_path):
        ck = torch.load(ck_path, weights_only=False)
        HP.update(ck["hp"])
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["opt"])
        steps, games, updates, pool = ck["steps"], ck["games"], ck["updates"], ck["pool"]
        run.event("resume", note=f"{steps:,} steps; the replay buffer starts empty again")
    else:
        run.event("run_start", note=f"Double + dueling DQN, {HP['n_step']}-step returns, self-play vs old snapshots + bots. hp={HP}")
    target = copy.deepcopy(net).eval()
    opp_net = dueling_net(HP["hidden"]).eval()
    buf = Buffer(HP["buffer"])
    N, nst, gam = HP["n_envs"], HP["n_step"], HP["gamma"]
    env = VecTron(N, seed=int(time.time()) % 100000 + 7)
    if not pool:
        pool.append(copy.deepcopy(net.state_dict()))
    recent = collections.deque(maxlen=400)
    t_start, s0, last_save, last_log = time.time(), steps, time.time(), time.time()
    hist = collections.deque(maxlen=nst)        # last n steps of (w, f, a, r, d)
    opp, opp_left = "space-filler", 0
    loss_v = 0.0

    def save():
        atomic_torch_save({"net": net.state_dict(), "opt": opt.state_dict(), "hp": HP, "steps": steps, "games": games,
                           "updates": updates, "pool": pool}, ck_path)
        run.publish_latest(ck_path, steps=steps, games=games)

    w, f = env.obs(0)
    while not run.stop:
        if run.paused():
            save()
            run.wait_if_paused()
        if opp_left <= 0:
            u = random.random()
            opp = "space-filler" if u < HP["p_bot_space"] else "avoid-walls" if u < HP["p_bot_space"] + HP["p_bot_avoid"] else "net"
            if opp == "net":
                opp_net.load_state_dict(pool[-1] if random.random() < 0.5 else random.choice(pool))
            opp_left = HP["opp_switch"]
        eps = max(HP["eps_end"], HP["eps_start"] - (HP["eps_start"] - HP["eps_end"]) * steps / HP["eps_steps"])
        with torch.no_grad():
            q = net(T(w), T(f))
            act = q.argmax(1).numpy()
            rnd = np.random.random(N) < eps
            act[rnd] = np.random.randint(0, 3, rnd.sum())
            if opp == "net":
                ow, of = env.obs(1)
                oact = opp_net(T(ow), T(of)).argmax(1).numpy()
                r2 = np.random.random(N) < 0.03
                oact[r2] = np.random.randint(0, 3, r2.sum())
            else:
                oact = env.bot(opp, 1)
        r, d, lens = env.step(act, oact)
        w2, f2 = env.obs(0)
        hist.append((w, f, act, r, d))
        # n-step transition starting nst-1 steps ago (truncated at episode ends)
        if len(hist) == nst:
            w0, f0, a0, _, _ = hist[0]
            R = np.zeros(N, np.float32)
            disc = np.ones(N, np.float32)
            done = np.zeros(N, bool)
            for (_, _, _, rk, dk) in hist:
                R += np.where(done, 0, disc * rk)
                disc = np.where(done, disc, disc * gam)
                done |= dk
            buf.add(w0.astype(np.uint8), f0, a0, R, w2.astype(np.uint8), f2, done.astype(np.float32), disc)
        for i in np.flatnonzero(d):
            recent.append((r[i], opp))
        games += int(d.sum())
        steps += N
        opp_left -= N
        w, f = w2, f2
        if len(buf) >= HP["learn_start"] and (steps // N) % HP["train_every"] == 0:
            bw, bf, ba, br, bw2, bf2, bd, bg = buf.sample(HP["batch"])
            with torch.no_grad():
                na = net(bw2, bf2).argmax(1, keepdim=True)
                tq = target(bw2, bf2).gather(1, na).squeeze(1)
                y = br + bg * (1 - bd) * tq
            qv = net(bw, bf).gather(1, ba[:, None]).squeeze(1)
            loss = F.smooth_l1_loss(qv, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
            opt.step()
            updates += 1
            loss_v = float(loss.detach())
            with torch.no_grad():
                for p, tp in zip(net.parameters(), target.parameters()):
                    tp.mul_(1 - HP["tau"]).add_(p, alpha=HP["tau"])
        if steps % HP["snapshot_every"] < N:
            pool.append(copy.deepcopy(net.state_dict()))
            pool[:] = pool[-HP["pool_size"]:]
        if time.time() - last_log > 15:
            def wr(kind):
                xs = [r for r, o in recent if o == kind]
                return (sum(1 for r in xs if r > 0) / len(xs)) if xs else ""
            run.stats({"steps": steps, "games": games, "updates": updates, "eps": round(eps, 4), "win_vs_self": wr("net"),
                       "win_vs_space": wr("space-filler"), "loss": loss_v, "sps": round((steps - s0) / max(1, time.time() - t_start))})
            last_log = time.time()
        if time.time() - last_save > 300:
            save()
            last_save = time.time()
        if a.minutes and time.time() - t_start > a.minutes * 60:
            break
    save()


if __name__ == "__main__":
    main()
