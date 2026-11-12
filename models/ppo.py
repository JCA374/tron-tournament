"""PPO with self-play: 64 games at once against a pool of its own older snapshots and the fixed bots.
    .venv/bin/python -m models.ppo --run ppo --cores 1 [--resume]"""
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
from tron.vecenv import VecTron

HP = dict(n_envs=64, n_steps=128, gamma=0.99, lam=0.95, clip=0.2, lr=3e-4, ent=0.01, vf=0.5, epochs=4, minibatch=1024,
          snapshot_every=25, pool_size=12, p_bot_space=0.15, p_bot_avoid=0.05, hidden=128)

# settings changed during the tournament (they win over the values saved in the checkpoint); see the log
OVERRIDE = dict(bot_main="territory", lr=1e-4)


def load_agent(path):
    ck = torch.load(path, weights_only=False)
    net = make_net(3, ck["hp"]["hidden"], value_head=True)
    net.load_state_dict(ck["net"])
    return NetAgent(net, "ppo")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="ppo")
    ap.add_argument("--cores", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--minutes", type=float, default=0)
    a = ap.parse_args()
    setup_threads(a.cores)
    run = Run(a.run, "ppo")
    ck_path = run.ckpt_path("pt")
    net = make_net(3, HP["hidden"], value_head=True)
    opt = torch.optim.Adam(net.parameters(), lr=HP["lr"])
    it, steps, games, pool = 0, 0, 0, []
    if a.resume and os.path.exists(ck_path):
        ck = torch.load(ck_path, weights_only=False)
        HP.update(ck["hp"])
        HP.update(OVERRIDE)
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["opt"])
        for g in opt.param_groups:          # the saved optimizer carries the old learning rate
            g["lr"] = HP["lr"]
        it, steps, games, pool = ck["it"], ck["steps"], ck["games"], ck["pool"]
        run.event("resume", note=f"iteration {it}, {steps:,} steps")
    else:
        run.event("run_start", note=f"PPO, {HP['n_envs']} parallel games, self-play vs a pool of old snapshots + bots. hp={HP}")
    opp_net = make_net(3, HP["hidden"], value_head=True).eval()
    env = VecTron(HP["n_envs"], seed=int(time.time()) % 100000)
    N, Tn = HP["n_envs"], HP["n_steps"]
    recent = collections.deque(maxlen=400)
    t_start, s0 = time.time(), steps

    def save():
        atomic_torch_save({"net": net.state_dict(), "opt": opt.state_dict(), "hp": HP, "it": it, "steps": steps,
                           "games": games, "pool": pool}, ck_path)
        play_path = run.ckpt_path("play.pt")
        atomic_torch_save({"net": net.state_dict(), "hp": HP}, play_path)
        run.publish_latest(play_path, steps=steps, games=games, it=it)

    if not pool:
        pool.append(copy.deepcopy(net.state_dict()))
    last_save = time.time()
    while not run.stop:
        run.wait_if_paused(save)
        # opponent for this iteration
        u = random.random()
        opp = HP.get("bot_main", "space-filler") if u < HP["p_bot_space"] else "avoid-walls" if u < HP["p_bot_space"] + HP["p_bot_avoid"] else "net"
        if opp == "net":
            opp_net.load_state_dict(pool[-1] if random.random() < 0.5 else random.choice(pool))
        bw = np.zeros((Tn, N, 3, 15, 15), np.float32)
        bf = np.zeros((Tn, N, 23), np.float32)
        ba = np.zeros((Tn, N), np.int64)
        blp = np.zeros((Tn, N), np.float32)
        bv = np.zeros((Tn, N), np.float32)
        br = np.zeros((Tn, N), np.float32)
        bd = np.zeros((Tn, N), np.float32)
        w, f = env.obs(0)
        for t in range(Tn):
            with torch.no_grad():
                logits, v = net(T(w), T(f))
                dist = torch.distributions.Categorical(logits=logits)
                act = dist.sample()
                if opp == "net":
                    ow, of = env.obs(1)
                    ol, _ = opp_net(T(ow), T(of))
                    oact = torch.distributions.Categorical(logits=ol).sample().numpy()
                else:
                    oact = env.bot(opp, 1)
            bw[t], bf[t], ba[t] = w, f, act.numpy()
            blp[t], bv[t] = dist.log_prob(act).numpy(), v.numpy()
            r, d, lens = env.step(act.numpy(), oact)
            br[t], bd[t] = r, d
            for i in np.flatnonzero(d):
                recent.append((r[i], opp))
            games += int(d.sum())
            w, f = env.obs(0)
        steps += N * Tn
        with torch.no_grad():
            _, last_v = net(T(w), T(f))
        last_v = last_v.numpy()
        adv = np.zeros((Tn, N), np.float32)
        gae = 0
        for t in reversed(range(Tn)):
            nv = last_v if t == Tn - 1 else bv[t + 1]
            nonterm = 1.0 - bd[t]
            delta = br[t] + HP["gamma"] * nv * nonterm - bv[t]
            gae = delta + HP["gamma"] * HP["lam"] * nonterm * gae
            adv[t] = gae
        ret = adv + bv
        W_, F_ = T(bw.reshape(-1, 3, 15, 15)), T(bf.reshape(-1, 23))
        A_, LP_, ADV, RET = T(ba.reshape(-1)), T(blp.reshape(-1)), T(adv.reshape(-1)), T(ret.reshape(-1))
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
        n = len(A_)
        for _ in range(HP["epochs"]):
            perm = torch.randperm(n)
            for s in range(0, n, HP["minibatch"]):
                idx = perm[s:s + HP["minibatch"]]
                logits, v = net(W_[idx], F_[idx])
                dist = torch.distributions.Categorical(logits=logits)
                ratio = (dist.log_prob(A_[idx]) - LP_[idx]).exp()
                pg = -torch.min(ratio * ADV[idx], ratio.clamp(1 - HP["clip"], 1 + HP["clip"]) * ADV[idx]).mean()
                vl = F.mse_loss(v, RET[idx])
                ent = dist.entropy().mean()
                loss = pg + HP["vf"] * vl - HP["ent"] * ent
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
        it += 1
        if it % HP["snapshot_every"] == 0:
            pool.append(copy.deepcopy(net.state_dict()))
            pool[:] = pool[-HP["pool_size"]:]
        def wr(kind):
            xs = [r for r, o in recent if (o == "net") == (kind == "net") and (kind == "net" or o == kind)]
            return (sum(1 for r in xs if r > 0) / len(xs)) if xs else ""
        run.stats({"it": it, "steps": steps, "games": games, "win_vs_self": wr("net"), "win_vs_space": wr(HP.get("bot_main", "space-filler")),
                   "entropy": float(ent.detach()), "value_loss": float(vl.detach()), "sps": round((steps - s0) / max(1, time.time() - t_start))})
        if time.time() - last_save > 300:
            save()
            last_save = time.time()
        if a.minutes and time.time() - t_start > a.minutes * 60:
            break
    save()


if __name__ == "__main__":
    main()
