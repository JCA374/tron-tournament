"""The arena library for the Play page: games between every pair of entrants (and free-for-alls), stored compactly so
the browser can replay and animate them (start positions + moves). Rebuilt every 2 hours with the newest PPO and
Mini-AlphaZero.
    .venv/bin/python -m tournament.arena [--procs 2]"""
import argparse
import glob
import itertools
import json
import multiprocessing as mp
import os
import shutil
import time

import numpy as np

from models.common import ROOT, load_agent
from tron import config, sim

RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "docs", "data", "arena.json")
FROZEN = os.path.join(RES, "arena")
GAMES_PER_PAIR = 6          # 3 start positions x both sides
GAMES_DUEL = 16             # PPO vs Mini-AlphaZero gets more
FFA_GAMES = 6
STARS = ("ppo", "az")


def entrants():
    """Newest checkpoints of the two still training; the retired / knocked-out ones from their last ranking round."""
    os.makedirs(FROZEN, exist_ok=True)
    rounds = sorted(glob.glob(os.path.join(RES, "rounds", "round_*.json")))
    last_spec = {}
    for p in rounds:
        for k, v in json.load(open(p))["specs"].items():
            if k in [m["id"] for m in config.CONTESTANTS]:
                last_spec[k] = v
    out = {}
    for m in config.CONTESTANTS:
        mid = m["id"]
        if mid in STARS:
            meta = json.load(open(os.path.join(ROOT, "runs", mid, "latest.json")))
            src = os.path.join(ROOT, meta["path"])
            dst = os.path.join(FROZEN, f"{mid}.{os.path.basename(src).split('.', 1)[1]}")
            shutil.copy(src, dst)
            out[mid] = {"kind": mid, "path": os.path.relpath(dst, ROOT)}
        elif mid in last_spec:
            out[mid] = last_spec[mid]
    for b in config.BOTS:
        out[b["id"]] = {"kind": b["id"]}
    return out


def _play(args):
    ids, specs, seed = args
    import torch
    torch.set_num_threads(1)
    agents = [load_agent(specs[i], seed=seed + k) for k, i in enumerate(ids)]
    st = sim.new_game(len(ids), seed)
    start = {"pos": st["pos"].tolist(), "dirs": st["dirs"].tolist()}
    moves = []
    while True:
        acts = [agents[b].act(st, b) if st["alive"][b] else 1 for b in range(len(ids))]
        moves.append("".join(map(str, acts)))
        r = sim.step(st, acts)
        if r is not None:
            break
    return {"ids": list(ids), "seed": int(seed), **start, "acts": "".join(moves), "result": int(r), "ticks": int(st["t"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=2)
    a = ap.parse_args()
    t0 = time.time()
    specs = entrants()
    ids = list(specs)
    jobs = []
    base = int(time.time()) % 100_000 * 1000
    for pi, (x, y) in enumerate(itertools.combinations(ids, 2)):
        n = GAMES_DUEL if {x, y} == set(STARS) else GAMES_PER_PAIR
        for k in range(n):
            seed = base + pi * 101 + k // 2
            jobs.append(((x, y) if k % 2 == 0 else (y, x), specs, seed))
    models = [m["id"] for m in config.CONTESTANTS if m["id"] in specs]
    for k in range(FFA_GAMES):
        order = list(np.random.default_rng(base + k).permutation(models))
        jobs.append((tuple(order), specs, base + 900_000 + k))
    # tree-search games first: they take longest
    jobs.sort(key=lambda j: -("az" in j[0]))
    with mp.get_context("fork").Pool(a.procs) as pool:
        games = list(pool.imap_unordered(_play, jobs))
    pairs, ffa = {}, []
    for g in games:
        if len(g["ids"]) > 2:
            ffa.append(g)
        else:
            pairs.setdefault("|".join(sorted(g["ids"])), []).append(g)
    info = []
    for m in config.CONTESTANTS + config.BOTS:
        if m["id"] not in specs:
            continue
        status = "training" if m["id"] in STARS else ("bot" if m in config.BOTS else "frozen")
        info.append({"id": m["id"], "label": m["label"], "color": m["color"], "method": m["method"], "status": status})
    out = {"updated": time.strftime("%Y-%m-%d %H:%M"), "seconds": round(time.time() - t0), "entrants": info,
           "pairs": pairs, "ffa": ffa, "board": sim.N}
    tmp = OUT + ".tmp"
    json.dump(out, open(tmp, "w"), separators=(",", ":"))
    os.replace(tmp, OUT)
    print(json.dumps({"games": len(games), "seconds": out["seconds"], "bytes": os.path.getsize(OUT)}))


if __name__ == "__main__":
    main()
