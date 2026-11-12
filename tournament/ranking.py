"""One ranking round: every active model's newest checkpoint (frozen copy) and the fixed bots play everyone else,
GAMES_PER_PAIR 1v1 games per pair, each start position played from both sides. Writes results/rounds/round_NNNN.json
and appends results/rating_history.jsonl.
    .venv/bin/python -m tournament.ranking --round 3 --models ppo,dqn,neat,cmaes,az --procs 3
    .venv/bin/python -m tournament.ranking --final --models ppo,dqn,az --games 200 --ffa 50"""
import argparse
import gzip
import itertools
import json
import multiprocessing as mp
import os
import shutil
import time

import numpy as np

from models.common import ROOT, load_agent
from tournament.match import bradley_terry, play_game
from tron import config

RES = os.path.join(ROOT, "results")


def snapshot(mid, tag):
    """Freeze the model's newest checkpoint for this round."""
    meta_p = os.path.join(ROOT, "runs", mid, "latest.json")
    if not os.path.exists(meta_p):
        return None
    meta = json.load(open(meta_p))
    src = os.path.join(ROOT, meta["path"])
    ext = os.path.basename(src).split(".", 1)[1]
    dst_dir = os.path.join(RES, "snapshots")
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, f"{mid}_{tag}.{ext}")
    shutil.copy(src, dst)
    return {"kind": meta["kind"], "path": os.path.relpath(dst, ROOT), "meta": {k: v for k, v in meta.items() if k not in ("path", "kind")}}


def _pair_job(args):
    (ia, sa), (ib, sb), seeds = args
    import torch
    torch.set_num_threads(1)
    A, B = load_agent(sa, seed=seeds[0]), load_agent(sb, seed=seeds[0] + 1)
    out = []
    for k, seed in enumerate(seeds):
        side = k % 2                 # each seed twice: once from each side of the mirrored start
        agents = [A, B] if side == 0 else [B, A]
        names = [ia, ib] if side == 0 else [ib, ia]
        rep = play_game(agents, [config.BY_ID.get(n, {"label": n})["label"] for n in names], [config.BY_ID.get(n, {"color": "#999999"})["color"] for n in names], seed)
        r = rep["result"]
        s = 0.5 if r < 0 else (1.0 if names[r] == ia else 0.0)
        rep["ids"] = names
        out.append((ia, ib, s, rep))
    return out


def _ffa_job(args):
    specs, ids, seed = args
    import torch
    torch.set_num_threads(1)
    agents = [load_agent(s, seed=seed + i) for i, s in enumerate(specs)]
    rep = play_game(agents, [config.BY_ID.get(n, {"label": n})["label"] for n in ids], [config.BY_ID.get(n, {"color": "#999999"})["color"] for n in ids], seed)
    rep["ids"] = ids
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--models", required=True)
    ap.add_argument("--games", type=int, default=config.GAMES_PER_PAIR)
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--ffa", type=int, default=0)
    ap.add_argument("--no-bots", action="store_true")
    ap.add_argument("--bots", default=",".join(b["id"] for b in config.BOTS), help="which fixed bots take part")
    a = ap.parse_args()
    t0 = time.time()
    tag = "final" if a.final else f"r{a.round:04d}"
    specs = {}
    for mid in [m for m in a.models.split(",") if m]:
        s = snapshot(mid, tag)
        if s:
            specs[mid] = s
    if not a.no_bots:
        for b in config.BOTS:
            if b["id"] in a.bots.split(","):
                specs[b["id"]] = {"kind": b["id"]}
    ids = list(specs)
    base = (a.round if not a.final else 999_999) * 1_000_003
    jobs = []
    for pi, (x, y) in enumerate(itertools.combinations(ids, 2)):
        seeds = [base + pi * 1009 + k // 2 for k in range(a.games)]
        for c in range(0, len(seeds), 10):          # chunks of 10 games (5 start positions x both sides)
            jobs.append(((x, specs[x]), (y, specs[y]), seeds[c:c + 10]))
    # longest jobs first (tree search plays slowest)
    jobs.sort(key=lambda j: -(("az" in (j[0][0], j[1][0])) * 10 + 1))
    with mp.get_context("fork").Pool(a.procs) as pool:
        res = [r for chunk in pool.imap_unordered(_pair_job, jobs) for r in chunk]
        ffa = []
        if a.ffa:
            fids = [i for i in ids if i in [m["id"] for m in config.CONTESTANTS]]
            fj = []
            for k in range(a.ffa):
                order = list(np.random.default_rng(k).permutation(fids))
                fj.append(([specs[i] for i in order], order, base + 777_000 + k))
            ffa = list(pool.imap_unordered(_ffa_job, fj))
    # The scale is pinned to a fixed bot. Space-filler first; once every top model beat it 20/20 the scale drifted, so the
    # stronger territory bot was added: rated on the old scale in its first round, then pinned at that value.
    anchor_p = os.path.join(RES, "anchor.json")
    games_ = [(x, y, s) for x, y, s, _ in res]
    if os.path.exists(anchor_p):
        an = json.load(open(anchor_p))
        ratings = bradley_terry(ids, games_, anchor=(an["id"], an["value"]))
    else:
        ratings = bradley_terry(ids, games_, anchor=config.ANCHOR)
        if "territory" in ratings and not a.final:
            json.dump({"id": "territory", "value": round(ratings["territory"], 1), "since_round": a.round}, open(anchor_p, "w"))
    k = {p: n for n, p in enumerate(ids)}
    sc = np.zeros((len(ids), len(ids)))
    cnt = np.zeros((len(ids), len(ids)))
    for x, y, s, _ in res:
        sc[k[x], k[y]] += s
        sc[k[y], k[x]] += 1 - s
        cnt[k[x], k[y]] += 1
        cnt[k[y], k[x]] += 1
    score = np.where(cnt > 0, sc / np.maximum(cnt, 1), np.nan)
    order = sorted(ids, key=lambda p: -ratings[p])
    oi = [k[p] for p in order]
    field = {p: float(np.nanmean(np.delete(score[k[p]], k[p]))) for p in ids}
    # highlights: longest game between #1 and #2 contestant, #1 vs #3, and the weakest contestant's longest win
    cont = [p for p in order if p in [m["id"] for m in config.CONTESTANTS]]
    rep_dir = os.path.join(RES, "replays")
    os.makedirs(rep_dir, exist_ok=True)
    hl = []

    def pick(cands, caption):
        if not cands:
            return
        rep = max(cands, key=lambda r: r["ticks"])
        p = os.path.join(rep_dir, f"{tag}_hl{len(hl)}.json")
        json.dump(rep, open(p, "w"))
        w = rep["result"]
        res_txt = "draw" if w < 0 else f"{rep['names'][w]} wins"
        hl.append({"path": os.path.relpath(p, ROOT), "caption": f"{caption}: {res_txt} after {rep['ticks']} ticks"})

    def games_between(p, q):
        return [r for x, y, s, r in res if {x, y} == {p, q}]
    if len(cont) >= 2:
        L = config.BY_ID
        pick(games_between(cont[0], cont[1]), f"{L[cont[0]]['label']} vs {L[cont[1]]['label']} (#1 vs #2), longest game")
    if len(cont) >= 3:
        pick(games_between(cont[0], cont[2]), f"{config.BY_ID[cont[0]]['label']} vs {config.BY_ID[cont[2]]['label']} (#1 vs #3), longest game")
    if cont:
        low = cont[-1]
        wins = [r for x, y, s, r in res if low in (x, y) and r["result"] >= 0 and r["ids"][r["result"]] == low]
        pick(wins, f"{config.BY_ID[low]['label']} (last place) — its longest win")
    for i, rep in enumerate(sorted(ffa, key=lambda r: -r["ticks"])[:2]):
        pick([rep], "Free-for-all")
    with gzip.open(os.path.join(rep_dir, f"{tag}_all.jsonl.gz"), "wt") as f:
        for x, y, s, r in res:
            f.write(json.dumps(r) + "\n")
        for r in ffa:
            f.write(json.dumps(r) + "\n")
    out = {"round": a.round if not a.final else "final", "t": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "ids": order,
           "ratings": ratings, "score": [[None if (i == j or np.isnan(score[i][j])) else round(float(score[i][j]), 3) for j in oi] for i in oi],
           "field_score": field, "games": len(res), "games_per_pair": a.games, "specs": specs, "highlights": hl,
           "seconds": round(time.time() - t0, 1)}
    if ffa:
        wins = {}
        for r in ffa:
            w = r["result"]
            key = r["ids"][w] if w >= 0 else "draw"
            wins[key] = wins.get(key, 0) + 1
        out["ffa"] = {"games": len(ffa), "wins": wins}
    os.makedirs(os.path.join(RES, "rounds"), exist_ok=True)
    name = "final.json" if a.final else f"round_{a.round:04d}.json"
    json.dump(out, open(os.path.join(RES, "rounds", name), "w"), indent=1)
    if not a.final:
        with open(os.path.join(RES, "rating_history.jsonl"), "a") as f:
            f.write(json.dumps({"t": out["t"], "round": a.round, "ratings": ratings, "games": len(res)}) + "\n")
    print(json.dumps({"round": out["round"], "seconds": out["seconds"], "ratings": {p: round(ratings[p]) for p in order}}))


if __name__ == "__main__":
    main()
