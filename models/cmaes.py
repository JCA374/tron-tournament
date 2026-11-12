"""CMA-ES: evolves the ~870 weights of a fixed small network (23 features -> 32 tanh -> 3).
Fitness = games against the fixed bots and a hall of fame of earlier champions (the same games for the whole population).
    .venv/bin/python -m models.cmaes --run cmaes --cores 1 [--resume]"""
import argparse
import multiprocessing as mp
import os
import pickle
import time

import cma
import numpy as np

from models import evo
from models.common import Run, atomic_pickle

HP = dict(hidden=32, popsize=32, sigma0=0.5, hof_size=6, n_bot_space=4, n_bot_avoid=2, n_hof_games=4,
          rivals=[], n_rival_games=0)


# settings changed during the tournament (they win over the values saved in the checkpoint); see the log
OVERRIDE = dict(n_bot_space=2, n_bot_avoid=0, n_hof_games=2, rivals=["ppo", "dqn", "neat"], n_rival_games=6)


def load_agent(path):
    d = pickle.load(open(path, "rb"))
    return evo.FeatAgent(evo.MLPPolicy(d["theta"], d["hidden"]), "cmaes")


def rival_specs(ids):
    """Other contestants' frozen checkpoints from the newest ranking round (results/snapshots)."""
    import glob
    import json
    from models.common import ROOT
    rounds = sorted(glob.glob(os.path.join(ROOT, "results", "rounds", "round_*.json")))
    if not rounds:
        return []
    specs = json.load(open(rounds[-1]))["specs"]
    return [specs[i] for i in ids if i in specs and os.path.exists(os.path.join(ROOT, specs[i]["path"]))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="cmaes")
    ap.add_argument("--cores", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--minutes", type=float, default=0)
    a = ap.parse_args()
    run = Run(a.run, "cmaes")
    state_path = run.ckpt_path("state.pkl")
    champ_path = run.ckpt_path("pkl")
    gen, games, hof = 0, 0, []
    if a.resume and os.path.exists(state_path):
        s = pickle.load(open(state_path, "rb"))
        es, gen, games, hof = s["es"], s["gen"], s["games"], s["hof"]
        HP.update({k: v for k, v in s["hp"].items() if k not in OVERRIDE})
        HP.update(OVERRIDE)
        run.event("resume", note=f"generation {gen}")
    else:
        n = evo.n_params(HP["hidden"])
        es = cma.CMAEvolutionStrategy(np.zeros(n), HP["sigma0"], {"popsize": HP["popsize"], "verbose": -9, "seed": 1})
        run.event("run_start", note=f"CMA-ES on a 23-{HP['hidden']}-3 network ({n} weights), population {HP['popsize']}. hp={HP}")
    pool = mp.Pool(a.cores) if a.cores > 1 else None
    t_start = time.time()
    best_theta = None

    def save():
        atomic_pickle({"es": es, "gen": gen, "games": games, "hof": hof, "hp": HP}, state_path)
        theta = best_theta if best_theta is not None else es.mean
        atomic_pickle({"theta": np.asarray(theta), "hidden": HP["hidden"]}, champ_path)
        run.publish_latest(champ_path, gen=gen, games=games)

    last_save = time.time()
    while not run.stop:
        if run.paused():
            save()
            run.wait_if_paused()
        t0 = time.time()
        sched = evo.schedule(len(hof), gen * 104729 + 3, HP["n_bot_space"], HP["n_bot_avoid"], HP["n_hof_games"])
        opps = ["space-filler", "avoid-walls"] + [("mlp", h, HP["hidden"]) for h in hof]
        rspecs = rival_specs(HP["rivals"]) if HP["n_rival_games"] else []
        if rspecs:
            rng = np.random.default_rng(gen * 31 + 5)
            for k in range(HP["n_rival_games"]):
                sched.append((len(opps) + k % len(rspecs), int(rng.integers(1 << 30)), k % 2))
            opps = opps + [("agent", r) for r in rspecs]
        xs = es.ask()
        jobs = [(("mlp", np.asarray(x, np.float32), HP["hidden"]), opps, sched) for x in xs]
        fits = pool.map(evo.fitness, jobs) if pool else list(map(evo.fitness, jobs))
        es.tell(xs, [-f for f in fits])
        gen += 1
        games += len(xs) * len(sched)
        # the distribution mean is usually the most reliable policy; the champion for the hall of fame is the mean
        best_theta = np.asarray(es.mean, np.float32)
        hof.append(best_theta.copy())
        hof[:] = hof[-HP["hof_size"]:]
        f = np.array(fits)
        run.stats({"gen": gen, "games": games, "best_fit": float(f.max()), "mean_fit": float(f.mean()),
                   "sigma": float(es.sigma), "sec_per_gen": round(time.time() - t0, 1)})
        if time.time() - last_save > 300 or gen == 1:
            save()
            last_save = time.time()
        if a.minutes and time.time() - t_start > a.minutes * 60:
            break
    save()


if __name__ == "__main__":
    main()
