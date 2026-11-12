"""NEAT (neat-python): evolves both the weights and the shape of a small network that reads the 23 features.
Fitness = games against the fixed bots and a hall of fame of earlier champions.
    .venv/bin/python -m models.neat --run neat --cores 1 [--resume]"""
import argparse
import multiprocessing as mp
import os
import pickle
import time

import neat
import numpy as np

from models import evo
from models.common import Run, atomic_pickle

CFG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "neat.cfg")
HP = dict(hof_size=6, n_bot_space=4, n_bot_avoid=2, n_hof_games=4)

# settings changed during the tournament (they win over the values saved in the checkpoint); see the log
OVERRIDE = dict(bot_main="territory")


def load_agent(path):
    d = pickle.load(open(path, "rb"))
    return evo.FeatAgent(evo.NeatPolicy(d["genome"], CFG), "neat")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="neat")
    ap.add_argument("--cores", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--minutes", type=float, default=0)
    a = ap.parse_args()
    run = Run(a.run, "neat")
    state_path = run.ckpt_path("state.pkl")
    champ_path = run.ckpt_path("pkl")
    config = evo.neat_config(CFG)
    gen, games, hof = 0, 0, []
    if a.resume and os.path.exists(state_path):
        s = pickle.load(open(state_path, "rb"))
        pop, gen, games, hof = s["pop"], s["gen"], s["games"], s["hof"]
        HP.update(s.get("hp", {}))
        HP.update(OVERRIDE)
        run.event("resume", note=f"generation {gen}")
    else:
        pop = neat.Population(config)
        run.event("run_start", note=f"NEAT, population {config.pop_size}, inputs = the 23 features, 3 outputs. hp={HP}")
    pool = mp.Pool(a.cores) if a.cores > 1 else None
    t_start = time.time()
    best_g = None

    def save():
        atomic_pickle({"pop": pop, "gen": gen, "games": games, "hof": hof, "hp": HP}, state_path)
        if best_g is not None:
            atomic_pickle({"genome": best_g}, champ_path)
            run.publish_latest(champ_path, gen=gen, games=games)

    last_save = time.time()
    while not run.stop:
        if run.paused():
            save()
            run.wait_if_paused()
        sched = evo.schedule(len(hof), gen * 7919 + 13, HP["n_bot_space"], HP["n_bot_avoid"], HP["n_hof_games"])
        opps = [HP.get("bot_main", "space-filler"), "avoid-walls"] + [("neat", g, CFG) for g in hof]
        stats = {}

        def eval_genomes(genomes, config):
            jobs = [(("neat", g, CFG), opps, sched) for _, g in genomes]
            fits = pool.map(evo.fitness, jobs) if pool else list(map(evo.fitness, jobs))
            for (_, g), fit in zip(genomes, fits):
                g.fitness = fit
            stats["fits"] = fits
            stats["best"] = max((g for _, g in genomes), key=lambda g: g.fitness)

        t0 = time.time()
        pop.run(eval_genomes, 1)
        gen += 1
        games += len(stats["fits"]) * len(sched)
        # generation champion: highest fitness this generation (noisy; the hall of fame keeps the last few)
        best_g = stats["best"]
        hof.append(best_g)
        hof[:] = hof[-HP["hof_size"]:]
        fits = np.array(stats["fits"])
        n_nodes = len(best_g.nodes)
        n_conn = sum(1 for c in best_g.connections.values() if c.enabled)
        run.stats({"gen": gen, "games": games, "best_fit": float(fits.max()), "mean_fit": float(fits.mean()),
                   "species": len(pop.species.species), "nodes": n_nodes, "connections": n_conn,
                   "sec_per_gen": round(time.time() - t0, 1)})
        if time.time() - last_save > 300 or gen == 1:
            save()
            last_save = time.time()
        if a.minutes and time.time() - t_start > a.minutes * 60:
            break
    save()


if __name__ == "__main__":
    main()
