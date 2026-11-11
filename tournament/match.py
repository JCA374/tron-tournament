"""Play one game between agents and keep a compact replay (seed + moves; the simulator is deterministic)."""
import numpy as np

from tron import sim


def play_game(agents, names, colors, seed):
    st = sim.new_game(len(agents), seed)
    for a in agents:
        a.reset()
    moves = []
    while True:
        acts = [agents[b].act(st, b) if st["alive"][b] else 1 for b in range(len(agents))]
        moves.append(acts)
        r = sim.step(st, acts)
        if r is not None:
            break
    return {"seed": int(seed), "names": list(names), "colors": list(colors), "actions": moves, "result": int(r),
            "ticks": int(st["t"])}


def bradley_terry(ids, results, anchor=None, iters=300):
    """results: list of (i, j, score of i) with score 1 / 0.5 / 0. A draw counts as half a win each way, and every pair
    that played gets one virtual draw so ratings stay finite. Returns Elo-scale ratings; anchor=(id, value) pins one."""
    k = {p: n for n, p in enumerate(ids)}
    n = len(ids)
    W = np.zeros((n, n))
    for i, j, s in results:
        W[k[i], k[j]] += s
        W[k[j], k[i]] += 1 - s
    played = (W + W.T) > 0
    W = W + 0.5 * played
    Nn = W + W.T
    g = np.ones(n)
    for _ in range(iters):
        wins = W.sum(1)
        den = (Nn / (g[:, None] + g[None, :] + 1e-300)).sum(1)
        g = np.where(den > 0, wins / np.maximum(den, 1e-300), g)
        g /= np.exp(np.mean(np.log(g)))
    elo = 400 * np.log10(g)
    if anchor and anchor[0] in k:
        elo += anchor[1] - elo[k[anchor[0]]]
    else:
        elo += 1500 - elo.mean()
    return {p: float(elo[k[p]]) for p in ids}
