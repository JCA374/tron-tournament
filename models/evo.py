"""Shared by NEAT and CMA-ES: policies that read only the 23 scalar features, and fitness by playing real games
against the fixed bots and a hall of fame of earlier champions. Games run in a process pool (one worker per core)."""
import numpy as np

from tron import sim

_NEAT_CFG = {}


def neat_config(path):
    if path not in _NEAT_CFG:
        import neat
        _NEAT_CFG[path] = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                                      neat.DefaultStagnation, path)
    return _NEAT_CFG[path]


def mlp_shapes(hidden):
    return [(sim.NF, hidden), (hidden,), (hidden, 3), (3,)]


def n_params(hidden):
    return sum(int(np.prod(s)) for s in mlp_shapes(hidden))


class MLPPolicy:
    def __init__(self, theta, hidden):
        ps, i = [], 0
        for s in mlp_shapes(hidden):
            k = int(np.prod(s))
            ps.append(np.asarray(theta[i:i + k], np.float32).reshape(s))
            i += k
        self.W1, self.b1, self.W2, self.b2 = ps

    def __call__(self, f):
        return np.tanh(f @ self.W1 + self.b1) @ self.W2 + self.b2


class NeatPolicy:
    def __init__(self, genome, cfg_path):
        import neat
        self.net = neat.nn.FeedForwardNetwork.create(genome, neat_config(cfg_path))

    def __call__(self, f):
        return np.asarray(self.net.activate(f.tolist()))


def make_policy(blob):
    kind = blob[0]
    if kind == "mlp":
        return MLPPolicy(blob[1], blob[2])
    if kind == "neat":
        return NeatPolicy(blob[1], blob[2])
    raise ValueError(kind)


class FeatAgent:
    """Tournament agent for a features-only policy: argmax of its three outputs."""

    def __init__(self, policy, name):
        self.policy, self.name = policy, name

    def reset(self):
        pass

    def act(self, state, b):
        _, f = sim.observe(state, b)
        return int(np.argmax(self.policy(f)))


def play(me, opp, seed, side, rng):
    """One 1v1 game; `me` is a policy (features -> 3 scores); opp a policy or a bot name.
    Returns (score for me: 1 / 0.5 / 0, ticks)."""
    st = sim.new_game(2, seed)
    b_me, b_op = side, 1 - side
    while True:
        acts = [1, 1]
        _, f = sim.observe(st, b_me)
        acts[b_me] = int(np.argmax(me(f)))
        if isinstance(opp, str):
            acts[b_op] = sim.BOTS[opp](st, b_op, rng)
        else:
            _, fo = sim.observe(st, b_op)
            acts[b_op] = int(np.argmax(opp(fo)))
        r = sim.step(st, acts)
        if r is not None:
            return (1.0 if r == b_me else 0.5 if r < 0 else 0.0), st["t"]


def fitness(args):
    """args: (candidate blob, [opponent blob or bot name], [(opp index, seed, side)]).
    Fitness = mean game score + a small survival bonus (up to 0.05) so early generations have a gradient."""
    blob, opps, games = args
    me = make_policy(blob)
    ops = [o if isinstance(o, str) else make_policy(o) for o in opps]
    rng = np.random.default_rng(games[0][1] if games else 0)
    tot, ticks, wins_bot = 0.0, 0, 0
    for oi, seed, side in games:
        s, t = play(me, ops[oi], seed, side, rng)
        tot += s
        ticks += min(t, 400)
    n = max(1, len(games))
    return tot / n + 0.05 * ticks / (400 * n)


def schedule(n_hof, gen_seed, n_bot_space=4, n_bot_avoid=2, n_hof_games=4):
    """The same games for every candidate in a generation (common random numbers). Opponent index 0 = space-filler,
    1 = avoid-walls, 2.. = hall of fame."""
    rng = np.random.default_rng(gen_seed)
    g = []
    for k in range(n_bot_space):
        g.append((0, int(rng.integers(1 << 30)), k % 2))
    for k in range(n_bot_avoid):
        g.append((1, int(rng.integers(1 << 30)), k % 2))
    for k in range(n_hof_games if n_hof else 0):
        g.append((2 + int(rng.integers(n_hof)), int(rng.integers(1 << 30)), k % 2))
    return g
