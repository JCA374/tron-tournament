"""Shared by all five trainers: the run folder (stats, events, checkpoints, pause file), the small CNN, and the
interface every model gives the tournament: load_agent(path) -> object with act(state, b) and reset()."""
import csv
import json
import os
import signal
import time

import numpy as np

from tron import sim

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def setup_threads(cores):
    os.environ.setdefault("NUMBA_NUM_THREADS", str(cores))
    import torch
    # always 1: on these tiny networks 2 torch threads made a DQN step 3x slower (39 -> 126 ms, Thu 12:40);
    # extra cores go to numba's parallel observation code instead
    torch.set_num_threads(1)
    try:
        import numba
        numba.set_num_threads(min(cores, numba.config.NUMBA_NUM_THREADS))
    except Exception:
        pass


class Run:
    """runs/<id>/: stats.csv (one row per iteration), events.jsonl, latest checkpoint, PAUSE file from the supervisor."""

    def __init__(self, run_id, kind=None):
        self.id = run_id
        self.kind = kind or run_id
        self.dir = os.path.join(ROOT, "runs", run_id)
        os.makedirs(os.path.join(self.dir, "ckpt"), exist_ok=True)
        self.stop = False
        self._fields = None
        self.t0 = time.time()
        signal.signal(signal.SIGTERM, self._sigterm)
        signal.signal(signal.SIGINT, self._sigterm)

    def _sigterm(self, *_):
        self.stop = True

    def event(self, kind, **kw):
        with open(os.path.join(self.dir, "events.jsonl"), "a") as f:
            f.write(json.dumps({"t": time.strftime("%Y-%m-%d %H:%M"), "kind": kind, **kw}) + "\n")

    def stats(self, row):
        path = os.path.join(self.dir, "stats.csv")
        row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **row}
        if self._fields is None:
            if os.path.exists(path) and os.path.getsize(path) > 0:
                with open(path) as f:
                    self._fields = next(csv.reader(f))
            else:
                self._fields = list(row)
                with open(path, "w", newline="") as f:
                    csv.writer(f).writerow(self._fields)
        with open(path, "a", newline="") as f:
            csv.DictWriter(f, self._fields, extrasaction="ignore").writerow(
                {k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()})

    def paused(self):
        return os.path.exists(os.path.join(self.dir, "PAUSE"))

    def wait_if_paused(self, on_pause=None):
        """Block while the supervisor's catch-up rule pauses this model. Returns seconds spent paused."""
        t = time.time()
        if self.paused() and on_pause:
            on_pause()
        while self.paused() and not self.stop:
            time.sleep(5)
        return time.time() - t

    def ckpt_path(self, ext):
        return os.path.join(self.dir, "ckpt", f"latest.{ext}")

    def publish_latest(self, path, **meta):
        """Tell the tournament which file is this model's newest playable checkpoint."""
        tmp = os.path.join(self.dir, "latest.json.tmp")
        json.dump({"kind": self.kind, "path": os.path.relpath(path, ROOT), "t": time.time(), **meta}, open(tmp, "w"))
        os.replace(tmp, os.path.join(self.dir, "latest.json"))


def atomic_torch_save(obj, path):
    import torch
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_pickle(obj, path):
    import pickle
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


# ---------------------------------------------------------------- the network shared by PPO, DQN and AlphaZero
def make_net(n_out, hidden=128, value_head=False):
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        """15x15 egocentric window (3 channels, flattened) + 23 scalar features -> 2-layer MLP -> heads.
        (A small CNN was tried first: 6x slower per update on this CPU-only machine.)"""

        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(nn.Linear(3 * sim.W * sim.W + sim.NF, 2 * hidden), nn.ReLU(),
                                    nn.Linear(2 * hidden, hidden), nn.ReLU())
            self.head = nn.Linear(hidden, n_out)
            self.vhead = nn.Linear(hidden, 1) if value_head else None

        def forward(self, win, feat):
            h = self.fc(torch.cat([win.flatten(1), feat], 1))
            if self.vhead is None:
                return self.head(h)
            return self.head(h), self.vhead(h).squeeze(1)

    return Net()


def T(x):
    import torch
    return torch.from_numpy(np.ascontiguousarray(x))


class NetAgent:
    """Greedy agent over a network's first output head (logits or Q-values)."""

    def __init__(self, net, name=""):
        self.net = net.eval()
        self.name = name

    def reset(self):
        pass

    def act(self, state, b):
        import torch
        w, f = sim.observe(state, b)
        with torch.no_grad():
            out = self.net(T(w[None]), T(f[None]))
            if isinstance(out, tuple):
                out = out[0]
        return int(out[0].argmax())


class BotAgent:
    def __init__(self, kind, seed=0):
        self.fn = sim.BOTS[kind]
        self.rng = np.random.default_rng(seed)
        self.name = kind

    def reset(self):
        pass

    def act(self, state, b):
        return self.fn(state, b, self.rng)


def load_agent(spec, seed=0):
    """spec: {"kind": "ppo"|...|bot name, "path": relative checkpoint path}."""
    kind = spec["kind"]
    if kind in sim.BOTS:
        return BotAgent(kind, seed)
    import importlib
    mod = importlib.import_module(f"models.{kind}")
    return mod.load_agent(os.path.join(ROOT, spec["path"]))
