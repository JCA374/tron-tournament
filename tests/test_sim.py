import numpy as np

from tron import sim
from tournament.match import bradley_terry


def test_mirror_start_is_symmetric():
    for seed in range(50):
        st = sim.new_game(2, seed)
        p, d = st["pos"], st["dirs"]
        assert (p[1] == sim.S - 1 - p[0]).all()
        assert d[1] == (d[0] + 2) % 4


def test_same_deterministic_bot_both_sides_stays_symmetric():
    rng = np.random.default_rng(0)
    for seed in range(20):
        st = sim.new_game(2, seed)
        while True:
            a = [sim.bot_space_filler(st, 0, rng), sim.bot_space_filler(st, 1, rng)]
            assert a[0] == a[1]                      # identical view -> identical choice
            r = sim.step(st, a)
            if r is not None:
                break
        assert r == -1                               # a mirror match can only end in a draw


def test_wall_and_head_on_collisions():
    st = sim.new_game(2, 0)
    st["grid"][1:-1, 1:-1] = 0
    st["pos"][:] = [[5, 1], [5, 3]]
    st["dirs"][:] = [1, 3]                           # facing each other with one empty cell between
    st["grid"][5, 1] = 2
    st["grid"][5, 3] = 3
    assert sim.step(st, [1, 1]) == -1                # both enter (5,2): both die, draw
    st = sim.new_game(2, 0)
    st["grid"][1:-1, 1:-1] = 0
    st["pos"][:] = [[1, 5], [20, 20]]
    st["dirs"][:] = [0, 0]
    st["grid"][1, 5] = 2
    st["grid"][20, 20] = 3
    assert sim.step(st, [1, 1]) == 1                 # bike 0 drives into the top wall


def test_observation_rotation_invariance():
    """The same situation rotated on the board gives the same egocentric observation."""
    st = sim.new_game(2, 3)
    w0, f0 = sim.observe(st, 0)
    w1, f1 = sim.observe(st, 1)                      # bike 1 is the 180-degree rotation of bike 0
    assert np.allclose(w0, w1) and np.allclose(f0, f1)


def test_bradley_terry_order_and_anchor():
    res = [("a", "b", 1.0)] * 15 + [("a", "b", 0.0)] * 5 + [("b", "c", 1.0)] * 15 + [("b", "c", 0.0)] * 5
    r = bradley_terry(["a", "b", "c"], res, anchor=("b", 1500))
    assert r["a"] > r["b"] == 1500 > r["c"]
