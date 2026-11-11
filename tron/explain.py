"""Pictures for the 'How it sees' page: one game position, and what bike 0 receives (rotated window + features)."""
import json
import os

import numpy as np
from PIL import Image, ImageDraw

from tron import render, sim

FEAT_NAMES = ["ray forward", "ray fwd-right", "ray right", "ray back-right", "ray back", "ray back-left", "ray left",
              "ray fwd-left", "free cells if left", "free cells if straight", "free cells if right",
              "territory if left", "territory if straight", "territory if right", "opponent ahead", "opponent to the right",
              "opponent distance", "opp. heading same", "opp. heading right", "opp. heading opposite", "opp. heading left",
              "live opponents", "walled off from everyone"]


def main(out_dir):
    for seed in range(100, 400):
        rng = np.random.default_rng(seed)
        st = sim.new_game(2, seed)
        acts, ok = [], True
        for _ in range(40):
            a = [sim.bot_avoid_walls(st, 0, rng), sim.bot_avoid_walls(st, 1, rng)]
            acts.append(a)
            if sim.step(st, a) is not None:
                ok = False
                break
        if ok and np.abs(st["pos"][0] - st["pos"][1]).max() <= 6:
            break
    rep = {"seed": seed, "names": ["You (bike 0)", "Opponent"], "colors": ["#4c6ef5", "#f76707"], "actions": acts, "result": -1}
    fr = render.frames(rep, max_frames=1000)
    fr[-2].save(os.path.join(out_dir, "senses_board.png"))
    w, f = sim.observe(st, 0)
    C = 18
    im = Image.new("RGB", (3 * (sim.W * C + 16), sim.W * C + 30), (14, 15, 20))
    d = ImageDraw.Draw(im)
    titles = ["obstacles (walls + all trails)", "your own trail", "opponent heads"]
    cols = [(200, 200, 210), (76, 110, 245), (247, 103, 7)]
    for c in range(3):
        x0 = c * (sim.W * C + 16)
        d.text((x0 + 2, 4), titles[c], fill=(220, 220, 225))
        for i in range(sim.W):
            for j in range(sim.W):
                v = w[c, i, j]
                col = cols[c] if v else (30, 32, 40)
                d.rectangle([x0 + j * C, 24 + i * C, x0 + j * C + C - 2, 24 + i * C + C - 2], fill=col)
        d.rectangle([x0 + sim.R * C - 1, 24 + sim.R * C - 1, x0 + sim.R * C + C - 1, 24 + sim.R * C + C - 1], outline=(255, 255, 255), width=2)
    im.save(os.path.join(out_dir, "senses_window.png"))
    json.dump([{"name": n, "value": round(float(v), 3)} for n, v in zip(FEAT_NAMES, f)],
              open(os.path.join(out_dir, "senses_features.json"), "w"))


if __name__ == "__main__":
    main(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "img"))
