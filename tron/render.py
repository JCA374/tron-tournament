"""Draw Tron games from replays (seed + actions) as GIFs and MP4 frames. No policy imports this module."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tron import sim

CELL = 11
HEAD = 34
BG = (14, 15, 20)
GRIDC = (24, 26, 33)
WALL = (38, 40, 48)


def hex2rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _font(size):
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def frames(replay, max_frames=320, scale=1):
    """replay: {"seed", "names", "colors", "actions": [[a0, a1, ...], ...], "result"}.
    Returns PIL frames: one per tick (or per few ticks for long games), the last one with the outcome written on it."""
    names, colors = replay["names"], [hex2rgb(c) for c in replay["colors"]]
    n = len(names)
    st = sim.new_game(n, replay["seed"])
    acts = replay["actions"]
    every = max(1, int(np.ceil(len(acts) / max_frames)))
    font, small = _font(13), _font(11)
    size = sim.S * CELL
    out = []

    def draw(final=False):
        im = Image.new("RGB", (size, size + HEAD), BG)
        d = ImageDraw.Draw(im)
        g = st["grid"]
        for y in range(sim.S):
            for x in range(sim.S):
                v = g[y, x]
                x0, y0 = x * CELL, HEAD + y * CELL
                if v == 1:
                    d.rectangle([x0, y0, x0 + CELL - 1, y0 + CELL - 1], fill=WALL)
                elif v >= 2:
                    c = colors[v - 2]
                    dim = c if st["alive"][v - 2] else tuple(int(k * .45) for k in c)
                    d.rectangle([x0 + 1, y0 + 1, x0 + CELL - 2, y0 + CELL - 2], fill=dim)
        for b in range(n):
            y, x = st["pos"][b]
            x0, y0 = x * CELL, HEAD + y * CELL
            if st["alive"][b]:
                d.rectangle([x0 - 1, y0 - 1, x0 + CELL, y0 + CELL], fill=(255, 255, 255), outline=colors[b], width=2)
            else:
                d.line([x0, y0, x0 + CELL, y0 + CELL], fill=(255, 80, 80), width=3)
                d.line([x0 + CELL, y0, x0, y0 + CELL], fill=(255, 80, 80), width=3)
        xx = 6
        for b in range(n):
            d.rectangle([xx, 11, xx + 10, 21], fill=colors[b])
            txt = names[b] + ("" if st["alive"][b] else " ✗")
            d.text((xx + 14, 9), txt, fill=(235, 235, 240) if st["alive"][b] else (120, 120, 125), font=font)
            xx += 22 + int(d.textlength(txt, font=font))
        d.text((size - 70, 11), f"tick {st['t']}", fill=(150, 150, 160), font=small)
        if final:
            r = replay.get("result", -1)
            msg = "draw" if r is None or r < 0 else f"{names[r]} wins"
            w = d.textlength(msg, font=_font(22))
            d.rectangle([size / 2 - w / 2 - 14, HEAD + size / 2 - 22, size / 2 + w / 2 + 14, HEAD + size / 2 + 20], fill=(0, 0, 0))
            d.text((size / 2 - w / 2, HEAD + size / 2 - 16), msg, fill=(255, 255, 255), font=_font(22))
        if scale != 1:
            im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
        return im

    out.append(draw())
    for i, a in enumerate(acts):
        sim.step(st, a)
        if (i + 1) % every == 0 or i == len(acts) - 1:
            out.append(draw())
    out.append(draw(final=True))
    return out


def save_gif(replay, path):
    fr = frames(replay)
    durs = [60] * (len(fr) - 1) + [2500]
    # no loop= argument: the GIF plays once and stops on the outcome (the page replays it on click)
    fr[0].save(path, save_all=True, append_images=fr[1:], duration=durs, optimize=True)
