"""Project hooks for publish.py (ml-project-site skill): replays, the Play tab and the Competition page data."""
import glob
import json
import os
import subprocess
import time

from tron import config

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, 'results')


def encode_mp4(frames, out, fps=25, hold=25):
    """frames: list of PIL images (same size, even width/height). Plays once; the last frame is held `hold` frames."""
    w, h = frames[0].size
    p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}',
                          '-r', str(fps), '-i', '-', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '28',
                          '-pix_fmt', 'yuv420p', '-movflags', '+faststart', out], stdin=subprocess.PIPE)
    for f in frames + [frames[-1]] * hold:
        p.stdin.write(f.convert('RGB').tobytes())
    p.stdin.close()
    p.wait()


def _jl(path):
    return [json.loads(l) for l in open(path) if l.strip()] if os.path.exists(path) else []


def replay_gif(policy_path, seed, out_gif):
    """Events carry `champion` = a saved replay JSON (results/replays/...). Render it as a GIF that plays once."""
    from tron import render
    render.save_gif(json.load(open(policy_path)), out_gif)


# ---------------------------------------------------------------- Play tab
def models():
    # Thu 22:10: the Play tab became the Arena (docs/arena.js replays games live in the browser), so no MP4 clips any more;
    # an empty list makes publish.py delete the old clips.
    return []


def games():
    p = os.path.join(RES, 'play_games.json')
    return json.load(open(p)) if os.path.exists(p) else []


def render_play(policy_path, game, out_mp4):
    """policy_path here is the model's snapshot spec; the game is played live and rendered."""
    import torch
    torch.set_num_threads(1)          # torch's default (8 threads) on busy cores made one clip take ~10 minutes
    from tournament import play
    from tron import render
    rep = play.play_tab_game(policy_path, game)
    fr = render.frames(rep, max_frames=600)
    encode_mp4(fr, out_mp4, fps=20, hold=30)
    me = rep['names'].index(rep['me'])
    r = rep['result']
    res = 'win' if r == me else 'draw' if r is None or r < 0 else 'loss'
    return {'ok': res == 'win', 'result': res, 'detail': f"{rep['ticks']} ticks"}


# ---------------------------------------------------------------- Competition page
def extra(data_dir):
    state = json.load(open(os.path.join(RES, 'tournament.json'))) if os.path.exists(os.path.join(RES, 'tournament.json')) else {}
    hist = _jl(os.path.join(RES, 'rating_history.jsonl'))
    rounds = sorted(glob.glob(os.path.join(RES, 'rounds', 'round_*.json')))
    last = json.load(open(rounds[-1])) if rounds else None
    mstate = state.get('models', {})
    rows = []
    for m in config.CONTESTANTS + config.BOTS:
        bot = m in config.BOTS
        rs = [h['ratings'].get(m['id']) for h in hist if h['ratings'].get(m['id']) is not None]
        rating = hist[-1]['ratings'].get(m['id']) if hist else None
        prev = hist[-2]['ratings'].get(m['id']) if len(hist) > 1 else None
        st = mstate.get(m['id'], {})
        rows.append({**m, 'bot': bot, 'rating': rating, 'delta': (rating - prev) if rating is not None and prev is not None else None,
                     'avg3': sum(rs[-3:]) / len(rs[-3:]) if rs else None,
                     'score': (last or {}).get('field_score', {}).get(m['id']),
                     'status': 'bot' if bot else st.get('status', 'waiting'), 'train': st.get('train', '')})
    rows.sort(key=lambda r: (r['rating'] is None, -(r['rating'] or 0)))
    # timeline: tournament events + replays
    ev = _jl(os.path.join(ROOT, 'runs', 'tournament', 'events.jsonl'))
    log = json.load(open(os.path.join(data_dir, 'log.json'))) if os.path.exists(os.path.join(data_dir, 'log.json')) else []
    replay_of = {(e.get('t'), e.get('kind')): e.get('replay') for e in log if e.get('run') == 'tournament'}
    timeline = [dict(e, replay=replay_of.get((e['t'], e['kind']))) for e in ev if e['kind'] != 'round_best']
    timeline.sort(key=lambda e: e['t'], reverse=True)
    reps = []
    if last:
        for i, r in enumerate(last.get('highlights', [])):
            name = f"round{last['round']:04d}_{i}.gif"
            path = os.path.join(ROOT, 'docs', 'replays', name)
            if not os.path.exists(path):
                try:
                    replay_gif(os.path.join(ROOT, r['path']), 0, path)
                except Exception as ex:
                    print('highlight render failed', ex)
                    continue
            reps.append({'src': f'replays/{name}', 'caption': r['caption']})
    comp = {'updated': time.strftime('%Y-%m-%d %H:%M'), 'schedule': config.SCHEDULE, 'rounds': len(hist),
            'games': sum(h.get('games', 0) for h in hist), 'note': state.get('note', ''),
            'models': rows, 'history': hist, 'pauses': state.get('pauses', []),
            'h2h': {'round': last['round'], 'ids': last['ids'], 'score': last['score']} if last else None,
            'replays': reps, 'events': timeline[:200]}
    json.dump(comp, open(os.path.join(data_dir, 'competition.json'), 'w'), separators=(',', ':'))
