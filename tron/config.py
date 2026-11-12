"""Tournament settings shared by the trainers, the supervisor and the website."""
CONTESTANTS = [
    {"id": "ppo", "label": "PPO", "color": "#4c6ef5", "method": "policy gradient, self-play vs old snapshots"},
    {"id": "dqn", "label": "DQN", "color": "#f76707", "method": "double + dueling deep Q-learning, self-play"},
    {"id": "neat", "label": "NEAT", "color": "#37b24d", "method": "neuroevolution that grows the network"},
    {"id": "cmaes", "label": "CMA-ES", "color": "#ae3ec9", "method": "evolution strategy on a fixed small network"},
    {"id": "az", "label": "Mini-AlphaZero", "color": "#f03e3e", "method": "tree search guided by a small network"},
]
BOTS = [
    {"id": "territory", "label": "Territory bot", "color": "#495057", "method": "claims the most space before the opponent (added Thu 01:45)"},
    {"id": "space-filler", "label": "Space-filler bot", "color": "#868e96", "method": "picks the move with the most free cells"},
    {"id": "avoid-walls", "label": "Avoid-walls bot", "color": "#adb5bd", "method": "random move that doesn't crash now"},
    {"id": "random", "label": "Random bot", "color": "#ced4da", "method": "random moves"},
]
BY_ID = {m["id"]: m for m in CONTESTANTS + BOTS}
ANCHOR = ("space-filler", 1500.0)   # until results/anchor.json exists: then the territory bot is pinned instead

SCHEDULE = [
    {"t": "2026-11-12T12:00:00+01:00", "kind": "knockout", "label": "knockout 1 (Thu 12:00)"},
    {"t": "2026-11-13T06:00:00+01:00", "kind": "knockout", "label": "knockout 2 (Fri 06:00)"},
    {"t": "2026-11-13T16:00:00+01:00", "kind": "final", "label": "the final starts (Fri 16:00)"},
    {"t": "2026-11-13T18:00:00+01:00", "kind": "end", "label": "the winner is announced (Fri 18:00)"},
]
ROUND_EVERY_MIN = 30
GAMES_PER_PAIR = 20
PAUSE_GAP, RESUME_GAP = 100.0, 50.0
TRAIN_CORES = 7          # cores 0-6 for training, core 7 for ranking rounds, the website and the supervisor
