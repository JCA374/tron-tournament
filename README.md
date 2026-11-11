# Tron Tournament

**🌐 Live site: https://jca374.github.io/tron-tournament/**

Five machine-learning methods (PPO, DQN, NEAT, CMA-ES and a small AlphaZero) teach themselves Tron light cycles on one
8-core laptop and compete from Wednesday 11 November evening until Friday 13 November, 18:00.

- Ranking round every 30 minutes: every model plays every other model and three fixed bots (20 games per pair).
- Catch-up rule: a leader more than 100 rating points ahead stops training until the gap is under 50.
- Knockouts: Thursday 12:00 and Friday 06:00. The final with three models: Friday 16:00-18:00.

The full brief is in [GOAL.md](GOAL.md).

| Folder | What |
|---|---|
| `tron/` | the game (numba), observations, fixed bots, replay rendering |
| `models/` | the five trainers |
| `tournament/` | ranking rounds, ratings, the supervisor (catch-up rule, knockouts, final) |
| `docs/` | the website (GitHub Pages), rebuilt every 15 minutes by `publish.py` |
