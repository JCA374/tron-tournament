# Tron Tournament

**🌐 Live site: https://jca374.github.io/tron-tournament/**

Five machine-learning methods (PPO, DQN, NEAT, CMA-ES and a small AlphaZero) taught themselves Tron light cycles on one
8-core laptop, from Wednesday 11 November 22:05 to Friday 13 November 13:20 (ended early).

## Result

**🏆 Mini-AlphaZero won the final against PPO: 168 games to 30, 2 draws (200 games, 100 start positions from both sides).**

| Place | Model | How it went |
|---|---|---|
| 1 | Mini-AlphaZero | Tree search (128 simulations per move) + a small network; network frozen since Fri 00:22 by the catch-up rule |
| 2 | PPO | Policy gradient with self-play and a territory bonus; beat every fixed bot, lost the duel |
| 3-4 | DQN, NEAT | Retired Thu 21:58 when the tournament became a duel |
| 5 | CMA-ES | Knocked out Thu 12:00 |

Watch every game of the final in the [Arena](https://jca374.github.io/tron-tournament/play.html?final=1). The final models are in
[`final_models/`](final_models/).

- Ranking round every 30 minutes: every model plays every other model and three fixed bots (20 games per pair).
- Catch-up rule: a leader more than 100 rating points ahead stops training until the gap is under 50.
- Knockout: Thursday 12:00. From Thursday 22:00 a duel of PPO and Mini-AlphaZero; the final was played Friday at 13:03.

The full brief is in [GOAL.md](GOAL.md).

| Folder | What |
|---|---|
| `tron/` | the game (numba), observations, fixed bots, replay rendering |
| `models/` | the five trainers |
| `tournament/` | ranking rounds, ratings, the supervisor (catch-up rule, knockouts, final) |
| `docs/` | the website (GitHub Pages), built by `publish.py` |
| `final_models/` | the last checkpoint of every model |
