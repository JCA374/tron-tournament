# Goal: Tron Tournament of 5 ML models (Wed 11 Nov evening → Fri 13 Nov 18:00)

Build and run an overnight-to-Friday tournament in /home/jonas/code/games/tournament on this
machine (8 CPU cores, 7.6 GB RAM, no GPU, PyTorch CPU). Five different ML methods learn to play
Tron light cycles and compete against each other. The best get stopped so the others can catch
up, the worst get cut, and on Friday evening 3 finalists remain. Everything is visible on a live
website with a log, a competition page, replays and podcast episodes.

## 1. The game: Tron light cycles

- 30×30 grid with walls around the edge. Each bike moves 1 cell per tick, actions: left / straight / right.
  Every bike leaves a permanent trail. A bike hitting a wall or any trail is out; head-on collisions
  kill both. Last bike alive wins; a draw if they die on the same tick or after 900 ticks (timeout).
- Modes: 1v1 (used for ranking) and free-for-all with 3–5 bikes (for show matches and the final).
- Random but mirrored start positions, so neither side gets an advantage.
- Observation (the same for all models): the grid around the bike, rotated so the bike always faces "up"
  (channels: walls/trails, own head, opponent heads), plus a few scalars (free space reachable per
  action via flood fill, distance to the opponent). Models may use less but not more.
- The simulator must be fast: aim for ≥200k steps/s per core. If Python is too slow, use the
  `ml-speedup` skill (bake-off, port the hot loop, exactness test against the Python reference).
- Fixed baseline bots for measuring absolute strength: `random`, `avoid-walls`, `space-filler`
  (picks the move with the most reachable cells). They are never knocked out and are in every ranking round.
- Replays are saved as compact JSON (moves per tick + seed), played back in the browser.

## 2. The 5 contestants

| # | Name | Method | Starting setup |
|---|---|---|---|
| 1 | PPO | Policy gradient, self-play against a pool of old snapshots | Small CNN + MLP |
| 2 | DQN | Double DQN + dueling + replay buffer, self-play pool | Same size network as PPO |
| 3 | NEAT | neat-python, fitness = results vs. pool + baselines | Inputs = compact features (rays + flood-fill scalars) |
| 4 | CMA-ES | Evolution strategy over the weights of a fixed small MLP | Same features as NEAT |
| 5 | Mini-AlphaZero | MCTS (e.g. 50–100 simulations) + small policy/value net | Search depth adjusted to the time budget |

Each model runs in its own process with a fair share of the CPU (taskset or torch threads). When a
model is knocked out, its cores go to those still in the tournament. Checkpoints are saved at least every
15 minutes, and everything must be able to resume after a crash or restart (supervisor + resume).

## 3. Tuning the models: you're allowed and expected to do this

The goal is that every model plays as well as its method allows in the time available, so the result
says something about the methods and not about a bad first setup. So:

- Check every 1–2 hours: is each model still improving (rating, win rate vs. baselines, loss curves)?
- Allowed: hyperparameters, network size, learning rate, reward shaping, opponent pool, exploration,
  MCTS simulations, population size, observation features (within the shared observation), bug fixes.
- Not allowed: changing a model to a different method family, copying weights between models,
  or giving one model more CPU than its share.
- A model that is clearly broken (e.g. stuck below `avoid-walls` after 2 hours) may be restarted
  from scratch with a better setup, but only before the first knockout.
- Every change goes in the log with the reason, what was measured before, and what happened after.
  Make one change at a time per model, so you can tell whether it helped.

## 4. Competition rules

- **Ranking rounds every 30 min:** round-robin 1v1, 20 games per pair (10 per start side) between all active
  models + baselines, each model's newest checkpoint. Elo/Bradley-Terry rating, stored history.
- **Catch-up rule:** if the leader is >100 Elo ahead of #2, the leader's training is paused (it still plays
  every ranking round). If the gap drops below 50, it starts training again. Pause and resume are logged as events.
- **Knockout 1:** Thu 12 Nov 12:00, the model with the lowest rating (average of the last 3 rounds) is out.
- **Knockout 2:** Fri 13 Nov 06:00, the next one is out.
- **Final:** Fri 13 Nov 16:00–18:00. Training stops at 16:00. The 3 finalists play a large 1v1 round-robin
  (200 games per pair) and 50 free-for-all games with 3 bikes. Then the winner is announced,
  with the final standings and the best replays.
- Knocked-out models are frozen at their last checkpoint and stay visible on the website (also in the "hall of fame").

## 5. Website: use the `ml-project-site` skill

GitHub Pages site, rebuilt and pushed every 15 minutes, with the skill's standard tabs plus a
**Competition** tab:

- **Competition:** leaderboard (rating, change since last round, status: training / paused (leader) /
  knocked out / finalist), rating chart over time for all models + baselines, head-to-head
  matrix (win %), countdown to the next knockout and the final, and the timeline of events
  (lead changes, pauses, knockouts, model changes).
- **Log:** journal + run events (what was changed and why, results, problems).
- **Play:** watch the newest models play Tron: choose 2–5 models and a fixed seed, or the round's
  best replays. Replays do not loop.
- **The model / How it sees / Curriculum / Speed / Lessons:** one section per contestant: method,
  network, observation, training curve, steps/s.

## 6. Podcasts: use the `project-podcast` skill

Make a short episode (2–4 min) when something big happens: the start of the tournament, the first lead change,
when a leader is paused, each knockout, a big comeback, a surprising change that works (or fails),
and the final with the winner. At least one episode per day. Be honest: also report when something went wrong
or an earlier claim was wrong.

## 7. Order of work

1. Tron simulator + baselines + replay format, tests (rules, mirroring, speed).
2. Common interface for all models (`act(obs)`, `save/load`) + the ranking runner.
3. The 5 training processes, short sanity test (5 min each: do they beat `random`?).
4. Supervisor: starts and resumes processes, CPU distribution, catch-up rule, knockouts on time.
5. Website + 15-min rebuild (cron), first podcast episode "the tournament starts".
6. Start the long run and then follow up with `/loop`: check health, tune models, log, podcasts.

Success = the tournament runs without interruption until Fri 18:00, all 5 models clearly beat
`space-filler` before the first knockout (or the log explains why not), the website shows the
whole race, and there is a winner with replays and a final podcast.
