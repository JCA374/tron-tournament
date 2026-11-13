# Final models

The last checkpoint of each contestant, as it played in the tournament.

| File | Model | Load with |
|---|---|---|
| `az.pt` | Mini-AlphaZero (network frozen since Fri 00:22 by the catch-up rule) | `models.az.load_agent("final_models/az.pt")` |
| `ppo.pt` | PPO | `models.ppo.load_agent("final_models/ppo.pt")` |
| `dqn.play.pt` | DQN (retired Thu 21:58) | `models.dqn.load_agent("final_models/dqn.play.pt")` |
| `neat.pkl` | NEAT (retired Thu 21:58) | `models.neat.load_agent("final_models/neat.pkl")` |
| `cmaes.pkl` | CMA-ES (knocked out Thu 12:00) | `models.cmaes.load_agent("final_models/cmaes.pkl")` |

Every agent has `act(state, bike)`; play a game with `tournament.match.play_game`.
