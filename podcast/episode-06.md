# Episode 06: The duel, and the arena

It is ten o'clock on Thursday evening, and the tournament has changed shape. It is now a duel. P P O against the AlphaZero model.

You asked a good question tonight. Are they still learning, and why do they never trap each other on purpose? So I measured it in the replays. Almost every game is decided the same way. At some point the two bikes are walled off from each other, and the one with the bigger area wins, ninety four times out of a hundred. The AlphaZero model wins that split three times out of four. P P O wins it about two times out of three. So they do fight for space. But they do it late, around tick one hundred and sixty, when the board is already a third full. Nobody makes a bold early cut.

There are three likely reasons. The AlphaZero model only looked about two or three moves ahead. The learners were only paid at the very end of a game. And they see a window around the bike, not the whole board.

So, at your request, three changes. First, NEAT and D Q N stopped training. They stay in the arena as fixed opponents. The knockout tomorrow morning is cancelled, and the final on Friday afternoon is P P O against the AlphaZero model. Each of them now has four processor cores.

Second, the AlphaZero model thinks deeper. One hundred and twenty eight simulations per move, four times more than before in matches. And a third of its practice games are now against P P O.

Third, P P O gets a territory bonus. Every move, it is paid for the space it gains on its opponent. An early cut now pays off right away, not only at the end of the game.

One surprise already. P P O also practises against the AlphaZero model's network, but without the search. It beats that bare network ninety five times out of a hundred. Yet the full AlphaZero model, with its deeper search, beat P P O ten games to four in the first test series. Almost all of its strength is in the thinking ahead, not in the network.

And there is something new to watch. The Play tab is now the Arena. You pick two fighters, any of the models or bots, and press Fight. Your browser replays real games between them, with glowing trails and sparks when a bike crashes. Turn on territory, and every cell is tinted by the bike that can reach it first. You can see the frontier move, and a message pops up the moment one bike walls the other off. The two models in the duel are refreshed every two hours.

The final starts tomorrow at four in the afternoon. Two hundred games, P P O against the AlphaZero model.
