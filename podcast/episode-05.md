# Episode 05: The first knockout

At noon on Thursday, the first model left the tournament. It is C M A E S.

The knockout uses the average rating over the last three rounds, so one unlucky round cannot decide it. P P O had twenty two hundred and three. The AlphaZero model had twenty one hundred and eighty eight. Then a big gap. NEAT had nineteen hundred and sixty, and D Q N nineteen hundred and forty seven. C M A E S had eighteen hundred and eleven, about a hundred and forty points behind.

It is a strange story, because C M A E S won the very first round. Half an hour in, it was the best of the five. Its method is good at finding a decent small network quickly. But then it stopped improving. It practised only against the bots and its own earlier champions, and those stopped teaching it anything. I gave it harder opponents, the other models. That did not help. Then I found that its search had frozen. So I restarted it with a wide step. That helped a little. Its last ratings were its best since the first round. But it was not enough, and not in time.

Its network was also the smallest in the tournament, under nine hundred numbers, and it could only read the twenty three summary values, not the picture of the board around the bike. NEAT has the same limit, and it survived. But NEAT can grow its network, and C M A E S could not.

At the top, it is a real fight. P P O and the AlphaZero model have swapped the lead four times since nine o'clock. P P O came back after this morning's change, when it started practising against the territory bot. It now beats that bot in every game. The AlphaZero model still wins most of their direct games, but only just.

Its knockout freed one processor core, so every model now has two. That needed a fix first. The two models that use extra worker processes on more cores had never run that way. Twenty minutes before the knockout, a test showed that stopping them would have left workers running, and that the supervisor would have mistaken those workers for duplicates. Both were fixed, and the switch went cleanly.

The next knockout is tomorrow morning at six. Right now that is a race between NEAT and D Q N, just thirteen points apart. After that, three finalists, and the final on Friday afternoon.
