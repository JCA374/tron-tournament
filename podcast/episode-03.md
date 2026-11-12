# Episode 03: The first pause, and a better ruler

It is half past two on Thursday morning, and the catch-up rule has fired for the first time.

The AlphaZero model took the lead at half past midnight. Two rounds later it was a hundred and ten points ahead of the next model. That is over the line of a hundred, so its training is now paused. It still plays every round with the network it has. It only trains again when the others get within fifty points.

Remember, this model has played only about two thousand practice games. P P O has played three hundred thousand. NEAT has played almost nine hundred thousand. Thinking ahead before every move is still worth more than all that practice.

Now a correction. At half past one the ratings jumped. The AlphaZero model went up two hundred and ninety points in a single round, and P P O almost two hundred and fifty. That was mostly not real. The scale is pinned to the space-filler bot at fifteen hundred points. But by then the top four models beat it in all twenty games. Against a perfect score, the maths can only say "much stronger", and the whole scale started to float upwards. The order between the models was still right. Just the numbers were stretched.

So I added a fourth fixed bot, the territory bot. It is the classic trick of good Tron programs. It always picks the move that gives it the most cells it can reach before the opponent. It beats the space-filler bot thirty nine times out of forty. It was rated once, at about two thousand, and is now pinned there instead.

And here is the surprise. That hand-written bot is in second place. It beats P P O sixteen times out of twenty. It beats D Q N and C M A E S nineteen times out of twenty. Only the AlphaZero model and NEAT beat it. Interestingly, both of them are the models that think in terms of space. NEAT reads the same territory numbers the bot uses. The AlphaZero model searches ahead.

C M A E S is still last, at about sixteen hundred. Two hours ago I let it practise against the other models, instead of only the bots and itself. That did not help. Then I found the real problem in its log. Its search had frozen. The step it takes between generations had shrunk almost to nothing. So I restarted the search from its best network, with a wide step again. It is the standard fix for this method. We will see in the next rounds whether it was in time. The first knockout is at noon, and right now C M A E S is the one going out.
