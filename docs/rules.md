# Fixed rules

The deck has 52 cards. Two through ten players receive five cards apiece; the next card establishes trump. Dealing starts left of dealer and moves in seat order. On the first pass, every non-dealer receives a packet of three and the dealer receives two. On the second pass, every non-dealer receives two and the dealer receives three.

The seat left of dealer leads, and each trick winner leads next. Each trick immediately awards five match points. The match ends on the trick that takes its winner to 25, so the final deal may contain fewer than five tricks. If nobody reaches 25, a complete deal contains five tricks and the dealer advances one seat.

## Rank and category

All tricks use this universal trump top: five of trumps, jack of trumps, Ace of Hearts. A♥ is always in the trump category. Ordinary trump court cards follow as ace, king, queen. Remaining red trump order is `10 9 8 7 6 4 3 2`; remaining black trump order is `2 3 4 6 7 8 9 10`.

Red non-trump led order is `K Q J 10 9 8 7 6 5 4 3 2 A`; black is `K Q J A 2 3 4 5 6 7 8 9 10`. Only trump-category cards and ordinary cards in the led category can win.

Because A♥ is categorically trump, leading it leads trump. A♥ is always legal to play, including when ordinary Hearts are led and Hearts are not trump, but it does not satisfy the obligation to follow Hearts.

## Trump reneging

The trump five may always be withheld. The trump jack may be withheld unless the trump five was led. A♥ may be withheld unless the trump five or trump jack was led. If any ordinary, non-protected trump is held after trump is led, the legal set is the held trump set; protected cards remain legal but need not be spent. If every held trump is protected, any card is legal.

## Mandatory rob

Before trick one, the player dealt the ace of the turned trump suit must rob. Dealer is not excluded. If the turned card is that ace, dealer robs. The turned card enters the eligible player’s hand, producing six cards, and the `ROB` action chooses a non-trump to exchange face down. The enabling ace and all other trumps are retained. The discard is known to the robber but hidden from every opponent. Selecting the discard is a strategic decision, not an automatic lowest-card rule.

A six-trump rob hand is possible in a legal deal. To keep the mandatory-rob engine total, that exceptional hand retains the enabling ace and must discard its weakest remaining trump. Play then begins with the seat left of dealer.

These constants and legality/ranking functions exist only in `rules.py`; reward shaping never changes them. The current contract is ruleset version 3.
