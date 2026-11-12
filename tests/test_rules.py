import unittest

from twenty_five import Card, Rank, Suit
from twenty_five import rules
from twenty_five.valuation import discard_value


class RuleTests(unittest.TestCase):
    def test_universal_top_three(self):
        for trump in Suit:
            ordered=[Card(trump,Rank.FIVE),Card(trump,Rank.JACK),Card(Suit.HEARTS,Rank.ACE)]
            self.assertEqual([rules.trump_strength(c,trump) for c in ordered],[100,99,98])

    def test_red_and_black_trump_order(self):
        red=[Rank.TEN,Rank.NINE,Rank.EIGHT,Rank.SEVEN,Rank.SIX,Rank.FOUR,Rank.THREE,Rank.TWO]
        black=[Rank.TWO,Rank.THREE,Rank.FOUR,Rank.SIX,Rank.SEVEN,Rank.EIGHT,Rank.NINE,Rank.TEN]
        for suit in Suit:
            order=red if suit.is_red else black
            values=[rules.trump_strength(Card(suit,r),suit) for r in order]
            self.assertEqual(values,sorted(values,reverse=True))

    def test_red_and_black_plain_order(self):
        red=[
            Rank.KING,Rank.QUEEN,Rank.JACK,Rank.TEN,Rank.NINE,Rank.EIGHT,
            Rank.SEVEN,Rank.SIX,Rank.FIVE,Rank.FOUR,Rank.THREE,Rank.TWO,Rank.ACE,
        ]
        black=[
            Rank.KING,Rank.QUEEN,Rank.JACK,Rank.ACE,Rank.TWO,Rank.THREE,
            Rank.FOUR,Rank.FIVE,Rank.SIX,Rank.SEVEN,Rank.EIGHT,Rank.NINE,Rank.TEN,
        ]
        for suit in Suit:
            order=red if suit.is_red else black
            values=[rules.plain_strength(Card(suit,r)) for r in order]
            self.assertEqual(values,sorted(values,reverse=True))

    def test_cards_have_no_misleading_natural_gameplay_order(self):
        with self.assertRaises(TypeError):
            Card(Suit.HEARTS,Rank.FIVE)<Card(Suit.HEARTS,Rank.JACK)

    def test_ace_hearts_is_always_legal_but_does_not_satisfy_following_hearts(self):
        hand=[Card(Suit.HEARTS,Rank.ACE),Card(Suit.CLUBS,Rank.TWO)]
        legal=rules.legal_cards(hand,Suit.SPADES,[(0,Card(Suit.HEARTS,Rank.KING))])
        self.assertEqual(set(legal),set(hand))
        hand.append(Card(Suit.HEARTS,Rank.TWO))
        self.assertEqual(
            set(rules.legal_cards(hand,Suit.SPADES,[(0,Card(Suit.HEARTS,Rank.KING))])),
            {Card(Suit.HEARTS,Rank.ACE),Card(Suit.HEARTS,Rank.TWO)},
        )

    def test_ace_hearts_led_is_trump(self):
        hand=[Card(Suit.SPADES,Rank.TWO),Card(Suit.HEARTS,Rank.TWO)]
        self.assertEqual(rules.legal_cards(hand,Suit.SPADES,[(0,Card(Suit.HEARTS,Rank.ACE))]),[Card(Suit.SPADES,Rank.TWO)])

    def test_protected_trump_reneging(self):
        trump=Suit.CLUBS; off=Card(Suit.DIAMONDS,Rank.TWO); led=Card(trump,Rank.KING)
        for protected in (Card(trump,Rank.FIVE),Card(trump,Rank.JACK),Card(Suit.HEARTS,Rank.ACE)):
            hand=[protected,off]; self.assertEqual(set(rules.legal_cards(hand,trump,[(0,led)])),set(hand))
        hand=[Card(trump,Rank.JACK),Card(trump,Rank.TWO),off]
        self.assertEqual(set(rules.legal_cards(hand,trump,[(0,led)])),{Card(trump,Rank.JACK),Card(trump,Rank.TWO)})

    def test_five_led_removes_jack_and_ace_protection(self):
        trump=Suit.SPADES; hand=[Card(trump,Rank.JACK),Card(Suit.HEARTS,Rank.ACE),Card(Suit.DIAMONDS,Rank.TWO)]
        self.assertEqual(set(rules.legal_cards(hand,trump,[(0,Card(trump,Rank.FIVE))])),set(hand[:2]))

    def test_jack_led_removes_ace_hearts_protection(self):
        trump=Suit.SPADES; ace=Card(Suit.HEARTS,Rank.ACE); off=Card(Suit.DIAMONDS,Rank.TWO)
        self.assertEqual(rules.legal_cards([ace,off],trump,[(0,Card(trump,Rank.JACK))]),[ace])

    def test_off_suit_cannot_win(self):
        trick=[(0,Card(Suit.DIAMONDS,Rank.TWO)),(1,Card(Suit.CLUBS,Rank.KING)),(2,Card(Suit.DIAMONDS,Rank.ACE))]
        self.assertEqual(rules.trick_winner(trick,Suit.SPADES),0)

    def test_discard_heuristic_protects_specials_and_values_voids(self):
        trump=Suit.SPADES; singleton=Card(Suit.DIAMONDS,Rank.TWO); pair_low=Card(Suit.CLUBS,Rank.TEN)
        hand=[Card(trump,Rank.FIVE),Card(trump,Rank.JACK),Card(Suit.HEARTS,Rank.ACE),singleton,pair_low,Card(Suit.CLUBS,Rank.KING)]
        self.assertLess(discard_value(singleton,hand,trump),discard_value(pair_low,hand,trump))
        self.assertGreater(discard_value(Card(trump,Rank.FIVE),hand,trump),discard_value(singleton,hand,trump))

    def test_rob_discards_are_non_trumps(self):
        trump=Suit.SPADES
        hand=[
            Card(trump,Rank.ACE),Card(trump,Rank.FIVE),Card(Suit.HEARTS,Rank.ACE),
            Card(Suit.CLUBS,Rank.TWO),Card(Suit.DIAMONDS,Rank.THREE),
        ]
        self.assertEqual(set(rules.legal_rob_discards(hand,trump)),{Card(Suit.CLUBS,Rank.TWO),Card(Suit.DIAMONDS,Rank.THREE)})

    def test_all_trump_rob_has_a_non_ace_fallback(self):
        trump=Suit.SPADES
        hand=[Card(trump,rank) for rank in (Rank.ACE,Rank.TWO,Rank.THREE,Rank.FOUR,Rank.SIX,Rank.SEVEN)]
        self.assertEqual(rules.legal_rob_discards(hand,trump),[Card(trump,Rank.SEVEN)])


if __name__=="__main__": unittest.main()
