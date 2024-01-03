import numpy as np
import random

#TODO can change everything so that the observation state is stored and updated as the game state changes rather than recalculating on every to obs call

'''
OBS structure:
1. On board
2. In hand
3. In play
4. In deck
5. In discard
6. In opp deck
7. In opp Play
8. In Void
'''

'''
Action Structure:
1. Buy / Kill
2. Play
3. Banish
4. Activate Construct
5. End Turn

'''

class Card:
    def __init__(self, type, indices, cost = 0, power_gain = 0, rune_gain=0, honor_gain=0, card_draw=0, dis_hand_banish=0, hand_banish=0, worth = 0):
        self.type = type
        self.power_gain = power_gain
        self.rune_gain = rune_gain
        self.honor_gain = honor_gain
        self.card_draw = card_draw
        self.dis_hand_banish = dis_hand_banish
        self.hand_banish = hand_banish
        self.indices = indices
        self.usage_count = 0
        self.cost = cost
        self.has_activated = False

ALL_CARDS = {
    "Apprentice": Card("hero",[x for x in range(0,16)], rune_gain=1),
    "Militia": Card("hero",[x for x in range(16,20)], power_gain=1),
    "Heavy Infantry": Card("hero",[x for x in range(20,30)], cost=2, power_gain=2, worth=1),
    "Mystic": Card("hero",[x for x in range(30,40)], cost=3, rune_gain=2, worth = 1),
    "Cultist": Card("monster",[x for x in range(40,55)], cost=2, worth=1),
    "Mechana Initiate": Card("hero",[x for x in range(40,43)], cost=1, power_gain=1, rune_gain=1, worth=1)
    
}
MAIN_DECK = ["Mech Hero"] * 3

INVERSE_CARD = {
}
for key in ALL_CARDS.keys():
    #add names to card type
    ALL_CARDS[key].name = key
    #create inverse card
    for x in ALL_CARDS[key].indices:
        INVERSE_CARD[x] = key
class Acsension:
    STATE_LEN = 287 #1607 #modify this value, monsters won't be in discard, deck, or hand. Everything else is 1. in our hand 2. in our deck 3. in our discard 4. in our opp deck 5. in 6 cards 6. not seen 7. gone.
    def __init__(self):
        self.reset()
        self.observation_shape = self.to_observation().shape
        self.action_space = 801 #Every monsters you can kill, or banish, in general you can: 1. play card 2. buy a card 3. discard a card 4. banish a card from your deck 5. banish a card from the 6 cards, there's also an action to end turn
  
    def reset(self):
        self.deck = MAIN_DECK.copy()
        random.shuffle(self.deck)
        self.board = self.deck[-6:]
        self.deck = self.deck[:-6]
        self.history = []
        self.first_player_deck = ["Apprentice"] * 8 + ["Militia"] * 2
        self.second_player_deck = ["Apprentice"] * 8 + ["Militia"] * 2
        random.shuffle(self.first_player_deck)
        random.shuffle(self.second_player_deck)
        self.first_player_hand = self.first_player_deck[:-5]
        self.first_player_deck = self.first_player_deck[-5:]
        self.second_player_hand = self.second_player_deck[:-5]
        self.second_player_deck = self.second_player_deck[-5:]
        self.first_player_disc = []
        self.second_player_disc = []
        self.first_player_InPlay = []
        self.second_player_InPlay = []
        self.void = []

        self.always_available = ["Heavy Infantry", "Mystic", "Cultist"]

        self.first_honor = 0
        self.second_honor = 0
        
        self.honor_left = 60

        self.curr_rune = 0
        self.curr_power = 0

        self.current_player = 1
    
    def __str__(self):
        o = []
        if self.current_player == 1:
            o.append("Current Player: 1")
        else:
            o.append("Current Player: 2")
        o.append("\nFirst Player Hand:")
        o.append(', '.join(self.first_player_hand))
        o.append("\nSecond Player Hand:")
        o.append(', '.join(self.second_player_hand))
        o.append("\nBoard")
        o.append(', '.join(self.board))
        return '\n'.join(o)
        
    def to_observation(self):
        #encode the entire board state
        observation = np.zeros(self.STATE_LEN)
        for key in ALL_CARDS:
            ALL_CARDS[key].usage_count = 0
        for x in self.board:
            card = ALL_CARDS[x]
            i = (card.indices[card.usage_count])*8 #TODO change 7 as this is number of states a card can be in with monster change
            observation[i] = 1
            card.usage_count += 1
        if self.current_player == 1:
            for x in self.first_player_hand:
                card = ALL_CARDS[x]
                i = (card.indices[card.usage_count])*8 #TODO change 7 as this is number of states a card can be in with monster change
                observation[i+1] = 1
                card.usage_count += 1
            for x in self.second_player_InPlay:
                card = ALL_CARDS[x]
                i = (card.indices[card.usage_count])*8 #TODO change 7 as this is number of states a card can be in with monster change
                observation[i+2] = 1
                card.usage_count += 1
            for x in self.first_player_deck:
                card = ALL_CARDS[x]
                i = (card.indices[card.usage_count])*8 #TODO change 7 as this is number of states a card can be in with monster change
                observation[i+3] = 1
                card.usage_count += 1
            for x in self.first_player_disc:
                card = ALL_CARDS[x]
                i = (card.indices[card.usage_count])*8 #TODO change 7 as this is number of states a card can be in with monster change
                observation[i+4] = 1
                card.usage_count += 1
            for x in self.second_player_deck + self.second_player_disc + self.second_player_hand:
                card = ALL_CARDS[x]
                i = (card.indices[card.usage_count])*8 #TODO change 7 as this is number of states a card can be in with monster change
                observation[i+5] = 1
                card.usage_count += 1
            for x in self.second_player_InPlay:
                card = ALL_CARDS[x]
                i = (card.indices[card.usage_count])*8 #TODO change 7 as this is number of states a card can be in with monster change
                observation[i+6] = 1
                card.usage_count += 1
            observation[-6] = self.first_honor / 60
            observation[-5] = self.second_honor / 60
        else:
            for x in self.second_player_hand:
                card = ALL_CARDS[x]
                i = (card.indices[card.usage_count])*8 #TODO change 7 as this is number of states a card can be in with monster change
                observation[i+1] = 1
                card.usage_count += 1
            for x in self.second_player_InPlay:
                card = ALL_CARDS[x]
                i = (card.indices[card.usage_count])*8 #TODO change 7 as this is number of states a card can be in with monster change
                observation[i+2] = 1
                card.usage_count += 1
            for x in self.second_player_deck:
                card = ALL_CARDS[x]
                i = (card.indices[card.usage_count])*8 #TODO change 7 as this is number of states a card can be in with monster change
                observation[i+3] = 1
                card.usage_count += 1
            for x in self.second_player_disc:
                card = ALL_CARDS[x]
                i = (card.indices[card.usage_count])*8 #TODO change 7 as this is number of states a card can be in with monster change
                observation[i+4] = 1
                card.usage_count += 1
            for x in self.first_player_deck + self.first_player_disc + self.first_player_hand:
                card = ALL_CARDS[x]
                i = (card.indices[card.usage_count])*8 #TODO change 7 as this is number of states a card can be in with monster change
                observation[i+5] = 1
                card.usage_count += 1
            for x in self.first_player_InPlay:
                card = ALL_CARDS[x]
                i = (card.indices[card.usage_count])*8 #TODO change 7 as this is number of states a card can be in with monster change
                observation[i+6] = 1
                card.usage_count += 1
            observation[-5] = self.first_honor / 60
            observation[-6] = self.second_honor / 60
        for x in self.void:
            card = ALL_CARDS[x]
            i = (card.indices[card.usage_count])*8 #TODO change 7 as this is number of states a card can be in with monster change
            observation[i+7] = 1
            card.usage_count += 1
        observation[-4] = self.curr_rune
        observation[-3] = self.curr_power
        return observation
        
    def get_legal_actions(self):
        for key in ALL_CARDS:
            ALL_CARDS[key].usage_count = 0
        actions = []
        for x in self.board + self.always_available:
            card = ALL_CARDS[x]
            i = (card.indices[card.usage_count])*4 #TODO change 6 as this is number of actions available
            if card.type == "hero" or card.type == "construct":
                if self.curr_rune >= card.cost:
                    actions.append(i)
            else:
                if self.curr_power >= card.cost:
                    actions.append(i)
            card.usage_count += 1
        if (self.current_player == 1):
            for x in self.first_player_hand:
                card = ALL_CARDS[x]
                i = (card.indices[card.usage_count])*4 + 1
                actions.append(i)
        else:
            for x in self.second_player_hand:
                card = ALL_CARDS[x]
                i = (card.indices[card.usage_count])*4 + 1
                actions.append(i)

        #TBD stuff with banishing
        #TBD stuff with constructs
        #TBD maybe you have to do something before you end turn
        actions.append(-1) #this is the EOT action

    def play_card(self, card):
        if (card.type == "hero"):
            self.curr_rune += card.rune_gain
            self.curr_power += card.power_gain
            self.honor_left -= card.honor_gain
            if self.current_player == 1:
                self.first_honor += card.honor_gain
                self.first_player_hand.remove(card.name)
                self.first_player_InPlay.append(card.name)
            else:
                self.second_honor += card.honor_gain
        if (card.type == "construct"):
            card.has_activated = False
            if self.current_player == 1:
                self.first_player_hand.remove(card.name)
                self.first_player_InPlay.append(card.name)
            else:
                self.second_player_hand.remove(card.name)
                self.second_player_InPlay.append(card.name)

    def buy_card(self, card):
        if card.type == "hero" or card.type == "construct":
            self.curr_rune -= card.cost
            if card.name not in self.always_available:
                self.board.remove(card.name)
            if self.current_player == 1:
                self.first_player_disc.append(card.name)
                self.first_honor += card.worth
            else:
                self.second_player_disc.append(card.name)
                self.second_honor += card.worth
        if card.type == "monster":
            self.curr_power -= card.cost
            if card.name not in self.always_available:
                self.board.remove(card.name)
                self.void.append(card.name)
            if self.current_player == 1:
                self.first_honor += card.worth
            else:
                self.second_honor += card.worth

            


        
    
    def step(self, action):
        if action == -1:
            self.current_player *= -1
            return
        card = INVERSE_CARD[action // 4]
        card_act = action % 4
        if card_act == 0:
            self.play_card(card)

        
    def undo_last_action(self):
        #undoes the last action and updates the game state
        pass
    def get_result(self):
        #returns the result of the game (-1 for second player win, 0 for draw, 1 for first player win), and None if the game is not over
        if self.honor_left <= 0:
            if self.second_honor > self.first_honor:
                return -1
            else:
                return 1
    def get_first_person_result(self):
        result = self.get_result()
        if result is not None:
            return result * self.turn
    @staticmethod
    def swap_result(result):
        return -result
    

g = Acsension()
obs = g.to_observation()
print(np.array_str(obs,max_line_width=27))

print(g)