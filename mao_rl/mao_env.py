
"""MAO game playground: all rules live here."""

import random
import numpy as np
from collections import deque

SUITS = "♠♥♦♣"
RANKS = "A23456789TJQK"

CARD2IDX = {f"{r}{s}": i for i, (r, s) in enumerate((r, s) for r in RANKS for s in SUITS)}
IDX2CARD = {v: k for k, v in CARD2IDX.items()}

def rank(idx): return RANKS[idx // 4]
def suit(idx): return SUITS[idx % 4]

class MaoEnv:
    """Gym‑style environment for the MAO card game."""

    def __init__(self, n_players: int = 3, penalty_draw: int = 1):
        self.n_players = n_players
        self.penalty_draw = penalty_draw

        self.state_size = 110          # 52+52+2+3+1
        self.action_space = 53         # 52 cards + 1 'draw'
        self.reset()

    # -------------------------------------------------------
    def reset(self, seed: int | None = None):
        if seed is not None:
            random.seed(seed)

        deck = list(range(52))
        random.shuffle(deck)

        self.hands = [deque(deck[i*7:(i+1)*7]) for i in range(self.n_players)]
        self.draw_pile = deque(deck[7*self.n_players:-1])
        self.discard   = [deck[-1]]

        self.current_player = 0
        self.direction = 1
        self.skip_next = False
        return self._obs()

    # -------------------------------------------------------
    def _obs(self):
        pid = self.current_player
        hand_vec = np.zeros(52)
        hand_vec[list(self.hands[pid])] = 1

        top_vec = np.zeros(52)
        top_vec[self.discard[-1]] = 1

        counts = np.array([len(h)/20 for i,h in enumerate(self.hands) if i!=pid])
        turn_ohe = np.eye(self.n_players)[pid]
        dir_flag = np.array([int(self.direction == -1)])

        return np.concatenate([hand_vec, top_vec, counts, turn_ohe, dir_flag])

    # -------------------------------------------------------
    def legal_actions(self, pid: int):
        top = self.discard[-1]
        return [c for c in self.hands[pid] if rank(c)==rank(top) or suit(c)==suit(top)]

    # -------------------------------------------------------
    def step(self, action: int):
        reward = 0.0
        done   = False
        pid = self.current_player
        pid = self.current_player
        legal = self.legal_actions(pid)

        # 1)  El jugador QUIERE robar:
        if action == 52:
            self._draw_card(pid)  # robar solo una
            self._advance_turn()
            return self._obs(), 0.0, False, {}

        # 2)  Jugada no permitida:
        if action not in legal:
            self._penalize(pid)
            self._advance_turn()
            return self._obs(), -1.0, False, {}

        if action == 52 or action not in self.legal_actions(pid):
            self._penalize(pid)
            reward = -1
        else:
            self.hands[pid].remove(action)
            self.discard.append(action)
            reward = +1

            r = rank(action)
            if r == 'A':
                self.skip_next = True
            elif r == 'K':
                self.direction *= -1

            if len(self.hands[pid]) == 0:
                done = True
                reward = 10

        if not done:
            self._advance_turn()

        return self._obs(), reward, done, {}

    def _draw_card(self, pid):
        if not self.draw_pile:  # rehacer mazo si falta
            if len(self.discard) > 1:
                top = self.discard.pop()
                tmp = list(self.discard);
                random.shuffle(tmp)
                self.draw_pile = deque(tmp)
                self.discard = [top]
        if self.draw_pile:
            self.hands[pid].append(self.draw_pile.popleft())

    # -------------------------------------------------------
    def _penalize(self, pid: int, n: int | None = None):
        n = n or self.penalty_draw
        for _ in range(n):
            if self.draw_pile:
                self.hands[pid].append(self.draw_pile.popleft())

        if not self.draw_pile:
            if len(self.discard) > 1:
                top = self.discard.pop()  # deja la carta de arriba
                tmp = list(self.discard);
                random.shuffle(tmp)
                self.draw_pile = deque(tmp)
                self.discard = [top]

    # -------------------------------------------------------
    def _advance_turn(self):
        step = self.direction
        nxt = (self.current_player + step) % self.n_players
        if self.skip_next:
            nxt = (nxt + step) % self.n_players
            self.skip_next = False
        self.current_player = nxt
