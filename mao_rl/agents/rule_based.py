
import random
from .base import AbstractAgent

class GreedyRuleAgent(AbstractAgent):
    def act(self, obs, legal_actions):
        return random.choice(legal_actions) if legal_actions else 52
