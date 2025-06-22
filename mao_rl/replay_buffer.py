
"""Little drawer to keep memories for replay.""" 
import random
from collections import deque, namedtuple

Transition = namedtuple("T", ("s", "a", "r", "s2", "d"))

class ReplayBuffer:
    def __init__(self, cap: int = 100_000):
        self.buf = deque(maxlen=cap)

    def push(self, *args):
        self.buf.append(Transition(*args))

    def sample(self, k: int):
        return random.sample(self.buf, k)

    def __len__(self):
        return len(self.buf)
