
import random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from hack.mao_rl.replay_buffer import ReplayBuffer

from .base import AbstractAgent

class QNet(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, action_size)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

class DQNAgent(AbstractAgent):
    def __init__(self, state_size, action_size, lr=1e-4, gamma=0.99,
                 eps_start=1.0, eps_end=0.1, eps_decay=5e-5):
        self.q      = QNet(state_size, action_size)
        self.target = QNet(state_size, action_size)
        self.target.load_state_dict(self.q.state_dict())
        self.opt    = torch.optim.Adam(self.q.parameters(), lr=lr)

        self.mem = ReplayBuffer()
        self.gamma = gamma
        self.eps, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay
        self.steps = 0

    def act(self, obs, legal_actions):
        self.steps += 1
        self.eps = max(self.eps_end, self.eps - self.eps_decay)
        if random.random() < self.eps:
            return random.choice(legal_actions) if legal_actions else 52
        with torch.no_grad():
            qvals = self.q(torch.tensor(obs, dtype=torch.float32))
        mask = torch.full_like(qvals, -1e9)
        mask[legal_actions] = 0
        return int(torch.argmax(qvals + mask))

    def learn(self, batch=256):
        if len(self.mem) < batch:
            return
        transitions = self.mem.sample(batch)
        s  = torch.tensor(np.stack([t.s  for t in transitions]), dtype=torch.float32)
        a  = torch.tensor([t.a for t in transitions]).unsqueeze(1)
        r  = torch.tensor([t.r for t in transitions], dtype=torch.float32)
        s2 = torch.tensor(np.stack([t.s2 for t in transitions]), dtype=torch.float32)
        d  = torch.tensor([t.d for t in transitions], dtype=torch.float32)

        q_pred = self.q(s).gather(1, a).squeeze()
        with torch.no_grad():
            q_next = self.target(s2).max(1)[0]
            y = r + self.gamma * q_next * (1-d)
        loss = F.mse_loss(q_pred, y)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        if self.steps % 500 == 0:
            self.target.load_state_dict(self.q.state_dict())
