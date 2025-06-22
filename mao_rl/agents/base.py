
class AbstractAgent:
    def act(self, obs, legal_actions):
        raise NotImplementedError
    def learn(self):  # optional
        pass
