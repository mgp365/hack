from mao_env import MaoEnv
from agents.rule_based import GreedyRuleAgent
from agents.dqn_agent import DQNAgent

# 1) Crear entorno y jugadores
env      = MaoEnv()
learner  = DQNAgent(env.state_size, 53)
teachers = [GreedyRuleAgent(), GreedyRuleAgent()]

EPISODES  = 1_000
MAX_TURNS = 1_000

wins = []                              # ← historial de victorias (1/0)

# 2) Bucle de partidas
for ep in range(EPISODES):
    state = env.reset()
    done  = False
    turns = 0

    while not done and turns < MAX_TURNS:
        turns += 1
        pid   = env.current_player
        legal = env.legal_actions(pid)

        action = learner.act(state, legal) if pid == 0 else teachers[pid-1].act(state, legal)
        next_state, reward, done, _ = env.step(action)

        if pid == 0:
            learner.mem.push(state, action, reward, next_state, done)
            learner.learn()

        state = next_state

    # guarda 1 si jugador-0 ganó, 0 en caso contrario
    wins.append(int(done and pid == 0 and reward == 10))

    if turns >= MAX_TURNS and not done:
        print(f"Episode {ep}: partida empatada (>{MAX_TURNS} turnos)")

    # progreso cada 100 partidas
    if ep % 100 == 0:
        win_rate = sum(wins[-100:]) / max(1, len(wins[-100:]))  # promedio ventana 100
        print(f"Episode {ep} terminado — ε = {learner.eps:.3f}  WinRate(últ.100) = {win_rate:.2f}")

print("Entrenamiento terminado 🎉")
