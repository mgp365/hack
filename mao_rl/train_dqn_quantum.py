import time
import torch

import numpy as np

# Importamos módulos locales necesarios:
from mao_env import MaoEnv
from agents.rule_based import GreedyRuleAgent
# --- MODIFICACIÓN CLAVE: Importar el agente DQN cuántico ---
from agents.dqn_agent_quantum import DQNAgent

# --- 1. CONFIGURACIÓN INICIAL Y PARÁMETROS ---
# Los simuladores cuánticos de Qiskit corren en CPU. Forzamos el uso de CPU.
device = torch.device("cpu")
print(f"Usando dispositivo: {device} (Requerido para el Agente Cuántico)")

# ... El resto de los hiperparámetros se mantiene igual ...
# Puedes ajustar la tasa de aprendizaje si es necesario
EPISODES                = 40     # ↓ de 100
MAX_TURNS               = 150    # ↓ de 200
BATCH_SIZE              = 16     # ↓ de 32
REPLAY_BUFFER_START_SIZE= 50     # ↓ de 100
LEARNING_FREQUENCY      = 2      # aprende el doble de veces
PRINT_EVERY_EPISODES    = 5
torch.set_num_threads(1)
# Frecuencia para imprimir resultados.

# --- 2. CONFIGURACIÓN DEL ENTORNO Y LOS AGENTES ---
env = MaoEnv()

# Aquí se instancia el agente cuántico. ¡El resto del código no lo nota!
learner = DQNAgent(env.state_size, env.action_space, device=device, lr=1e-1)

teachers = [GreedyRuleAgent(), GreedyRuleAgent()]

# ... EL RESTO DEL SCRIPT DE ENTRENAMIENTO ES EXACTAMENTE IGUAL ...
# El bucle de entrenamiento, la lógica de impresión,
# gracias a la abstracción de la clase DQNAgent y TorchConnector.
loss_history = []
illegal_moves_this_episode = 0
total_steps = 0
start_time = time.time()

# --- 3. CICLO PRINCIPAL DE ENTRENAMIENTO ---
# Este es el ciclo principal donde ocurre el entrenamiento. Aquí el agente jugará y aprenderá de cada episodio.
for ep in range(EPISODES):
    # Reiniciar el entorno al inicio de cada episodio. Regresa el estado inicial.
    s = env.reset()
    done = False
    turns = 0
    illegal_moves_this_episode = 0

    # Bucle interno. Este bucle controla cada turno dentro del episodio.
    while not done and turns < MAX_TURNS:
        turns += 1
        total_steps += 1
        pid = env.current_player

        # Consultar qué acciones son legales para el jugador actual.
        legal = env.legal_actions(pid)

        # Elegir la acción:
        a = learner.act(s, legal) if pid == 0 else teachers[pid - 1].act(s, legal)

        # Ejecutar la acción seleccionada en el entorno.
        s2, r, done, _ = env.step(a)

        # Si la acción fue realizada por el agente DQN (pid == 0):
        if pid == 0:
            if r == -3.0:
                illegal_moves_this_episode += 1
            learner.mem.push(s, a, r, s2, done)
            if total_steps % LEARNING_FREQUENCY == 0 and len(learner.mem) > REPLAY_BUFFER_START_SIZE:
                loss_value = learner.learn(batch_size=BATCH_SIZE)
                if loss_value is not None:
                    loss_history.append(loss_value)
        s = s2

    # --- 4. IMPRESIÓN PERIÓDICA DE RESULTADOS ---
    if (ep + 1) % PRINT_EVERY_EPISODES == 0:
        end_time = time.time()
        elapsed_time = end_time - start_time
        eps_per_sec = (ep + 1) / elapsed_time
        avg_loss = np.mean(loss_history[-1000:]) if loss_history else 0
        print(
            f"Episodio {ep + 1:6d} | "
            f"ε={learner.eps:.3f} | "
            f"Ilegales: {illegal_moves_this_episode:2d} | "
            f"Buffer: {len(learner.mem):6d} | "
            f"Loss (MSE): {avg_loss:.4f} | "
            f"Eps/seg: {eps_per_sec:.2f}"
        )

print(f"\nEntrenamiento finalizado en {(time.time() - start_time) / 60:.2f} minutos.")

MODEL_PATH = "quantum_dqn_mao_final.pth"
print(f"Guardando los pesos del modelo en: {MODEL_PATH}")
torch.save(learner.q.state_dict(), MODEL_PATH)
print("Modelo guardado exitosamente.")