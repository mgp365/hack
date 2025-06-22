# --- Implementación de un modelo DQN para entrenar un agente en el entorno Mao ---
# Este código permite entrenar un agente DQN (Deep Q-Network) dentro de un entorno simulado.
# Documentamos cada instrucción para que sea adecuada para principiantes.

# --- Importación de bibliotecas necesarias ---
import time  # Biblioteca estándar de Python para medir tiempos y gestionar retrasos.
import torch  # Biblioteca avanzada para realizar cálculos matemáticos y aprendizaje profundo.
from collections import deque  # Estructura de datos tipo "cola" para manejar historial (como el buffer de memoria).
import numpy as np  # Biblioteca para trabajar con operaciones matemáticas y arreglos numéricos.

# Importamos módulos locales necesarios:
from mao_env import MaoEnv  # Define el entorno simulado donde el agente interactuará.
from agents.rule_based import GreedyRuleAgent  # Agente basado en reglas lógicas. Sirve como oponente del agente DQN.
from agents.dqn_agent import DQNAgent  # Modelo del agente inteligente basado en deep learning.

# --- 1. CONFIGURACIÓN INICIAL Y PARÁMETROS ---
# Detectar el hardware disponible. Si hay una GPU (procesador gráfico), se usará para cálculos más rápidos.
# Caso contrario, se usará el procesador (CPU).
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")  # Mostrar en pantalla si se usa GPU o CPU.

# Definición de hiperparámetros:
# Los hiperparámetros son valores que configuran el comportamiento del modelo y el entrenamiento.
EPISODES = 20_000  # Número total de episodios (partidas) que se entrenarán.
MAX_TURNS = 2000  # Número máximo de turnos permitidos dentro de un episodio.
LEARNING_FREQUENCY = 4  # Frecuencia de aprendizaje: el agente aprende cada 4 pasos, no en cada paso.
BATCH_SIZE = 512  # Número de muestras que el modelo procesará en cada paso de aprendizaje.
REPLAY_BUFFER_START_SIZE = 10000  # Tamaño mínimo del buffer de memoria antes de iniciar el aprendizaje.
PRINT_EVERY_EPISODES = 50  # Imprimir resultados cada 50 episodios.

# --- 2. CONFIGURACIÓN DEL ENTORNO Y LOS AGENTES ---
# Crear una instancia del entorno `MaoEnv`. Aquí es donde el agente realizará las acciones y recibirá recompensas.
env = MaoEnv()

# Crear al agente DQN (Deep Q-Network).
# Este agente será el que aprenderá las mejores estrategias en base al estado actual y las recompensas recibidas.
learner = DQNAgent(env.state_size, env.action_space, device=device)

# Crear "agentes maestros". Estos son agentes basados en reglas predefinidas ("Greedy"), que actuarán de competidores del DQN.
#teachers = [GreedyRuleAgent(), GreedyRuleAgent()]  # Se configuran dos agentes basados en reglas.

# --- MODIFICACIÓN 1: Añadir una lista para guardar el historial de la pérdida ---
loss_history = []
# (El resto de las variables de seguimiento se mantienen igual)
illegal_moves_this_episode = 0  # El número de movimientos ilegales hechos por el agente en el episodio actual.
total_steps = 0  # Número total de pasos realizados durante
start_time = time.time()  # Registrar el momento de inicio del entrenamiento.

# --- 3. CICLO PRINCIPAL DE ENTRENAMIENTO ---
# Este es el ciclo principal donde ocurre el entrenamiento. Aquí el agente jugará y aprenderá de cada episodio.
for ep in range(EPISODES):  # Iterar sobre el número total de episodios.
    # Reiniciar el entorno al inicio de cada episodio. Regresa el estado inicial.
    s = env.reset()
    done = False  # Indica si el episodio ha terminado.
    turns = 0  # Contador de turnos dentro del episodio actual.
    illegal_moves_this_episode = 0  # Reiniciar contador de movimientos ilegales.

    # Bucle interno. Este bucle controla cada turno dentro del episodio.
    while not done and turns < MAX_TURNS:
        turns += 1  # Incrementar el número de turnos realizados durante el episodio.
        total_steps += 1  # Incrementar el contador global de pasos.
        pid = env.current_player  # Identificar al jugador que tiene el turno actual.

        # Consultar qué acciones son legales para el jugador actual.
        legal = env.legal_actions(pid)

        # Elegir la acción:
        # - Si el jugador actual es el agente DQN (pid == 0), tomará la acción basada en su red neuronal.
        # - Si el jugador es un "teacher" (oponente), tomará la decisión basada en reglas predefinidas.
        a = learner.act(s, legal) if pid == 0 else teachers[pid - 1].act(s, legal)

        # Ejecutar la acción seleccionada en el entorno.
        s2, r, done, _ = env.step(a)

        # Si la acción fue realizada por el agente DQN (pid == 0):
        if pid == 0:
            # Verificar si la acción fue ilegal. Si la recompensa es -3.0, el movimiento fue ilegal.
            if r == -3.0:
                illegal_moves_this_episode += 1

            # Guardar la experiencia en la memoria del agente.
            learner.mem.push(s, a, r, s2, done)

            # --- APRENDIZAJE CON FRECUENCIA REDUCIDA ---
            if total_steps % LEARNING_FREQUENCY == 0 and len(learner.mem) > REPLAY_BUFFER_START_SIZE:
                # --- MODIFICACIÓN 2: Capturar el valor de pérdida devuelto por la función learn ---
                loss_value = learner.learn(batch_size=BATCH_SIZE)
                # Si el aprendizaje ocurrió (no devolvió None), guardamos el valor de la pérdida.
                if loss_value is not None:
                    loss_history.append(loss_value)

        s = s2

    # --- 4. IMPRESIÓN PERIÓDICA DE RESULTADOS ---
    if (ep + 1) % PRINT_EVERY_EPISODES == 0:
        end_time = time.time()
        elapsed_time = end_time - start_time
        eps_per_sec = (ep + 1) / elapsed_time

        # --- MODIFICACIÓN 3: Calcular la pérdida promedio y añadirla a la impresión ---
        # Calculamos el promedio de las últimas 1000 mediciones de pérdida para tener un valor estable.
        avg_loss = np.mean(loss_history[-1000:]) if loss_history else 0

        print(
            f"Episodio {ep + 1:6d} | "
            f"ε={learner.eps:.3f} | "
            f"Ilegales: {illegal_moves_this_episode:2d} | "
            f"Buffer: {len(learner.mem):6d} | "
            f"Loss (MSE): {avg_loss:.4f} | "  # <-- Nueva métrica añadida aquí
            f"Eps/seg: {eps_per_sec:.2f}"
        )

print(f"\nEntrenamiento finalizado en {(time.time() - start_time) / 60:.2f} minutos.")