# Importamos las bibliotecas necesarias
import random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from hack.mao_rl.replay_buffer import ReplayBuffer

from .base import AbstractAgent
# --- MODIFICACIÓN CLAVE: Importar la red cuántica ---
from .quantum_qnet import QuantumQNet


# Clase DQNAgent (la lógica interna no cambia, solo la red que usa)
class DQNAgent(AbstractAgent):
    """
    Esta clase es idéntica a la anterior, pero ahora utiliza QuantumQNet.
    """

    def __init__(self, state_size, action_size, device, lr=1e-3, gamma=0.99,
                 eps_start=1.0, eps_end=0.05, eps_decay=2e-6):
        """
        Inicializa los atributos del agente.
        """
        # A diferencia del DQN clásico, el agente cuántico se ejecutará en la CPU
        # porque los simuladores de Qiskit no usan GPU.
        self.device = torch.device("cpu")

        # --- MODIFICACIÓN CLAVE: Instanciar QuantumQNet en lugar de QNet ---
        # Los parámetros `q` y `target` ahora son redes cuánticas.
        self.q = QuantumQNet(state_size, action_size).to(self.device)
        self.target = QuantumQNet(state_size, action_size).to(self.device)
        # --- Fin de la modificación ---

        self.target.load_state_dict(self.q.state_dict())
        # Nota: La tasa de aprendizaje (lr) puede necesitar ser más alta para modelos cuánticos.
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)

        self.mem = ReplayBuffer()
        self.gamma = gamma
        self.eps, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay
        self.steps = 0

    # ... EL RESTO DE LA CLASE (act, learn) ES EXACTAMENTE IGUAL ...
    # ... NO NECESITAS CAMBIAR NADA MÁS EN ESTA CLASE ...
    def act(self, obs, legal_actions):
        """

        basada en una política epsilon-greedy:
        - Con probabilidad epsilon, se elige una acción aleatoria (exploración).
        - Con (1 - epsilon), se usa la red Q para elegir la mejor acción (explotación).

        Parámetros:
            - obs: Observación actual del estado (entrada a la red Q).
            - legal_actions: Lista de acciones válidas en el estado actual.

        Retorna:
            - Una acción seleccionada (entero).
        """
        self.steps += 1
        self.eps = max(self.eps_end, self.eps - self.eps_decay)
        if random.random() < self.eps:
            # Exploración: Elegir una acción aleatoria válida
            return random.choice(legal_actions) if legal_actions else 52

        # Explotación: Predecir valores Q con la red, considerando solo las acciones legales
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            qvals = self.q(obs_t)

        # Aplicamos un "máscara" para filtrar acciones no legales
        mask = torch.full_like(qvals, -float('inf'))
        if legal_actions:
            mask[legal_actions] = 0
        else:
            mask[52] = 0  # Solo permite robar si no hay acciones legales disponibles

        return int(torch.argmax(qvals + mask))

    def learn(self, batch_size=512):
        """

        - Aplica aprendizaje supervisado sobre la red Q en base a la ecuación bellman:
          Q(s, a) = r + γ * max_a' Q(s', a').

        Parámetros:
            - batch_size: Número de transiciones a muestrear del buffer.

        Implementación:
            - Si no se tienen suficientes datos en el buffer, no se realiza ningún aprendizaje.
            - Actualiza la red objetivo cada 1000 pasos para mayor estabilidad.
        """
        if len(self.mem) < batch_size:
            # MODIFICACIÓN 1: Devolvemos None explícitamente si no hay suficiente memoria para aprender.
            return None

        # Muestreamos un lote de transiciones (batch) del buffer
        transitions = self.mem.sample(batch_size)
        s = torch.tensor(np.stack([t.s for t in transitions]), dtype=torch.float32, device=self.device)
        a = torch.tensor([t.a for t in transitions], device=self.device).unsqueeze(1)
        r = torch.tensor([t.r for t in transitions], dtype=torch.float32, device=self.device)
        s2 = torch.tensor(np.stack([t.s2 for t in transitions]), dtype=torch.float32, device=self.device)
        d = torch.tensor([t.d for t in transitions], dtype=torch.float32, device=self.device)  # Terminal sí/no

        # Computamos las predicciones de Q y el objetivo (target)
        q_pred = self.q(s).gather(1, a).squeeze()
        with torch.no_grad():
            q_next = self.target(s2).max(1)[0]
            y = r + self.gamma * q_next * (1 - d)

        # Calculamos el error cuadrático medio como función de pérdida
        loss = F.mse_loss(q_pred, y)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        # Copia los parámetros a la red objetivo cada 1000 pasos
        if self.steps % 1000 == 0:
            self.target.load_state_dict(self.q.state_dict())

        # MODIFICACIÓN 2: Devolvemos el valor numérico de la pérdida.
        # .item() extrae el número de un tensor de PyTorch, que es lo que necesitamos.
        return loss.item()