# Importamos las bibliotecas necesarias
import random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from hack.mao_rl.replay_buffer import ReplayBuffer

from .base import AbstractAgent


# Clase QNet - Red neuronal para estimar valores Q
class QNet(nn.Module):
    """
    Clase que define la arquitectura de una red neuronal profunda (DQN),
    utilizada para aproximar la función Q en Aprendizaje por Refuerzo (Reinforcement Learning).

    Atributos:
        - state_size: Dimensión del estado (entrada).
        - action_size: Número de acciones posibles (salida).
        - fc1 y fc2: Capas lineales con 256 neuronas cada una para procesar los datos.
        - out: Capa de salida que genera los valores Q para cada acción.
    """

    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, action_size)

    def forward(self, x):
        """
        Definimos cómo los datos serán propagados a través de la red neuronal.
        Se aplican funciones de activación ReLU en las capas ocultas.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


# Clase DQNAgent - Agente que implementa el algoritmo DQN
class DQNAgent(AbstractAgent):
    """
    Clase que implementa un Agente DQN (Deep Q-Network) para Aprendizaje por Refuerzo.
    Combina una red neuronal para predecir valores Q con un enfoque de exploración-explotación.

    Atributos principales:
        - Modelo principal (q) y modelo objetivo (target).
        - Buffer de repetición (ReplayBuffer).
        - Parámetros del algoritmo, como la tasa de aprendizaje, el descuento gamma,
          y el mecanismo de epsilon-greedy para exploración.
    """

    def __init__(self, state_size, action_size, device, lr=1e-4, gamma=0.99,
                 eps_start=1.0, eps_end=0.05, eps_decay=2e-6):
        """
        Inicializa los atributos del agente:
        - Las redes QNet y la red objetivo se inicializan y se copian entre sí.
        - Se define un optimizador Adam sobre los parámetros de la red principal.
        - Se utiliza un buffer de repetición (ReplayBuffer) para almacenar transiciones pasadas.
        - La política de exploración-explotación depende de un epsilon que decrementa gradualmente.
        """
        self.device = device
        self.q = QNet(state_size, action_size).to(self.device)
        self.target = QNet(state_size, action_size).to(self.device)
        self.target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)

        # Inicializamos parámetros auxiliares
        self.mem = ReplayBuffer()
        self.gamma = gamma
        self.eps, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay
        self.steps = 0

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