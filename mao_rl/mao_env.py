import random  # Librería para generar valores aleatorios (mezclar mazos, por ejemplo)
import numpy as np  # Utilizado para cálculos matemáticos y manejo eficiente de datos.
from collections import deque  # Estructura de datos eficiente para pilas/colores (mazos, manos, etc.)

# Definición de los palos y rangos de las cartas (♥, ♠, etc. y A, 2, 3, etc.)
SUITS = "♠♥♦♣"  # Palos: Espadas, Corazones, Diamantes, Tréboles
RANKS = "A23456789TJQK"  # Rangos: As, Números del 2 al 10, J, Q, K

# Mapea cada carta (como "A♠") a un índice único (0 a 51)
CARD2IDX = {f"{r}{s}": i for i, (r, s) in
            enumerate((r, s) for r in RANKS for s in SUITS)}


# Devuelve el rango (número/letra) de una carta a partir de su índice
def rank(i):
    return RANKS[i // 4]


# Devuelve el palo (espadas, corazones, etc.) de una carta a partir de su índice
def suit(i):
    return SUITS[i % 4]


# Estas son acciones especiales en el juego
ROB_ACTION, MAO_ACTION = 52, 53  # 52 = Robar carta, 53 = Declarar "Mao"


# ───────────────────────────────────────────────────────────────────────────────
# Clase principal que representa el entorno del juego Mao
# ───────────────────────────────────────────────────────────────────────────────
class MaoEnv:
    def __init__(self, n_players=3, penalty_draw=1):
        # Inicializa el juego con el número de jugadores y la penalización estándar.
        # `n_players` = Número de jugadores en el juego
        # `penalty_draw` = Número de cartas penalizadas por errores
        self.n_players, self.penalty_draw = n_players, penalty_draw

        # Tamaño del estado y el espacio de acciones
        self.state_size = 116  # Cantidad de información sobre el estado del juego.
        self.action_space = 54  # 52 cartas + 2 acciones extra (ROB_ACTION y MAO_ACTION)

        # Llama a la función para reiniciar el juego
        self.reset()

    # ───────────── reset ─────────────
    def reset(self, seed=None):
        # Reinicia el juego con un mazo mezclado y distribuye las cartas.
        if seed is not None:
            random.seed(seed)  # Asegura resultados reproducibles si se usa una semilla.

        # Crear y mezclar el mazo de 52 cartas.
        deck = list(range(52))  # 52 cartas (indices de 0 a 51)
        random.shuffle(deck)

        # Repartir 7 cartas a cada jugador.
        self.hands = [deque(deck[i * 7:(i + 1) * 7]) for i in range(self.n_players)]

        # Las cartas restantes estarán disponibles para robos (pila de robo).
        self.draw_pile = deque(deck[7 * self.n_players:-1])

        # La última carta será la carta inicial del descarte.
        self.discard = [deck[-1]]

        # Inicializar estado del juego.
        self.current_player = 0  # Jugador actual (empieza en el índice 0)
        self.direction = 1  # Dirección del turno (1 = horario, -1 = antihorario)
        self.skip_next = False  # Controla si el siguiente jugador pierde su turno.
        self.declared_mao = False  # Verdadero si el jugador declaró "Mao".
        self.forced_suit, self.forced_counter = None, 0  # Control de palo obligado.
        self.stalemate = 0  # Número de turnos sin progreso (para evitar estancamiento).

        # Devuelve la observación inicial (estado del juego).
        return self._obs()

    # ───────────── observación del juego ─────────────
    def _obs(self):
        # Devuelve una representación numérica del estado actual del juego.
        # La observación incluye detalles como:
        # - Cartas del jugador actual.
        # - Carta superior del descarte.
        # - Tamaño de las manos de los demás jugadores.
        # - Estado del palo obligado, si existe.

        # Obtiene el ID del jugador actual.
        pid = self.current_player

        # Representación de cartas en la mano del jugador actual.
        hand_vec = np.zeros(52)
        hand_vec[list(self.hands[pid])] = 1

        # Representación de la carta superior del descarte.
        top_vec = np.zeros(52)
        top_vec[self.discard[-1]] = 1

        # Representación del tamaño de las manos de los otros jugadores.
        counts = np.array([len(h) / 20 for i, h in enumerate(self.hands) if i != pid])

        # Representación codificada del turno del jugador.
        turn_ohe = np.eye(self.n_players)[pid]

        # Indicador de la dirección del juego (1 = horario, 0 = antihorario).
        dirflag = np.array([int(self.direction == -1)])

        # Indicador de si se ha declarado Mao.
        mao_flag = np.array([int(self.declared_mao)])

        # Representación del palo obligado (si existe).
        suit_vec = np.zeros(5)  # 4 palos + opción de "sin palo obligado".
        suit_vec[0] = 1  # Valor inicial.
        if self.forced_suit:
            suit_vec[:] = 0  # Resetea los valores.
            suit_vec[SUITS.index(self.forced_suit) + 1] = 1

        # Concatenar todos estos valores en una observación completa.
        return np.concatenate([hand_vec, top_vec, counts, turn_ohe, dirflag, mao_flag, suit_vec])

    # ───────────── acciones legales ─────────
    def legal_actions(self, pid):
        # Devuelve la lista de cartas o acciones válidas para un jugador.
        if pid != self.current_player:
            return []  # Si no es el turno del jugador, no puede hacer nada.

        top = self.discard[-1]  # Carta superior del descarte.

        # Si hay un palo obligado, el jugador solo puede jugar cartas de ese palo.
        if self.forced_suit:
            return [c for c in self.hands[pid] if suit(c) == self.forced_suit]

        # Si no hay restricciones adicionales, el jugador puede jugar cartas que:
        # - Coincidan en palo o rango con la carta superior del descarte.
        return [c for c in self.hands[pid]
                if suit(c) == suit(top) or rank(c) == rank(top)]

    # ───────────── step ─────────────
    # ───────────── step ─────────────
    def step(self, action):
        # Realiza una acción en el juego, actualiza el estado y devuelve los resultados.
        # `action` es la decisión tomada por el jugador actual.
        pid = self.current_player
        legal = self.legal_actions(pid)

        # --- Lógica de seguridad para el palo forzado ---
        # Si un palo ha sido forzado por demasiado tiempo, se anula para evitar bloqueos.
        self.forced_counter = self.forced_counter + 1 if self.forced_suit else 0
        if self.forced_counter > 300:
            self.forced_suit, self.forced_counter = None, 0

        # --- CASO 1: El jugador declara "MAO" ---
        if action == MAO_ACTION:
            # Si el jugador tiene 1 o 2 cartas, su declaración es válida.
            if 1 <= len(self.hands[pid]) <= 2:
                self.declared_mao = True
            # Es un movimiento que no termina el turno, pero cuenta como una acción.
            self._advance_turn()
            self.stalemate += 1
            # Se devuelve el nuevo estado, sin recompensa y sin que el juego termine.
            return self._obs(), 0.0, False, {}

        # --- CASO 2: El jugador elige robar una carta ---
        if action == ROB_ACTION:
            card = self._draw_card(pid)  # El jugador roba una carta.
            # Se asigna una recompensa base (negativa si se estaba en un palo forzado).
            reward = -3 if self.forced_suit else 0.0
            played = False  # Para rastrear si la carta robada se jugó.
            if card is not None:  # Si se pudo robar una carta...
                top = self.discard[-1]
                # Se comprueba si la carta robada se puede jugar inmediatamente.
                playable = (suit(card) == self.forced_suit) if self.forced_suit else \
                    (suit(card) == suit(top) or rank(card) == rank(top))
                if playable:
                    # Si se puede jugar, se juega.
                    played = True
                    self.hands[pid].remove(card)
                    self.discard.append(card)
                    reward += 1.0  # Recompensa por jugar.
                    self._apply_special(card)  # Aplica el efecto de la carta.
                    if len(self.hands[pid]) == 0:  # Si era la última carta, el jugador gana.
                        return self._obs(), 10.0, True, {}  # Recompensa máxima por ganar.
                    reward += -0.02 * len(self.hands[pid])  # Pequeño incentivo por tener menos cartas.
            # Se avanza el turno. Si no se jugó carta, aumenta el contador de estancamiento.
            self._advance_turn()
            self.stalemate = 0 if played else self.stalemate + 1
            return self._obs(), reward, False, {}

        # --- CASO 3: El jugador realiza una jugada ilegal ---
        if action not in legal:
            self._penalize(pid)  # Se penaliza al jugador haciéndole robar.
            self._advance_turn()
            self.stalemate += 1
            # Se devuelve una recompensa negativa fuerte para enseñar al agente a no hacer esto.
            return self._obs(), -3.0, False, {}

        # --- CASO 4: El jugador realiza una jugada legal ---
        # Se mueve la carta de la mano al descarte.
        self.hands[pid].remove(action)
        self.discard.append(action)
        reward, done = 1.0, False  # Recompensa base por una jugada correcta.
        self.stalemate = 0  # Se resetea el contador de estancamiento.

        # Lógica para la declaración de "MAO".
        if len(self.hands[pid]) == 1:  # Si al jugador le queda una carta...
            if self.declared_mao:
                reward += 2.0  # ...y lo declaró correctamente, recibe un bonus.
            else:
                self._penalize(pid, 2)
                reward -= 3.0  # ...y no lo declaró, es penalizado.
        else:
            self.declared_mao = False  # Resetea la declaración si tiene más de una carta.

        self._apply_special(action)  # Aplica el efecto de la carta jugada.

        # Comprueba si el jugador ha ganado.
        if len(self.hands[pid]) == 0:
            return self._obs(), 10.0, True, {}  # Recompensa máxima por ganar.

        reward += -0.02 * len(self.hands[pid])  # Incentivo por tener menos cartas.
        self._advance_turn()
        return self._obs(), reward, False, {}

        # ───────── Funciones de Ayuda (Helpers) ─────────

    def _apply_special(self, card):
        # Aplica los efectos de las cartas especiales (As, Rey, Reina, Jota).
        r = rank(card)
        if r in ("A", "Q"):
            self.skip_next = True  # As y Reina saltan al siguiente jugador.
        elif r == "K":
            self.direction *= -1  # Rey cambia la dirección del juego.
        elif r == "J":
            self.forced_suit, self.forced_counter = suit(card), 0  # Jota fuerza a jugar su palo.
        elif self.forced_suit and suit(card) == self.forced_suit:
            self.forced_suit, self.forced_counter = None, 0  # Jugar el palo forzado anula el efecto.

    def _advance_turn(self):
        # Determina a quién le toca jugar después.
        step = self.direction
        nxt = (self.current_player + step) % self.n_players
        if self.skip_next:
            nxt = (nxt + step) % self.n_players
            self.skip_next = False
        self.current_player = nxt

        # Lógica de seguridad "anti-estancamiento".
        # Si todos los jugadores pasan, se "quema" una carta para cambiar el juego.
        if self.stalemate >= self.n_players:
            if len(self.discard) > 1:
                self.discard.pop()
            if not self.draw_pile:
                self._reshuffle_discard_into_draw()
            if self.draw_pile:
                self.discard.append(self.draw_pile.popleft())
            self.stalemate = 0

    def _draw_card(self, pid):
        # Hace que un jugador robe una carta de la pila.
        if not self.draw_pile:
            self._reshuffle_discard_into_draw()  # Si no hay cartas, baraja el descarte.
        if self.draw_pile:
            card = self.draw_pile.popleft()
            self.hands[pid].append(card)
            return card
        return None  # Si no hay cartas en ningún lado, no puede robar.

    def _penalize(self, pid, n=None):
        # Aplica una penalización a un jugador, haciéndole robar cartas.
        for _ in range(n or self.penalty_draw):
            self._draw_card(pid)

    def _reshuffle_discard_into_draw(self, force_all=False):
        # Cuando la pila de robar se acaba, toma las cartas del descarte, las baraja y crea una nueva pila.
        if len(self.discard) <= 1:
            return
        keep_n = 1 if force_all else 10
        keep = [self.discard.pop() for _ in range(min(keep_n, len(self.discard) - 1))]
        tmp = list(self.discard)
        random.shuffle(tmp)
        self.draw_pile = deque(tmp)
        self.discard = list(reversed(keep))
