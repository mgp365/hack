# graficar_circuito.py



# --- 1. Configuración ---
# Define los parámetros para crear una instancia de la red cuántica.
# Deben coincidir con los que usas en el entrenamiento.
# graficar_circuito.py

import matplotlib.pyplot as plt
from agents.quantum_qnet import QuantumQNet

# --- 1. Configuración de la Arquitectura ---
# Define los parámetros para construir el circuito.
# No se necesita un modelo entrenado.
STATE_SIZE = 116   # Tamaño del estado del entorno Mao
ACTION_SIZE = 54   # Número de acciones posibles en el entorno Mao
NUM_QUBITS = 4     # El número de qubits para el que quieres ver el diseño.

print(f"Generando el plano del circuito para {NUM_QUBITS} qubits...")

# --- 2. Construir el Circuito ---
# Creamos una instancia de QuantumQNet para que construya el circuito (self.qc) en su interior.
quantum_net = QuantumQNet(STATE_SIZE, ACTION_SIZE, n_qubits=NUM_QUBITS)

# --- 3. Graficar la Estructura ---
# Usamos el méo .draw() con 'mpl' para una imagen de alta calidad.
# 'fold=-1' asegura que el circuito se dibuje en una sola línea ancha.
figure = quantum_net.qc.draw('mpl', fold=-1)

# --- 4. Guardar la Imagen ---
file_name = f'plano_circuito_{NUM_QUBITS}_qubits.png'
figure.savefig(file_name, dpi=300) # dpi=300 para mayor resolución

print(f"\n¡Plano del circuito guardado como '{file_name}'!")

# Para mostrar el gráfico en una ventana emergente, puedes usar:
# plt.show()

# Si quieres que la imagen se muestre en una ventana al ejecutar el script,
# descomenta la siguiente línea:
plt.show()