
import torch
import numpy as np
import matplotlib.pyplot as plt
from agents.quantum_qnet import QuantumQNet

# --- 1. Configuración del Modelo Entrenado ---
# ¡IMPORTANTE! Estos parámetros deben coincidir con los del modelo que fue guardado.
STATE_SIZE = 116    # Tamaño del estado del entorno Mao
ACTION_SIZE = 54    # Número de acciones posibles en el entorno Mao
NUM_QUBITS = 4      # El modelo fue entrenado con 4 qubits.
MODEL_PATH = "quantum_dqn_mao_final.pth" # Ruta al archivo de pesos guardado.

print(f"Cargando el modelo entrenado desde '{MODEL_PATH}'...")

# --- 2. Cargar la Arquitectura y los Pesos ---
# Creamos una instancia del modelo con la misma arquitectura.
quantum_net = QuantumQNet(STATE_SIZE, ACTION_SIZE, n_qubits=NUM_QUBITS)
# Cargamos los pesos guardados.
quantum_net.load_state_dict(torch.load(MODEL_PATH))
# Ponemos el modelo en modo de evaluación.
quantum_net.eval()
print("Modelo y pesos cargados exitosamente.")

# --- 3. Vincular los Pesos al Circuito ---
# Extraemos los pesos numéricos del ansatz cuántico.
final_quantum_weights = quantum_net.qnn_torch._weight.detach().numpy()

# Mapeamos los parámetros simbólicos (θ) a sus valores numéricos aprendidos.
param_dict = dict(zip(quantum_net.ansatz.parameters, final_quantum_weights))

# Creamos una versión del circuito con los valores numéricos ya aplicados.
bound_circuit = quantum_net.qc.bind_parameters(param_dict)
print("Pesos numéricos vinculados al circuito.")

# --- 4. Graficar el Circuito con Pesos Finales ---
# Dibujamos este nuevo circuito "vinculado".
figure = bound_circuit.draw('mpl', fold=-1)

# --- 5. Guardar la Imagen ---
file_name = f'circuito_con_pesos_finales_{NUM_QUBITS}_qubits.png'
figure.savefig(file_name, dpi=300) # dpi=300 para mayor resolución

print(f"\n¡Gráfico con pesos finales guardado como '{file_name}'!")