# hack/mao_rl/graficar_circuito.py


from hack.mao_rl.agents.quantum_qnet import QuantumQNet

STATE_SIZE, ACTION_SIZE, N_QUBITS = 110, 54, 2
net = QuantumQNet(STATE_SIZE, ACTION_SIZE, n_qubits=N_QUBITS)

# 1) Una sola descomposición (muestra el nivel intermedio de library gates)
fig1 = net.circuit.decompose().draw('mpl', fold=-1)
fig1.savefig("circuito_descompuesto.png", dpi=150)

# 2) Descomposición total hasta puertas básicas
full = net.circuit.decompose().decompose()      # repite las veces que quieras
fig2 = full.draw('mpl', fold=-1)
fig2.savefig("circuito_bajo_nivel.png", dpi=150)
