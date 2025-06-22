# quantum_qnet.py
# Optimizado ≤ 10 min – sin __future__ – API moderna de Estimator


import torch
import torch.nn as nn
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector


class QuantumQNet(nn.Module):
    """Red híbrida clásica-cuántica para un agente DQN."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        *,
        n_qubits: int = 2,
        reps_featmap: int = 1,
        reps_ansatz: int = 1,
        shots: int = 256,
        backend_method: str = "statevector",
    ) -> None:
        super().__init__()

        # ── 0 · Proyección clásica ───────────────────────────────────────────
        self.input_scaling = nn.Linear(state_size, n_qubits)

        # ── 1 · Circuito cuántico ────────────────────────────────────────────
        feature_map = ZZFeatureMap(feature_dimension=n_qubits,
                                   reps=reps_featmap)
        ansatz = RealAmplitudes(num_qubits=n_qubits,
                                reps=reps_ansatz)

        qc = QuantumCircuit(n_qubits)
        qc.compose(feature_map, inplace=True)
        qc.compose(ansatz,       inplace=True)

        # Guardamos el circuito como atributo “público”
        self.circuit: QuantumCircuit = qc

        # Estimator sin args  →  se configuran después
        aer_estimator = AerEstimator()
        aer_estimator.set_options(method=backend_method)
        if backend_method != "statevector":
            aer_estimator.set_options(shots=shots)

        qnn = EstimatorQNN(
            circuit=qc,
            estimator=aer_estimator,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
        )
        self.qnn_torch = TorchConnector(qnn)

        # ── 2 · Expansión a logits de acción ────────────────────────────────
        self.output_layer = nn.Linear(1, action_size)

    # -----------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parámetros
        ----------
        x : Tensor (batch, state_size)

        Devuelve
        --------
        logits : Tensor (batch, action_size)
        """
        x = self.input_scaling(x)    # (B, n_qubits)
        q = self.qnn_torch(x)        # (B, 1)
        return self.output_layer(q)  # (B, action_size)

    # -----------------------------------------------------------------------
    # MÉ “PRO” PARA DIBUJAR EL CIRCUITO
    # -----------------------------------------------------------------------
    def draw(self, *args, **kwargs):
        """
        Empaqueta `QuantumCircuit.draw`.

        Ejemplo
        -------
        >>> net = QuantumQNet(110, 54)
        >>> fig = net.draw('mpl', fold=-1)
        """
        return self.circuit.draw(*args, **kwargs)


# Test rápido ---------------------------------------------------------------
if __name__ == "__main__":
    torch.set_num_threads(1)
    net = QuantumQNet(state_size=110, action_size=54)
    dummy = torch.randn(8, 110)
    print("Shape salida:", net(dummy).shape)   # → (8, 54)

    # Dibujo de ejemplo (solo en local con matplotlib):
    # fig = net.draw('mpl', fold=-1)
    # fig.savefig('quantum_circuit.png')
