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

        # ── Capa 0 · Proyección clásica ───────────────────────────────────────
        self.input_scaling = nn.Linear(state_size, n_qubits)

        # ── Capa 1 · Circuito cuántico ────────────────────────────────────────
        self.feature_map = ZZFeatureMap(feature_dimension=n_qubits,
                                        reps=reps_featmap)
        self.ansatz = RealAmplitudes(num_qubits=n_qubits,
                                     reps=reps_ansatz)

        qc = QuantumCircuit(n_qubits)
        qc.compose(self.feature_map, inplace=True)
        qc.compose(self.ansatz,    inplace=True)

        # Estimator sin argumentos; luego configuramos opciones.
        aer_estimator = AerEstimator()
        aer_estimator.set_options(method=backend_method)
        if backend_method != "statevector":
            aer_estimator.set_options(shots=shots)

        # QNN + conector PyTorch
        qnn = EstimatorQNN(
            circuit=qc,
            estimator=aer_estimator,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
        )
        self.qnn_torch = TorchConnector(qnn)

        # ── Capa 2 · Expansión a logits de acción ────────────────────────────
        self.output_layer = nn.Linear(1, action_size)

    # -----------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x (batch, state_size) → logits (batch, action_size)"""
        x = self.input_scaling(x)   # (B, n_qubits)
        q = self.qnn_torch(x)       # (B, 1)
        return self.output_layer(q) # (B, action_size)


# Test rápido ---------------------------------------------------------------
if __name__ == "__main__":
    torch.set_num_threads(1)
    net = QuantumQNet(state_size=110, action_size=54)
    dummy = torch.randn(8, 110)
    print("Shape salida:", net(dummy).shape)   # → (8, 54)