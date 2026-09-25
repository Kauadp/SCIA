import torch
import torch.nn as nn

from pathlib import Path


# ============================================================
# CAMINHOS
# ============================================================

RAIZ = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    RAIZ
    / "artifacts"
    / "models"
    / "scia_mlp_k25.pt"
)


# ============================================================
# CONFIGURAÇÕES
# ============================================================

INPUT_DIMENSION = 100417
NUM_CLASSES = 13


# ============================================================
# ARQUITETURA DO MLP
# ============================================================

class SparseMLP(nn.Module):

    def __init__(
        self,
        input_dim=INPUT_DIMENSION,
        num_classes=NUM_CLASSES
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                input_dim,
                256
            ),

            nn.ReLU(),

            nn.Dropout(0.4),

            nn.Linear(
                256,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                128,
                num_classes
            )
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# CARREGAMENTO DO MODELO
# ============================================================

def carregar_modelo():

    modelo = SparseMLP()

    state_dict = torch.load(
        MODEL_PATH,
        map_location="cpu"
    )

    modelo.load_state_dict(
        state_dict
    )

    modelo.eval()

    return modelo


# ============================================================
# MODELO CARREGADO
# ============================================================

modelo = carregar_modelo()


# ============================================================
# PREDIÇÃO
# ============================================================

def prever(X):

    # O MLP foi treinado com entrada densa.
    X_dense = X.toarray()

    X_tensor = torch.tensor(
        X_dense,
        dtype=torch.float32
    )

    with torch.no_grad():

        logits = modelo(
            X_tensor
        )

        probabilidades = torch.softmax(
            logits,
            dim=1
        )

    return probabilidades.numpy()
