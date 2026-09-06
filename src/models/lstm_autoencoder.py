"""
lstm_autoencoder.py
-------------------
Stage 3: Deep Learning Dual-Stream LSTM Autoencoder (LSTM-AE)
Project: Motion Consistency Evaluation in Unmodified Surveillance Video

This module implements two independent LSTM Autoencoders:
  - Stream A LSTM-AE: Models temporal patterns in Frame Differencing motion.
  - Stream B LSTM-AE: Models temporal patterns in Optical Flow Magnitude.

Each Autoencoder consists of:
  - Encoder: 2-layer LSTM mapping 16-frame sequence (T=16) to latent vector h_T.
  - Decoder: 2-layer LSTM reconstructing the 16-frame sequence from h_T.
  - Loss: Mean Squared Error (MSE) reconstruction loss e_A and e_B.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Try importing PyTorch; provide fallback if not installed
TRY_TORCH = True
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    TRY_TORCH = False
    torch = None  # type: ignore
    nn = None  # type: ignore


# ──────────────────────────────────────────────────────────────────────────────
# PyTorch Neural Network Modules (if PyTorch available)
# ──────────────────────────────────────────────────────────────────────────────

if TRY_TORCH and torch is not None:

    class PyTorchLSTMEncoder(nn.Module):
        """
        LSTM Encoder module.
        Maps input sequence (Batch, T=16, Input_Dim) -> Latent vector h_T (Batch, Hidden_Dim).
        """
        def __init__(self, input_dim: int = 1, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.1):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers

            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x shape: (B, T, D)
            _, (h_n, _) = self.lstm(x)
            # Take last layer's hidden state: (B, Hidden_Dim)
            latent = h_n[-1]
            return latent


    class PyTorchLSTMDecoder(nn.Module):
        """
        LSTM Decoder module.
        Maps Latent vector h_T (Batch, Hidden_Dim) -> Reconstructed sequence (Batch, T=16, Input_Dim).
        """
        def __init__(self, output_dim: int = 1, hidden_dim: int = 128, num_layers: int = 2, seq_len: int = 16, dropout: float = 0.1):
            super().__init__()
            self.output_dim = output_dim
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.seq_len = seq_len

            self.lstm = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, latent: torch.Tensor) -> torch.Tensor:
            # latent shape: (B, Hidden_Dim) -> repeat for T time steps: (B, T, Hidden_Dim)
            repeated = latent.unsqueeze(1).repeat(1, self.seq_len, 1)
            decoder_out, _ = self.lstm(repeated)
            reconstructed = self.fc(decoder_out)  # (B, T, Output_Dim)
            return reconstructed


    class PyTorchLSTMAutoencoder(nn.Module):
        """
        Complete PyTorch LSTM Autoencoder combining Encoder and Decoder.
        """
        def __init__(self, input_dim: int = 1, hidden_dim: int = 128, num_layers: int = 2, seq_len: int = 16):
            super().__init__()
            self.seq_len = seq_len
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim

            self.encoder = PyTorchLSTMEncoder(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers)
            self.decoder = PyTorchLSTMDecoder(output_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, seq_len=seq_len)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            latent = self.encoder(x)
            reconstructed = self.decoder(latent)
            return reconstructed

        def get_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
            """
            Compute joint MSE + temporal velocity reconstruction error per clip:
            Returns shape: (Batch,)
            """
            reconstructed = self.forward(x)
            mse_per_clip = torch.mean((x - reconstructed) ** 2, dim=(1, 2))
            if x.shape[1] > 1:
                diff_true = x[:, 1:] - x[:, :-1]
                diff_pred = reconstructed[:, 1:] - reconstructed[:, :-1]
                temp_err_per_clip = torch.mean((diff_true - diff_pred) ** 2, dim=(1, 2))
                return mse_per_clip + 2.0 * temp_err_per_clip
            return mse_per_clip


# ──────────────────────────────────────────────────────────────────────────────
# Stage 3 Wrapper: Dual Stream LSTM Autoencoder
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingHistory:
    history: Dict[str, List[float]]


class DualStreamLSTMAutoencoder:
    """
    High-level interface for training and evaluating two independent
    LSTM Autoencoders (Stream A: Frame Diff, Stream B: Optical Flow).
    Automatically uses PyTorch when available, with a SVD fallback.
    """
    def __init__(
        self,
        seq_len: int = 16,
        input_dim: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 2,
        learning_rate: float = 1e-3,
        device: str = "cpu"
    ):
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.device = device
        self.is_pytorch = (TRY_TORCH and torch is not None)

        if self.is_pytorch:
            self.model = PyTorchLSTMAutoencoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                seq_len=seq_len
            ).to(device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            self.criterion = nn.MSELoss()
        else:
            # Fallback SVD baseline if torch is not installed
            self.mean_: Optional[np.ndarray] = None
            self.components_: Optional[np.ndarray] = None

    def fit(
        self,
        train_sequences: np.ndarray,
        epochs: int = 30,
        batch_size: int = 16,
        validation_split: float = 0.2
    ) -> TrainingHistory:
        """
        Train the LSTM Autoencoder on normal clips.
        train_sequences shape: (N, T=16, Feature_Dim) or (N, T=16)
        """
        # Ensure shape (N, T, D)
        if train_sequences.ndim == 2:
            train_sequences = train_sequences[..., np.newaxis]

        N, T, D = train_sequences.shape

        if self.is_pytorch:
            val_size = max(1, int(N * validation_split))
            train_x = torch.tensor(train_sequences[:-val_size], dtype=torch.float32)
            val_x = torch.tensor(train_sequences[-val_size:], dtype=torch.float32)

            train_loader = DataLoader(TensorDataset(train_x), batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(TensorDataset(val_x), batch_size=batch_size, shuffle=False)

            train_losses: List[float] = []
            val_losses: List[float] = []

            self.model.train()
            for epoch in range(epochs):
                epoch_loss = 0.0
                for (batch_x,) in train_loader:
                    batch_x = batch_x.to(self.device)
                    self.optimizer.zero_grad()
                    reconstructed = self.model(batch_x)
                    mse_loss = self.criterion(reconstructed, batch_x)
                    # Temporal velocity difference loss
                    diff_true = batch_x[:, 1:] - batch_x[:, :-1]
                    diff_pred = reconstructed[:, 1:] - reconstructed[:, :-1]
                    temp_loss = self.criterion(diff_pred, diff_true)
                    loss = mse_loss + 1.0 * temp_loss
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.item() * len(batch_x)

                epoch_loss /= len(train_x)
                train_losses.append(epoch_loss)

                # Validation loss
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for (batch_x,) in val_loader:
                        batch_x = batch_x.to(self.device)
                        reconstructed = self.model(batch_x)
                        mse_loss = self.criterion(reconstructed, batch_x)
                        diff_true = batch_x[:, 1:] - batch_x[:, :-1]
                        diff_pred = reconstructed[:, 1:] - reconstructed[:, :-1]
                        temp_loss = self.criterion(diff_pred, diff_true)
                        loss = mse_loss + 1.0 * temp_loss
                        val_loss += loss.item() * len(batch_x)
                val_loss /= len(val_x)
                val_losses.append(val_loss)
                self.model.train()

            return TrainingHistory(history={"loss": train_losses, "val_loss": val_losses})

        else:
            # Fallback SVD fit
            flat = train_sequences.reshape(N, -1).astype(np.float32)
            self.mean_ = flat.mean(axis=0)
            centered = flat - self.mean_
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            latent_dim = max(1, min(self.hidden_dim, vt.shape[0]))
            self.components_ = vt[:latent_dim]
            train_err = np.mean(np.square(centered - (centered @ vt[:latent_dim].T @ vt[:latent_dim])))
            return TrainingHistory(history={"loss": [float(train_err)], "val_loss": [float(train_err)]})

    def reconstruction_error(self, sequences: np.ndarray) -> np.ndarray:
        """
        Compute reconstruction error e_i per clip.
        Returns 1D array of shape (N,) containing float MSE values.
        """
        if sequences.ndim == 2:
            sequences = sequences[..., np.newaxis]

        if self.is_pytorch:
            self.model.eval()
            tensor_x = torch.tensor(sequences, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                errors = self.model.get_reconstruction_error(tensor_x)
            return errors.cpu().numpy().astype(np.float32)
        else:
            flat = sequences.reshape(sequences.shape[0], -1).astype(np.float32)
            centered = flat - self.mean_
            latent = centered @ self.components_.T
            reconstructed = latent @ self.components_ + self.mean_
            errors = np.mean(np.square(flat - reconstructed), axis=1)
            return errors.astype(np.float32)

    def save(self, path: Union[str, Path]):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.is_pytorch:
            torch.save(self.model.state_dict(), str(path))
        else:
            with open(path, "wb") as f:
                pickle.dump({"mean_": self.mean_, "components_": self.components_}, f)

    def load(self, path: Union[str, Path]):
        path = Path(path)
        if self.is_pytorch:
            self.model.load_state_dict(torch.load(str(path), map_location=self.device))
            self.model.eval()
        else:
            with open(path, "rb") as f:
                data = pickle.load(f)
                self.mean_ = data["mean_"]
                self.components_ = data["components_"]
