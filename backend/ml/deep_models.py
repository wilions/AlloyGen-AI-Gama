"""Deep learning model wrappers for AlloyGen 2.0.

Provides sklearn-compatible wrappers for:
- CrabNet-style attention model (self-attention on element embeddings)
- TabNet (attention-based tabular model)
- DeepMLP (3-layer feedforward with BatchNorm, Dropout, residual connections)

All models implement fit(), predict(), score(), and feature_importances_.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, accuracy_score

logger = logging.getLogger(__name__)

# Check PyTorch availability
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.info("PyTorch not installed — deep learning models will be unavailable")

# Check TabNet availability
try:
    from pytorch_tabnet.tab_model import TabNetRegressor as _TabNetRegressor
    from pytorch_tabnet.tab_model import TabNetClassifier as _TabNetClassifier
    HAS_TABNET = True
except ImportError:
    HAS_TABNET = False


# ---------------------------------------------------------------------------
# DeepMLP (PyTorch)
# ---------------------------------------------------------------------------
if HAS_TORCH:
    class _MLPNet(nn.Module):
        """3-layer MLP with BatchNorm, Dropout, and residual connection."""

        def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 1, dropout: float = 0.2):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.bn1 = nn.BatchNorm1d(hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.bn2 = nn.BatchNorm1d(hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, hidden_dim)
            self.bn3 = nn.BatchNorm1d(hidden_dim)
            self.out = nn.Linear(hidden_dim, output_dim)
            self.dropout = nn.Dropout(dropout)
            self.relu = nn.ReLU()

        def forward(self, x):
            h1 = self.relu(self.bn1(self.fc1(x)))
            h1 = self.dropout(h1)
            h2 = self.relu(self.bn2(self.fc2(h1)))
            h2 = self.dropout(h2)
            # Residual connection
            h3 = self.relu(self.bn3(self.fc3(h2))) + h1
            h3 = self.dropout(h3)
            return self.out(h3)


class DeepMLPRegressor(BaseEstimator, RegressorMixin):
    """PyTorch MLP regressor with sklearn interface."""

    def __init__(
        self,
        hidden_dim: int = 128,
        epochs: int = 200,
        lr: float = 1e-3,
        batch_size: int = 64,
        dropout: float = 0.2,
        random_state: int = 42,
    ):
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.dropout = dropout
        self.random_state = random_state
        self.scaler_ = StandardScaler()
        self.model_ = None
        self.feature_importances_ = None

    def fit(self, X, y):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for DeepMLPRegressor")

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32).ravel()

        X_scaled = self.scaler_.fit_transform(X_arr)
        X_t = torch.tensor(X_scaled, dtype=torch.float32)
        y_t = torch.tensor(y_arr, dtype=torch.float32).unsqueeze(1)

        self.model_ = _MLPNet(X_arr.shape[1], self.hidden_dim, 1, self.dropout)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        criterion = nn.MSELoss()

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model_.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = self.model_(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            scheduler.step(epoch_loss)

        # Compute feature importance via gradient-based sensitivity
        self._compute_feature_importance(X_t, y_t)
        return self

    def _compute_feature_importance(self, X_t, y_t):
        """Gradient-based feature importance."""
        self.model_.eval()
        X_req = X_t.clone().requires_grad_(True)
        pred = self.model_(X_req)
        pred.sum().backward()
        grads = X_req.grad.abs().mean(dim=0).detach().numpy()
        self.feature_importances_ = grads / (grads.sum() + 1e-10)

    def predict(self, X):
        if not HAS_TORCH or self.model_ is None:
            raise RuntimeError("Model not fitted")
        X_arr = np.asarray(X, dtype=np.float32)
        X_scaled = self.scaler_.transform(X_arr)
        X_t = torch.tensor(X_scaled, dtype=torch.float32)
        self.model_.eval()
        with torch.no_grad():
            return self.model_(X_t).numpy().ravel()

    def score(self, X, y):
        return r2_score(y, self.predict(X))

    def __getstate__(self):
        state = self.__dict__.copy()
        if self.model_ is not None:
            state["_model_state_dict"] = self.model_.state_dict()
            state["_input_dim"] = list(self.model_.fc1.parameters())[0].shape[1]
            del state["model_"]
        return state

    def __setstate__(self, state):
        model_state = state.pop("_model_state_dict", None)
        input_dim = state.pop("_input_dim", None)
        self.__dict__.update(state)
        if model_state is not None and input_dim is not None and HAS_TORCH:
            self.model_ = _MLPNet(input_dim, self.hidden_dim, 1, self.dropout)
            self.model_.load_state_dict(model_state)
            self.model_.eval()
        else:
            self.model_ = None


# ---------------------------------------------------------------------------
# TabNet wrapper
# ---------------------------------------------------------------------------
class TabNetRegressorWrapper(BaseEstimator, RegressorMixin):
    """Sklearn-compatible wrapper for TabNet regressor."""

    def __init__(
        self,
        n_d: int = 16,
        n_a: int = 16,
        n_steps: int = 3,
        max_epochs: int = 100,
        patience: int = 15,
        random_state: int = 42,
    ):
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.max_epochs = max_epochs
        self.patience = patience
        self.random_state = random_state
        self.model_ = None
        self.feature_importances_ = None

    def fit(self, X, y):
        if not HAS_TABNET:
            raise ImportError("pytorch-tabnet is required for TabNetRegressorWrapper")

        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        self.model_ = _TabNetRegressor(
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            seed=self.random_state,
            verbose=0,
        )
        self.model_.fit(
            X_arr, y_arr,
            max_epochs=self.max_epochs,
            patience=self.patience,
            batch_size=min(256, len(X_arr)),
        )
        self.feature_importances_ = self.model_.feature_importances_
        return self

    def predict(self, X):
        if self.model_ is None:
            raise RuntimeError("Model not fitted")
        X_arr = np.asarray(X, dtype=np.float32)
        return self.model_.predict(X_arr).ravel()

    def score(self, X, y):
        return r2_score(y, self.predict(X))


# ---------------------------------------------------------------------------
# CrabNet-style attention model
# ---------------------------------------------------------------------------
if HAS_TORCH:
    class _ElementAttentionNet(nn.Module):
        """Simplified CrabNet-style attention on element features.

        Instead of element embeddings (which require element-level input),
        this applies self-attention over feature groups to learn which
        input features matter most for each prediction.
        """

        def __init__(self, input_dim: int, n_heads: int = 4, d_model: int = 64, output_dim: int = 1):
            super().__init__()
            self.projection = nn.Linear(input_dim, d_model)
            # Reshape input into "tokens" by splitting features into groups
            self.n_tokens = min(input_dim, 16)
            self.token_dim = d_model
            self.token_proj = nn.Linear(
                (input_dim + self.n_tokens - 1) // self.n_tokens,
                d_model,
            )
            self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            self.norm = nn.LayerNorm(d_model)
            self.fc = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(d_model, output_dim),
            )
            self.attention_weights_ = None

        def forward(self, x):
            batch_size = x.shape[0]
            input_dim = x.shape[1]

            # Split features into n_tokens groups
            chunk_size = (input_dim + self.n_tokens - 1) // self.n_tokens
            # Pad input if needed
            if input_dim % chunk_size != 0:
                pad_size = chunk_size * self.n_tokens - input_dim
                x_padded = torch.nn.functional.pad(x, (0, pad_size))
            else:
                x_padded = x

            # Reshape to (batch, n_tokens, chunk_size)
            tokens = x_padded.view(batch_size, self.n_tokens, -1)
            # Truncate to expected chunk size if needed
            tokens = tokens[:, :, :chunk_size]
            tokens = self.token_proj(tokens)

            # Self-attention
            attn_out, attn_weights = self.attention(tokens, tokens, tokens)
            self.attention_weights_ = attn_weights.detach()
            tokens = self.norm(tokens + attn_out)

            # Pool across tokens
            pooled = tokens.mean(dim=1)
            return self.fc(pooled)


class CrabNetStyleRegressor(BaseEstimator, RegressorMixin):
    """CrabNet-inspired attention model with sklearn interface.

    Provides attention heatmaps showing which feature groups drive predictions.
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        epochs: int = 200,
        lr: float = 1e-3,
        batch_size: int = 64,
        random_state: int = 42,
    ):
        self.d_model = d_model
        self.n_heads = n_heads
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.random_state = random_state
        self.scaler_ = StandardScaler()
        self.model_ = None
        self.feature_importances_ = None
        self.attention_heatmap_ = None

    def fit(self, X, y):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for CrabNetStyleRegressor")

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32).ravel()

        X_scaled = self.scaler_.fit_transform(X_arr)
        X_t = torch.tensor(X_scaled, dtype=torch.float32)
        y_t = torch.tensor(y_arr, dtype=torch.float32).unsqueeze(1)

        self.model_ = _ElementAttentionNet(X_arr.shape[1], self.n_heads, self.d_model, 1)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        criterion = nn.MSELoss()

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model_.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = self.model_(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            scheduler.step(epoch_loss)

        # Compute attention heatmap and feature importance
        self.model_.eval()
        with torch.no_grad():
            self.model_(X_t)
            if self.model_.attention_weights_ is not None:
                self.attention_heatmap_ = self.model_.attention_weights_.mean(dim=0).numpy()

        # Gradient-based feature importance
        self._compute_feature_importance(X_t)
        return self

    def _compute_feature_importance(self, X_t):
        self.model_.eval()
        X_req = X_t.clone().requires_grad_(True)
        pred = self.model_(X_req)
        pred.sum().backward()
        grads = X_req.grad.abs().mean(dim=0).detach().numpy()
        self.feature_importances_ = grads / (grads.sum() + 1e-10)

    def predict(self, X):
        if not HAS_TORCH or self.model_ is None:
            raise RuntimeError("Model not fitted")
        X_arr = np.asarray(X, dtype=np.float32)
        X_scaled = self.scaler_.transform(X_arr)
        X_t = torch.tensor(X_scaled, dtype=torch.float32)
        self.model_.eval()
        with torch.no_grad():
            return self.model_(X_t).numpy().ravel()

    def score(self, X, y):
        return r2_score(y, self.predict(X))

    def __getstate__(self):
        state = self.__dict__.copy()
        if self.model_ is not None:
            state["_model_state_dict"] = self.model_.state_dict()
            state["_input_dim"] = list(self.model_.projection.parameters())[0].shape[1]
            del state["model_"]
        return state

    def __setstate__(self, state):
        model_state = state.pop("_model_state_dict", None)
        input_dim = state.pop("_input_dim", None)
        self.__dict__.update(state)
        if model_state is not None and input_dim is not None and HAS_TORCH:
            self.model_ = _ElementAttentionNet(input_dim, self.n_heads, self.d_model, 1)
            self.model_.load_state_dict(model_state)
            self.model_.eval()
        else:
            self.model_ = None
