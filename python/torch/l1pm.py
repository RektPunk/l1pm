import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from numpy.typing import NDArray


class L1PMDense(nn.Module):
    """Custom Layer that ensures non-crossing multi-quantile outputs."""

    def __init__(self, in_features: int, r: int):
        super().__init__()
        self.in_features = in_features
        self.r = r

        # Output Layer Weights (Initialized with small random normal)
        self.delta_coef_mat = nn.Parameter(torch.randn(in_features, r) * 0.05)
        self.delta_0_mat = nn.Parameter(torch.randn(1, r) * 0.05)

        # Placeholder to dynamically store the L1 penalty during the forward pass
        self.register_buffer("_penalty", torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Concatenate intercepts and weights: (h + 1, r)
        delta_mat = torch.cat([self.delta_0_mat, self.delta_coef_mat], dim=0)

        # Cumulative sum along columns for monotonicity: (h + 1, r)
        beta_mat = torch.cumsum(delta_mat, dim=1)

        # Slice out the weight and intercept variations for constraint validation
        # delta_vec: (h, r - 1)  -> Weight variations excluding the first quantile
        # delta_0_vec: (1, r - 1)   -> Intercept variations excluding the first quantile
        delta_vec = delta_mat[1:, 1:]  # (h, r - 1)
        delta_0_vec = delta_mat[0:1, 1:]  # (1, r - 1)

        # Isolate negative weight
        # delta_minus_vec: max(0, -delta_vec) (h, r - 1)
        delta_minus_vec = torch.clamp(-delta_vec, min=0.0)

        # Sum the negative weight across all hidden neurons (h) for each quantile
        # delta_minus_vec_sum: (r - 1,) -> Summed results per quantile column
        delta_minus_vec_sum = torch.sum(delta_minus_vec, dim=0)

        # Clip intercept to be greater than or equal to the sum of negative weights
        # This lower bound guarantees that predicted quantiles never cross.
        # delta_0_vec_clipped: (1, r - 1)
        delta_0_vec_clipped = torch.clamp(delta_0_vec, min=delta_minus_vec_sum)

        if self.training:
            # Store the L1 penalty internally to pass it up to the optimizer
            self._penalty = torch.mean(torch.abs(delta_0_vec - delta_0_vec_clipped))

            # Return unconstrained prediction for smoother training landscape
            return torch.matmul(x, beta_mat[1:, :]) + beta_mat[0, :]
        else:
            term1 = torch.matmul(x, beta_mat[1:, :])
            term2 = torch.cumsum(
                torch.cat([beta_mat[0:1, 0:1], delta_0_vec_clipped], dim=1),
                dim=1,
            )
            return term1 + term2

    @property
    def penalty(self) -> torch.Tensor:
        """Getter to safely retrieve the penalty for loss function computation."""
        return self._penalty


class L1PMModel(nn.Module):
    """Model definition using the custom L1PMDense layer."""

    def __init__(self, input_dim: int, hidden_dim1: int, hidden_dim2: int, r: int):
        super().__init__()

        # Feature extractor part
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.Sigmoid(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.Sigmoid(),
        )

        # Our proposed custom layer
        self.output_layer = L1PMDense(in_features=hidden_dim2, r=r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        return self.output_layer(features)


class L1PMRegressor:
    """Wrapper for training and predicting with L1PMModel."""

    def __init__(
        self,
        tau: NDArray[np.float64],
        hidden_dim1: int,
        hidden_dim2: int,
        learning_rate: float = 0.01,
        max_deep_iter: int = 5000,
        l1_penalty: float = 0.1,
        l2_penalty: float = 0.0,
        device: str = "cpu",
    ) -> None:
        self.tau: NDArray[np.float64] = tau
        self.hidden_dim1: int = hidden_dim1
        self.hidden_dim2: int = hidden_dim2
        self.learning_rate: float = learning_rate
        self.max_deep_iter: int = max_deep_iter
        self.l1_penalty: float = l1_penalty
        self.l2_penalty: float = l2_penalty
        self.device: str = device

        # Model instance placeholder
        self.model: L1PMModel | None = None

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> "L1PMRegressor":
        """Trains the internal L1PMModel using the provided dataset."""
        X_tensor: torch.Tensor = torch.tensor(
            X, dtype=torch.float32, device=self.device
        )
        y_tensor: torch.Tensor = torch.tensor(
            y, dtype=torch.float32, device=self.device
        )

        if y_tensor.dim() == 1:
            y_tensor = y_tensor.unsqueeze(1)

        input_dim: int = X.shape[1]

        # Format tau vector to shape (1, r) for matrix broadcasting
        tau_dynamic = torch.tensor(
            self.tau, dtype=torch.float32, device=self.device
        ).view(1, -1)
        r: int = len(self.tau)

        self.model = L1PMModel(input_dim, self.hidden_dim1, self.hidden_dim2, r).to(
            self.device
        )
        optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate)

        # Optimization loop (Full-batch training without DataLoader)
        self.model.train()
        for epoch in range(self.max_deep_iter):
            optimizer.zero_grad()

            # Forward pass: Generate predictions -> Shape: (batch_size, r)
            y_pred = self.model(X_tensor)

            # Compute residuals using matrix broadcasting -> Shape: (batch_size, r)
            diff_y = y_tensor - y_pred

            # Calculate standard multi-quantile loss via broadcasting.
            loss = torch.mean(
                diff_y * (tau_dynamic - (torch.sign(-diff_y) + 1.0) / 2.0)
            )

            # Optimized L2 regularization (Ridge penalty)
            l2_reg = sum(torch.mean(param**2) for param in self.model.parameters())

            # Retrieve the L1 non-crossing penalty dynamically computed
            # inside the L1PMDense forward pass to prevent quantile inversion.
            l1pm_penalty = self.model.output_layer.penalty

            # Formulate the total objective function (Loss + L2 Ridge + L1 Penalty)
            objective_fun = (
                loss + self.l2_penalty * l2_reg + self.l1_penalty * l1pm_penalty
            )

            objective_fun.backward()
            optimizer.step()

            if (epoch + 1) % 1000 == 0:
                print(
                    f"Epoch [{epoch + 1}/{self.max_deep_iter}] "
                    f"Loss: {objective_fun.item():.4f}"
                )

        return self

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float32]:
        """Predicts multiple quantiles for the given input features."""
        if self.model is None:
            raise RuntimeError(
                "The model has not been trained yet. Please call .fit() first."
            )

        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            predictions: NDArray[np.float32] = self.model(X_tensor).cpu().numpy()

        return predictions
