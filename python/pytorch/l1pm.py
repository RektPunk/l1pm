import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.optim as optim


class L1PMModel(nn.Module):
    """Model definition for L1PM"""

    def __init__(self, input_dim, hidden_dim1, hidden_dim2, r):
        super().__init__()
        self.r = r
        self.hidden_dim2 = hidden_dim2

        # Hidden Layers
        self.dense1 = nn.Linear(input_dim, hidden_dim1)
        self.dense2 = nn.Linear(hidden_dim1, hidden_dim2)

        # Sigmoid activation
        self.sigmoid = nn.Sigmoid()

        # Output Layer Weights
        # Initialized with random normal distribution
        self.delta_coef_mat = nn.Parameter(torch.randn(hidden_dim2, r) * 0.05)
        self.delta_0_mat = nn.Parameter(torch.randn(1, r) * 0.05)

    def _get_mats_and_constraints(self):
        """Helper function to concatenate weights and compute constraints"""
        # delta_mat dim: (hidden_dim2 + 1, r)
        delta_mat = torch.cat([self.delta_0_mat, self.delta_coef_mat], dim=0)

        beta_mat = torch.cumsum(delta_mat, dim=1)

        delta_vec = delta_mat[1:, 1:]
        delta_0_vec = delta_mat[0:1, 1:]

        # delta_minus_vec = max(0, -delta_vec)
        delta_minus_vec = torch.clamp(-delta_vec, min=0.0)
        delta_minus_vec_sum = torch.sum(delta_minus_vec, dim=0)
        delta_0_vec_clipped = torch.clamp(delta_0_vec, min=delta_minus_vec_sum)

        return beta_mat, delta_0_vec, delta_0_vec_clipped

    def forward(self, x, training=True):
        h1 = self.sigmoid(self.dense1(x))
        feature_vec = self.sigmoid(self.dense2(h1))

        beta_mat, _, delta_0_vec_clipped = self._get_mats_and_constraints()

        if training:
            # Use unconstrained predicted_y during training for loss calculation
            predicted_y = torch.matmul(feature_vec, beta_mat[1:, :]) + beta_mat[0, :]
            # Transpose and reshape to shape: (batch_size * r, 1)
            predicted_y_tiled = predicted_y.t().reshape(-1, 1)
            return predicted_y_tiled
        else:
            # Return modified predicted_y with non-crossing constraints for inference
            term1 = torch.matmul(feature_vec, beta_mat[1:, :])
            term2 = torch.cumsum(
                torch.cat([beta_mat[0:1, 0:1], delta_0_vec_clipped], dim=1), dim=1
            )
            return term1 + term2


def l1_p(
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    test_X: npt.NDArray[np.float64],
    valid_X: npt.NDArray[np.float64],
    tau: npt.NDArray[np.float64],
    hidden_dim1: int,
    hidden_dim2: int,
    learning_rate: float,
    max_deep_iter: int,
    lambda_obj: float,
    penalty: float = 0.0,
):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    valid_X_tensor = torch.tensor(valid_X, dtype=torch.float32)
    test_X_tensor = torch.tensor(test_X, dtype=torch.float32)
    tau_tensor = torch.tensor(tau, dtype=torch.float32)

    input_dim = X.shape[1]
    r = len(tau)

    # Initialize model and optimizer
    model = L1PMModel(input_dim, hidden_dim1, hidden_dim2, r)
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

    # Optimization loop (Full-batch training without DataLoader)
    model.train()
    for epoch in range(max_deep_iter):
        optimizer.zero_grad()
        y_pred = model(X_tensor, training=True)

        # y_true_tiled shape: (batch_size * r, 1)
        y_true_tiled = y_tensor.repeat(r, 1)
        diff_y = y_true_tiled - y_pred

        # Dynamically tile tau vector based on current batch size
        tau_dynamic = tau_tensor.repeat_interleave(X_tensor.shape[0]).view(-1, 1)

        # Calculate standard quantile loss
        loss = torch.mean(diff_y * (tau_dynamic - (torch.sign(-diff_y) + 1.0) / 2.0))

        # Extract constraint variables for penalization
        beta_mat, delta_0_vec, delta_0_vec_clipped = model._get_mats_and_constraints()

        # Formulate the final objective function (Loss + L2 Ridge + L1 Penalty)
        objective_fun = (
            loss
            + penalty
            * (
                torch.mean(model.dense1.weight**2)
                + torch.mean(model.dense2.weight**2)
                + torch.mean(model.delta_coef_mat**2)
            )
            + lambda_obj * torch.mean(torch.abs(delta_0_vec - delta_0_vec_clipped))
        )

        # Backward pass and weight update
        objective_fun.backward()
        optimizer.step()

        if (epoch + 1) % 1000 == 0:
            print(
                f"Epoch [{epoch + 1}/{max_deep_iter}], Objective Loss: {objective_fun.item():.4f}"
            )

    # Switch to evaluation mode for inference
    model.eval()
    with torch.no_grad():
        y_predict = model(X_tensor, training=False).numpy()
        y_valid_predict = model(valid_X_tensor, training=False).numpy()
        y_test_predict = model(test_X_tensor, training=False).numpy()

    return {
        "y_predict": y_predict,
        "y_valid_predict": y_valid_predict,
        "y_test_predict": y_test_predict,
    }
