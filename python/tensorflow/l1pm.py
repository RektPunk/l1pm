import keras
import numpy as np
import tensorflow as tf
from keras import layers
from numpy.typing import NDArray


class L1PMDense(layers.Layer):
    """Custom Layer that ensures non-crossing multi-quantile outputs."""

    def __init__(
        self, in_features: int, r: int, l1_penalty: float, l2_penalty: float, **kwargs
    ):
        super().__init__(**kwargs)
        self.in_features = in_features
        self.r = r
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty

    def build(self, input_shape):
        # Output Layer Weights (Initialized with small random normal)
        # delta_coef_mat: (in_features, r) -> Weight variations between features and quantiles
        self.delta_coef_mat = self.add_weight(
            name="delta_coef_mat",
            shape=(self.in_features, self.r),
            initializer=keras.initializers.RandomNormal(stddev=0.05),
            trainable=True,
        )
        # delta_0_mat: (1, r) -> Intercept variations for the baseline quantiles
        self.delta_0_mat = self.add_weight(
            name="delta_0_mat",
            shape=(1, self.r),
            initializer=keras.initializers.RandomNormal(stddev=0.05),
            trainable=True,
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # Concatenate intercepts and weights -> Shape: (in_features + 1, r)
        delta_mat = tf.concat([self.delta_0_mat, self.delta_coef_mat], axis=0)

        # Compute monotonic coefficients over quantiles via cumulative sum -> Shape: (in_features + 1, r)
        # (Transposed back and forth because tf.cumsum defaults to column-wise if not configured)
        beta_mat = tf.transpose(tf.cumsum(tf.transpose(delta_mat), axis=0))

        # Slice out the weight and intercept variations for constraint validation
        # delta_vec: (in_features, r - 1) -> Weight variations excluding the first quantile
        # delta_0_vec: (1, r - 1) -> Intercept variations excluding the first quantile
        delta_vec = delta_mat[1:, 1:]
        delta_0_vec = delta_mat[0:1, 1:]

        # Isolate negative weights -> Shape: (in_features, r - 1)
        delta_minus_vec = tf.maximum(0.0, -delta_vec)

        # Sum the negative weights across all hidden neurons for each quantile -> Shape: (1, r - 1)
        delta_minus_vec_sum = tf.reduce_sum(delta_minus_vec, axis=0, keepdims=True)

        # Clip intercept to be greater than or equal to the sum of negative weights
        # This lower bound guarantees that predicted quantiles never cross.
        # delta_0_vec_clipped: (1, r - 1)
        delta_0_vec_clipped = tf.clip_by_value(
            delta_0_vec,
            clip_value_min=delta_minus_vec_sum,
            clip_value_max=float("inf"),
        )

        # Apply Ridge L2 Regularization for the output layer weights internally
        if self.l2_penalty > 0.0:
            self.add_loss(
                self.l2_penalty * tf.reduce_mean(tf.square(self.delta_coef_mat))
            )

        # Track L1 Non-crossing Penalty constraint mismatch internally
        if self.l1_penalty > 0.0:
            l1_penalty_val = self.l1_penalty * tf.reduce_mean(
                tf.abs(delta_0_vec - delta_0_vec_clipped)
            )
            self.add_loss(l1_penalty_val)

        # Return unconstrained prediction during training for smoother gradient landscape
        # Return constrained prediction during evaluation to enforce non-crossing guarantee
        if self.trainable:
            # Same logic as PyTorch's 'term1 + beta_mat[0, :]' using broadcasting
            return tf.matmul(inputs, beta_mat[1:, :]) + beta_mat[0:1, :]
        else:
            # Build valid monotonic intercepts -> Shape: (1, r)
            intercept_0 = delta_mat[0:1, 0:1]
            intercepts_modified = tf.cumsum(
                tf.concat([intercept_0, delta_0_vec_clipped], axis=1), axis=1
            )
            # Return final non-crossing output -> Shape: (batch_size, r)
            return tf.matmul(inputs, beta_mat[1:, :]) + intercepts_modified


def L1PMModel(
    input_dim: int,
    hidden_dim1: int,
    hidden_dim2: int,
    r: int,
    l1_penalty: float,
    l2_penalty: float,
) -> keras.Sequential:
    """Model definition using the custom L1PMDense layer."""

    # Feature extractor parts with optional L2 regularizers
    dense1 = layers.Dense(
        hidden_dim1,
        activation="sigmoid",
        kernel_initializer="random_normal",
        kernel_regularizer=keras.regularizers.l2(l2_penalty)
        if l2_penalty > 0.0
        else None,
    )
    dense2 = layers.Dense(
        hidden_dim2,
        activation="sigmoid",
        kernel_initializer="random_normal",
        kernel_regularizer=keras.regularizers.l2(l2_penalty)
        if l2_penalty > 0.0
        else None,
    )
    # Our proposed custom layer
    output_layer = L1PMDense(
        in_features=hidden_dim2,
        r=r,
        l1_penalty=l1_penalty,
        l2_penalty=l2_penalty,
    )

    return keras.Sequential(
        [layers.Input(shape=(input_dim,)), dense1, dense2, output_layer]
    )


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
    ):
        # Format tau vector to shape (1, r) for matrix broadcasting
        self.tau = tf.constant(tau, dtype=tf.float32)
        self.tau_dynamic = tf.reshape(self.tau, [1, -1])  # (1, r)

        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.learning_rate = learning_rate
        self.max_deep_iter = max_deep_iter
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.model = None

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> "L1PMRegressor":
        """Trains the internal L1PMModel using the provided dataset."""
        in_features = X.shape[1]

        # Model instance initialization
        self.model = L1PMModel(
            in_features,
            self.hidden_dim1,
            self.hidden_dim2,
            len(self.tau),
            self.l1_penalty,
            self.l2_penalty,
        )
        optimizer = keras.optimizers.RMSprop(learning_rate=self.learning_rate)

        # Cast input arrays to TensorFlow tensors to boost execution speed
        X_tensor = tf.cast(X, tf.float32)  # Shape: (batch_size, in_features)
        y_tensor = tf.cast(y, tf.float32)  # Shape: (batch_size,) or (batch_size, 1)

        # Ensure target shape is strictly 2D (batch_size, 1) to prevent broadcasting explosion
        if len(y_tensor.shape) == 1:
            y_tensor = tf.expand_dims(y_tensor, axis=-1)

        assert self.model is not None
        model_local = self.model

        # Static graph compilation via tf.function to match PyTorch's speed (and potentially exceed it)
        @tf.function
        def train_step(X_batch, y_batch):
            with tf.GradientTape() as tape:
                # Forward pass: Generate predictions -> Shape: (batch_size, r)
                y_pred = model_local(X_batch, training=True)

                # Compute residuals using matrix broadcasting -> Shape: (batch_size, r)
                diff_y = y_batch - y_pred

                # Calculate standard multi-quantile loss via broadcasting -> Scalar Tensor
                base_loss = tf.reduce_mean(
                    diff_y * (self.tau_dynamic - (tf.sign(-diff_y) + 1.0) / 2.0)
                )

                # Formulate the total objective function (Loss + Custom Layer Internal Penalties)
                total_loss = base_loss + sum(model_local.losses)

            gradients = tape.gradient(total_loss, model_local.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model_local.trainable_variables))
            return total_loss

        # Optimization loop (Full-batch training without Keras .fit() wrapper overhead)
        print("Training started...")
        self.model.trainable = True  # Ensure training mode is activated
        for epoch in range(self.max_deep_iter):
            loss_val = train_step(X_tensor, y_tensor)

            if (epoch + 1) % 1000 == 0 or epoch == 0:
                print(
                    f"Epoch [{epoch + 1}/{self.max_deep_iter}] - Loss: {loss_val.numpy():.4f}"
                )

        return self

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float32]:
        """Predicts multiple quantiles for the given input features."""
        if self.model is None:
            raise ValueError(
                "The model has not been trained yet. Please call .fit() first."
            )

        X_tensor = tf.cast(X, tf.float32)
        self.model.trainable = (
            False  # Deactivate training mode to trigger monotonic projection logic
        )

        # Call the model directly to prevent unnecessary batching overhead during inference
        predictions = self.model(X_tensor, training=False)
        return predictions.numpy()
