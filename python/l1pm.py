import keras
import numpy as np
import numpy.typing as npt
import tensorflow as tf
from keras import layers


class OutputLayer(layers.Layer):
    def __init__(self, units: int, input_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.input_dim = input_dim

    def build(self, input_shape):
        self.delta_coef_mat = self.add_weight(
            name="delta_coef_mat",
            shape=(self.input_dim, self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.delta_0_mat = self.add_weight(
            name="delta_0_mat",
            shape=(1, self.units),
            initializer="random_normal",
            trainable=True,
        )

    def call(self, inputs: tf.Tensor):
        # delta_mat = [delta_0_mat; delta_coef_mat] (dim: input_dim + 1, units)
        delta_mat = tf.concat([self.delta_0_mat, self.delta_coef_mat], axis=0)

        # beta_mat = tf.transpose(tf.cumsum(tf.transpose(delta_mat), axis=0))
        # This is cumsum across units for each feature (including intercept)
        beta_mat = tf.transpose(tf.cumsum(tf.transpose(delta_mat), axis=0))

        delta_vec = delta_mat[1:, 1:]
        delta_0_vec = delta_mat[0:1, 1:]

        delta_minus_vec = tf.maximum(0.0, -delta_vec)
        delta_minus_vec_sum = tf.reduce_sum(delta_minus_vec, axis=0)

        delta_0_vec_clipped = tf.clip_by_value(
            delta_0_vec,
            clip_value_min=delta_minus_vec_sum,
            clip_value_max=float("inf"),
        )

        intercept_0 = delta_mat[0:1, 0:1]
        intercepts_modified = tf.cumsum(
            tf.concat([intercept_0, delta_0_vec_clipped], axis=1), axis=1
        )

        # Use modified intercepts to ensure non-crossing for the intercept part
        predicted_y_modified = tf.matmul(inputs, beta_mat[1:, :]) + intercepts_modified

        return predicted_y_modified


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
    input_dim = X.shape[1]
    n = X.shape[0]
    r = len(tau)

    tau_tf = tf.constant(tau, dtype=tf.float32)

    # Model definition
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(
                hidden_dim1, activation="sigmoid", kernel_initializer="random_normal"
            ),
            layers.Dense(
                hidden_dim2, activation="sigmoid", kernel_initializer="random_normal"
            ),
            OutputLayer(units=r, input_dim=hidden_dim2),
        ]
    )

    def quantile_loss(y_true, y_pred):
        diff_y = y_true - y_pred
        loss = tf.reduce_mean(diff_y * (tau_tf - (tf.sign(-diff_y) + 1.0) / 2.0))

        output_layer = model.layers[-1]
        dense1 = model.layers[0]
        dense2 = model.layers[1]

        delta_mat = tf.concat(
            [output_layer.delta_0_mat, output_layer.delta_coef_mat], axis=0
        )
        delta_vec = delta_mat[1:, 1:]
        delta_0_vec = delta_mat[0:1, 1:]

        delta_minus_vec = tf.maximum(0.0, -delta_vec)
        delta_minus_vec_sum = tf.reduce_sum(delta_minus_vec, axis=0)
        delta_0_vec_clipped = tf.clip_by_value(
            delta_0_vec,
            clip_value_min=delta_minus_vec_sum,
            clip_value_max=tf.constant(
                float("inf"), shape=delta_minus_vec_sum.shape, dtype=tf.float32
            ),
        )

        res = (
            loss
            + penalty
            * (
                tf.reduce_mean(tf.square(dense1.kernel))
                + tf.reduce_mean(tf.square(dense2.kernel))
                + tf.reduce_mean(tf.square(output_layer.delta_coef_mat))
            )
            + lambda_obj * tf.reduce_mean(tf.abs(delta_0_vec - delta_0_vec_clipped))
        )

        return res

    optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
    model.compile(loss=quantile_loss, optimizer=optimizer)

    model.fit(X, y, epochs=max_deep_iter, batch_size=n)

    y_predict = model.predict(X, batch_size=n)
    y_valid_predict = model.predict(valid_X, batch_size=len(valid_X))
    y_test_predict = model.predict(test_X, batch_size=len(test_X))

    return {
        "y_predict": y_predict,
        "y_valid_predict": y_valid_predict,
        "y_test_predict": y_test_predict,
    }
