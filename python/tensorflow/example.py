import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from l1pm import L1PMRegressor

if __name__ == "__main__":
    # Set seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    ### train data
    n = 1000
    input_dim = 1
    x_data = np.random.uniform(-1, 1, (n, input_dim))

    # np.sinc(x) is defined as sin(pi*x) / (pi*x)
    sincx = np.sinc(x_data)
    Z = sincx.reshape(n, input_dim)
    # heteroscedastic noise
    ep = np.random.normal(0, 0.1 * np.exp(1 - x_data)).reshape(n, 1)
    y_data = Z + ep

    ### test data
    x_test_data = np.random.uniform(-1, 1, (n, input_dim))
    sincx_test = np.sinc(x_test_data)
    y_test = sincx_test.reshape(n, input_dim) + np.random.normal(
        0, 0.1 * np.exp(1 - x_test_data)
    ).reshape(n, 1)

    ### Model fitting
    tau_vec = np.arange(0.1, 1.0, 0.1)  # 0.1, 0.2, ..., 0.9
    regressor = L1PMRegressor(
        tau=tau_vec,
        hidden_dim1=4,
        hidden_dim2=4,
        learning_rate=0.005,
        max_deep_iter=5000,
        l1_penalty=5.0,
        l2_penalty=0.0,
    )
    regressor.fit(X=x_data, y=y_data)

    ### Plotting
    y_pred = regressor.predict(x_test_data)

    plt.figure(figsize=(10, 6))
    plt.scatter(x_test_data, y_test, color="gray", alpha=0.3, label="Test Data")

    colors = plt.cm.rainbow(np.linspace(0, 1, len(tau_vec)))
    for i in range(y_pred.shape[1]):
        plt.scatter(x_test_data, y_pred[:, i], color=colors[i], s=5)

    plt.title("L1PM Quantile Regression")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
