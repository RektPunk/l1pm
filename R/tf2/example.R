### required package
reticulate::use_condaenv(condaenv = 'r-reticulate')
require(tensorflow)
# require(l1pm)
source("./l1pm.R")
rm(list = ls())

### train data
n = 1000
input_dim = 1
x_data = matrix(runif( n* input_dim, -1, 1), n, input_dim)
sincx = sin(pi * x_data) / (pi * x_data)
Z = matrix(sincx, nrow = n, ncol = input_dim)
ep = rnorm(n, mean = 0, sd = 0.1 * exp(1 - x_data)) ## example 1
y_data = Z + ep

### valid data
x_valid_data = matrix(runif( n * input_dim, -1, 1), n, input_dim)
sincx_valid = sin(pi * x_valid_data) / (pi * x_valid_data)
y_valid = matrix(sincx_valid, nrow = n, ncol = input_dim) +
  rnorm(n, mean = 0, sd = 0.1 * exp(1 - x_valid_data))

### test data
x_test_data = matrix(runif( n * input_dim, -1, 1), n, input_dim)
sincx_test = sin(pi * x_test_data) / (pi * x_test_data)
y_test = matrix(sincx_test, nrow = n, ncol = input_dim) +
  rnorm(n, mean = 0, sd = 0.1 * exp(1 - x_test_data))

### Model fitting
tau_vec = seq(0.1, 0.9, 0.1)
fit_result = l1_p(X = x_data,
                  y = y_data,
                  test_X = x_test_data,
                  valid_X = x_valid_data,
                  tau = tau_vec,
                  hidden_dim1 = 4,
                  hidden_dim2 = 4,
                  learning_rate = 0.005,
                  max_deep_iter = 5000,
                  penalty = 0,
                  lambda_obj = 5)


plot(x_test_data, y_test)
# plot(x_data, y)
col_vec = rainbow(length(tau_vec))
for(i in 1:ncol(fit_result$y_test_predict))
{
  points(x_test_data, fit_result$y_test_predict[, i], col = col_vec[i])
  # points(x_data, fit_result$y_predict[, i], col = col_vec[i])
}

