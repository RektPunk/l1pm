library(tensorflow)
library(keras3)

l1_p <- function(X, y, test_X, valid_X, tau, hidden_dim1, hidden_dim2, learning_rate, max_deep_iter, lambda_obj, penalty = 0) {
  input_dim <- ncol(X)
  n <- nrow(X)
  r <- length(tau)
  p <- hidden_dim2 + 1
  tau_mat <- matrix(rep(tau, each = n), ncol = 1)
  
  input_x <- layer_input(shape = input_dim)
  output_y <- tf$constant(y, dtype = tf$float32)
  output_y_tiled <- tf$tile(output_y, multiples = tf$constant(as.integer(c(r, 1)), dtype = tf$int32))
  tau_tf <- tf$constant(tau_mat, dtype = tf$float32)
  
  # Layer 1
  hidden_layer_1 <- layer_dense(units = hidden_dim1, activation = "sigmoid", kernel_initializer = initializer_random_normal())
  
  # Layer 2
  hidden_layer_2 <- layer_dense(units = hidden_dim2, activation = "sigmoid", kernel_initializer = initializer_random_normal())
  
  # Output layer
  outputLayer = Layer(
    classname = 'OutputLayer',
    initialize = function(units, input_dim) {
      super$initialize()
      self$units = units
      self$input_dim = input_dim
      self$delta_coef_mat = self$add_weight(
        shape = shape(input_dim, units),
        initializer = 'random_normal',
        trainable = TRUE
      )
      self$delta_0_mat = self$add_weight(
        shape = shape(1, units),
        initializer = 'random_normal',
        trainable = TRUE
      )
    },
    call = function(inputs) {
      delta_mat = tf$concat(list(self$delta_0_mat, self$delta_coef_mat), axis = 0L)
      beta_mat <- tf$transpose(tf$cumsum(tf$transpose(delta_mat)))
      
      delta_vec <- delta_mat[2:(self$input_dim + 1), 2:self$units]
      self$delta_0_vec <- delta_mat[1, 2:self$units, drop = FALSE]
      delta_minus_vec <- tf$maximum(0, -delta_vec)
      delta_minus_vec_sum <- tf$reduce_sum(delta_minus_vec, axis = 0L)
      self$delta_0_vec_clipped <- tf$clip_by_value(
        self$delta_0_vec,
        clip_value_min = delta_minus_vec_sum,
        clip_value_max = tf$constant(Inf, shape = delta_minus_vec_sum$shape, dtype = tf$float32)
      )
      predicted_y_modified <- tf$matmul(inputs, beta_mat[2:(self$input_dim + 1), ]) +
        tf$cumsum(tf$concat(list(beta_mat[1, 1, drop = FALSE], self$delta_0_vec_clipped), axis = 1L), axis = 1L)
      predicted_y <- tf$matmul(inputs, beta_mat[2:(self$input_dim + 1), ]) + beta_mat[1, ]
      predicted_y_tiled <- tf$reshape(tf$transpose(predicted_y), shape = tf$constant(as.integer(c(-1L, 1)), dtype = tf$int32))
      return(predicted_y_tiled)
    }
  )
  output_layer = outputLayer(units = r, input_dim = hidden_dim2)
  model = keras_model_sequential()
  model %>% hidden_layer_1 %>% hidden_layer_2 %>% output_layer
  
  quantile_loss = function(output_y, predicted_y_tiled) {
    output_y_tiled <- tf$tile(output_y, multiples = tf$constant(as.integer(c(r, 1)), dtype = tf$int32))
    diff_y <- output_y_tiled - predicted_y_tiled
    loss = tf$reduce_mean(diff_y * (tau_tf - (tf$sign(-diff_y) + 1) / 2))

    delta_mat = tf$concat(list(output_layer$delta_0_mat, output_layer$delta_coef_mat), axis = 0L)
    beta_mat <- tf$transpose(tf$cumsum(tf$transpose(delta_mat)))
    delta_vec <- delta_mat[2:p, 2:r]
    delta_0_vec <- delta_mat[1, 2:r, drop = FALSE]
    delta_minus_vec <- tf$maximum(0, -delta_vec)
    delta_minus_vec_sum <- tf$reduce_sum(delta_minus_vec, axis = 0L)
    delta_0_vec_clipped <- tf$clip_by_value(
      delta_0_vec,
      clip_value_min = delta_minus_vec_sum,
      clip_value_max = tf$constant(Inf, shape = delta_minus_vec_sum$shape, dtype = tf$float32))

    res <-  loss +
      penalty * (
        tf$reduce_mean(tf$square(hidden_layer_1$kernel)) + 
        tf$reduce_mean(tf$square(hidden_layer_2$kernel)) +
        tf$reduce_mean(tf$square(output_layer$delta_coef_mat))
      ) +
      lambda_obj * tf$reduce_mean(tf$abs(delta_0_vec - delta_0_vec_clipped))
    return(loss)
  }

  optimizer <- optimizer_rmsprop(learning_rate = learning_rate)
  model = model %>% compile(loss = quantile_loss, optimizer = optimizer)
  model %>% fit(X, output_y, epochs = max_deep_iter, batch_size = n)
  
  y_predict <- model %>% predict(X, batch_size = n)
  y_valid_predict <- model %>% predict(valid_X, batch_size = nrow(valid_X))
  y_test_predict <- model %>% predict(test_X, batch_size = nrow(test_X))
  
  barrier_result <- list(
    y_predict = matrix(y_predict, n, r), 
    y_valid_predict = matrix(y_valid_predict, nrow(valid_X), r),
    y_test_predict = matrix(y_test_predict, nrow(test_X), r)
  )
  return(barrier_result)
}
