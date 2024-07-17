##################### Pacakges required ######################
if(!require(dplyr)) install.packages('dplyr'); require(dplyr)
if(!require(qrnn)) install.packages('qrnn'); require(qrnn)
if(!require(quadprog)) install.packages("quadprog"); require(quadprog)

################### install tensorflow ####################
if(!require(Rcpp)) install.packages('Rcpp'); require(Rcpp)
if(!require(devtools)) install.packages('devtools'); require(devtools)
if(!require(reticulate)) install.packages('reticulate'); require(reticulate)

reticulate::install_miniconda()
reticulate::conda_list()
reticulate::use_condaenv(condaenv = 'r-reticulate')

if(!require(tensorflow)) install.packages('tensorflow'); require(tensorflow)
if(!require(keras3)) install.packages('keras3'); require(keras3)
tensorflow::install_tensorflow(version = '2.16', method = 'conda', envname = 'r-reticulate')

#################### install l1pm ###########################
reticulate::conda_list()
reticulate::use_condaenv(condaenv = 'r-reticulate')

require(tensorflow)
tf$constant('Hellow')

