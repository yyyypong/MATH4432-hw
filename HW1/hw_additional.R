library(FNN)
library(ggplot2)
rm(list=ls())

ntrain <- 50  # Training
ntest <- 500  # Test
nrep <- 100   # repeat 100 times
p <- 20
puse <- c(1, 2, 3, 4, 10, 20)  # number of predictors
k <- c(1:9)

sigma <- 0.5

# Generate training and test data
Xtrain <- matrix(runif(ntrain * p, -1, 1), ntrain, p)
Xtest <- matrix(runif(ntest * p, -1, 1), ntest, p)
y0 <- sin(2 * Xtrain[, 1])  # Only the first predictor is related to y
ytest <- sin(2 * Xtest[, 1])

out_knn <- data.frame()  # Output results
out_lm <- data.frame()

for (i in 1:length(puse)) {
  yhat_lm <- matrix(0, ntest, nrep)
  yhat_knn <- array(0, dim = c(ntest, nrep, length(k)))  # 3D array for KNN predictions
  
  for (l in 1:nrep) {
    y <- y0 + rnorm(ntrain, 0, sigma)
    
    # Select the first 'puse[i]' predictors for training and testing
    Xtrain_subset <- Xtrain[, 1:puse[i], drop = FALSE]
    Xtest_subset <- Xtest[, 1:puse[i], drop = FALSE]
    
    # Fit linear regression using lm function
    Xtrain_df <- as.data.frame(Xtrain_subset)
    Xtest_df <- as.data.frame(Xtest_subset)
    fit_lm <- lm(y ~ ., data = Xtrain_df)
    yhat_lm[, l] <- predict(fit_lm, newdata = Xtest_df)
    
    # Fit KNN using knn.reg function for different k
    for (j in 1:length(k)) {
      fit_knn <- knn.reg(Xtrain_subset, Xtest_subset, y, k = k[j])
      yhat_knn[, l, j] <- fit_knn$pred
    }
    
    cat(i, "-th p, ", l, "-th repetition finished. \n")
  }
  
  # Compute bias and variance of linear regression
  ybar_lm <- rowMeans(yhat_lm)  # E(f^hat)
  biasSQ_lm <- mean((ytest - ybar_lm)^2)  # Bias^2
  variance_lm <- mean(apply(yhat_lm, 1, var))  # Variance
  err_lm <- biasSQ_lm + variance_lm  # Total MSE
  
  out_lm <- rbind(out_lm, data.frame(error = biasSQ_lm, component = "squared-bias", p = paste0("p = ", puse[i])))
  out_lm <- rbind(out_lm, data.frame(error = variance_lm, component = "variance", p = paste0("p = ", puse[i])))
  out_lm <- rbind(out_lm, data.frame(error = err_lm, component = "MSE", p = paste0("p = ", puse[i])))
  
  # Compute bias and variance of KNN regression
  for (j in 1:length(k)) {
    ybar_knn <- rowMeans(yhat_knn[, , j])  # E(f^hat)
    biasSQ_knn <- mean((ytest - ybar_knn)^2)  # Bias^2
    variance_knn <- mean(apply(yhat_knn[, , j], 1, var))  # Variance
    err_knn <- biasSQ_knn + variance_knn  # Total MSE
    
    out_knn <- rbind(out_knn, data.frame(error = biasSQ_knn, component = "squared-bias", k = k[j], p = paste0("p = ", puse[i])))
    out_knn <- rbind(out_knn, data.frame(error = variance_knn, component = "variance", k = k[j], p = paste0("p = ", puse[i])))
    out_knn <- rbind(out_knn, data.frame(error = err_knn, component = "MSE", k = k[j], p = paste0("p = ", puse[i])))
  }
}