# Correlation_Analysis
Some basics about correlation analysis based on code, including resampling, standard error, lsqfit etc.

## 1. Resampling

### 1.1. What is resampling?

Resampling is a method of estimating the precision of sample statistics (medians, variances, percentiles) by using subsets of available data (bootstrapping) or drawing randomly with replacement from a set of data points (permutation).

### 1.2. Why resampling?

Resampling is used to estimate the precision of sample statistics (medians, variances, percentiles) by using subsets of available data (bootstrapping) or drawing randomly with replacement from a set of data points (permutation).

### 1.3. Jackknife

The jackknife is a resampling technique especially useful for variance and bias estimation. The jackknife predates other common resampling methods such as the bootstrap. The jackknife estimator of a parameter is found by systematically leaving out each observation from a dataset and calculating the estimate and then finding the average of these calculations. 

### 1.4. Bootstrap

The bootstrap is a resampling technique used to estimate statistics on a population by sampling a dataset with replacement. It can be used to estimate summary statistics such as the mean or standard deviation. It is used in applied machine learning to estimate the skill of machine learning models when making predictions on data not included in the training data.


## 2. Standard Error

### 2.1. What is standard error?

The standard error (SE) of a statistic (usually an estimate of a parameter) is the standard deviation of its sampling distribution or an estimate of that standard deviation. If the parameter or the statistic is the mean, it is called the standard error of the mean (SEM).

### 2.2. Standard error of the mean in the raw data

### 2.3. Standard error of the mean in the Jackknifed data

### 2.4. Standard error of the mean in the Bootstrapped data


## 3. Correlation

### 3.1. Correlation is determined when averaging over the samples

### 3.2. Gvar list can preserve all correlations as well as reconstruct distributions

### 3.3. Fit once with gvar list is better than fit N times on each sample


## 4. Least Square Fit

### 4.1. What is least square fit?

In statistics, the method of least squares is a standard approach to the approximate solution of overdetermined systems, i.e., sets of equations in which there are more equations than unknowns. "Least squares" means that the overall solution minimizes the sum of the squares of the errors made in solving every single equation.

### 4.2. How to do least square fit with lsqfit package?

### 4.3. What is a good fit?



