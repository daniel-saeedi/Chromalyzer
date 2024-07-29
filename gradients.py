import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import logging
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.interpolate import interp1d
from sklearn.neighbors import NearestNeighbors
from numba import jit
from scipy import sparse
import pandas as pd

df = pd.read_csv('combined_peaks_named.csv')
df.describe()

def gradient_calculator_gaussian_kde(x, data, bandwidth):
    D = data.shape[1]
    N = data.shape[0]
    grad = np.zeros((1, D))

    h = bandwidth

    C = 1/(N*h**D * np.sqrt(2*np.pi)**D)
    F = -1/(2*h**2)
    for i in range(D):
        for n in range(N):
            grad[0, i] += 2 * C * F * (x[i] - data[n, i]) * np.exp(F * np.sum((x - data[n])**2))
    
    return grad


from scipy import stats
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
import numpy as np
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

df1 = df[df['class'] == 0]
df2 = df[df['class'] == 1]

features_all = df[['RT1_center', 'RT2_center', 'm/z']].to_numpy()
features_class0_ = df1[['RT1_center', 'RT2_center', 'm/z']].to_numpy()
features_class1_ = df2[['RT1_center', 'RT2_center', 'm/z']].to_numpy()

# Normalize all features
scaler = StandardScaler()
scaler.fit(features_all)
biotic_peaks_normalized = scaler.transform(features_class1_)
abiotic_peaks_normalized = scaler.transform(features_class0_)


# Define the bandwidth values to test
bandwidths = np.linspace(0.1, 1.0, 30)

# Set up the grid search
grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                    {'bandwidth': bandwidths},
                    cv=5)  # 5-fold cross-validation

# Fit the grid search
grid.fit(biotic_peaks_normalized)

# Get the best bandwidth
best_bandwidth = grid.best_estimator_.bandwidth
print(f"Best bandwidth: {best_bandwidth}")

# Fit KDE with the best bandwidth
kde = KernelDensity(bandwidth=best_bandwidth, kernel='gaussian')
kde.fit(biotic_peaks_normalized)
best_bandwidth

import multiprocessing as mp
from functools import partial

def parallel_gradient_calculation(features, biotic_peaks_normalized, best_bandwidth):
    with mp.Pool() as pool:
        gradient_calculator = partial(gradient_calculator_gaussian_kde, 
                                      data=biotic_peaks_normalized, 
                                      bandwidth=best_bandwidth)
        gradients = pool.map(gradient_calculator, features)
    return np.array(gradients).reshape(-1, 3)

# 'Learning' rate - controls how influential gradients are
alpha = 1e-3

gradients_at_time_t = []

points_at_time_t = []

# Initialization

features = abiotic_peaks_normalized.copy()

time_steps = 10000
for t in range(time_steps):
    # Compute gradient and move in that direction
    gradients = parallel_gradient_calculation(features, biotic_peaks_normalized, best_bandwidth)

    # Prevent exploding gradients due to approximation error
    # gradients = np.clip(gradients, -1, 1)

    # normalize gradients
    gradients_normalized = gradients / np.linalg.norm(gradients, axis=1)[:, None]

    features += (alpha) * gradients_normalized

    # Update new feature locations
    points_at_time_t.append(scaler.inverse_transform(features))

    gradients_at_time_t.append(gradients * scaler.scale_)
    if t % 10 == 0:
        print('Step ', t)


# Save gradients_at_time_t
np.save('gradients_at_time_t.npy', np.array(gradients_at_time_t) )
np.save('points_at_time_t.npy', np.array(points_at_time_t) )