#from prophet import Prophet
from scipy.stats import weibull_min, norm, t
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn import metrics
import math
import pandas as pd
import numpy as np
import nevergrad as ng
import random
#from django.db import models
import json
import nlopt
import nlopt
import numpy as np

# Define the objective function
def objective_function(x, grad):
    if grad.size > 0:
        grad[0] = 0.0
        grad[1] = 0.5 / (np.sqrt(x[1]) + 1e-8)  # Adding a small epsilon to avoid division by zero
    return np.sqrt(x[1])

# Initialize the optimizer
opt = nlopt.opt(nlopt.LD_MMA, 2)
opt.set_min_objective(objective_function)

# Define bounds for the variables
opt.set_lower_bounds([0, 0])
opt.set_upper_bounds([1, 1])

# Set initial guess
x0 = [0.5, 0.5]

# Set optimization parameters
opt.set_xtol_rel(1e-4)

# Run the optimization
x_opt = opt.optimize(x0)
minf = opt.last_optimum_value()

print("Optimal solution:", x_opt)
print("Optimal objective value:", minf)

