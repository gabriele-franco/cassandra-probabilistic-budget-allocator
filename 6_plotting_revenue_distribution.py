import matplotlib.pyplot as plt

# Plotting the total revenue with 95% confidence intervals
plt.figure(figsize=(10, 6))

total_revenue = 41341
bound_upper = 51034
bound_lower = 31648
import numpy as np
import matplotlib.pyplot as plt

# Given parameters

# we assume 95% probability of being between the interval (equivalent to 2 standard deviations)
mu = total_revenue
num_standard_deviations = 2
sigma = (bound_upper - total_revenue) / num_standard_deviations  # Standard deviation

# Generate a range of values
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 10000)

# Calculate the Gaussian distribution
y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# Plot the distribution
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Gaussian Distribution')

# Highlight the 95% confidence interval
plt.fill_between(x, y, where=(x >= bound_lower) & (x <= bound_upper), color='gray', alpha=0.3, label='95% Confidence Interval')

# Add vertical lines at mu, lower bound, and upper bound
plt.axvline(mu, color='red', linestyle='--', alpha=0.8, label='Mean Predicted Revenue')
plt.axvline(bound_lower, color='green', linestyle='--', alpha=0.8, label='Lower Bound')
plt.axvline(bound_upper, color='green', linestyle='--', alpha=0.8, label='Upper Bound')

# Add labels and title
plt.title('Gaussian Distribution with 95% Confidence Interval')
plt.xlabel('Revenue')
plt.ylabel('Probability Density')
plt.legend()
#plt.grid(True)
# Display the plot
plt.show()

