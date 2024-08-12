import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Set random seed for reproducibility
np.random.seed(42)

# Generate many low saturation values clustered together (e.g., between 1 and 10)
low_saturation = np.random.normal(loc=5, scale=1, size=50)
low_saturation = np.clip(low_saturation, 1, 10)  # Ensure values stay within desired range

# Generate a single middle saturation value (e.g., around 50)
middle_saturation = np.array([50])

# Generate many high saturation values clustered together (e.g., between 90 and 100)
high_saturation = np.random.normal(loc=95, scale=2, size=50)
high_saturation = np.clip(high_saturation, 90, 100)  # Ensure values stay within desired range

# Combine all saturation values
saturation_values = np.concatenate([low_saturation, middle_saturation, high_saturation])

# Calculate corresponding revenue values
revenue_values = 10 * saturation_values

# Create a DataFrame
artificial_data = pd.DataFrame({
    'saturation': saturation_values,
    'revenue': revenue_values
})

# Sort by saturation values
artificial_data = artificial_data.sort_values(by='saturation').reset_index(drop=True)

# Step 1: Calculate the local density using Gaussian Kernel Density Estimation (KDE)
kde = gaussian_kde(artificial_data['saturation'], bw_method='scott')  # 'scott' is the default bandwidth method
artificial_data['density'] = kde(artificial_data['saturation'])

# Step 2: Calculate confidence based on density (higher density = higher confidence)
artificial_data['confidence'] = (artificial_data['density'] - artificial_data['density'].min()) / \
                                (artificial_data['density'].max() - artificial_data['density'].min())

# Step 3: Calculate the upper and lower bounds on revenue
margin_factor = 0.2  # You can adjust this factor to increase/decrease the bounds
artificial_data['lower_bound'] = artificial_data['revenue'] - margin_factor * (1 - artificial_data['confidence']) * artificial_data['revenue']
artificial_data['upper_bound'] = artificial_data['revenue'] + margin_factor * (1 - artificial_data['confidence']) * artificial_data['revenue']

# Step 4: Plotting
plt.scatter(artificial_data['saturation'], artificial_data['revenue'], label='Actual Revenue', color='blue')

# Plot the upper and lower bounds as lines
plt.plot(artificial_data['saturation'], artificial_data['lower_bound'], label='Lower Bound', linestyle='--', color='red')
plt.plot(artificial_data['saturation'], artificial_data['upper_bound'], label='Upper Bound', linestyle='--', color='green')

# Labeling the axes and title
plt.xlabel('Saturation')
plt.ylabel('Revenue')
plt.title('Saturation vs Revenue with Confidence Bounds')

# save to data_viz as image
plt.savefig('data_viz/confidence_bounds.png')