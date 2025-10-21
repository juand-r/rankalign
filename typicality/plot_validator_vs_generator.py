import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')
# Load the data
print("Loading data...")
df = pd.read_csv("hypernym_predictions/cars_combined_clean_predictions_with_validator.csv")
print(f"Loaded {len(df)} rows")

# Subsample 1/10th of the data randomly
print("Subsampling 1/10th of data...")
np.random.seed(42)
df_sample = df.sample(frac=0.1, random_state=42)
print(f"Subsampled to {len(df_sample)} rows")

# Extract generator and validator scores
x = df_sample['log_prob'].values
y = df_sample['validator_log_prob'].values

# Create figure with scatter plot
fig, ax = plt.subplots(figsize=(10, 10))

# Scatter plot with small dots
ax.scatter(x, y, s=1, alpha=0.5, c='blue', rasterized=True)

# Labels and formatting
ax.set_xlabel('Generator Score (log_prob)', fontsize=12)
ax.set_ylabel('Validator Score (validator_log_prob)', fontsize=12)
ax.set_title('Generator vs Validator Scores for "cars" Hypernyms\n(1/10 subsample)', fontsize=14)
ax.grid(True, alpha=0.3)

plt.tight_layout()
print("Showing plot...")
plt.show()

