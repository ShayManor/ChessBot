import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV with your experiments.
df = pd.read_csv('data.csv')

# Convert hyperparameters to numeric types
df['Layers'] = pd.to_numeric(df['Layers'])
df['Epocs'] = pd.to_numeric(df['Epocs'])
df['Hidden Dimensions'] = pd.to_numeric(df['Hidden Dimensions'])

# Filter out any experiments with epocs less than 25.
df = df[df['Epocs'] >= 25]

###############################################
# Plot 1: Overall effect of Layers (aggregated)
###############################################
grouped_layers = df.groupby('Layers').agg({
    'Error Rate': 'mean',
    'Average Time': 'mean'
}).reset_index()

fig, ax1 = plt.subplots(figsize=(8,6))
ax2 = ax1.twinx()
ax1.plot(grouped_layers['Layers'], grouped_layers['Error Rate'],
         'g-o', linewidth=2, markersize=8, label='Error Rate')
ax2.plot(grouped_layers['Layers'], grouped_layers['Average Time'],
         'b-s', linewidth=2, markersize=8, label='Average Time')
ax1.set_xlabel('Layers', fontsize=12)
ax1.set_ylabel('Error Rate', color='green', fontsize=12)
ax2.set_ylabel('Average Time (s)', color='blue', fontsize=12)
plt.title('Average Error Rate & Time vs. Layers', fontsize=14)
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper right')
plt.tight_layout()
plt.show()

###############################################
# Plot 2: Overall effect of Hidden Dimensions (aggregated)
###############################################
grouped_hidden = df.groupby('Hidden Dimensions').agg({
    'Error Rate': 'mean',
    'Average Time': 'mean'
}).reset_index()

fig, ax1 = plt.subplots(figsize=(8,6))
ax2 = ax1.twinx()
ax1.plot(grouped_hidden['Hidden Dimensions'], grouped_hidden['Error Rate'],
         'g-o', linewidth=2, markersize=8, label='Error Rate')
ax2.plot(grouped_hidden['Hidden Dimensions'], grouped_hidden['Average Time'],
         'b-s', linewidth=2, markersize=8, label='Average Time')
ax1.set_xlabel('Hidden Dimensions', fontsize=12)
ax1.set_ylabel('Error Rate', color='green', fontsize=12)
ax2.set_ylabel('Average Time (s)', color='blue', fontsize=12)
plt.title('Average Error Rate & Time vs. Hidden Dimensions', fontsize=14)
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper right')
plt.tight_layout()
plt.show()

###########################################################
# Plot 3: Effect of Layers, grouped by Hidden Dimensions bins
###########################################################
# Create bins for Hidden Dimensions (you can adjust edges as needed)
hidden_bins = [0, 1024, 2048, 4096, 10000]  # For example: [low, mid, high, very high]
hidden_labels = ["Low", "Medium", "High", "Very High"]
df['HiddenDim_bin'] = pd.cut(df['Hidden Dimensions'], bins=hidden_bins, labels=hidden_labels, include_lowest=True)

unique_hd_bins = sorted(df['HiddenDim_bin'].dropna().unique(), key=lambda x: x)
for bin_val in unique_hd_bins:
    subset = df[df['HiddenDim_bin'] == bin_val]
    grouped = subset.groupby('Layers').agg({'Error Rate':'mean', 'Average Time':'mean'}).reset_index()
    if grouped.empty:
        continue
    fig, ax1 = plt.subplots(figsize=(8,6))
    ax2 = ax1.twinx()
    ax1.plot(grouped['Layers'], grouped['Error Rate'],
             'g-o', linewidth=2, markersize=8, label='Error Rate')
    ax2.plot(grouped['Layers'], grouped['Average Time'],
             'b-s', linewidth=2, markersize=8, label='Average Time')
    ax1.set_xlabel('Layers', fontsize=12)
    ax1.set_ylabel('Error Rate', color='green', fontsize=12)
    ax2.set_ylabel('Average Time (s)', color='blue', fontsize=12)
    plt.title(f'Layers vs. Outcomes (Hidden Dimensions: {bin_val})', fontsize=14)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')
    plt.tight_layout()
    plt.show()

###########################################################
# Plot 4: Effect of Hidden Dimensions, grouped by Layers bins
###########################################################
# Create bins for Layers (adjust edges if necessary)
layers_bins = [0, 4, 7, 15]
layers_labels = ["Low", "Medium", "High"]
df['Layers_bin'] = pd.cut(df['Layers'], bins=layers_bins, labels=layers_labels, include_lowest=True)

unique_layers_bins = sorted(df['Layers_bin'].dropna().unique(), key=lambda x: x)
for bin_val in unique_layers_bins:
    subset = df[df['Layers_bin'] == bin_val]
    grouped = subset.groupby('Hidden Dimensions').agg({'Error Rate':'mean', 'Average Time':'mean'}).reset_index()
    if grouped.empty:
        continue
    fig, ax1 = plt.subplots(figsize=(8,6))
    ax2 = ax1.twinx()
    ax1.plot(grouped['Hidden Dimensions'], grouped['Error Rate'],
             'g-o', linewidth=2, markersize=8, label='Error Rate')
    ax2.plot(grouped['Hidden Dimensions'], grouped['Average Time'],
             'b-s', linewidth=2, markersize=8, label='Average Time')
    ax1.set_xlabel('Hidden Dimensions', fontsize=12)
    ax1.set_ylabel('Error Rate', color='green', fontsize=12)
    ax2.set_ylabel('Average Time (s)', color='blue', fontsize=12)
    plt.title(f'Hidden Dimensions vs. Outcomes (Layers: {bin_val})', fontsize=14)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')
    plt.tight_layout()
    plt.show()
