import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming your CSV data is stored in a file named 'data.csv'
df = pd.read_csv('simulated7b_results.csv')

# Filter out rows where time is -1
df_filtered = df[df['time'] != -1]

# Calculate average and standard deviation for each group of batch_size and fff_depth
df_result = df_filtered.groupby(['batch_size', 'fff_depth']).agg({'time': ['mean', 'std']}).reset_index()

# Rename the columns
df_result.columns = ['batch_size', 'fff_depth', 'average_time', 'std_dev']

df_result.to_csv('simulated7b_results_averaged.csv', index=False)

# Set seaborn style for a more academic look
sns.set(style="whitegrid", palette="pastel")

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

df_result['batch_size'] *= 2

# Filter dataframe to include only powers of 2 from 1 to 32
powers_of_2 = [2**i for i in range(1,7,1)]  # Powers of 2 from 2^1 to 2^6
filtered_df = df_result[df_result['batch_size'].isin(powers_of_2)]

# Filter dataframe to include only specified depths
selected_depths = [-1, 0, 1, 6, 11, 12]
filtered_df = df_result[df_result['fff_depth'].isin(selected_depths)]


# Iterate over each fff_depth to create lines
for depth, data in filtered_df.groupby('fff_depth'):
    batch_sizes = data['batch_size']
    average_times = data['average_time']
    std_devs = data['std_dev']
    
    # Customize legend labels
    legend_label = 'Baseline' if depth == -1 else 'No FFN' if depth == 0 else f'Depth {depth}'

    # Plot the line
    ax.plot(batch_sizes, average_times, label=legend_label, marker='o')

    # Add error bars for standard deviation
    ax.errorbar(batch_sizes, average_times, yerr=std_devs, linestyle='None', color='black', capsize=5)

# Set labels and title
ax.set_xscale('log', base=2)  # Set X-axis to log scale with base 2
ax.set_xticks(powers_of_2)  # Set X-axis ticks to powers of 2
ax.set_xticklabels(powers_of_2)  # Set X-axis tick labels to powers of 2
ax.set_xlabel('Batch Size')
ax.set_ylabel('Average Time (s)')
ax.set_title('Llama 2 7b - FFN Study')

# Add legend
ax.legend()

# Show the plot
plt.savefig('simulated7b_results.png')

