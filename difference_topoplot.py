import pandas as pd
import numpy as np
from utils.plot_topomaps import plot_marker_topography
import matplotlib.pyplot as plt

# Define the path to the large CSV file
csv_file_path = 'results/nice_markers/all_subjects_per_electrode.csv'

# Define the markers and conditions
markers = ['a', 'a_n', 'wSMI_1']  # Add more markers as needed
conditions = ['onoff_binary', 'selfother_binary', 'valence', 'time', 'confidence']

# Function to process each chunk of the CSV file
def process_chunk(chunk, markers, conditions):
    # Initialize a dictionary to store averages
    averages = {marker: {condition: [] for condition in conditions} for marker in markers}
    
    # Iterate over each row in the chunk
    for index, row in chunk.iterrows():
        for marker in markers:
            for condition in conditions:
                # Calculate the average for each condition
                condition_value = row[condition]
                marker_value = row[f'{marker}_{row.channel}']
                averages[marker][condition].append((condition_value, marker_value))
    
    # Calculate the average for each marker and condition
    for marker in markers:
        for condition in conditions:
            averages[marker][condition] = np.mean([val for cond, val in averages[marker][condition] if cond == 1])
    
    return averages

# Read the CSV file in chunks
chunk_size = 10000  # Adjust the chunk size as needed
chunks = pd.read_csv(csv_file_path, chunksize=chunk_size)

# Initialize a dictionary to store all averages
all_averages = {marker: {condition: [] for condition in conditions} for marker in markers}

# Process each chunk
for chunk in chunks:
    chunk_averages = process_chunk(chunk, markers, conditions)
    for marker in markers:
        for condition in conditions:
            all_averages[marker][condition].append(chunk_averages[marker][condition])

# Calculate the overall average for each marker and condition
for marker in markers:
    for condition in conditions:
        all_averages[marker][condition] = np.mean(all_averages[marker][condition])

# Plot the topoplots
for marker in markers:
    for condition in conditions:
        # Plot condition 1
        fig1 = plot_marker_topography(df, marker, title=f'{marker} - {condition} Condition 1')
        plt.show()
        
        # Plot condition 2
        fig2 = plot_marker_topography(df, marker, title=f'{marker} - {condition} Condition 2')
        plt.show()
        
        # Plot the difference
        difference = all_averages[marker][condition] - all_averages[marker][condition]
        fig3 = plot_marker_topography(df, marker, title=f'{marker} - {condition} Difference')
        plt.show()

# Save the plots if needed
# fig1.savefig('condition1.png')
# fig2.savefig('condition2.png')
# fig3.savefig('difference.png') 