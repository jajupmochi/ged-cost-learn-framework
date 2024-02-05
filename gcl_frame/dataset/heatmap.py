"""
heatmap



@Author: linlin
@Date: 20.11.23
"""

import numpy as np
import matplotlib.pyplot as plt

# Your data
data = np.array([
    [13.4, 10.6, 5.9, 6.4, 5.9, 8.2, 7.4, 8.2],
    [29.2, 30.4, 15.0, 13.0, 16.8, 14.3, 14.0, 14.6],
    [25.3, 36.2, 24.8, 20.1, 19.4, 22.1, 26.1, 21.0],
    [24.4, 40.0, 26.8, 26.4, 25.8, 26.7, 28.5, 26.9],
    [80.0, 74.3, 80.0, 81.4, 81.4, 84.3, 84.3, 87.1],
    [69.0, 71.0, 68.0, 68.0, 74.0, 71.0, 67.0, 68.0],
    [80.0, 81.6, 78.9, 82.6, 84.7, 81.1, 80.0, 80.5],
    [71.4, 71.7, 70.7, 71.0, 70.0, 72.4, 70.3, 69.7],
    [56.3, 56.0, 59.4, 55.7, 55.1, 60.0, 57.4, 54.0],
    [84.3, 84.3, 91.8, 90.1, None, None, 82.6, 82.6],
])

# Mask NaN values
data = np.ma.masked_invalid(data)

# Plotting the heatmap
plt.figure(figsize=(10, 6))
cmap = plt.get_cmap("RdYlGn")  # Red-Yellow-Green colormap
plt.imshow(data, cmap=cmap, interpolation="none")

# Adding colorbar
cbar = plt.colorbar()
cbar.set_label("Values")

# Adding labels and ticks
plt.xticks(np.arange(data.shape[1]), ["Random", "Expert", "Target", "Path", "Treelet", "WLSubtree", "GCN", "GAT"])
plt.yticks(np.arange(data.shape[0]), ["Alkane", "Acyclic", "Redox", "Redox_ox", "MAO", "PAH", "MUTAG", "Monoterp", "PTC_MR", "Letter-h"])

# Displaying the values in each cell
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        plt.text(j, i, f"{data[i, j]:.1f}", ha='center', va='center', color='black' if data[i, j] > data.min() + data.ptp() / 2.0 else 'white')

# Adding title and showing the plot
plt.title("Heatmap of Your Data")
plt.show()
