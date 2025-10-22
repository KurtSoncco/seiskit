import os

import matplotlib.pyplot as plt
import numpy as np

from seiskit.ttf.TTF import TTF

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load the real data
base_data = "results/Case_1a/BaseRight.txt"
top_data = "results/Case_1a/SurfaceRight.txt"

# Read the data
base_data_array = np.loadtxt(base_data)
top_data_array = np.loadtxt(top_data)

# compute TTF
dt = base_data_array[1, 0] - base_data_array[0, 0]
base_acc = base_data_array[:, 1]
top_acc = top_data_array[:, 1]
freq, tf = TTF(
    surface_acc=top_acc,
    base_acc=base_acc,
    dt=dt,
)

# Plot the TTF
plt.figure(figsize=(8, 6))
plt.loglog(freq, tf)
plt.title("Transfer Function (TF) between Surface and Base Acceleration")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Transfer Function (TF)")
plt.grid(which="both", linestyle="--", linewidth=0.5)
plt.xlim(0.1, 2.5)
plt.show()
