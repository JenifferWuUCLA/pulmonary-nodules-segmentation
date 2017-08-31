import csv, os
# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# display plots in this notebook
%matplotlib inline

# set display defaults
plt.rcParams['figure.figsize'] = (10, 10)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap

csv_path = "/home/ucla/Downloads/tianchi-2D/csv"
statistics_file = os.path.join(csv_path, "statistics.csv")

headers = ['seriesuid', 'coordX-error', 'coordY-error', 'coordZ-error', 'diameter_mm-error']
df = pd.read_csv(statistics_file, names=headers)
print (df)

x = df['seriesuid']
y = df['coordX-error']
z = df['coordX-error']
w = df['coordX-error']
u = df['coordX-error']

# plot
plt.plot(x, y, label="$coordX-error$", color="red", linewidth=2)
plt.plot(x, z, label="$coordX-error$", color="green", linewidth=2)
plt.plot(x, w, label="$coordX-error$", color="blue", linewidth=2)
plt.plot(x, u, "b--", label="$coordX-error$", color="yellow", linewidth=2)

plt.show()