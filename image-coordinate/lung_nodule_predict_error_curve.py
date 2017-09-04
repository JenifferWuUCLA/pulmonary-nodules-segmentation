import os
# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import pandas as pd
import matplotlib.pyplot as plt
# display plots in this notebook
%matplotlib inline

# set display defaults
plt.rcParams['figure.figsize'] = (10, 10)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap

csv_path = "/home/ucla/Downloads/tianchi-2D/csv"
# csv_path = "/home/jenifferwu/IMAGE_MASKS_DATA/z-nerve/csv"
statistics_file = os.path.join(csv_path, "statistics.csv")

headers = ['seriesuid', 'coordX-error', 'coordY-error', 'coordZ-error', 'diameter_mm-error']
df = pd.read_csv(statistics_file, names=headers)
print (df)

seriesuid = df['seriesuid']
coordX_error = df['coordX-error'].astype(float)
coordY_error = df['coordY-error'].astype(float)
coordZ_error = df['coordZ-error'].astype(float)
diameter_mm_error = df['diameter_mm-error'].astype(float)

x = range(len(seriesuid))
y_1 = coordX_error
y_2 = coordY_error
y_3 = coordZ_error
d = diameter_mm_error
# plt.plot(x, y, 'ro-')
# plt.xticks(x, seriesuid, rotation=45)
# plt.margins(0.08)
# plt.subplots_adjust(bottom=0.15)
# plt.show()

# plot
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(x, y_1, 'ro-', label='$y = coordX-error', color="purple", linewidth=2)
ax.plot(x, y_2, 'ro-', label='$y = coordY-error', color="green", linewidth=2)
ax.plot(x, y_3, 'ro-', label='$y = coordZ-error', color="blue", linewidth=2)
ax.plot(x, d, 'ro-', label='$y = diameter_mm-error', color="red", linewidth=2)
plt.title('Legend lung nodule prediction error.')
ax.legend()

plt.show()