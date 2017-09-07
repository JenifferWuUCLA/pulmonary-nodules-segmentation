import os
# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import pandas as pd
import matplotlib.pyplot as plt

# display plots in this notebook
# %matplotlib inline

# set display defaults
plt.rcParams['figure.figsize'] = (10, 10)  # large images
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
# Create four subplots sharing y axis
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharey=True)

ax1.plot(x, y_1, 'ko-')
ax1.set(title='Lung nodule prediction error', ylabel='CoordX Error (mm)')

ax2.plot(x, y_2, 'r.-')
ax2.set(ylabel='CoordY Error (mm)')

ax3.plot(x, y_3, 'o-')
ax3.set(ylabel='CoordZ Error (mm)')

ax4.plot(x, d, '.-')
ax4.set(xlabel='seriesuid', ylabel='Diameter_mm Error (mm)')

plt.show()