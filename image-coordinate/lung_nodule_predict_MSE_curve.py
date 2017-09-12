import os
# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
from sklearn import metrics
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
# statistics_file = os.path.join(csv_path, "statistics_cmp.csv")
statistics_file = os.path.join(csv_path, "statistics_MSE.csv")

headers = ['seriesuid', 'true_coordX', 'true_coordY', 'true_coordZ', 'true_diameter_mm', 'pred_coordX', 'pred_coordY',
           "pred_coordZ", "pred_diameter_mm"]
df = pd.read_csv(statistics_file, names=headers)
print (df)

seriesuid = df['seriesuid']

true_coordX = df['true_coordX'].astype(float)
true_coordY = df['true_coordY'].astype(float)
true_coordZ = df['true_coordZ'].astype(float)
true_diameter_mm = df['true_diameter_mm'].astype(float)

pred_coordX = df['pred_coordX'].astype(float)
pred_coordY = df['pred_coordY'].astype(float)
pred_coordZ = df['pred_coordZ'].astype(float)
pred_diameter_mm = df['pred_diameter_mm'].astype(float)

x = range(len(seriesuid))

coordX_error_mean = np.mean(true_coordX - pred_coordX)
coordY_error_mean = np.mean(true_coordY - pred_coordY)
coordZ_error_mean = np.mean(true_coordZ - pred_coordZ)
diameter_mm_error_mean = np.mean(true_diameter_mm - pred_diameter_mm)
print('======================================Mean of lung nodule prediction error====================================')
print('Mean of CoordX Error: {:.3f}'.format(coordX_error_mean))
print('Mean of CoordY Error: {:.3f}'.format(coordY_error_mean))
print('Mean of CoordZ Error: {:.3f}'.format(coordZ_error_mean))
print('Mean of Diameter_mm Error: {:.3f}'.format(diameter_mm_error_mean))

coordX_error_std = np.std(true_coordX - pred_coordX)
coordY_error_std = np.std(true_coordY - pred_coordY)
coordZ_error_std = np.std(true_coordZ - pred_coordZ)
diameter_mm_error_std = np.std(true_diameter_mm - pred_diameter_mm)
print('====================================Standard deviation  of lung nodule prediction error=======================')
print('Standard deviation of CoordX Error: {:.3f}'.format(coordX_error_std))
print('Standard deviation of CoordY Error: {:.3f}'.format(coordY_error_std))
print('Standard deviation of CoordZ Error: {:.3f}'.format(coordZ_error_std))
print('Standard deviation of Diameter_mm Error: {:.3f}'.format(diameter_mm_error_std))

coordX_error_var = np.var(true_coordX - pred_coordX)
coordY_error_var = np.var(true_coordY - pred_coordY)
coordZ_error_var = np.var(true_coordZ - pred_coordZ)
diameter_mm_error_var = np.var(true_diameter_mm - pred_diameter_mm)
print('=====================================Variance of lung nodule prediction error=================================')
print('Variance of CoordX Error: {:.3f}'.format(coordX_error_var))
print('Variance of CoordY Error: {:.3f}'.format(coordY_error_var))
print('Variance of CoordZ Error: {:.3f}'.format(coordZ_error_var))
print('Variance of Diameter_mm Error: {:.3f}'.format(diameter_mm_error_var))

coordX_MSE = metrics.mean_squared_error(true_coordX, pred_coordX)
coordY_MSE = metrics.mean_squared_error(true_coordY, pred_coordY)
coordZ_MSE = metrics.mean_squared_error(true_coordZ, pred_coordZ)
diameter_mm_MSE = metrics.mean_squared_error(true_diameter_mm, pred_diameter_mm)
print('===========================================MSE of lung nodule prediction=====================================')
print('CoordX Mean Square Error: {:.3f}'.format(coordX_MSE))
print('CoordY Mean Square Error: {:.3f}'.format(coordY_MSE))
print('CoordZ Mean Square Error: {:.3f}'.format(coordZ_MSE))
print('Diameter_mm Mean Square Error: {:.3f}'.format(diameter_mm_MSE))