import numpy as np
import matplotlib.pyplot as plt

vasicStepFile = "/mnt/local/CMSPIX28/Scurve/data/ChipVersion1_ChipID9_SuperPix2/2025.03.21_16.28.18_MatrixCalibration_vMin0.001_vMax0.400_vStep0.01000_nSample32.000_vdda0.900_VTH0.800_BXCLK10.00/vasic_steps.npy"
analyzeFile = "/mnt/local/CMSPIX28/Scurve/data/ChipVersion1_ChipID9_SuperPix2/2025.03.21_16.28.18_MatrixCalibration_vMin0.001_vMax0.400_vStep0.01000_nSample32.000_vdda0.900_VTH0.800_BXCLK10.00/plots/scurve_data.npz"

v = np.load(vasicStepFile)
x = np.load(analyzeFile)
s = x["scurve"]

fig, ax = plt.subplots(figsize=(6,6))
pixel = 0
bit = 2
nSettingsToPlot = 500 # to plot all settings use nSettingsToPlot = s.shape[2]
plt.plot(v, s[pixel, bit][:nSettingsToPlot].T)
plt.show()