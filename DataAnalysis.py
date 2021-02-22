import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

scan = np.loadtxt("scan.txt")
#print(scan)


#plt.figure(figsize = (10,10))
#plt.imshow(scan, alpha=0.8, cmap='coolwarm', aspect="auto")

intScan = sum(scan)#integrated scan data
TOF = np.linspace(0,5000,5000)

plt.figure(dpi=100)
plt.plot(TOF,intScan)

#find peaks in raw data
indices = find_peaks(intScan, threshold=70)[0]
print(indices)
plt.plot(indices, [intScan[i] for i in indices], "rx")