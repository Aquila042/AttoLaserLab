import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import minimize 

scan = np.loadtxt("scan.txt")

#print(scan)


#plt.figure(figsize = (10,10))
#plt.imshow(scan, alpha=0.8, cmap='coolwarm', aspect="auto")

intScan = sum(scan)#integrated scan data
TOF = np.loadtxt("TOF.txt")
t = np.linspace(0,5000,5000)


#plt.figure(dpi=150)
#plt.plot(t,intScan)

#find peaks in raw data
startLook = 4000
indices = find_peaks(intScan[startLook:], threshold=20)[0]
indices = [x + startLook for x in indices]
#print(indices)
#plt.plot(indices, [intScan[i] for i in indices], "rx")

#curated peaks
#plt.xlim([2300,5000])
#plt.ylim(0, 1000)
bands = [433, 443, 454, 466, 478, 492, 507, 524, 540, 560, 580, 602, 627, 654,
         682, 717, 755, 801, 856, 918, 993, 1095, 1215, 1402, 1670, 2229, 4248]


#plt.plot(bands, [intScan[i] for i in bands], "g.")

#note m_e = 1 in au
m_e = 9.10953*1e-31
Veff = 1.178185# au or 32.06 #eV
L = 2 #m
Hartre = 4.35974*1e-18 #J is one au energy

bandTOF = [TOF[i]*1e-9 for i in bands]

def kinEnergy(gamma, tTOF):
    return((L + gamma)**2/(2*tTOF**2) - Veff)

#bandDiff = [bands[n+1] - bands[n] for n in range(len(bands)-1)]

def bandDiff(gamma, bandCenter = bandTOF):
    #returns the average deviation from proper normalization of 1.55 eV diff
    #per band as a function of the gamma parameter.
    diff = []
    bandE = [kinEnergy(gamma, b) for b in bandCenter]
    for n in range(len(bandE)-1):
        diff.append(abs(float(bandE[n+1] - bandE[n])))
    #print(diff)
    goodness = abs(sum([0.057 - d for d in diff]))/len(diff)
    return(goodness)

optimalGamma = minimize(bandDiff, -0.5, args=(bandTOF), tol = 1e-10).x
print(optimalGamma, bandDiff(optimalGamma))
#Note, as far as I can tell gamma = -0.35680721

plt.plot(np.linspace(-2, 2, 1000), [bandDiff(g) for g in np.linspace(-2, 2, 1000)])
plt.plot(optimalGamma, bandDiff(optimalGamma), "rx")
plt.yscale("log")

plt.figure(dpi=100)
calEnergy = [kinEnergy(optimalGamma,t) for t in TOF]
plt.plot(calEnergy, intScan)

