import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import minimize, curve_fit
#from scipy.stats import linregress

scan = np.loadtxt("scan.txt")

#print(scan)

bands = [433, 443, 454, 466, 478, 492, 507, 524, 540, 560, 580, 602, 627, 654,
         682, 717, 755, 801, 856, 918, 993, 1095, 1215, 1402, 1670, 2229, 4248]
plt.figure(figsize = (10,10))
plt.imshow(scan, alpha=0.8, cmap='coolwarm', aspect="auto")

intScan = sum(scan)#integrated scan data
TOF = np.loadtxt("TOF.txt")
t = np.linspace(0,5000,5000)


plt.figure(dpi=150)
plt.plot(t,intScan)

#find peaks in raw data
startLook = 4000
#indices = find_peaks(intScan[startLook:], threshold=20)[0]
#indices = [x + startLook for x in indices]
#print(indices)
#plt.plot(indices, [intScan[i] for i in indices], "rx")

#curated peaks
#plt.xlim([2300,5000])
#plt.ylim(0, 1000)
bands = [433, 443, 454, 466, 478, 492, 507, 524, 540, 560, 580, 602, 627, 654,
         682, 717, 755, 801, 856, 918, 993, 1095, 1215, 1402, 1670, 2229, 4248]


plt.plot(bands, [intScan[i] for i in bands], "g.")
plt.xlabel("TOF (ns)")
plt.ylabel("Intensity (arb. units)")

#note m_e = 1 in au
m_e = 9.10953*1e-31
Veff = 1.178185# au or 32.06 #eV
L = 2 #m
Hartree = 4.35974*1e-18 #J is one au energy

bandTOF = [TOF[i]*1e-9 for i in bands]

def kinEnergy(gamma, tTOF):
    return((m_e*(L + gamma)**2/(2*tTOF**2) + Veff*Hartree)/Hartree)

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

plt.figure(dpi=150)
plt.plot(np.linspace(-1, 1, 1000), [bandDiff(g) for g in np.linspace(-1, 1, 1000)])
plt.plot(optimalGamma, bandDiff(optimalGamma), "rx")
plt.yscale("log")
plt.xlabel("gamma factor (m)")
plt.ylabel("calibration score")

plt.figure(dpi=150)
calEnergy = [kinEnergy(optimalGamma,t) for t in TOF*1e-9]
plt.plot(calEnergy, intScan)
plt.xlim(1, 3.5)
plt.plot([float(calEnergy[i]) for i in bands], [intScan[i] for i in bands], "rx")
plt.xlabel("E_kin (au)")
plt.ylabel("Intensity (arb. units)")
#%%

sideBands = [bands[i*2] for i in range(int(len(bands)/2 + 1))]
delay = np.loadtxt("delay.txt") #in fs

def sideBandF(t, a, b, c, omega):
    return(a + b*np.cos(omega*t + c))

sideBandCE = np.array([float(calEnergy[i]) for i in sideBands]) #center energy of each band


sideBandIntensity = []
activeSB = 13
for n in range(50):
    sideBandIntensity.append(float((np.transpose(scan)[sideBands[activeSB] - 1][n] +
                              np.transpose(scan)[sideBands[activeSB]][n] +
                              np.transpose(scan)[sideBands[activeSB] + 1][n])/3))


qIndex = [46 - 2*j for j in range(14)]
omegaQ = 0.057*28

#OBS fit needs a good p0 guess to be accurate

fit = curve_fit(sideBandF, delay, sideBandIntensity, p0 = (3, 2, 0, 0.98), bounds = [0, np.inf])
para = fit[0]
print(para)


plt.figure(dpi=100)
plt.plot(delay, sideBandIntensity, ".")
plt.plot(np.linspace(delay[0], delay[-1], 500), [sideBandF(y, para[0], para[1], para[2], para[3]) for y in np.linspace(0, 50, 500)])

print(fit[0][2])
print(np.sqrt(np.diag(fit[1]))[2])
#side band phases found from the above regresion
#%%
sbPhase = [0.5839685139096004, 5.9459518118470176e-18, 2.7998216661937906e-16, 0.004831329978085194, 0.47054149214539936,
           0.5895038166673301, 0.9239772189004821, 0.937013891261136, 1.389932618549851, 1.5449515245808427,
           1.8557349919270725, 2.255696101490715, 2.4018044336588003, 2.6317644371219098]
sbPhaseErr = [0.31466253656470217, 0.29384742174800565, 0.19517440210571102, 0.17750055014014526, 0.14423812602105737,
              0.09604367666409845, 0.08313459285020776, 0.0669713560537302, 0.08729350912829566, 0.08154113637509512,
              0.095390950932269, 0.11332368946020593, 0.19300789098949825, 0.6775244234739115]
#the last value comes from a very weak signal

def lin(x, a, b):
    return(a + b*x)

phaseDep = curve_fit(lin, sideBandCE, sbPhase, sigma = sbPhaseErr)
linParam = phaseDep[0]
linCov =phaseDep[1]

print(linParam)
print(np.sqrt(np.diag(linCov)))

plt.figure(dpi=150)
plt.errorbar(sideBandCE, sbPhase, sbPhaseErr, fmt = ".")
plt.plot(np.linspace(sideBandCE[0], sideBandCE[-1]), [lin(x, linParam[0], linParam[1])
                                 for x in np.linspace(sideBandCE[0], sideBandCE[-1])])
plt.xlabel("Side band E_kin (au)")
plt.ylabel("Side band Phase")

#%%
#harmonic dependence of the phase
deltaPhase = 0.11571176477495296
deltaPhaseErr = 0.010804025322714456

#%% Renorm plotting
#Create vector from row in scan transforming x according to kinEnergy()

def reCal(row, N, cutoff = (1, 5000)):
    calX = [kinEnergy(-0.35680722, (x+1)*1e-9) for x in range(cutoff[0], cutoff[1])]
    minE = kinEnergy(-0.35680722, cutoff[1]*1e-9)
    maxE = kinEnergy(-0.35680722, cutoff[0]*1e-9)
    bins = np.linspace(minE, maxE, N)
    trimRow = row[cutoff[0]: cutoff[1]]
    
    reBin = []
    for d in range(N):
        reBin.append([0])
    
    binIndx = np.digitize(calX, bins)
    
    for i in range(len(calX)):
        reBin[binIndx[i]].append(trimRow[i])
    
    normReBin = [sum(m)/len(m) for m in reBin]
    return(normReBin, (minE, maxE))
    

calScan = []
N = 300
for r in range(len(scan)):
    calR = reCal(scan[r], N, cutoff = (450, 2500))
    calScan.append(calR[0])
EBound = calR[1]

calScanA = np.array(calScan)
plt.figure(dpi=150)
plt.imshow(calScanA, alpha=0.8, cmap='coolwarm', aspect="auto")
plt.xticks(np.array([0, 0.2, 0.4, 0.6, 0.8, 1])*N, [round(l, 4) for l in np.linspace(EBound[0], EBound[1], 6)])
plt.xlabel("Ekin (a.u.)")
plt.yticks([0, 10, 20, 30, 40 ,50], [round(l, 4) for l in np.linspace(20, 29.8, 6)])
plt.ylabel("delay (fs)")
        
    
   