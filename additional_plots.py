import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from math import ceil
import pickle
plt.rcParams.update({'font.size': 18})

rough_dpcm = int(2044 / 5)
smoothing_size_cm = 0.2
smoothing_size_px = (ceil(smoothing_size_cm*rough_dpcm)//2)*2+1

materials = ['G2-GmundCotton']

paper = {
    'N': 'N-NormalPaper',
    'G1': 'G1-GmundColor',
    'G2': 'G2-GmundCotton',
    'P1': 'P1-Papieroffizin'
    #'P2': 'P2-Papieroffizin'
}

times = {
    '0h': 0,
    '2h': 2,
    '4h': 4,
    '10h': 10,
    '10h-1h': 11,
    '10h-2h': 12,
    '10h-3h': 13,
    '10h-4h': 14
}

acro = {
    'N': 'W1',
    'N1': 'W1',
    'G1': 'W2',
    'G2' : 'C1',
    'P1': 'H1',
    'P2': 'H1'
}

full = {
    'W1': 'Wood, standard copy paper',
    'W2': 'Wood, machine-made',
    'C1': 'Cotton, machine-made',
    'C2': 'Cotton, hand-crafted',
    'H1': 'Hemp and Linen, hand-crafted'
}

color = {
    'N': '#1f77b4',
    'N1': '#ff7f0e',
    'G1': '#2ca02c',
    'G2': '#d62728',
    'P1': '#9467bd',
    'X':  '#8c564b',
    'Y':  '#e377c2'
}
for a in acro:
    if a in color:
        color[acro[a]] = color[a]

fig,ax = plt.subplots()

def smooth(c, k_size=51):
    k = np.ones(k_size) / k_size
    r = np.convolve(c, k, mode='same')
    for i in range(k_size//2):
        r[i]  = np.mean(c[:(i+k_size//2)])
        r[-i] = np.mean(c[-((i+k_size//2)+1):])
    r2 = medfilt(r, 3)
    return r2
    

for p in paper:
    base = np.load(os.path.join('plots', 'npy', '%s_%d.npy' % (p, times['0h'])))
    all_times = {}
    for t in times:
        if t!='10h' and t!='0h':
            continue
        
        try:
            sample = np.load(os.path.join('plots', 'npy', '%s_%d.npy' % (p, times[t])))
        except:
            print('No file %s' % os.path.join('plots', 'npy', '%s_%d.npy' % (p, times[t])))
            continue
        
        if t=='0h':
            white = sample
            for c in range(3):
                white[:, c] -= np.min(white[:, c])
        
        if t!='10h':
            continue
        
        sample -= white
        
        for c in range(3):
            sample[:, c] = smooth(sample[:, c])
        cm = [x/rough_dpcm*10 for x in range(sample.shape[0])]
        
        r = sample[:, 0]
        g = sample[:, 1]
        b = sample[:, 2]
        plt.plot(cm, (r+g)/2 / b, color=color[p], label='%s' % full[acro[p]])
        print(p, color[p])

plt.legend(loc="upper right")

plt.title('Yellowing coefficient F, 10 h irradiation')
plt.xlabel('Distance to sample side in mm')
plt.ylabel('F := mean(red, green) / blue')
fig.set_size_inches(10, 6)
plt.ylim(ymax = 1.05, ymin = 0.99)
plt.xlim(xmax = 35, xmin = 0)
plt.savefig(os.path.join('plots', 'y_all_norm2_10.png'), dpi=1200)
