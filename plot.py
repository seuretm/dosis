import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from math import ceil
import pickle

rough_dpcm = int(2044 / 5)
smoothing_size_cm = 0.2
smoothing_size_px = (ceil(smoothing_size_cm*rough_dpcm)//2)*2+1

materials = ['G2-GmundCotton']

paper = {
    'N': 'N-NormalPaper',
    'G1': 'G1-GmundColor',
    'G2': 'G2-GmundCotton',
    'P1': 'P1-Papieroffizin',
    'P2': 'P2-Papieroffizin'
}

acro = {
    'N': 'W1',
    'N1': 'W1',
    'G1': 'W2',
    'G2' : 'C1',
    'P1': 'H1',
    'P2': 'H1'
}

times = {
    '0h': 0,
    '10h': 10,
    '10h-1h': 11,
    '10h-2h': 12,
    '10h-3h': 13,
    '10h-4h': 14,
    '2h': 2,
    '4h': 4
}

def smooth(c, k_size=51):
    k = np.ones(k_size) / k_size
    r = np.convolve(c, k, mode='same')
    for i in range(k_size//2):
        r[i] = np.mean(c[:(i+k_size//2)])
    return r

prev = None
for p in paper:
    for t in times:
        dir_path = os.path.join('crops', paper[p], '%s_%s' % (p, t))
        if not os.path.exists(dir_path):
            print('No folder %s' % dir_path)
            continue
        print(dir_path)
        samples = []
        raw = []
        for filename in sorted(os.listdir(dir_path)):
            leaf = filename.split('_')[-1].split('-')[0]
            if leaf=='00' or not filename.endswith('npy') or not 'sample' in filename:
                continue
            white  = np.load(os.path.join(dir_path, filename.replace('sample', 'white')))
            sample = np.load(os.path.join(dir_path, filename))
            raw.append(sample)
            samples.append(sample / white)
        data = np.stack(samples)
        raw  = np.stack(raw)
        # mean along samples
        data = np.mean(data, axis=0)
        raw  = np.mean(raw,  axis=0)
        # mean along Y axis
        data = np.mean(data, axis=0)
        raw  = np.mean(raw,  axis=0)
        
        # From left to right
        data = np.flip(data, axis=0)
        raw  = np.flip(raw,  axis=0)
        
        cm = [x/rough_dpcm for x in range(data.shape[0])]
        
        # just for now, mean along channels
        #data = np.mean(data, axis=1)
        plt.clf()
        plt.title('Paper %s, hour %d' % (acro[p], times[t]))
        plt.plot(cm, smooth(data[:, 0]), 'r', label='Red channel', linestyle=':')
        plt.plot(cm, smooth(data[:, 1]), 'g', label='Green channel', linestyle='--')
        plt.plot(cm, smooth(data[:, 2]), 'b', label='Blue channel')
        plt.legend(loc="lower right")
        plt.xlabel('Distance to the side [cm]')
        plt.ylabel('Comparison to reference (%)')
        plt.ylim(ymax = 1.025, ymin = 0.925)
        plt.xlim(xmax = 3.5, xmin = 0)
        
        #plt.show()
        plt.savefig(os.path.join('plots', '%s_%d.png' % (p, times[t])))
        
        np.save(os.path.join('plots', 'npy', '%s_%d.npy' % (p, times[t])), data)
        
