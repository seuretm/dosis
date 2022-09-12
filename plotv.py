from matplotlib.pyplot import figure
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from math import ceil
import pickle
plt.rcParams.update({'font.size': 18})

fin = open('final/dat.dat', 'rb')

rough_dpcm = int(2044 / 5)
smoothing_size_cm = 0.2
smoothing_size_px = (ceil(smoothing_size_cm*rough_dpcm)//2)*2+1

materials = ['G2', 'N1', 'N']

paper = {
    #'N': 'Normal paper',
    'N1': 'Normal paper',
    #'G1': 'G1-GmundColor',
    'G2': 'G2-GmundCotton',
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
    'C1': 'Cotton, machine-made',
    'C2': 'Cotton, hand-crafted'
}

color = {
    'N': '#1f77b4',
    #'N1': '#ff7f0e',
    'N1': '#1f77b4',
    'G1': '#2ca02c',
    'G2': '#d62728',
    'P1': '#9467bd',
    'W1':  '#8c564b',
    'C2':  '#ff8c00'
}
for a in acro:
    if a in color:
        color[acro[a]] = color[a]

times = {
    '0h': 0,
    # ~ '2h': 2,
    # ~ '4h': 4,
    # ~ '0h-4h': 4,
    '10h': 10,
    '10h-1h': 11,
    '10h-2h': 12,
    '10h-3h': 13,
    '10h-4h': 14,
}

def get_rgb(sample):
    mr = np.mean(sample[:, :, 0])
    mg = np.mean(sample[:, :, 1])
    mb = np.mean(sample[:, :, 2])
    return [mr, mg, mb]

all_together = {}
xtra2 = {}

prev = None
for p in paper:
    print(p)
    measures = {}
    compare = {}
    for t in times:
        dir_path = os.path.join('cropsv', p, '%s_%s' % (p, t))
        if not os.path.exists(dir_path):
            print('No folder %s' % dir_path)
            continue
        print('Entering', dir_path)
        for filename in sorted(os.listdir(dir_path)):
            leaf = filename.split('_')[-1].split('-')[0]
            if leaf=='00' or not filename.endswith('npy') or not 'sample' in filename:
                continue
            leaf = int(leaf)
            white  = np.load(os.path.join(dir_path, filename.replace('sample', 'comparison')))
            sample = np.load(os.path.join(dir_path, filename))
            white = get_rgb(white)
            sample = get_rgb(sample)
            if not leaf in measures:
                measures[leaf] = {}
                compare[leaf] = {}
            measures[leaf][times[t]] = sample
            compare[leaf][times[t]]  = white
    
    
    for leaf in measures:
        r = []
        g = []
        b = []
        t = []
        for tt in measures[leaf]:
            t.append(tt)
            r.append(100*measures[leaf][tt][0] / compare[leaf][tt][0])
            g.append(100*measures[leaf][tt][1] / compare[leaf][tt][1])
            b.append(100*measures[leaf][tt][2] / compare[leaf][tt][2])
        r = np.array(r)
        g = np.array(g)
        b = np.array(b)
        
        if leaf==1:
            all_together[acro[p]] = (t, (r+g)/2 / b)
        
print(all_together)

plt.clf()
fig,ax = plt.subplots()
plt.title('Yellowing coefficient F over time')
for l in all_together:
    plt.plot(all_together[l][0], all_together[l][1], color=color[l], label=full[l])
    print(l, color[l])
t = pickle.load(fin)
y = pickle.load(fin)
yy = []
tt = []
n = 0
for ttt in [int(x) for x in t]:
    if ttt==0 or ttt>=10:
        tt.append(ttt)
        yy.append(y[n])
    n += 1
print(yy)
plt.plot(tt, yy, color=color['C2'], label=full['C2'])
plt.axvline(x=10, color='black')
ax.text(7.3, 1.00, 'X-ray on')
ax.text(10.5, 1.00, 'X-ray off')

#plt.ylim(ymax = 1.08, ymin = 0.99)
plt.legend(loc="upper left", prop={'size': 15})
plt.xlabel('Irradiation time in hours')
plt.ylabel('F := mean(red, green) / blue')
plt.subplots_adjust(left=0.15, right=0.85, top=0.90, bottom=0.13)
fig.set_size_inches(10, 6)
plt.savefig(os.path.join('plotv', 'F_all_together.png'), dpi=1200)
