import matplotlib
import pickle
import os
import json
import pickle
import numpy as np
import colour

from tqdm import tqdm
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

jspath = os.path.join('..', '..', 'dosis', 'cropsahd', 'measures.json')

def get_rgb(sample):
    mr = np.mean(sample[:, :, 0])
    mg = np.mean(sample[:, :, 1])
    mb = np.mean(sample[:, :, 2])
    return [mr, mg, mb]

leaves = [str(n) for n in range(1, 13)]

if not os.path.exists(jspath):
    times = {
        '0 h before Measurement': 0,
        '2 h': 2,
        '4 h': 4,
        '6 h': 6,
        '8 h': 8,
        '10 h standard': 10,
        '10 h and 1 h After Measurement standard': 11,
        '10 h and 2 h after Measurement standard': 12
    }
    leaves = set([x for x in range(1, 13)])

    measures = {}

    for timefolder in tqdm(times):
        y = []
        for fname in os.listdir(os.path.join('..', '..', 'dosis', 'cropsahd', timefolder)):
            if not fname.endswith('-sample.npy'):
                continue
            n = int(fname.split('(')[1].split(')')[0])
            if n==0:
                continue
            sample = np.load(os.path.join('..', '..', 'dosis', 'cropsahd', timefolder, fname))
            right  = np.load(os.path.join('..', '..', 'dosis', 'cropsahd', timefolder, fname.replace('-sample.npy', '-comparison.npy')))
            
            sr, sg, sb = get_rgb(sample)
            rr, rg, rb = get_rgb(right)
            
            if not n in measures:
                measures[n] = {}
            measures[n][times[timefolder]] = ([get_rgb(sample),get_rgb(right)])
    json.dump(measures, open(jspath, 'wt'), indent=2)
    print('Data computed')
else:
    measures = json.load(open(jspath, 'rt'))
    print('Data loaded')


out  = open('last/dat.dat', 'wb')
leafwise = []
#for leaf_n in ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10'):
for leaf_n in ('1', ): # for the paper, we want only he plot for page 1
    t = [time for time in measures[leaf_n]]
    r = []
    g = []
    b = []
    for time in measures[leaf_n]:
        r.append(measures[leaf_n][time][0][0] / measures[leaf_n][time][1][0] * 100)
        g.append(measures[leaf_n][time][0][1] / measures[leaf_n][time][1][1] * 100)
        b.append(measures[leaf_n][time][0][2] / measures[leaf_n][time][1][2] * 100)
    plt.clf()
    
    fig,ax = plt.subplots()
    ax.plot(t, r, color='red', label='Red', linestyle=':')
    ax.plot(t, g, color='green', label='Green', linestyle='--')
    ax.plot(t, b, color='blue', label='Blue')
    #ax.ylim(ymax = 100, ymin = 93)
    
    bx=ax.twinx()
    bx.plot(t, (np.array(r)+g)/2 / b, color='#ff8c00', label='F')
    print(leaf_n)
    if leaf_n=='1':
        print('Leaf 1 dumped')
        pickle.dump(t, out)
        pickle.dump((np.array(r)+g)/2 / b, out)
    leafwise.append(((np.array(r)+g)/2 / b)[5])
    
    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = bx.get_legend_handles_labels()
    lines = lines_1 + lines_2
    labels = labels_1 + labels_2
    ax.legend(lines, labels, loc=6, prop={'size': 16})
    
    ax.set_ylabel('Sample color intensity (%)')
    ax.set_xlabel('Irradiation time in hours')
    bx.set_ylabel('F := mean(red, green) / blue')
    
    #plt.legend(loc="center left")
    plt.xlabel('Irradiation time in hours')
    #plt.ylabel('Sample color intensity (%)')
    plt.title('Color gradient over time, hand-crafted cotton paper')
    plt.axvline(x=5, color='black')
    ax.text(2.6, 99.5, 'X-ray on')
    ax.text(5.5, 99.5, 'X-ray off')
    
    #bx.set_ylim(ymax = 1.06, ymin = 1.00)
    
    plt.subplots_adjust(left=0.15, right=0.85, top=0.94, bottom=0.13)
    fig.set_size_inches(10, 6)
    
    plt.savefig(os.path.join('last', 'leaf-%d-color.png' % int(leaf_n)), dpi=1200)
    #plt.show()
pickle.dump(leafwise, open('last/leaves.dat', 'wb'))


for time in t:
    r = []
    g = []
    b = []
    l = []
    ms = sorted([int(m) for m in measures])
    for i, leaf_n in enumerate(ms):
        leaf_n = str(leaf_n)
        r.append(measures[leaf_n][time][0][0] / measures[leaf_n][time][1][0] * 100)
        g.append(measures[leaf_n][time][0][1] / measures[leaf_n][time][1][1] * 100)
        b.append(measures[leaf_n][time][0][2] / measures[leaf_n][time][1][2] * 100)
        l.append(i+1)
    r = np.array(r)
    g = np.array(g)
    b = np.array(b)
    
    plt.clf()
    fig,ax = plt.subplots()
    ax.plot(l, r, color='red', label='Red', linestyle=':')
    ax.plot(l, g, color='green', label='Green', linestyle='--')
    ax.plot(l, b, color='blue', label='Blue')
    
    bx=ax.twinx()
    bx.plot(l, (r+g)/2 - b, color='black', label='F')
    
    if time=='10':
        pickle.dump(l, out)
        pickle.dump((r+g)/2 - b, out)
        print('Second dump')
    
    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = bx.get_legend_handles_labels()
    lines = lines_1 + lines_2
    labels = labels_1 + labels_2
    ax.legend(lines, labels, loc=0)
    
    #plt.legend(loc="lower right")
    ax.set_ylim(ymax = 100, ymin = 93)
    bx.set_ylim(ymax = 4, ymin = 2)
    bx.set_yticks([2, 2.5, 3, 3.5, 4])
    
    plt.title('Paper C2')
    plt.xlabel('Leaf position in the stack')
    ax.set_ylabel('Sample color intensity (%)')
    bx.set_ylabel('F := mean(red, green) - blue')
    plt.savefig(os.path.join('last', 'time-%d-leaves.png' % int(time)))
    #plt.show()

out.close()
