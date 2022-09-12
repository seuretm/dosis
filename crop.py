# produces JPG images out of the CR3 files, and crops the relevant parts
# of these images. Stores the crops as numpy arrays.
# Note that this script processes measures/*. It expects the paper to
# have been irradiated from the side, not from the front.

import matplotlib
import os
import matplotlib.pyplot as plt
import imageio
import rawpy
import numpy as np
import cv2 as cv

from tqdm import tqdm
from scipy.signal import medfilt
from scipy.signal import find_peaks

def find_n_peaks(arr, qty):
    md   = medfilt(arr, kernel_size=31)
    diff = np.abs(arr-md)
    peaks, _ = find_peaks(diff, distance=100)
    indices = (-diff[peaks]).argsort()[:qty]
    return sorted(peaks[indices])

def analyze(gray):
    v = find_n_peaks(np.mean(gray[1600:1700,:], axis=0), 6)
    
    w = 200
    cx = (v[3]+v[4])//2
    h = find_n_peaks(np.mean(gray[:, (cx-w):(cx+w)], axis=1), 3)
    
    return {
        'first':  [v[1], v[2]],
        'second': [v[3], v[4]],
        'vertical': [h[1], h[2]]
    }

def get_areas(a):
    strip = 0

def produce_crop(neutral, gray, rgb, out):
    a = analyze(gray)
    right = a['second'][1]
    w = 200
    top = a['vertical'][0]
    bot = a['vertical'][1]
    mid = (top+bot)//2
    ox = 15
    #res = rgb[(mid-w):(mid+w), (right-1500-ox):(right-ox), :] - neutral[(mid-w):(mid+w), (right-1500-ox):(right-ox), :]
    neu = neutral[(mid-w):(mid+w), (right-1500-ox):(right-ox), :]
    res = rgb[(mid-w):(mid+w), (right-1500-ox):(right-ox), :]# - neu
    np.save('%s-sample.npy' % out, res)
    
    white = np.mean(np.mean(np.mean(np.stack([rgb[(top+ox):(top+ox+2*w), (right-ox-2*w):(right-ox), :], rgb[(bot-ox-2*w):(bot-ox), (right-ox-2*w):(right-ox), :]]), axis=0), axis=0), axis=0)# - np.mean(np.mean(neu, axis=0), axis=0)
    np.save('%s-white.npy' % out, white)
    
    imageio.imwrite('%s-rgb.jpg' % out, rgb)
    rgb[(mid-w):(mid+w), (right-1500-ox):(right-ox), 0] = 0
    imageio.imwrite('%s-rgb2.jpg' % out, rgb)

def produce_top_crop(neutral, gray, rgb, out):
    a = analyze(gray)
    left  = a['second'][0]
    right = a['second'][1]
    w = 200
    top = a['vertical'][0]
    mid = (left+right)//2
    ox = 15
    neu = np.flip(neutral[(top+ox):(top+1500+ox), (mid-w):(mid+w), :].transpose((1, 0, 2)), axis=1)
    res = np.flip(rgb[(top+ox):(top+1500+ox), (mid-w):(mid+w), :].transpose((1, 0, 2)), axis=1)# - neu
    np.save('%s-sample.npy' % out, res)
    
    white = np.mean(np.mean(np.mean(np.stack([rgb[(top+300):(top+300+2*w), (left+ox):(left+ox+2*w), :], rgb[(top+ox):(top+ox+2*w), (right-ox-2*w):(right-ox), :]]), axis=0), axis=0), axis=0)# - np.mean(np.mean(neu, axis=0), axis=0)
    np.save('%s-white.npy' % out, white)
    
    imageio.imwrite('%s-rgb.jpg' % out, rgb)
    rgb[(top+ox):(top+1500+ox), (mid-w):(mid+w), 0] = 0
    imageio.imwrite('%s-rgb2.jpg' % out, rgb)


if not os.path.exists('crops'):
    os.mkdir('crops')

todo = []
for paper_type in ('G1-GmundColor', 'N-NormalPaper', 'G2-GmundCotton', 'P1-Papieroffizin', 'P2-Papieroffizin'):
    base = os.path.join('measures', paper_type)
    for fold in os.listdir(base):
        src_fold = os.path.join(base, fold, 'Standard_50_dark')
        dst_fold = os.path.join('crops', paper_type, fold)
        if not os.path.exists(os.path.join('crops', paper_type)):
            os.mkdir(os.path.join('crops', paper_type))
        if not os.path.exists(dst_fold):
            os.mkdir(dst_fold)
        for fname in sorted(os.listdir(src_fold)):
            if 'comparison' in fname:
                continue
            if fname.endswith('CR3'):
                no_ext = fname.split('.')[0]
                todo.append((os.path.join(src_fold, fname), os.path.join(dst_fold, no_ext)))

need_top_crop = set(['G2_10h', 'G2_10h-1h', 'G2_10h-2h', 'G2_10h-3h', 'G2_10h-4h'])

params = rawpy.Params(user_flip=0, use_camera_wb=True, bright=0.5)
dz_path = None
for task in tqdm(todo):
    fname = task[0]
    out   = task[1]
    
    if fname.endswith('_00.CR3'):
        dz_path = fname
        neutral = None
    
    if os.path.exists('%s-sample.npy' % out):
        continue # already done :-)
    
    if neutral is None:
        neutral = rawpy.imread(fname).postprocess(params=params)
    
    raw = rawpy.imread(fname)
    rgb  = np.array(raw.postprocess(params=params))
    gray = np.mean(rgb, axis=2)
    if out.split('/')[2] in need_top_crop:
        produce_top_crop(neutral, gray, rgb, out)
    else:
        produce_crop(neutral, gray, rgb, out)
    
