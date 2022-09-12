# produces JPG images out of the CR3 files, and crops the relevant parts
# of these images. Stores the crops as numpy arrays.
# Note that this script processes measuresv/*. It expects the paper to
# have been irradiated from the front, not from the side.

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

params = rawpy.Params(user_flip=0, use_camera_wb=True, bright=0.5)

def segment_landscape(fname, out, raw, rgb, rgb2, gray, ox):
    row = np.mean(gray[2100:2300, :],axis=0)
    md = medfilt(row, kernel_size=31)
    diff = np.abs(row-md)
    peaks, _ = find_peaks(diff, distance=100)
    indices = (-diff[peaks]).argsort()[:6]
    vertical_lines = sorted(peaks[indices])

    col = np.mean(gray[:, 2100:2900],axis=1)
    md = medfilt(col, kernel_size=31)
    diff = np.abs(col-md)
    peaks, _ = find_peaks(diff, distance=100)
    indices = (-diff[peaks]).argsort()[:3]
    horizontal_lines = sorted(peaks[indices])

    cx1 = 2*(np.mean(vertical_lines[1:3])/2).astype(np.int32) + ox
    cx2 = 2*(np.mean(vertical_lines[3:5])/2).astype(np.int32) + ox

    cy = 2*(np.mean(horizontal_lines[1:3])/2).astype(np.int32)

    hs = 128
    raw_image = np.array(raw.raw_image)
    
    crop_1 = rgb[(cy-hs):(cy+hs), (cx1-hs):(cx1+hs), :]
    crop_2 = rgb[(cy-hs):(cy+hs), (cx2-hs):(cx2+hs), :]
    
    rgb2[(cy-hs):(cy+hs), (cx1-hs):(cx1+hs), 0] = 0
    rgb2[(cy-hs):(cy+hs), (cx2-hs):(cx2+hs), 1] = 0
    
    np.save('%s-sample.npy' % out, crop_2)
    np.save('%s-comparison.npy' % out, crop_1)
    imageio.imwrite('%s-rgb.jpg' % out, rgb)
    imageio.imwrite('%s-rgb2.jpg' % out, rgb2)

todo = []
for paper_type in ('G2', 'N', 'NX', 'N1'):
    if paper_type=='N1 (vertical)':
        offset_x = 290
    else:
        offset_x = 0
    base = os.path.join('measuresv', paper_type)
    for fold in os.listdir(base):
        src_fold = os.path.join(base, fold, 'Standard_50_dark')
        dst_fold = os.path.join('cropsv', paper_type, fold)
        if not os.path.exists(os.path.join('cropsv', paper_type)):
            os.mkdir(os.path.join('cropsv', paper_type))
        if not os.path.exists(dst_fold):
            os.mkdir(dst_fold)
        for fname in sorted(os.listdir(src_fold)):
            if 'comparison' in fname:
                continue
            if fname.endswith('CR3'):
                no_ext = fname.split('.')[0]
                todo.append((os.path.join(src_fold, fname), os.path.join(dst_fold, no_ext), offset_x))

for task in tqdm(todo):
    fname = task[0]
    out   = task[1]
    ox    = task[2]
    
    if os.path.exists('%s-sample.npy' % out):
        continue # already done :-)
    
    raw = rawpy.imread(fname)
    rgb  = np.array(raw.postprocess(params=params))
    rgb2 = np.array(raw.postprocess(params=params))
    gray = np.mean(rgb, axis=2)
    
    segment_landscape(fname, out, raw, rgb, rgb2, gray, ox)
