from utils import *
from tiles import *
from aicspylibczi import *
from PIL import Image
import numpy as np
import os 
from skimage.transform import resize
import matplotlib.pyplot as plt
import pickle
import rasterio
import rasterio.features
import rasterio.warp
import json 
from skimage.morphology import binary_dilation, binary_erosion
from plot_functions import single_image_plot
import logging

filenames = [
    '0_6',
    '2_2',
    '4_1',
    '5_1',
]

def read_czi(czi_path, idx, crop_dict):
    # read czi file
    path = os.path.join(czi_path, filenames[idx][0]+"_"+filenames[idx][-1]+".jpg")
    img = Image.open(path)
    im = np.asarray(img).astype(float) 
   
    # split channels 
    _, g, _=splitChannels( im )
    _, s, _=splitChannels( convertHSV(img) )
    
    # crop 
    bb = crop_dict[filenames[idx]]['czi']
    g = g[bb['1']:bb['2'], bb['3']:bb['4']]
    s = s[bb['1']:bb['2'], bb['3']:bb['4']]
    im = im[bb['1']:bb['2'], bb['3']:bb['4']]
    return im, g, s


def read_tif(idx, crop_dict, czi_shape):
    # read tif file
    logging.getLogger("rasterio").setLevel(logging.WARNING)
    path = os.path.join("annotations", filenames[idx]+".tif")
    data = rasterio.open(path)
    red  = data.read(1)
    green= data.read(2)
    blue = data.read(3)
    tif = np.stack((red,green,blue), axis=-1)

    # crop bounding box
    bb = crop_dict[filenames[idx]]['annotation']
    tif = tif[bb['1']:bb['2'], bb['3']:bb['4']]
   
    # resize to full czi shape
    tif = resize(tif, czi_shape)
    # remove the white background ( temporary solution with tifs ) 
    tif = np.where(tif == 255,0,tif)
    
    return tif

def exclusion_mask():
    red  = np.where( red <= red.min()+100, 1, 0 )
    blue = np.where( blue <= blue.min()+100, 1, 0 )
    green  = np.where( green <= green.min()+100, 1, 0 )
    ex = np.logical_and(red, np.logical_and( blue, green ))
    return None



def get_color(im, color='purple'):
    '''select annotations given their color on the tif file
       purple : vessels
       green : roi 
       yellow : ctl
    '''
    r,g,b = splitChannels(im)
    max_red = np.max(r)
    max_green = np.max(g)
    max_blue = np.max(b)

    red = np.where(r==max_red, 1, 0)
    blue = np.where(b==max_blue, 1, 0)
    green = np.where(g==max_green, 1, 0)

    # Get RoI
    if color=='green':
        return np.logical_and(1-blue, np.logical_and(1-red, green))

    # Get Vessels
    if color=='purple':
        return np.logical_and(red, np.logical_and(blue,1-green))

    # Get CT-Lymphocytes
    if color=='yellow':
        return np.logical_and(red, np.logical_and(green, 1-blue))
    
    return None


def roi_mask(tif):
    # roi annotation in 'green' : contours
    roi = get_color(tif, color='green')
    roi = binary_dilation(roi, disk(20))
    roi= 1-roi
    # return the 'inside' : the shape mask
    for i in range(roi.shape[0]):
        mins = np.where(roi[i]== roi[i].min())[0]
        roi[i, : mins[1]] = 0
        roi[i, mins[1]:mins[-1]] = 1
        roi[i, mins[-1]:] = 0
    roi = binary_dilation(roi, disk(10))
    return roi

def vessels_mask(tif):
    # vessels annotations in 'purple'
    mask = get_color(tif, color="purple")
    mask = binary_dilation(mask, disk(2))
    return mask

def apply_mask(im, mask, bg=0):
    '''bg = unselected area color value 0:255'''
    res = np.where(mask, im, bg)
    return res