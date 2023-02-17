from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import *
from skimage import measure
import os
from aicspylibczi import *
import pickle
import math
from tqdm import tqdm
import json


def load_json(filename='crop.json'): 
    with open(filename, "r") as io_str:
        data = json.load(io_str)
    return(data)

def convertHSV(image):   
    return np.array(image.convert('HSV'))

def splitChannels(image):
    c1=image[:,:,0]
    c2=image[:,:,1]
    c3=image[:,:,2]
    return c1, c2, c3


def get_bounding_box_from_contour(contours):
    bbs=[]
    for contour in contours:
        bbs.append([contour[:,0].min(), contour[:,1].min(),contour[:,0].max(),contour[:,1].max()])
    return np.array(bbs).astype('int')

def crop_tile(bb, g, size):
    
    center = bb[0]+(bb[2]-bb[0])//2, bb[1]+(bb[3]-bb[1])//2
    xmin = max(0,center[0]-(size//2))
    xmax = min(center[0]+(size//2), g.shape[0])
    ymin = max(0, center[1]-(size//2))
    ymax = min(center[1]+(size//2), g.shape[1])
    return g[xmin:xmax, ymin:ymax], [xmin, ymin,xmax,ymax]

    
def imadjust(x,c,d,gamma=1):
    ''' normalizes the image between c and d '''
    y = (((x - x.min()) / (x.max() - x.min())) ** gamma) * (d - c) + c
    return y