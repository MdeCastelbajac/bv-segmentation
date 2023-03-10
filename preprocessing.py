import os 
import json 
import rasterio
import numpy as np
from PIL import Image
from tqdm import tqdm
from aicspylibczi import *
from utils import splitChannels, convertHSV, imadjust
from skimage.morphology import binary_dilation, binary_erosion, disk
from skimage.transform import resize


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
    # crop 
    bb = crop_dict[filenames[idx]]['czi']
    g = g[bb['1']:bb['2'], bb['3']:bb['4']]
    g = imadjust(g, 0, 255)
    im = im[bb['1']:bb['2'], bb['3']:bb['4']]
    # normalize
    for i in range(3):
       im[:,:,i] = imadjust(im[:,:,i], 0, 255)

    return im, g


def read_tif(idx, crop_dict, czi_shape):
    '''opens tif annotations and returns a numpy array'''
    # read tif file
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
    # remove the white background 
    tif = np.where(tif == 255,0,tif)
    
    return tif

def get_color(im, color='purple'):
    '''selects annotations given their color on the tif file
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
    if color == 'green':
      # Get RoI
        return np.logical_and(1-blue, np.logical_and(1-red, green))
      # Get Vessels
    elif color == 'purple':
        return np.logical_and(red, np.logical_and(blue,1-green))
      # Get CT-Lymphocytes
    elif color == 'yellow':
        return np.logical_and(red, np.logical_and(green, 1-blue))
    
    return None


def roi_mask(tif):
    ''' selects ROI mask in a given TIF file ''' 
    # roi annotation in 'green' : contours
    roi = get_color(tif, color='green')
    roi = binary_dilation(roi, disk(20))
    roi= 1-roi
    # return the 'inside' : the shape mask
    # we only have the contour
    for i in range(roi.shape[0]):
        mins = np.where(roi[i]== roi[i].min())[0]
        roi[i, : mins[1]] = 0
        roi[i, mins[1]:mins[-1]] = 1
        roi[i, mins[-1]:] = 0
    roi = binary_dilation(roi, disk(5))
    roi = binary_dilation(roi, disk(5))
    roi = binary_erosion(roi, disk(5))

    return roi

def vessels_mask(tif):
    ''' selects vessels mask in a given TIF file '''
    # vessels annotations in 'purple'
    mask = get_color(tif, color="purple")
    mask = binary_dilation(mask, disk(2))
    return mask


def getScene( czi, scene=0, scale_factor=0.1 ):
    '''returns array from tiles of the scene in the czi file
    '''
    boundingBoxes=czi.get_all_mosaic_tile_bounding_boxes(S=scene)
    minx=float('inf')
    miny=float('inf')
    maxx=-float('inf')
    maxy=-float('inf')
    for key in boundingBoxes:
        boundingBox=boundingBoxes[key]
        x,y,w,h = boundingBox.x, boundingBox.y, boundingBox.w, boundingBox.h
        minx = min(minx, x)
        maxx = max(maxx, x+w)
        miny = min(miny, y)
        maxy = max(maxy, y+h)
    scene=czi.read_mosaic(C=0, region=[minx,miny,maxx-minx,maxy-miny], scale_factor=scale_factor)
    
    return scene[0, :, :, :]


def bgrToRgb( arr ):
    return arr[:,:,::-1]


def splitCziInScenes( path, fileNumber=0 ):
    """saves each czi mosaic scene as its own image.
       path (str), savePath (str), fileNumber (int) -> None
    """
    # Variables 
    fileNumber=str(fileNumber)
    scenesPath=os.path.join('./prat/scenes/', fileNumber)
    czi=CziFile( path )
    sceneNumber=czi.get_dims_shape()[-1]['S'][1]
    for s in tqdm(range(sceneNumber)):
        scene = getScene( czi, scene=s, scale_factor=0.1 )
        scene = bgrToRgb( scene )
        saveImg( scene, scenesPath, id=s )

    return 


def saveImg( arr, path, id=0 ):
    '''saves array to path as jpeg'''
    img = Image.fromarray(arr)
    img.save(os.path.join(path,str(id)+".jpg"))
