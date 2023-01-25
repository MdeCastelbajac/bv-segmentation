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


def load_json(): 
    with open("crop.json", "r") as io_str:
        data = json.load(io_str)
    return(data)

def convertHSV(image):   
    return np.array(image.convert('HSV'))

def splitChannels(image):
    c1=image[:,:,0]
    c2=image[:,:,1]
    c3=image[:,:,2]
    return c1, c2, c3

    
def load_json(): 
    with open("crop.json", "r") as io_str:
        data = json.load(io_str)
    return(data)

def get_vessels_bounding_box(contours):
    bbs=[]
    for contour in contours:
        bbs.append([contour[:,0].min(), contour[:,1].min(),contour[:,0].max(),contour[:,1].max()])
    return np.array(bbs).astype('int')


def binaryThreshold(image, threshold=127, highPass=True):
    if highPass:
        return np.where( image>threshold, 1, 0 )
    return np.where( image<=threshold, 1, 0 )

def invertImage(image):
    return np.max(image)-image

# should be for a given component, if several
def edgeLength(image):
    return np.size( measure.find_contours( image, 0 ))

def surface(image):
    return np.shape( np.where(image) )[1]

def smoothness(image):
    return surface(image)/edgeLength(image)

def asf( image, N=3 ):
    for i in range(N):
        strel=disk(N+1)
        image=opening( closing( image, strel ), strel)
    return image

def bgrToRgb( arr ):
    return arr[:,:,::-1]
    
def saveImg( arr, path, id=0 ):
    '''saves array to path as jpeg'''
    img = Image.fromarray(arr)
    img.save(os.path.join(path,str(id)+".jpg"))

def getScene( czi, scene=0, scale_factor=0.1 ):
    '''returns array from tiles of the scene in the czi file'''
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


def fillBackgroundG(im):
    # find bg value
    bgValue = 0
    i=0
    while im[i,i]==0 or im[i,i]==255 or im[i,i]!=im[i+1,i+1]:
        i+=1
    bgValue = im[i,i]
    # fill with bg value
    im=np.where(im<=50, bgValue, im)
    im=np.where(im>=bgValue+10, bgValue, im)
    return im, bgValue
