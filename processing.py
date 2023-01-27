from skimage import segmentation
from skimage.morphology import *
from skimage.measure import *
import numpy as np 
from plot_functions import *
from skimage.segmentation.morphsnakes import disk_level_set
from tqdm import tqdm
from skimage.segmentation import watershed
from skimage.feature import peak_local_max, canny
from scipy import ndimage as ndi
from skimage.filters import rank
from skimage.util import img_as_ubyte
from skimage import exposure
from skimage.measure import label, find_contours, regionprops, regionprops_table

def crop_tile(bb, g, size):
    center = bb[0]+(bb[2]-bb[0])//2, bb[1]+(bb[3]-bb[1])//2
    xmin = max(0,center[0]-size//2)
    xmax = min(center[0]+size//2, g.shape[0])
    ymin = max(0, center[1]-size//2)
    ymax = min(center[1]+size//2, g.shape[1])
    return g[xmin:xmax, ymin:ymax], [xmin, ymin,xmax,ymax]


def closed_mask(bb, tile_s, tile_g, t=120):
    im = np.where(tile_s >= min(t, tile_s.max()), 1, 0)
    im2 =  np.where(tile_g < 100, 1, 0)
    im = np.logical_or(im, im2)
    single_image_plot(im)
    im = binary_dilation(im, disk(1))
    im = area_opening(im, 100)
    for i in range(2):
      im = binary_dilation(im, disk(2))
      im = binary_erosion(im, disk(2))
      im = binary_dilation(im, disk(1))
    im = area_opening(im, 500)    
    label_im = label(im)
    regions = regionprops(label_im)
    return im, regions


def clear_mask(bb, tile_g):
    tile_g_ = area_opening(tile_g, 2000)
    tile_g_ = area_closing(tile_g_, 2000)
  
    mask = np.where(tile_g_>=tile_g_.max()-50 ,1,0)
    mask = binary_erosion(mask,disk(5))

    reconstructed = segmentation.morphological_geodesic_active_contour(tile_g, 10, mask,smoothing=1, balloon=0.5)
    reconstructed = binary_dilation(reconstructed, disk(2))

    out =area_opening(reconstructed, 10000)
    reconstructed = np.logical_and(reconstructed, 1-out)

    return reconstructed

def is_duplicate(bb1, bb2, th=50):
    return (bb1[2]-bb2[0] <= th) and (bb1[3]-bb2[1] <= th)

def clear_bb_list(bbs, th=150):
    '''remove duplicates'''
    bb_list = []
    i=0
    while i < len(bbs)-1:
        bb_list.append(bbs[i])
        if is_duplicate(bbs[i], bbs[i+1], th):
            i+=1
        i+=1
    return bb_list
    
    
def ctl_clot_filter(im, t, a):
    im1 = im.copy() 
    mask = np.where(im1 < t, 1, 0)
    mask = area_opening( binary_dilation(mask, disk(1)), area_threshold=a, connectivity=1 )
    mask = binary_dilation(mask, disk(6))
    return mask

def clear_contours(bbs, masks):
    new_bbs=[]
    new_masks=[]
    for i in range(len(bbs)):
        contours = np.array(find_contours(masks[i]))
        if contours.shape[0] == 0:
            continue
        new_bbs.append(bbs[i])
        new_masks.append(masks[i])
    return np.array(new_bbs), np.array(new_masks)

def find_closed_vessels(bbs, s,g, tile_size=256):
    '''complete pipeline to return predicted vessels mask in a czi scene'''
    final_mask = np.zeros(s.shape)
    new_bbs = []
    masks = []
    for bb in tqdm(bbs):
        tile, new_bb = crop_tile(bb, s, tile_size)
        tile_g,_ = crop_tile(bb, g, tile_size)
        mask, regions = closed_mask(bb, tile, tile_g)
        final_mask[new_bb[0]:new_bb[2], new_bb[1]:new_bb[3]] = mask
        new_bbs.append(new_bb)
        masks.append(mask)
    new_bbs, masks = clear_contours(new_bbs, masks)
    return final_mask, new_bbs, masks

def find_clear_vessels(bbs, g, tile_size=256):
    '''complete pipeline to return predicted vessels mask in a czi scene'''
    final_mask = np.zeros(g.shape)
    new_bbs = []
    masks = []
    for bb in tqdm(bbs):
        tile, new_bb= crop_tile(bb, g, tile_size)
        mask, regions = clear_mask(bb, tile)
        final_mask[new_bb[0]:new_bb[2], new_bb[1]:new_bb[3]] = mask
        new_bbs.append(new_bb)
        masks.append(mask)
    new_bbs, masks = clear_contours(new_bbs, masks)
    return final_mask, new_bbs, masks