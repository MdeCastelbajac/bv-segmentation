from skimage import segmentation
from skimage.morphology import *
from skimage.measure import *
import numpy as np 
from plot_functions import *
from tqdm import tqdm


def crop_tile(bb, g, size):
    center = bb[0]+(bb[2]-bb[0])//2, bb[1]+(bb[3]-bb[1])//2
    xmin = center[0]-size//2 
    xmax = center[0]+size//2 
    ymin = center[1]-size//2 
    ymax = center[1]+size//2 
    return g[xmin:xmax, ymin:ymax], center

def clot_mask(bb, tile_g):
    clot = area_closing(area_opening(tile_g, 400), 400)
    clot = 255 - clot
    mask = area_closing( clot, area_threshold=100)
    
    mask = np.where(mask==mask.max(),1,0)
    mask = binary_dilation(mask,disk(2))
    return mask

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


def find_vessels(bbs, g, tile_size=256):
    '''complete pipeline to return predicted vessels mask in a czi scene'''
    final_mask = np.zeros(g.shape)
    new_bbs = []
    masks = []
    for bb in tqdm(bbs):
        tile, center = crop_tile(bb, g, tile_size)
        print(tile.shape)
        mask = clot_mask(bb, tile)
        final_mask[center[0]-tile_size//2:center[0]+tile_size//2,center[1]-tile_size//2:center[1]+tile_size//2] = mask
        new_bbs.append(np.array((center[0]-tile_size//2, center[1]-tile_size//2, center[0]+tile_size//2,center[1]+tile_size//2)))
        masks.append(mask)
    return final_mask, new_bbs, masks
    