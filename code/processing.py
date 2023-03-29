from itertools import filterfalse
import numpy as np
from scipy.stats.distributions import truncexpon_gen 
from tqdm import tqdm
from utils import crop_tile, imadjust
from plot_functions import implot
from skimage import segmentation, morphology, measure, filters



def closing_rec(im, size):
  im2 = morphology.closing(im, morphology.disk(size))
  return morphology.reconstruction(im2, im, method='erosion')


def get_closed_vessels_mask(g, max_value=100, min_area=100, max_area=1000, max_hole=8000):
    im =  np.where(g < max_value, 1, 0)
    im = morphology.binary_dilation(im, morphology.disk(1))

    # remove small structures
    im = morphology.area_opening(im, min_area)
    # remove small holes
    for i in range(2):
      im = morphology.binary_dilation(im, morphology.disk(2))
      im = morphology.binary_erosion(im, morphology.disk(2))
      im = morphology.binary_dilation(im, morphology.disk(2))
    im = morphology.binary_dilation(im, morphology.disk(2))

    # remove aberrant structures
    removed = morphology.area_opening(im, max_area)
    im = np.logical_and(im, 1-removed)

    # try recovering holes
    holes=1-morphology.dilation(im,morphology.disk(10))
    out=morphology.area_opening(holes, max_hole)
    holes=np.logical_and(holes,1-out)
    holes=segmentation.clear_border(holes)
    if holes.sum==im.size:
        holes=np.zeros(g.shape)
    return im, holes



def get_opened_vessels_mask( g, factor, min_area, max_area ):
    g = morphology.opening(g, morphology.disk(1))
    g = imadjust(g,0,255)
    g =closing_rec(g, 10)
    thresh = factor * g.max()
    mask = np.where(g>=thresh,1,0)
    mask =morphology.area_opening(mask, min_area)
    out =morphology.area_opening(mask, max_area)
    mask = np.logical_and(mask, 1-out)
    for i in range(2):
      mask = morphology.binary_dilation(mask, morphology.disk(2))
      mask = morphology.binary_erosion(mask, morphology.disk(2))
    mask = morphology.binary_dilation(mask, morphology.disk(2))
    mask = segmentation.clear_border(mask)
    return mask 



def get_vessels_mask( bbs, g, tile_size=256, max_value=100, min_area=100, max_area=1000, factor=0.8):
    ''' get all vessels masks ''' 
    closed_vessels = np.zeros(g.shape)
    opened_from_closed_vessels = np.zeros(g.shape)
    opened_vessels = np.zeros(g.shape)
    new_bbs = []
    bbs = clear_bb_list(bbs, tile_size)
    for bb in tqdm(bbs):
        tile_g, new_bb = crop_tile(bb, g, tile_size)
        closed, opened_from_closed = get_closed_vessels_mask( np.where(tile_g<=5,255,tile_g), max_value, min_area, max_area )
        opened = get_opened_vessels_mask( np.where(tile_g==255,0,tile_g), factor, min_area, max_area )
        
        closed_vessels[new_bb[0]:new_bb[2], new_bb[1]:new_bb[3]] = closed
        opened_from_closed_vessels[new_bb[0]:new_bb[2], new_bb[1]:new_bb[3]] = opened_from_closed
        opened_vessels[new_bb[0]:new_bb[2], new_bb[1]:new_bb[3]] = opened

        new_bbs.append(new_bb)
    opened_vessels = opened_vessels - morphology.area_opening(opened_vessels, max_area)
    return closed_vessels, opened_from_closed_vessels, opened_vessels, new_bbs



def merge_duplicates(bb1, bb2, th=50):
    if (np.abs(bb1[2]-bb2[0]) <= th) and (np.abs(bb1[3]-bb2[1]) <= th):
      return [min(bb1[0],bb2[0]),min(bb1[1],bb2[1]), max(bb1[2],bb2[2]), max(bb1[3],bb2[3])], True
    else:
      return bb1, False

def clear_bb_list(bbs, th):
    bb_list = []
    i=0
    while i < len(bbs)-1:
        cpt = 1
        bb, merged = merge_duplicates(bbs[i], bbs[i+cpt], th=th)
        while merged:
          cpt+=1
          if (i+cpt)>=len(bbs):
            merged = False
            break
          bb, merged = merge_duplicates(bb, bbs[i+cpt], th=th)
        if not merged:
          bb_list.append(bb)
          i+=cpt
    return bb_list
    
    
def get_init_mask(im, t, a, low_ctl=filterfalse):
    ''' get global CTL or Lumen mask'''
    im1 = im.copy()
    if low_ctl:
      im = np.where(im==255, 0, im)
      im1 = np.where(im1==255, 0, im1)
      im1 = 255-im1
    else :
      im = np.where(im<=10, 255, im)
      im1 = np.where(im1<=10, 255, im1)

    mask = np.where(im1 < t, 1, 0)
    mask = morphology.area_opening( mask, area_threshold=a, connectivity=1 )
    contours = measure.find_contours(mask)
    return mask, contours, im
