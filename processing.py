from scipy.ndimage import gaussian_filter
from skimage import segmentation, morphology, measure, filters
import numpy as np 
from utils import crop_tile
from plot_functions import implot
from tqdm import tqdm
import matplotlib.pyplot as plt 

# from skimage.segmentation import watershed
# from skimage.measure import label, find_contours, regionprops, regionprops_table

# Two functions that respectively recover closed and opened vessels
# inside a givem patch
def asf(img, N=3, size=1):
  im = img.copy()
  for i in range(N):
    im = morphology.opening(im, morphology.disk(size))
    im = morphology.closing(im, morphology.disk(size))

  return im 


def closed_mask(bb, tile_g, tile_s, t=100):
    # im = np.where(tile_s >= t, 1, 0)
    im =  np.where(tile_g < 100, 1, 0)
    im2 = np.where(tile_s > 225, 1, 0 )  
    im = np.logical_or(im, im2)
    im = morphology.binary_dilation(im, morphology.disk(1))
    im = morphology.area_opening(im, 100)

    for i in range(2):
      im = morphology.binary_dilation(im, morphology.disk(2))
      im = morphology.binary_erosion(im, morphology.disk(2))
      im = morphology.binary_dilation(im, morphology.disk(2))
    # im = morphology.binary_dilation(im, morphology.disk(5))

    im = morphology.area_opening(im, 1000)    
    return im


def open_mask(bb, tile_g):

    tile_g_ = morphology.area_opening(tile_g, 2000)
    mask = asf(tile_g, N=50, size=3) 
    mask = np.where(mask>=mask.max()-50 ,1,0)

    mask = segmentation.morphological_geodesic_active_contour(tile_g, 10, mask,smoothing=1, balloon=0.5)

    # mask = morphology.binary_dilation(mask, morphology.disk(2))
    # mask = morphology.binary_erosion(mask, morphology.disk(2))
    # mask = morphology.binary_dilation(mask, morphology.disk(2))
    # mask = morphology.binary_dilation(mask, morphology.disk(2))
    # mask = morphology.binary_erosion(mask, morphology.disk(2))
    # mask = morphology.binary_dilation(mask, morphology.disk(2))

    reconstructed =morphology.area_opening(mask, 2000)
    out =morphology.area_opening(mask, 8000)
    reconstructed = np.logical_and(reconstructed, 1-out)

    return reconstructed 


# Two functions to avoid segmenting the same vessels twice, 
# merge and remove close or duplicated patches
def merge_duplicates(bb1, bb2, th=50):
    if (bb1[2]-bb2[0] <= th) and (bb1[3]-bb2[1] <= th):
      return [min(bb1[0],bb2[0]),min(bb1[1],bb2[1]), max(bb1[2],bb2[2]), max(bb1[3],bb2[3])], 2
    else:
      return bb1, 1

# def clear_bb_list(g, bbs, masks, th=150):
#     final_mask = np.zeros(g.shape)
#     bb_list = []
#     i=0
#     while i < len(bbs)-1:
#         bb, n = merge_duplicates(bbs[i], bbs[i+1])
#         bb_list.append(bb)
#         i+=n

#     return full_mask
    
    
def ctl_clot_mask(im, t, a):
    ''' get global CTL clot mask'''
    im1 = im.copy() 
    mask = np.where(im1 < t, 1, 0)
    mask = morphology.area_opening( morphology.binary_dilation(mask, morphology.disk(1)), area_threshold=a, connectivity=1 )
    mask = morphology.binary_dilation(mask, morphology.disk(6))
    return mask

# remove bbs where we did not find any vessel
def clear_contours(bbs, masks):
    new_bbs=[]
    new_masks=[]
    for i in range(len(bbs)):
        contours = np.array(measure.find_contours(masks[i]),dtype=object)
        if contours.shape[0] != 0:
          new_bbs.append(bbs[i])
          new_masks.append(masks[i])
    # new_bbs=np.array(new_bbs,dtype=object)
    # new_masks=np.array(new_masks,dtype=object)
    return new_bbs,new_masks


def find_closed_vessels(bbs, g, s, tile_size=256):
    '''complete pipeline to return predicted vessels mask in a czi scene'''
    final_mask = np.zeros(g.shape)
    new_bbs = []
    masks = []
    for bb in tqdm(bbs):
        tile_g,new_bb = crop_tile(bb, g, tile_size)
        tile_s ,_ = crop_tile(bb, s, tile_size)

        mask = closed_mask(bb, tile_g, tile_s)
        final_mask[new_bb[0]:new_bb[2], new_bb[1]:new_bb[3]] = mask
        new_bbs.append(new_bb)
        masks.append(mask)
    new_bbs, masks = clear_contours(new_bbs, masks)
    return final_mask, new_bbs, masks

def find_opened_vessels(bbs, g, tile_size=256):
    '''complete pipeline to return predicted vessels mask in a czi scene'''
    final_mask = np.zeros(g.shape)
    new_bbs = []
    masks = []
    for bb in tqdm(bbs):
        tile, new_bb= crop_tile(bb, g, tile_size)
        mask = open_mask(bb, tile)
        final_mask[new_bb[0]:new_bb[2], new_bb[1]:new_bb[3]] = mask
        new_bbs.append(new_bb)
        masks.append(mask)
    new_bbs, masks = clear_contours(new_bbs, masks)
    return final_mask, new_bbs, masks
