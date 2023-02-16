import numpy as np 
from tqdm import tqdm
from utils import crop_tile
from plot_functions import implot
from skimage import segmentation, morphology, measure, filters


# def opening_rec(im, size):
#   im2 = morphology.opening(im, morphology.disk(size))
#   return morphology.reconstruction(im2,im, method='dilation')

# def closing_rec(im, size):
#   im2 = morphology.closing(im, morphology.disk(size))
#   return morphology.reconstruction(im2, im, method='erosion')


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
    # remove aberrant structures
    removed = morphology.area_opening(im, max_area)
    im = np.logical_and(im, 1-removed)
    # try recovering holes
    holes=1-morphology.dilation(im,morphology.disk(10))
    out=morphology.area_opening(holes, max_hole)
    holes=morphology.area_opening(holes, max_area)
    holes=np.logical_and(holes,1-out)

    if holes.sum() != 0:   
      return im, holes
    return im, None


def get_vessels_mask( bbs, g, tile_size=256, max_value=100, min_area=100, max_area=1000, max_hole=8000 ):
    ''' get all vessels masks ''' 
    closed_vessels = np.zeros(g.shape)
    opened_from_closed_vessels = np.zeros(g.shape)
    opened_vessels = np.zeros(g.shape)
    new_bbs = []
    bbs = clear_bb_list(bbs, tile_size/2)
    for bb in tqdm(bbs):
        tile_g, new_bb = crop_tile(bb, g, tile_size)
        closed, opened_from_closed = get_closed_vessels_mask( g, max_value, min_area, max_area, max_hole )
        # opened = get_opened_vessels( g, ... )
        
        closed_vessels[new_bb[0]:new_bb[2], new_bb[1]:new_bb[3]] = closed
        opened_from_closed_vessels[new_bb[0]:new_bb[2], new_bb[1]:new_bb[3]] = opened_from_closed
        # opened_vessels[new_bb[0]:new_bb[2], new_bb[1]:new_bb[3]] = opened

        new_bbs.append(new_bb)
    new_bbs, masks = clear_contours(new_bbs, masks)
    return closed_vessels, opened_from_closed_vessels, opened_vessels, new_bbs






# def open_mask(bb, tile_g):
#     tile_g = morphology.opening(tile_g, morphology.disk(1))
#     mask=opening_rec(tile_g, 10)
#     mask = np.where(mask>=(0.9*mask.max()),1,0)

#     mask =morphology.area_opening(mask, 1000)
#     out =morphology.area_opening(mask, 10000)
#     mask = np.logical_and(mask, 1-out)
#     for i in range(2):
#       mask = morphology.binary_dilation(mask, morphology.disk(2))
#       mask = morphology.binary_erosion(mask, morphology.disk(2))
#     mask = morphology.binary_dilation(mask, morphology.disk(2))

#     return mask 


# def open_mask(bb, tile_g):
#     tile_g = morphology.closing(tile_g, morphology.disk(1))
#     mask=opening_rec(tile_g, 10)
#     plt.imshow(mask, cmap='gray')
#     plt.show()

#     # mask = asf(tile_g, N=50, size=3) 
#     # mask = morphology.area_opening(mask, 1000)
#     mask = np.where(mask>=(0.97*mask.max()),1,0)
#     plt.imshow(mask, cmap='gray')
#     plt.show()
#     # mask = segmentation.morphological_geodesic_active_contour(255-tile_g,20, mask,smoothing=1, balloon=0.2)
#     mask =morphology.area_opening(mask, 1000)
#     out =morphology.area_opening(mask, 8000)
#     mask = np.logical_and(mask, 1-out)
   
#     mask = morphology.binary_closing(mask,morphology.disk(5))
#     plt.imshow(mask, cmap='gray')
#     plt.show()
    
#     n = len(measure.find_contours(mask))
#     if n > 3:
#       mask = segmentation.clear_border(mask)
#     return mask 


# Two functions to avoid segmenting the same vessels twice, 
# merge and remove close or duplicated patches



def merge_duplicates(bb1, bb2, th=50):
    if (bb1[2]-bb2[0] <= th) and (bb1[3]-bb2[1] <= th):
      return [min(bb1[0],bb2[0]),min(bb1[1],bb2[1]), max(bb1[2],bb2[2]), max(bb1[3],bb2[3])], 2
    else:
      return bb1, 1

def clear_bb_list(bbs, th=150):
    bb_list = []
    i=0
    while i < len(bbs)-1:
        bb, n = merge_duplicates(bbs[i], bbs[i+1])
        bb_list.append(bb)
        i+=n
    return bb_list
    
    
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
    bbs = clear_bb_list(bbs, th=150)
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
    bbs = clear_bb_list(bbs, th=150)
    for bb in tqdm(bbs):
        tile, new_bb= crop_tile(bb, g, tile_size)
        mask = open_mask(bb, tile)
        final_mask[new_bb[0]:new_bb[2], new_bb[1]:new_bb[3]] = mask
        new_bbs.append(new_bb)
        masks.append(mask)
    new_bbs, masks = clear_contours(new_bbs, masks)
    return final_mask, new_bbs, masks
