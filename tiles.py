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
from skimage.filters import threshold_otsu
from utils import *

def make_tiles(img, mask, tile_size=256):
    '''
    img: np.ndarray with dtype np.uint8 and shape (width, height, channel)
    mask: np.ndarray with dtype np.uint9 and shape (width, height)
    '''
    w_i, h_i = img.shape
    w_m, h_m = mask.shape
    
    pad0, pad1 = (tile_size - w_i%tile_size) % tile_size, (tile_size - h_i%tile_size) % tile_size
    
    padding_i = [[pad0//2, pad0-pad0//2], [pad1//2, pad1-pad1//2]]
    padding_m = [[pad0//2, pad0-pad0//2], [pad1//2, pad1-pad1//2]]
    
    img = np.pad(img, padding_i, mode='constant', constant_values=255)
    img = img.reshape(img.shape[0]//tile_size, tile_size, img.shape[1]//tile_size, tile_size)
    img = img.transpose(0, 2, 1, 3).reshape(-1, tile_size, tile_size)
    
    mask = np.pad(mask, padding_m, mode='constant', constant_values=255)
    mask = mask.reshape(mask.shape[0]//tile_size, tile_size, mask.shape[1]//tile_size, tile_size)
    rows, cols = mask.shape[0], mask.shape[2]
    mask = mask.transpose(0, 2, 1, 3).reshape(-1, tile_size, tile_size)
    
    num_tiles = len(mask)
    return img, mask, rows, cols


def visualize_tiles(img, mask, img_tiles, mask_tiles, rows, cols):
    plt.figure(figsize=(9.5,9.5))
    plt.imshow(img)
    plt.imshow(mask, cmap='gray', alpha=0.9)
    plt.title(f'Scaled Image + Mask\nImage Size: {img.shape}')
    plt.xticks([])
    plt.yticks([])
    plt.show()
    
    fig, ax = plt.subplots(rows, cols)
    num_tiles, size, size2 = img_tiles.shape
    for i in range(rows):
        for j in range(cols):
            img_crop = img_tiles[i*cols+j]
            mask_crop = mask_tiles[i*cols+j]
            ax[i,j].imshow(img_crop)
            ax[i,j].imshow(mask_crop, cmap='gray',alpha=1.)
#             fig.tight_layout()
            ax[i,j].axis('off') 
        fig.suptitle(f'Image + Mask \nNum Tiles: {img_tiles.shape[0]}\nTile Size: {img_tiles.shape[1]}')
        fig.tight_layout()
#         plt.show()
    plt.show()
    
    
    
    
  # def select_tiles_in_image(tiled_image, mask_tiles, keep_dark=True, tile_size=256):
#     '''select tiles not belonging to background'''
#     selected_tiles = []
#     if not keep_dark:
#         mask_tiles=255-mask_tiles
#     for i in range(tiled_image.shape[0]):
#         if mask_tiles[i].sum() <= tile_size**2 / 1.3:
#             selected_tiles.append(tiled_image[i])
            
#     return np.array(selected_tiles)
