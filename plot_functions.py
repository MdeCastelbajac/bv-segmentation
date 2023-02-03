import matplotlib.pyplot as plt
from skimage import measure
import numpy as np
from utils import get_vessels_bounding_box, crop_tile


def implot(images,labels,cmap='gray',figsize=(20,10)):

  rows = int(np.sqrt(len(images)))
  if rows==1:# single image plot
    plt.figure(figsize=figsize)
    fig = plt.imshow(images[0], cmap=cmap)
    plt.axis('off')
    plt.title(labels[0])
    plt.show()
    return
  # multi image plot
  cols = len(images)//rows
  fig, ax = plt.subplots(rows,cols, figsize=figsize)
  for j in range(cols):
    for i in range(rows):
      idx=i+j*rows
      ax[i,j].set_title(labels[idx])
      ax[i,j].imshow(images[idx],cmap=cmap)
      ax[i,j].axis('off')
  plt.show()

  return


def plot_contours(full_mask, vessels, im, color='indianred',figsize=(40,40)):
    contour = measure.find_contours(full_mask)
    # gt = measure.find_contours(vessels)
    im = im.astype('int')
    # global plot
    plt.figure(figsize=figsize)
    plt.imshow(im)
    # for g in gt:
      # x = g[:,0]
      # y = g[:,1]
      # plt.plot(y, x, linewidth=2, color='olive')
    plt.axis('off')
    for c in contour:
      x = c[:,0]
      y = c[:,1]
      plt.plot(y, x, linewidth=5, color=color)
   
    plt.show()
    # fig, ax = plt.subplots(rows,cols, figsize=figsize)
    # for j in range(cols):
    #   for i in range(rows):
    #     idx=i+j*rows
    #     c = contour[idx]
    #     bb = bbs[idx]
    #     tile, new_bb = crop_tile(bb, im, tile_size)
    #     ax[i,j].imshow(tile)

    #     x = c[:,0]-new_bb[0]
    #     y = c[:,1]-new_bb[1]
    #     ax[i,j].plot(y, x, linewidth=2, color=color)
    #     ax[i,j].axis('off')
    # plt.show()
    return
    