import matplotlib.pyplot as plt
from skimage import measure
import numpy as np
from utils import get_bounding_box_from_contour, crop_tile


def implot(images,labels,cmap='gray',figsize=(20,10), rows=2):
  # single image plot
  if type(images) == np.ndarray:
    plt.figure(figsize=figsize)
    fig = plt.imshow(images, cmap=cmap)
    plt.axis('off')
    plt.title(labels)
    plt.show()
    return
  # multi image plot
  cols = len(images)//rows
  fig, ax = plt.subplots(rows,cols, figsize=figsize)
  if cols == 1:
    for i in range(rows):
      idx=i
      ax[i].set_title(labels[idx])
      ax[i].imshow(images[idx],cmap=cmap)
      ax[i].axis('off')
    plt.show()
    return
  elif rows == 1:
    for i in range(cols):
      idx=i
      ax[i].set_title(labels[idx])
      ax[i].imshow(images[idx],cmap=cmap)
      ax[i].axis('off')
    plt.show()
    return
  for j in range(cols):
    for i in range(rows):
      idx=i+j*rows
      ax[i,j].set_title(labels[idx])
      ax[i,j].imshow(images[idx],cmap=cmap)
      ax[i,j].axis('off')
  plt.show()

  return


def plot_contours(vessels, im, color='indianred',figsize=(20,20)):
    contours = [measure.find_contours(v) for v in vessels]
    plt.figure( figsize=figsize )
    plt.imshow(im.astype('int'))
    plt.axis('off')
    for i in range(3):
      contour = contours[i]
      for c in contour:
        x = c[:,0]
        y = c[:,1]
        plt.plot(y, x, linewidth=2, color=color[i])
    plt.show()
    return
    