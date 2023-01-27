import matplotlib.pyplot as plt
from skimage import measure
from processing import crop_tile

def  preprocessing_plot(tif, g, h, roi, vessels, clipped_g, clipped_h):
    
    fig, ax = plt.subplots(2,3)
    ax[0,0].imshow(g, cmap='gray')
    ax[0,0].axis('off')
    ax[0,0].title.set_text('green channel')
    ax[1,0].imshow(h,cmap='gray')
    ax[1,0].axis('off')
    ax[1,0].title.set_text('hue channel')
    ax[0,1].imshow(clipped_g,cmap='gray')
    ax[0,1].axis('off')
    ax[0,1].title.set_text('clipped G channel')
    ax[1,1].imshow(clipped_h,cmap='gray')
    ax[1,1].axis('off')
    ax[1,1].title.set_text('clipped S channel')
    ax[0,2].imshow(roi,cmap='gray')
    ax[0,2].axis('off')
    ax[0,2].title.set_text('RoI mask')
    ax[1,2].imshow(vessels,cmap='gray')
    ax[1,2].axis('off')
    ax[1,2].title.set_text('Vessels mask')

 
    plt.show()
    return

def single_image_plot(im, title=''):
    plt.figure()
    fig = plt.imshow(im, cmap='gray')
    plt.axis('off')
    plt.title(title)
    plt.show()
    return


def contour_plot(masks, bbs, tif, im, tiles):
    fig, ax=plt.subplots(2,tiles//2,figsize=(20,10))
    for i in range(0,tiles,2):
        contour = measure.find_contours(masks[i])
        contour2 = measure.find_contours(masks[i+1])
        bb = bbs[i]
        bb2 = bbs[i+1]
        tile = im[bb[0]:bb[2], bb[1]:bb[3]]
        tile2 = im[bb2[0]:bb2[2], bb2[1]:bb2[3]]
        a = tif[bb[0]:bb[2], bb[1]:bb[3]]
        a2 = tif[bb2[0]:bb2[2], bb2[1]:bb2[3]]
        # ax[0, i%tiles//2].imshow(a, cmap='bone', alpha=0.5)
        # ax[1, i%tiles//2].imshow(a2, cmap='bone', alpha=0.5)
        ax[0, i%tiles//2].imshow(tile.astype('int'),alpha=0.8)
        ax[1, i%tiles//2].imshow(tile2.astype('int'),alpha=0.8)
 
        ax[1, i%tiles//2].axis('off')
        ax[0, i%tiles//2].axis('off')
        if contour != []:
          contour = contour[0]
          x = contour[:,0]
          y = contour[:,1]
          ax[0, i%tiles//2].plot(y, x, linewidth=2, color='indianred')
        if contour2 != []:
          contour2 = contour2[0]
          x2 = contour2[:,0]
          y2 = contour2[:,1]
          ax[1, i%tiles//2].plot(y2, x2, linewidth=2, color='indianred')
       
       
       
    plt.show()
    
    return
    
def plotChannels(c1, c2, c3):
    fig, ax=plt.subplots(1,3,figsize=(20,20))
    ax[0].imshow(c1, cmap='gray')
    ax[1].imshow(c2, cmap='gray')
    ax[2].imshow(c3, cmap='gray')
    plt.show()
    
    