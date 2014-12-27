#!/usr/bin/python3
import numpy as np


def HOGFeatures(img,numGrad=9,cellSize=(16,16),blockSize=(3,3),normalise=False):
    img=np.atleast_2d(img)
    if img.ndim>2:
        raise ValueError('Currently only supports grey-level images')

    if img.dtype.kind=='u':#convert uint image to float
        img=img.astype('float')
    """
    The first stage applies an optional global image normalisation(sqrt or log)
    equalisation that is designed to reduce the influence of illumination
    effects. In practice we use gamma (power law) compression, either
    computing the square root or the log of each colour channel.
    Image texture strength is typically proportional to the local surface
    illumination so this compression helps to reduce the effects of local
    shadowing and illumination variations.
    """
    if normalise:
        img=np.sqrt(img)

    """
    The second stage computes first order image gradients with centered operators[-1,0,1] and [-1,0,1]^T.
    These capture contour, silhouette and some texture information, while providing
    further resistance to illumination variations. The locally dominant
    colour channel is used, which provides colour invariance to a large
    extent. Variant methods may also include second order image derivatives,
    which act as primitive bar detectors - a useful feature for capturing,
    e.g. bar like structures in bicycles and limbs in humans.
    """
    row,col=img.shape#size of the image
    conx_data=np.zeros((row,col))
    cony_data=np.zeros((row,col))
    for cid in range(1,col-1):#horizonal gradient with operator [-1,0,1]
        conx_data[:,cid]=img[:,cid+1]-img[:,cid-1]
    for rid in range(1,row-1):#vertical gradient with operator [-1,0,1]^T
        cony_data[rid,:]=img[rid+1,:]-img[rid-1,:]
    con_data=np.sqrt(conx_data**2+cony_data**2) #magnitude of gradient
    angle_data=np.abs(np.arctan2(cony_data,conx_data))%np.pi#undirected
                
##    newImg=Image.new('L',(row,col),0)
##    for rid in range(0,row):
##        for cid in range(0,col):
##            newImg.putpixel((rid,cid),con_data[rid,cid])
##    newImg.show()
##    newImg.save('test.jpg','JPEG')

    """
    The third stage aims to produce an encoding that is sensitive to
    local image content while remaining resistant to small changes in
    pose or appearance. The adopted method pools gradient orientation
    information locally in the same way as the SIFT [Lowe 2004]
    feature. The image window is divided into small spatial regions,
    called "cells". For each cell we accumulate a local 1-D histogram
    of gradient or edge orientations over all the pixels in the
    cell. This combined cell-level 1-D histogram forms the basic
    "orientation histogram" representation. Each orientation histogram
    divides the gradient angle range into a fixed number of
    predetermined bins. The gradient magnitudes of the pixels in the
    cell are used to vote into the orientation histogram.
    """
    cellsizex,cellsizey=cellSize
    blocksizex,blocksizey=blockSize #number of cells a block contains
    n_cellx=row//cellsizex #number of cells along x-axis
    n_celly=col//cellsizey #number of cells along y-axis
    hogWidth=n_celly-blocksizey+1
    hogHeight=n_cellx-blocksizex+1
    hogFea=np.zeros((hogWidth*hogHeight,blocksizex*blocksizey*numGrad))

    base=np.pi/numGrad #the width of the range of angle split
    angle_index=np.floor(angle_data/base).astype('int') #generate the index for each orientation
    
    #histgram of gradient for each cell
    con_cell_data=np.zeros((n_cellx,n_celly,numGrad))
    for rid in range(n_cellx):
        for cid in range(n_celly):
            start_x=rid*cellsizex
            end_x=start_x+cellsizex
            start_y=cid*cellsizey
            end_y=start_y+cellsizey
            cell_angle_index=angle_index[start_x:end_x,start_y:end_y]
            cell_con=con_data[start_x:end_x,start_y:end_y]
            for ori_index in range(numGrad):
                pos=np.where(cell_angle_index==ori_index,cell_angle_index,-1)
                cell_con_filter=np.where(pos>-1,cell_con,0)
                con_cell_data[rid,cid,ori_index]=np.sum(cell_con_filter)#sum of all elements
    #collect histogram of gradient across cells
    blockcnt=0
    for block_rid in range(hogHeight):
        for block_cid in range(hogWidth):
            b_startx=block_rid
            b_endx=b_startx+blocksizex
            b_starty=block_cid
            b_endy=b_starty+blocksizey
            block_con=con_cell_data[b_startx:b_endx,b_starty:b_endy,:]
            hogFea[blockcnt]=block_con.flatten()#cascade of HoG for the cells in a block
            blockcnt+=1
    """
    The fourth stage computes normalisation with L2-norm, which takes local groups of
    cells and contrast normalises their overall responses before passing
    to next stage. Normalisation introduces better invariance to illumination,
    shadowing, and edge contrast. It is performed by accumulating a measure
    of local histogram "energy" over local groups of cells that we call
    "blocks". The result is used to normalise each cell in the block.
    Typically each individual cell is shared between several blocks, but
    its normalisations are block dependent and thus different. The cell
    thus appears several times in the final output vector with different
    normalisations. This may seem redundant but it improves the performance.
    We refer to the normalised block descriptors as Histogram of Oriented
    Gradient (HOG) descriptors.
    """
    eps = 1e-5
    for rid in range(hogHeight*hogWidth):
        denominator = np.sqrt(np.sum(hogFea[rid] ** 2) + eps)
        hogFea[rid] = hogFea[rid] / denominator

    return hogFea.flatten()
