#!/usr/bin/python3
import sys,os
import math
import numpy as np
from PIL import Image


def LBPFeatures(img,radius=1,npoints=8,window_size=(3,3),mode='uniform'):
    """
    LBPFeatures(img,radius=1,npoints=8,window_size=(3,3),mode='normal')
    Extract Local Binary Pattern(LBP) feature for a grey-level image
    Parameters
    ----------
    image:(M,N) ndarray
        input image(greyscale)
    radius:float
        radius of a circle
    npoints:int
        number of sampling points on a circle
    window_size: 2 tuple(int,int)
        number of sampling windows
    mode:'normal','uniform','uniform-ror'
        'normal':the original LBP descriptors(2^P patterns)
        'uniform':extention to LBP with uniform patterns(P(P-1)+2 patterns)
        'uniform-ror':extention to LBP with both rotatin invarient and uniform patterns(P+1 patterns)
    Returns
    ---------
    lbp:ndarray
        LBP feature for the image as a 1D array
    """
    img=np.atleast_2d(img)
    if img.ndim>2:
        raise ValueError('Currently only supports grey-level images')
    if mode not in ('normal','uniform','uniform-ror'):
        raise ValueError("Invalid mode for LBP features:'normal','uniform','uniform-ror'")
    row,col=img.shape
    table=LBP_genHistTable(npoints, mode)
    sorted_keys=sorted(table.keys())
    nDim=len(table) #dimensionality of the histogram
    w_numx,w_numy=window_size
    w_sizex=row//w_numx
    w_sizey=col//w_numy
    if(w_sizex<2*radius or w_sizey<2*radius):
        raise ValueError('Radius is large for the scaning window')
    lbp_hist=np.zeros((w_numx,w_numy,nDim))
    
    binary_pattern=np.zeros(npoints).astype('int')
    coordinate_bias=np.zeros((2,npoints))
    theta=2*math.pi*np.arange(npoints)/npoints
    coordinate_bias[0]=-np.sin(theta)
    coordinate_bias[1]=np.cos(theta)
    coordinate_bias*=radius
    
    eps=1e-15
    val=0 #value for a specific pixel
    for w_rid in range(w_numx):
        for w_cid in range(w_numy):
            w_startx=w_rid*w_sizex
            w_endx=w_startx+w_sizex
            w_starty=w_cid*w_sizey
            w_endy=w_starty+w_sizey
            window=img[w_startx:w_endx,w_starty:w_endy]
            p_startx=int(np.ceil(radius))
            p_endx=w_sizex-1-p_startx
            p_starty=int(np.ceil(radius))
            p_endy=w_sizey-1-p_starty
            for p_rid in range(p_startx,p_endx+1):
                for p_cid in range(p_starty,p_endy+1):
                    binary_pattern[:]=0 #clear the binary pattern
                    for point_id in range(npoints):
                        new_x=p_rid+coordinate_bias[0,point_id]
                        new_y=p_cid+coordinate_bias[1,point_id]
                        
                        new_x_r=np.round(new_x)
                        new_y_r=np.round(new_y)
                        if np.abs(new_x_r-new_x)<eps and np.abs(new_y_r-new_y)<eps:
                            val=window[new_x_r,new_y_r]
                        else: #bi-linarly interpolation for this point
                            x0=int(math.floor(new_x))
                            x1=int(math.ceil(new_x))
                            y0=int(math.floor(new_y))
                            y1=int(math.ceil(new_y))
                            
                            f00=window[x0,y0]
                            f01=window[x1,y0]
                            f10=window[x0,y1]
                            f11=window[x1,y1]
                            val1=(x1-new_x)*f00+(new_x-x0)*f01
                            val2=(x1-new_x)*f10+(new_x-x0)*f11
                            val=(y1-new_y)*val1+(new_y-y0)*val2
                        
                        if val>=window[p_rid,p_cid]:
                            binary_pattern[point_id]=1
                    index=-1
                    if mode=='normal':
                        index=LBP_binary2int(binary_pattern)
                    elif LBP_hopCounter(binary_pattern)<=2:
                        if mode=='uniform-ror':
                            binary_pattern=LBP_rotate4Min(binary_pattern)
                        index=LBP_binary2int(binary_pattern)
                    try:
                        table[index]+=1
                    except KeyError:
                        raise ValueError('Invalid key:%d' %index)
                    
            lbp_hist[w_rid,w_cid]=np.asarray([table[x] for x in sorted_keys])
            lbp_hist[w_rid,w_cid]/=np.sum(lbp_hist[w_rid,w_cid])
            #print([(x,table[x]) for x in sorted(table.keys())])
            for x in table.keys():#clear the data in the histogram
                table[x]=0
    return lbp_hist.flatten()


def LBP_hopCounter(binaryArr):
    sz=binaryArr.size
    cnt=0
    for i in range(1,sz):
        if(binaryArr[i]!=binaryArr[i-1]):
            cnt+=1
    if(binaryArr[0]!=binaryArr[sz-1]):
        cnt+=1
    return cnt


def LBP_rotate4Min(binaryArr):
    sz=binaryArr.size
    res=np.zeros(sz)
    maxPos=0
    maxCnt=0
    start=0
    while(start<sz and binaryArr[start]!=1):
        start+=1
    if start<sz:
        buf=np.zeros(sz).astype('int')
        buf[start]=0
        pos=(start-1+sz)%sz
        while(pos!=start):
            if(binaryArr[pos]==1):
                buf[pos]=0
            else:
                buf[pos]=buf[(pos+1)%sz]+1
            if buf[pos]>maxCnt:
                maxPos=pos
                maxCnt=buf[pos]
            pos=(pos-1+sz)%sz
        validDigit=sz-maxCnt
        copypos=(maxPos-validDigit+sz)%sz
        for idx in range(validDigit):
            res[maxCnt+idx]=binaryArr[(copypos+idx)%sz]
    return res


def LBP_binary2int(binaryArr):
    sz=binaryArr.size
    res=0
    for i in range(sz):
        if binaryArr[sz-i-1]:
            res+=1<<i
    return res


def LBP_genHistTable(npoints,mode):
    table={}
    if mode=='normal':
        table={x:0 for x in range(2**npoints)}
    else:
        keys=[-1,0,2**npoints-1] #-1 for non-uniform patterns
        buf=np.zeros(npoints).astype('int')
        for num in range(1,npoints):
            buf[num-1]=1
            key=-1
            if mode=='uniform-ror':
                pattern=LBP_rotate4Min(buf)
                key=LBP_binary2int(pattern)
                keys.append(key)
            else:#'uniform'
                for i in range(npoints):#shift in a circle
                    key=LBP_binary2int(buf)
                    keys.append(key)
                    buf[i],buf[(i+num)%npoints]=buf[(i+num)%npoints],buf[i] #swap 
        for key in keys:
            table[key]=0
    return table
