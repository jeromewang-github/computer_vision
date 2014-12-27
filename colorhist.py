#!/usr/bin/python3
import numpy as np

def colorHistogram(img,nRange=4,window_sz=(5,5)):
    img=np.atleast_2d(img)
    if img.ndim!=3:
        raise ValueError('Currently only supports RGB images')
    row,col,_=img.shape
    dim=nRange**3
    base=256/nRange
    num_wx,num_wy=window_sz
    sz_x=row//num_wx
    sz_y=col//num_wy
    nElem_per_win=sz_x*sz_y
    res=np.zeros((num_wx,num_wy,dim))
    for w_rid in range(num_wx):
        for w_cid in range(num_wy):
            start_x=w_rid*sz_x
            end_x=start_x+sz_x
            start_y=w_cid*sz_y
            end_y=start_y+sz_y
            window=img[start_x:end_x,start_y:end_y,:]
            for p_rid in range(sz_x):
                for p_cid in range(sz_y):
                    val=window[p_rid,p_cid]
                    #x=int(val[0]/base)
                    #y=int(val[1]/base)
                    #z=int(val[2]/base)
                    x,y,z=map(lambda x:int(x/base),val)
                    index=x*nRange*nRange+y*nRange+z
                    res[w_rid,w_cid,index]+=1
            res[w_rid,w_cid]/=nElem_per_win
    return res.flatten()
