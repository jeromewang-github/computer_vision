#!/usr/bin/python3
#encoding=utf-8
'''
Bag of visual words (BoVW) is a popular technique for image classification 
inspired by models used in natural language processing. BoVW downplays word 
arrangement (spatial information in the image) and classifies based on a histogram 
of the frequency of visual words. The set of visual words forms a visual vocabulary, 
which is constructed by clustering a large corpus of features.

@author: jerome
@Constact:yunfeiwang@hust.edu.cn
'''

import numpy as np
from Metric import distEuclidean
from sklearn import mixture
from sklearn.cluster import KMeans,MiniBatchKMeans


def BoVWFeatures(dataLearn,dataEncode,nClus,method='kMeans',encode='hard',distfun=distEuclidean):
    '''
    BoVWFeatures(dataLearn,dataEncode,nClus,encode='hard',distfun=distEuclidean)
    @Parameters:
    dataLearn: nSmp*nDim ndarray, low level features
    dataEncode: N*nDim ndarray, low level features to be encoded
    nClus: int, number of word in the visual dictionary
    encode: string 'soft' or 'hard',the way used to encode the faetures
    distfun: metric used to compute distance.(distEuclidean or distCosine)
    @Return:
    ndarray of size (nClus,)
    '''
    #1.Visual vocabulary construction
    print('Clustering for visual word dictionary...')
    centers=BoVWDictionary(dataLearn, nClus, method, distfun)
        
    #2. Extract the BoVW representation for an image
    print('Generating BoVW features...')
    desc=BoVWEncoding(dataEncode, centers, encode, distfun)
    return desc


def BoVWDictionary(dataLearn,nClus,method='kMeans',distfun=distEuclidean):
    '''
    BoVWDictionary(dataLearn,nClus,distfun=distEuclidean)
    @Parameters:
    dataLearn: nSmp*nDim ndarray, low level features
    nClus: int, number of word in the visual dictionary
    distfun: metric used to compute distance.(distEuclidean or distCosine)
    @Return:
    ndarray of size (nClus,nDim)
    '''
    #Visual vocabulary construction
    method=str.lower(method)
    if method not in ('kmeans','gmm'):
        raise ValueError('Invalid method for constructing visual dictionary')
    
    if method=='kmeans':
        nSmp=dataLearn.shape[0]
        if nSmp<3e4:
            km=KMeans(n_clusters=nClus,init='k-means++',n_init=3,n_jobs=-1)#use all the cpus
        else:
            km=MiniBatchKMeans(n_clusters=nClus,init='k-means++',n_init=3)
        km.fit(dataLearn)
        centers=km.cluster_centers_
    else:
        gmm=mixture.GMM(n_components=nClus)
        gmm.fit(dataLearn)
        centers=gmm.means_
    return centers


def BoVWEncoding(dataEncode,centers,encode='hard',distfun=distEuclidean):
    '''
    BoVWEncoding(dataEncode,centers,encode='hard',distfun=distEuclidean)
    Extract the BoVW representation for an image
    @Parameters:
    dataEncode: N*nDim ndarray, low level features to be encoded
    centers: nClus*nDim ndarray, clustering centers as the coodbook in the visual dictionary
    encode: string 'soft' or 'hard',the way used to encode the faetures
    distfun: metric used to compute distance.(distEuclidean or distCosine)
    @Return:
    ndarray of size (nClus,)
    '''
    encode=str.lower(encode)
    if encode not in ('soft','hard'):
        raise ValueError('Invalid encoding scheme(\'soft\' or \'hard\')')
    nClus=centers.shape[0]
    desc=np.zeros(nClus)
    nSmp=dataEncode.shape[0]
    if encode=='hard':
        for i in range(nSmp):
#             dist_iter=map(lambda x,y:(distfun(dataEncode[i],x),y),centers,range(nSmp))
#             nn=max(dist_iter,key=lambda x:x[0])[1]
            dist_iter=map(lambda x:distfun(dataEncode[i],x),centers)
#             nn=np.argmin(list(dist_iter)) #this code do the following things
            dist=np.Inf
            nn=-1
            for (cnt,item) in enumerate(dist_iter):
                if item<dist:
                    dist=item
                    nn=cnt       
            desc[nn]+=1
    else:#soft Assignment
        for i in range(nSmp):
            dist=np.array(list(map(lambda x:np.exp(-distfun(dataEncode[i],x)),centers)))
            dist/=np.sum(dist)
            desc+=dist
    desc/=np.sum(desc)#normalize the histogram
    return desc
    
if __name__=='__main__':
    dataLearn=np.random.rand(100,8)
    dataEncode=np.random.rand(10000,8)
    nClus=10
    desc=BoVWFeatures(dataLearn,dataEncode,nClus,method='GMM',encode='hard')
    print(desc)
