#!/usr/bin/python3
#encoding=utf-8
'''
Vector of Locally Aggregated Descriptors (VLAD) image encoding is a feature
encoding and pooling method. VLAD is similar to Fisher vectors but (i) it does not store 
second-order information about the features and (ii) it typically use KMeans instead of 
GMMs to generate the feature vocabulary (although the latter is also an option). 
VLAD encodes a set of local feature descriptors I=(x1,…,xn) extracted from an image using a 
dictionary built with a clustering method such as Gaussian Mixture Models (GMM)
or K-means clustering. Let qik be the strength of the association of data vector
xi to cluster μk, such that q_{ik}≥0 and \sum_k(q_{ik})=1. The association may be either 
soft (e.g. obtained as the posterior probabilities of the GMM clusters) or 
hard (e.g. obtained by vector quantization with K-means).
μk are the cluster means, vectors of the same dimension as the data xi. VLAD encodes 
feature x by considering the residuals v_k=\sum_{i=1}^N q_{ik}(xi−μk).
The residulas are concatenated together to obtain the vector.

VLAD normalization
VLFeat VLAD implementation supports a number of different normalization strategies. 
These are optionally applied in this order:
Component-wise mass normalization. Each vector vk is divided by the total mass of features
 associated to it \sum_i q_{ik}.
Square-rooting. The function sign(z)sqrt(|z|)is applied to all scalar components of the VLAD 
descriptor.
Component-wise l2 normalization. The vectors vk are divided by their norm ∥v_k∥_2.
Global l2 normalization. The VLAD descriptor Φ^(I) is divided by its norm ∥Φ^(I)∥_2.

Ref. H. Jegou, M. Douze, C. Schmid, and P. Perez. Aggregating local descriptors into 
a compact image representation. In Proc. CVPR, 2010.
L. Liu, L. Wang, and X. Liu. In defense of soft-assignment coding.In ICCV, pages 2486–2493, 2011.

@author: jerome
@Constact:yunfeiwang@hust.edu.cn
'''

import numpy as np
from sklearn.cluster import KMeans,MiniBatchKMeans
from sklearn import mixture
from Metric import distEuclidean


def VLADFeatures(dataLearn,dataEncode,nClus,method='kMeans',encode='hard',distfun=distEuclidean,normalize=0):
    """
    VLADFeatures(dataLearn,dataEncode,nClus,method='kMeans',encode='hard',distfun=distEuclidean,normalize=0)
    @Parameters:
    dataLearn: M*N ndarray,each row is a sample.
    dataEncode:P*N ndarray,each row is a sample,which is the data to be encoded.
    nClus: number of viusual words in the visual word dictinary.
    method: string, 'kMeans' or 'GMM',clustering algorithm used to create visual dictionary
    encode: hard or soft, used to assign each sample to cluster centers in hard or soft way.
    distfun: metric used to compute distance.(distEuclidean or distCosine)
    normalize: the normalization method
        0-Component-wise mass normalization. 
        1-Square-rooting.
        2-Component-wise l2 normalization.
        3-Global l2 normalization.
    @Return:
        (k*dim,) ndarray, feature encoded in VLAD
    """
    #1.Learn visual word dictionary with k-Means or GMM
    print("Clustering for visual word dictionary...")
    centers=VLADDictionary(dataLearn, nClus, method, distfun)
    print('Generating VLAD features...')
    vlad=VLADEncoding(dataEncode,centers,encode,distfun,normalize)
    return vlad


def VLADDictionary(dataLearn,nClus,method='kMeans',distfun=distEuclidean):
    """
    VLADDictionary(dataLearn,nClus,method='kMeans',encode='hard',distfun=distEuclidean)
    @Parameters:
    dataLearn: M*N ndarray,each row is a sample.
    nClus: number of viusual words in the visual word dictinary.
    method: string, 'kMeans' or 'GMM',clustering algorithm used to create visual dictionary
    encode: hard or soft, used to assign each sample to cluster centers in hard or soft way.
    distfun: metric used to compute distance.(distEuclidean or distCosine)
    @Return:
        (k,dim) ndarray, cluster centers treated as codebook in viusual dictionary
    """
    method=str.lower(method)
    if method not in ('kmeans','gmm'):
        raise ValueError('Invalid clustering method for constructing visual dictionary')
    
    centers=None
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


def VLADEncoding(dataEncode,centers,encode='hard',distfun=distEuclidean,normalize=0):
    """
    VLADFeatures(dataLearn,dataEncode,nClus,method='kMeans',encode='hard',distfun=distEuclidean,normalize=0)
    @Parameters:
    dataEncode:P*N ndarray,each row is a sample,which is the data to be encoded.
    centers: nClus*nDim ndarray, clustering centers as the coodbook in the visual dictionary
    encode: hard or soft, used to assign each sample to cluster centers in hard or soft way.
    distfun: metric used to compute distance.(distEuclidean or distCosine)
    normalize: the normalization method
        0-Component-wise mass normalization. 
        1-Square-rooting.
        2-Component-wise l2 normalization.
        3-Global l2 normalization.
    @Return:
        (k*dim,) ndarray, feature encoded in VLAD
    """
    if encode not in('hard','soft'):
        raise ValueError('Invalid value for VQ(hard or soft)')
    if dataEncode.ndim==1:
        dataEncode=dataEncode[:,np.newaxis]
    nSmp,nDim=dataEncode.shape
    vlad=np.zeros((nClus,nDim))#VLAD descriptors
    
    if encode=='hard':
        #2.Vector quantization with hard or soft Assignment
        vq=np.zeros(nSmp)
        for idx in range(nSmp):
            mindist=np.Inf
            nn=-1
            dist_iter=map(lambda x:distfun(dataEncode[idx],x),centers)
            for (cnt,dist) in enumerate(dist_iter):
                if dist<mindist:
                    mindist=dist
                    nn=cnt
            vq[idx]=nn
        #3.Accumulate the residuals between descriptors and cluster centers 
        for i in range(nClus):
            idx=vq==i
            data_diff=dataEncode[idx]-centers[i]
            vlad[i]=np.sum(data_diff,axis=0)
    else:#VQ='soft'
        #2.Vector quantization with hard or soft Assignment
        vq=np.zeros((nSmp,nClus))
        for idx in range(nSmp):
            vq[idx]=np.array(list(map(lambda x:np.exp(-distfun(dataEncode[idx],x)),centers)))
            vq[idx]/=np.sum(vq[idx])
        #3.Accumulate the residuals between descriptors and cluster centers 
        for k in range(nClus):
            diff_data=dataEncode-centers[k]
            for i in range(nSmp):
                diff_data[i]*=vq[i,k]
            vlad[k]=np.sum(diff_data,axis=0)
    #4.Normalize the finish the final encoding procedure
    if normalize==0:
        #Each vector vk is divided by the total mass of features associated to it \sum_i q_{ik}
        for i in range(nClus):
            totalmass=sum(vq==i)
            vlad[i]/=totalmass
    elif normalize==1:#Apply sign(z)sqrt(|z|)is applied to all scalar components
        vlad=np.sign(vlad)*np.sqrt(np.abs(vlad))
    elif normalize==2:#Vectors vk are divided by their norm ∥v_k∥_2.
        for i in range(nClus):
            vlad[i]=vlad[i]/np.sqrt(np.sum(vlad[i]**2))
    elif normalize==3:#Component-wise l2 normalization.
        vlad/=np.sqrt(np.sum(vlad**2))
    else:
        raise ValueError('Invalid normalization option.')
        
    return vlad.flatten()

        
if __name__=='__main__':
    dataLearn=np.random.rand(1000,1)
    dataEncode=np.random.rand(100,1)
    nClus=5
    vlad=VLADFeatures(dataLearn,dataEncode,nClus,method='kmeans',encode='soft',normalize=3)
    print(vlad)
