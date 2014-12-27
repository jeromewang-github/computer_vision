#!/usr/bin/python3
#encoding=utf-8
'''
The Fisher Vector(FV) is an extension of the BOV representation， obtained by 
pooling local image features.It is frequently used as a global image descriptor 
in visual classification. Instead of characterizing an image by the number of 
occurrences of each visual word, it is characterized by a gradient vector derived 
from a generative probabilistic model(Gaussian Mixture Model). The gradient of the 
log-likelihood describes the contribution of the parameters to the generation process.
@author: jerome
@Constact:yunfeiwang@hust.edu.cn
'''

import numpy as np
from sklearn import mixture

EPS=np.finfo(float).eps

def FVFeatures(dataLearn,dataEncode,nClus,normalize=0):
    '''
    Extracting Fisher vector descriptors for given data.
    FVFeatures(dataLearn,dataEncode,nClus,distfun=distEuclidean,normalize=0)
    @Parameters:
    dataLearn: n*dim ndarray,used to train a GMM model
    dataEncode: m*dim ndarray,data to be encoded
    nClus: number of cluster centers in GMM
    normalize: normalization strategy 
        0-Power normalization. 
        Applying the function sqrt(|z|)signz to each dimension of the vector Φ(I). 
        Other additive kernels can also be used at an increased space or time cost.
        1-L2 Normalization. 
        Before using the representation in a linear model (e.g. SVM), the vector Φ(I) is 
        further normalized by the l2 norm (note that the standard Fisher 
        vector is normalized by the number of encoded feature vectors).
    '''
    print('Clustering for visual word dictionary..')
    gmm=FV_trainGMM(dataLearn,nClus)
    print('Generating Fisher Vector features...')
    fv_desc=FVEncoding(dataEncode, gmm.weights_, gmm.means_, gmm.covars_, normalize)
    return fv_desc


def FV_trainGMM(dataLearn,nClus):
    #1.model dictionary of visual words with GMM
    gmm=mixture.GMM(n_components=nClus)
    gmm.fit(dataLearn)
    return gmm

def FVEncoding(dataEncode,gmm,normalize=0):
    '''
    Encoding the given data with fisher vector descriptors.
    FVEncoding(dataEncode,Phi,Mu,Sigma,distfun=distEuclidean,normalize=0)
    @Parameters:
    dataEncode: m*dim ndarray,data to be encoded
    gmm: Gaussian Mixture Model trained
    normalize: normalization strategy 
        0-Power normalization. 
        Applying the function sqrt(|z|)signz to each dimension of the vector Φ(I). 
        Other additive kernels can also be used at an increased space or time cost.
        1-Normalization. 
        Before using the representation in a linear model (e.g. SVM), the vector Φ(I) is 
        further normalized by the l2 norm (note that the standard Fisher 
        vector is normalized by the number of encoded feature vectors).
    '''
    if normalize not in (0,1):
        raise ValueError('Invalid normalization option')
    
    if dataEncode.ndim==1:
        dataEncode=dataEncode[:,np.newaxis]
    nSmp,nDim=dataEncode.shape
    Phi,Mu,Sigma=gmm.weights_, gmm.means_, gmm.covars_
    nClus=len(Phi)
    prob=np.empty((nSmp,nClus))
    for i in range(nClus):
        prob[:,i]=-0.5*(np.sum((dataEncode**2)/Sigma[i],axis=1)
                        +np.sum((Mu[i]**2)/Sigma[i])
                        -2*np.sum(dataEncode*Mu[i]/Sigma[i],axis=1)
                        +nDim*np.log(2*np.pi)+np.sum(np.log(Sigma[i])))
        
    prob+=np.log(Phi)
    prob=np.exp(prob)
    for i in range(nSmp):
        prob[i]/=np.sum(prob[i])
        
    fv_desc=np.empty((nDim,2*nClus))
    for i in range(nClus):
        t=((dataEncode-Mu[i])/Sigma[i]).T
        fv_desc[:,2*i]=np.sum(t*prob[:,i],axis=1)/(nSmp*np.sqrt(Phi[i]))
        fv_desc[:,2*i+1]=np.sum((t**2-1)*prob[:,i],axis=1)/(nSmp*np.sqrt(2*Phi[i]))
        
    fv_desc=fv_desc.flatten()
    
    if normalize==0:
        fv_desc=np.sign(fv_desc)*np.sqrt(np.abs(fv_desc))
    elif normalize==1:
        fv_desc/=np.sum(fv_desc**2)
    return fv_desc
        

if __name__=='__main__':
    nSmp=1000
    nDim=3
    dataLearn=np.random.rand(nSmp,nDim)
    fv=FVFeatures(dataLearn, dataLearn,3,1)
    print(fv)