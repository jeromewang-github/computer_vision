#!/usr/bin/python3
import os,platform,subprocess
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def getSystemInfo():
    sys_str=platform.system()
    arch_str=platform.architecture()[0]
    return sys_str+'_'+arch_str


def SIFTFeatures(filename,outfile,params='--edge-thresh 10 --peak-thresh 5'):
    if not os.path.exists(filename):
        raise ValueError('File %s does not exists!' %filename)
    if filename[-3:]!='pgm':#must use the image in pgm format
        img=Image.open(filename).convert('L')
        img.save('tmp.pgm')
        tmpfilename=os.path.abspath('tmp.pgm')
    workdir=os.path.abspath('.')
    siftdir='.'
    sysinfo=getSystemInfo().lower()
    exe_str='sift'
    if sysinfo=='windows_64bit':
        siftdir='sift_bin/win64'
    elif sysinfo=='windows_32bit':
        siftdir='sift_bin/win32'
    elif sysinfo=='linux_64bit':
        siftdir='sift_bin/glnxa64'
    elif sysinfo=='linux_32bit':
        siftdir='sift_bin/glnx86'
    elif sysinfo=='darwin_64bit':
        siftdir='sift_bin/maci64'
    elif sysinfo=='darwin_32bit':
        siftdir='sift_bin/maci'
    else:
        raise ValueError('Unknown platform %s' %sysinfo)
    os.chdir(siftdir)
    call_cmd=str(exe_str+' '+tmpfilename+' --output='+outfile+' '+params)
    #os.system(call_cmd)
    subprocess.call(call_cmd)
    sift_fea=np.genfromtxt(outfile,dtype=np.float)
    location=sift_fea[:,:4] #(row,col,scale,orientation of each feature
    sift=sift_fea[:,4:]
    pos=filename.rfind(r'/')
    print(filename[pos+1:])
    os.chdir(workdir)
    return sift,location


def plot_features(img,locs,circle=False):
    """
    Show image with features.
    locs:row,col,scale,orientation of each feature
    """
    def draw_circle(pos,radius):
        t=np.arange(0,1.01,.01)*2*np.pi
        x=radius*np.cos(t)+pos[0]
        y=radius*np.sin(t)+pos[1]
        plt.plot(x,y,'b',linewidth=2)
        
    plt.figure()
    plt.gray()
    plt.imshow(img)
    if circle:
        for p in locs:
            draw_circle(p[:2],p[2])
    else:
        plt.plot(locs[:,0],locs[:,1],'ob')
    plt.axis('off')
    plt.show()

    
if __name__=='__main__':
    filename='25.jpg'
    img=Image.open(filename)
    img=img.convert('L')
    im=np.array(img)
    fea,locs=SIFTFeatures(filename,'tmp')
    plot_features(im,locs,False)
