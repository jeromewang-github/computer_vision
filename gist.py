#!/usr/bin/python3
import os,platform,subprocess
from PIL import Image
import numpy as np

def getSystemInfo():
    sys_str=platform.system()
    arch_str=platform.architecture()[0]
    return sys_str+'_'+arch_str

def GISTFeatures(filename,params='-nblocks 4 -orientationsPerScale 8,8,4',tmpfile='gist_tmp.ppm'):
    if not os.path.exists(filename):
        raise ValueError('File %s does not exists!' %filename)
    pos=filename.rfind('/')
    print(filename[pos+1:])
    if filename[-3:] not in ('ppm','pgm'):
        im=Image.open(filename)
        im.save(tmpfile)
    tmpfilename=os.path.abspath(tmpfile)
    tmpfilename=tmpfilename.replace('\\','/')
    workdir=os.path.abspath('.')
    
    sysinfo=getSystemInfo().lower()
    gist_path='.'
    gist_exe='gist'
    if sysinfo=='windows_64bit':
        gist_path='gist_bin/win64'
    elif sysinfo=='windwos_32bit':
        gist_path='gist_bin/win32'
    elif sysinfo=='linux_64bit':
        gist_path='gist_bin/glnxa64'
    elif sysinfo=='linux_32bit':
        gist_path='gist_bin/glnx32'
    else:
        raise ValueError('Unknown platform %s' %sysinfo)
    
    os.chdir(gist_path)
    cmd_str=' '.join([gist_exe,params,tmpfilename])
    res_str=subprocess.check_output(cmd_str.split())
    gist_fea=np.asarray(res_str.split()).astype('float')
    os.chdir(workdir)
    return gist_fea

if __name__=='__main__':
    filename='bird.jpg'
    gist_fea=GISTFeatures(filename)
    print(gist_fea)
    print(len(gist_fea))