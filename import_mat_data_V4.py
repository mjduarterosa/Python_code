# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 11:12:21 2016

Reads series of .mat data and saves in single numpy array

@author: mariarosa
"""

# Import packages
import scipy.io
import numpy as np
import glob
import h5py

def import_mat_data( dname):
    # Read mat data
    all_files=glob.glob(dname)
    nf=len(all_files)

    mat=scipy.io.loadmat(all_files[0])
    data_tmp=mat[mat.keys()[2]]
    data=np.zeros((nf,data_tmp.shape[0],data_tmp.shape[1]))

    i=0
    for name in all_files:    
        mat=scipy.io.loadmat(name)
        data[i]=mat[mat.keys()[2]]
        i=i+1
        
    return data
    
def import_mat_data_h5( dname):
        
    all_files=glob.glob(dname)
    nf=len(all_files)

    mat=h5py.File(all_files[0])
    var=mat.items()
    var=var[0][1].value
    data=np.zeros((nf,var.shape[0],var.shape[1]))

    i=0
    for name in all_files:
        mat=h5py.File(name)
        var=mat.items()
        data[i]=var[0][1].value
        i=i+1

    return data
    
    