# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 13:05:44 2016

@author: mariarosa
"""
import numpy as np
import import_mat_data
import csv

# Import files

dname_UK='/Users/mariarosa/Documents/Work/pain_data/all_data/CorrelationMatrices_V5/UK_Cam_V5/ROICorrelation_*'
dname_JP='/Users/mariarosa/Documents/Work/pain_data/all_data/CorrelationMatrices_V5/JP_CiN_V5/ROICorrelation_*'
dname_US='/Users/mariarosa/Documents/Work/pain_data/all_data/CorrelationMatrices_V5/US_OP_V5/ROICorrelation_*'

data_UK=import_mat_data.import_mat_data( dname_UK)
data_JP=import_mat_data.import_mat_data( dname_JP)
data_US=import_mat_data.import_mat_data( dname_US)

P_UK=import_mat_data.glob.glob(dname_UK)
P_UK=[i for i in range(len(P_UK)) if (P_UK[i][-6:-5]=='P') ]

P_JP=import_mat_data.glob.glob(dname_JP)
P_JP=[i for i in range(len(P_JP)) if (P_JP[i][-7:-6]=='P') ]

P_US=import_mat_data.glob.glob(dname_US)
P_US=[i for i in range(len(P_US)) if (P_US[i][-7:-6]=='P') ]

tmp=data_UK[0]
tmp=tmp[np.tril(tmp,-1)!=0]
conn=len(tmp)

# Create labels

Y_UK=np.zeros(data_UK.shape[0])
Y_JP=np.zeros(data_JP.shape[0])
Y_US=np.zeros(data_US.shape[0])

Y_UK[P_UK]=1
Y_JP[P_JP]=1
Y_US[P_US]=1

# Load data (lower triangular matrix)

X_UK=np.zeros((data_UK.shape[0],conn))
X_JP=np.zeros((data_JP.shape[0],conn))
X_US=np.zeros((data_US.shape[0],conn))

# Save data
for i in range(data_UK.shape[0]):
    tmp=data_UK[i]
    tmp=tmp[np.tril(tmp,-1)!=0]
    X_UK[i,::]=tmp

for i in range(data_JP.shape[0]):
    tmp=data_JP[i]
    tmp=tmp[np.tril(tmp,-1)!=0]
    X_JP[i,::]=tmp

for i in range(data_US.shape[0]):
    tmp=data_US[i]
    tmp=tmp[np.tril(tmp,-1)!=0]
    X_US[i,::]=tmp
    

outdname_UK='/Users/mariarosa/Documents/Work/pain_data/all_data/CorrelationMatrices_V5/UK_Cam_V5/ROICorrelation.csv'
outdname_JP='/Users/mariarosa/Documents/Work/pain_data/all_data/CorrelationMatrices_V5/JP_CiN_V5/ROICorrelation.csv'
outdname_US='/Users/mariarosa/Documents/Work/pain_data/all_data/CorrelationMatrices_V5/US_OP_V5/ROICorrelation.csv'

writer = csv.writer(open(outdname_UK, 'w'))
for row in X_UK:
    writer.writerow(row)
    
writer = csv.writer(open(outdname_JP, 'w'))
for row in X_JP:
    writer.writerow(row)
    
X_US = np.delete(X_US, (21), axis = 0) # Delete subject with NaNs
Y_US = np.delete(Y_US, (21), axis = 0) # Delete subject with NaNs

writer = csv.writer(open(outdname_US, 'w'))
for row in X_US:
    writer.writerow(row)


