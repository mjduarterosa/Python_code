# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 09:13:29 2016

@author: mariarosa
"""

import matplotlib.pyplot as plt
import load_pain_data as pd
from sklearn import preprocessing

# Pain data

X_UK = pd.X_UK
X_JP = pd.X_JP
#X_UK = preprocessing.scale(pd.X_UK)
#X_JP = preprocessing.scale(pd.X_JP)

# Box plots with custom fill colors
all_data=[]
all_data.append(X_UK.mean(axis=1))
all_data.append(X_JP.mean(axis=1))


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

# rectangular box plot
bplot1 = axes[0].boxplot(all_data,
                         vert=True,   # vertical box aligmnent
                         patch_artist=True)   # fill with color

# notch shape box plot
bplot2 = axes[1].boxplot(all_data,
                         notch=True,  # notch shape
                         vert=True,   # vertical box aligmnent
                         patch_artist=True)   # fill with color

# fill with colors
colors = ['lightblue', 'lightgreen']
for bplot in (bplot1, bplot2):
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

# adding horizontal grid lines
for ax in axes:
    ax.yaxis.grid(True)
    ax.set_xticks([y+1 for y in range(len(all_data))], )
    ax.set_xlabel('xlabel')
    ax.set_ylabel('ylabel')

# add x-tick labels
plt.setp(axes, xticks=[y+1 for y in range(len(all_data))],
         xticklabels=['x1', 'x2'])

plt.show()