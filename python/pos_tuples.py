#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 18:12:50 2020

@author: philipp
"""

import numpy as np


def pos_array_to_pos_tuples(pos):
    n = len(pos)
    
    tuples = []
    
    for i in range(n):
        new_tuple = (pos[i, 0], pos[i, 1], pos[i, 2])
        tuples.append(new_tuple)
        
        
    return tuples
        

n = 10

dc =  np.array([1.0, 0.0, 0.0])

pos = np.array([0.0,0.0,0.0])

hist = []
for i in range(n):
    hist.append(pos)
    
    pos = pos + dc
    
hist = np.array(hist)

print(hist)

hist_tuples = pos_array_to_pos_tuples(hist)

print(hist_tuples)
