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
        

pos1 = np.array([0,0,0])
pos2 = np.array([1,0,0])
pos3 = np.array([2,0,0])
pos4 = np.array([3,0,0])

pos = np.array([pos1, pos2, pos3, pos4])

print(pos.shape)
    
pos_tuples = pos_array_to_pos_tuples(pos)

print(pos_tuples)
    
    