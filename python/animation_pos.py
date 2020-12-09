#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 10:19:39 2020

@author: philipp
"""


import numpy as np
from scipy.linalg import expm

"""
CONSTANTS
"""

# standard basis of so(3)
L1 = np.array([[0,0,0],[0,0,-1],[0,1,0]])
L2 = np.array([[0,0,1],[0,0,0],[-1,0,0]])
L3 = np.array([[0,-1,0 ],[1,0,0],[0,0,0]])

basis_so3 = (L1, L2, L3)

# arms of the standard tetrahedron
z1 = np.array([np.sqrt(8/9), 0, -1.0/3.0])
z2 = np.array([-np.sqrt(2/9), np.sqrt(2/3), - 1.0/3.0])
z3 = np.array([-np.sqrt(2/9), -np.sqrt(2/3), - 1.0/3.0])
z4 = np.array([0,0,1])

def pos_array_to_pos_tuples(pos):
    n = len(pos)
    
    tuples = []
    
    for i in range(n):
        new_tuple = (pos[i, 0], pos[i, 1], pos[i, 2])
        tuples.append(new_tuple)
        
        
    return tuples


def calculate_sphere_positions(xi0, dc, dR, xi, n_strokes, fps):
    # discretization
    T = n_strokes*2*np.pi
    
    n_keyframes = int(T*fps)
    
    dcn = dc/n_keyframes
    
    dRn = 1/n_keyframes*(dR[0]*L1 + dR[1]*L2 + dR[2]*L3)
    
    dt = 1/fps
    
    # starting positions
    
    hist1 = []
    
    hist2 = []
    
    hist3 = []

    hist4 = []

    histc = []
    
    for k in range(n_keyframes):
        newpos1 = k*dcn + np.dot(expm(k*dRn), (xi0 + xi(k*dt)[0])*z1 )
        hist1.append(newpos1)
        
        newpos2 = k*dcn + np.dot(expm(k*dRn), (xi0 + xi(k*dt)[1])*z2 )
        hist2.append(newpos2)
        
        newpos3 = k*dcn + np.dot(expm(k*dRn), (xi0 + xi(k*dt)[2])*z3)
        hist3.append(newpos3)
        
        newpos4 = k*dcn + np.dot(expm(k*dRn), (xi0 + xi(k*dt)[3])*z4 )
        hist4.append(newpos4)
        
        newposc = k*dcn
        histc.append(newposc)
        
        
    
    hist1 = np.array(hist1)
    hist1_tuples = pos_array_to_pos_tuples(hist1)
    
    hist2 = np.array(hist2)
    hist2_tuples = pos_array_to_pos_tuples(hist2)
    
    hist3 = np.array(hist3)
    hist3_tuples = pos_array_to_pos_tuples(hist3)
    
    hist4 = np.array(hist4)
    hist4_tuples = pos_array_to_pos_tuples(hist4)
    
    histc = np.array(histc)
    histc_tuples = pos_array_to_pos_tuples(histc)
    
    
    return (hist1_tuples, hist2_tuples, hist3_tuples, hist4_tuples, histc_tuples, n_keyframes)
    
    


    