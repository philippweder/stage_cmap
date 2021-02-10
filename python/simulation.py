#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 11:28:08 2021

@author: philipp
"""

from spr4 import spr4
import numpy as np

def Rz(t):
    return np.array([
        [np.cos(t), -np.sin(t), 0],
        [np.sin(t), np.cos(t), 0],
        [0, 0, 1]
        ])

a = 0.001
xi0 = 100

swimmer = spr4(a, xi0)
eps = 1/1000000
dp1 = np.array([0,0,1,0,0,0])*eps
dp2 = np.array([0,0,0,0,0,1])*eps
dp3 = np.array([0,0,1,0,0,1])*eps

dps = [dp1, dp2, dp3]

n_res = 500
n_strokes = 1

c0 = np.array([0,0,0])
R0 = np.eye(3)

sols = []

# for dp in dps:
#     new_sol_c, _, _, _  = swimmer.optimal_curve_trajectory(dp, n_strokes, n_res, c0, R0, log=True)
#     sols.append(new_sol_c)



# swimmer.trajectories_plot(sols, highlights=True)


R1 = Rz(np.pi/2)
R2 = Rz(np.pi)
R3 = Rz(3*np.pi/2)

Rs = [R0, R1, R2, R3]

sols = []

for R in Rs:
    new_sol_c, _, _, _  = swimmer.optimal_curve_trajectory(dp1, n_strokes, n_res, c0, R, log=True)
    sols.append(new_sol_c)
    
swimmer.trajectories_plot(sols, highlights=True)
