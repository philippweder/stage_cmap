#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 14:21:13 2021

@author: philipp
"""

import numpy as np
from scipy.integrate import quad, solve_ivp


from spr4 import spr4


a = 0.01
xi0 = 10

swimmer = spr4(a, xi0)
Fc0 = swimmer.Fc0

dp = np.array([1,0,0,1,0,0])
dc = dp[:3]
xi, dxidt, coeffs = swimmer.optimal_curve(dp, full=True)




def rhs(t, c):
    
    dcdt = np.dot(Fc0, dxidt(t))
    
    return dcdt


c0 = np.array([0, 0, 0])
t_span = [0, 2*np.pi]
sol_c = solve_ivp(rhs, t_span, c0, method="DOP853")
Dc = sol_c.y[:,-1] - sol_c.y[:,0]

print(Dc)
print(sol_c.t)
print(2*np.pi)




