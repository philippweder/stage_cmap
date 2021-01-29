#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 17:17:05 2021

@author: philipp
"""

from spr4 import spr4
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import norm

xi0 = 1

A = np.linspace(1/1000, 1/10, 50)

dp = np.array([0,0,1,0,0,1])
dR = dp[3:]

# A_corr_norms= []
# for a in A:
#     swimmer = spr4(a, xi0)
#     xi, dxidt, coeffs = swimmer.optimal_curve(dp, full=True)
#     new_corr =  swimmer.correction(dR, coeffs)
#     A_corr_norms.append(norm(new_corr))
    

# A_corr_norms = np.array(A_corr_norms)

# plt.figure()
# plt.loglog(A, A_corr_norms)
# plt.show()


# a = 1
Xi = np.linspace(1, 500, 50)
# Xi_corr_norms = []
# for xi0 in Xi:
#     swimmer = spr4(a, xi0)
#     xi, dxidt, coeffs = swimmer.optimal_curve(dp, full=True)
#     new_corr = swimmer.correction(dR, coeffs)
#     Xi_corr_norms.append(norm(new_corr))

# Xi_corr_norms = np.array(Xi_corr_norms)

# plt.figure()
# plt.loglog(Xi, Xi_corr_norms)
# plt.show()
    

AA , XX = np.meshgrid(A, Xi)

corr_norms = []

for a in A:
    a_corr_norms = []
    for xi0 in Xi:
        swimmer = spr4(a, xi0)
        xi, dxidt, coeffs = swimmer.optimal_curve(dp, full=True)
        new_a_corr = swimmer.correction(dR, coeffs)
        a_corr_norms.append(norm(new_a_corr))
    
    a_corr_norms = np.array(a_corr_norms)
    corr_norms.append(a_corr_norms)

corr_norms = np.array(corr_norms)

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(AA, XX, corr_norms)
plt.xlabel('$a$')
plt.ylabel('$xi_0$')
plt.title("$||\zeta||$ depending on $a$ and $xi$")
plt.show()

        
    