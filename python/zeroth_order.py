#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 16:41:29 2021

@author: philipp
"""
import numpy as np
from odeintw import odeintw
from scipy.integrate import solve_ivp
from numpy.linalg import det

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


from spr4 import spr4

a = 0.01
xi0 = 1.0

swimmer = spr4(a, xi0)
w = np.array([0,0,0,1,0,0])
U = swimmer.U
Lg12inv = swimmer.Lg12inv
Lg = swimmer.Lg
Lh = swimmer.Lh
tildeLg = swimmer.tildeLg

dp1 = 1/np.sqrt(np.linalg.det(Lg))*np.matmul(Lh, tildeLg).dot(w)

xi, dxidt, coeffs = swimmer.optimal_curve(dp1, full=True)


dp2 = swimmer.net_displacement(xi, dxidt)

print("----------------", "\n")
print(dp1)
print("----------------", "\n")
print(dp2)
print("----------------", "\n")





R0 = np.eye(3)
c0 = np.array([0,0,0])
p0 = np.vstack((c0, R0))


n_points = 1000
n_strokes = 1
T = np.linspace(0, 2*np.pi*n_strokes, n_points*n_strokes, endpoint=True)

sol = odeintw(swimmer.rhs, p0, T, args= (xi, dxidt))

sol_pos = sol[:, 0, :]

print("----------------", "\n")
print(sol_pos[-1,:] - sol_pos[0,:])
print("----------------", "\n")

#3D Plotting
fig = plt.figure()
ax1 = plt.axes(projection="3d")

#Labeling
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')

ax1.plot3D(sol_pos[:,0], sol_pos[:,1], sol_pos[:,2], color="b")

plt.show()

fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=True, sharex=True)
for i in range(3):
    axs[i].plot(T, sol_pos[:, i], color="b")
fig.suptitle('Plot of coordinates')
plt.show()

sol_theta = sol[:, 1:, :]

dets = []
for n in range(len(sol_theta)):
    dets.append(det(sol_theta[n,:]))

dets = np.array(dets)