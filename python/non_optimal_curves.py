#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 19:17:55 2021

@author: philipp
"""

import numpy as np
from odeintw import odeintw
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


from spr4 import spr4


# standard basis of R^4
e1 = np.array([1,0,0,0])
e2 = np.array([0,1,0,0])
e3 = np.array([0,0,1,0])
e4 = np.array([0,0,0,1])
        
basis_r4 = (e1, e2, e3, e4)

# standard basis of R^3
f1 = np.array([1,0,0])
f2 = np.array([0,1,0])
f3 = np.array([0,0,1])


# standard basis of so(3)
L1 = np.array([[0,0,0],[0,0,-1],[0,1,0]])
L2 = np.array([[0,0,1],[0,0,0],[-1,0,0]])
L3 = np.array([[0,-1,0 ],[1,0,0],[0,0,0]])

basis_so3 = (L1, L2, L3)
a = 0.01
xi0 = 1

swimmer = spr4(a, xi0)
U = swimmer.U
Lg12inv = swimmer.Lg12inv
Lg = swimmer.Lg
Lh = swimmer.Lh
ULginv12 = np.matmul(U, Lg12inv)
tildeLg = swimmer.tildeLg



w = np.array([1,1,1,0,0,1])
dp = 1/np.sqrt(np.linalg.det(Lg))*np.matmul(Lh, tildeLg).dot(w)

xi_opt, dxidt_opt, _ = swimmer.optimal_curve(dp, full=True)
                                                           

amp = 1

# construction of the non-optimal curve with 4 modes
v1 = e1
u1 = e4

v2 = e2
u2 = e4

v3 = e3
u3 = e4

v4 = e1
u4 = e2

a1 = (1/np.sqrt(2*np.pi))*np.matmul(ULginv12, u1)
b1 = (1/np.sqrt(2*np.pi))*np.matmul(ULginv12, v1)

a2 = (1/np.sqrt(2*2*np.pi))*np.matmul(ULginv12, u2)
b2 = (1/np.sqrt(2*2*np.pi))*np.matmul(ULginv12, v2)

a3 = (1/np.sqrt(3*2*np.pi))*np.matmul(ULginv12, u3)
b3 = (1/np.sqrt(3*2*np.pi))*np.matmul(ULginv12, v3)

a4 = (1/np.sqrt(4*2*np.pi))*np.matmul(ULginv12, u2)
b4 = (1/np.sqrt(4*2*np.pi))*np.matmul(ULginv12, v2)

xi_4terms = lambda t: np.cos(t)*a1 + np.sin(t)*b1 + np.cos(2*t)*a2 + np.sin(2*t)*b2 \
    + np.cos(3*t)*a3 + np.sin(3*t)*b3 + np.cos(4*t)*a4 + np.sin(4*t)*b4


dxidt_4terms = lambda t: -np.sin(t)*a1 + np.cos(t)*b1 -2*np.sin(2*t)*a2 + 2*np.cos(2*t)*b2 \
    - 3* np.sin(3*t)*a3 + 3* np.cos(3*t)*b3 - 4* np.sin(4*t)*a4 + 4*np.cos(4*t)*b4
    
# construction of the non-optimal curve with 3 modes

y1 = e1 + e2 + e3
x1 = e4

y2 = e2
x2 = e3

y3 = e1
x3 = e2

c1 = (1/np.sqrt(2*np.pi))*np.matmul(ULginv12, x1)
d1 = (1/np.sqrt(2*np.pi))*np.matmul(ULginv12, y1)

c2 = (1/np.sqrt(2*2*np.pi))*np.matmul(ULginv12, x2)
d2 = (1/np.sqrt(2*2*np.pi))*np.matmul(ULginv12, y2)

c3 = (1/np.sqrt(3*2*np.pi))*np.matmul(ULginv12, x3)
d3 = (1/np.sqrt(3*2*np.pi))*np.matmul(ULginv12, y3)


xi_3terms = lambda t: np.cos(t)*c1 + np.sin(t)*d1 + np.cos(2*t)*c2 + np.sin(2*t)*d2 \
    + np.cos(3*t)*c3 + np.sin(3*t)*d3


dxidt_3terms = lambda t: -np.sin(t)*c1 + np.cos(t)*d1 -2*np.sin(2*t)*c2 + 2*np.cos(2*t)*d2 \
    - 3* np.sin(3*t)*c3 + 3* np.cos(3*t)*d3
    
# construction of the non-optimal curve with 2 modes

s1 = e1 + e2 + e3
r1 = e4


s2 = e2
r2 = e3 - e1


g1 = (1/np.sqrt(2*np.pi))*np.matmul(ULginv12, r1)
h1 = (1/np.sqrt(2*np.pi))*np.matmul(ULginv12, s1)

g2 = (1/np.sqrt(2*2*np.pi))*np.matmul(ULginv12, r2)
h2 = (1/np.sqrt(2*2*np.pi))*np.matmul(ULginv12, s2)


xi_2terms = lambda t: np.cos(t)*g1 + np.sin(t)*h1 + np.cos(2*t)*g2 + np.sin(2*t)*h2


dxidt_2terms = lambda t: -np.sin(t)*g1 + np.cos(t)*h1 -2*np.sin(2*t)*g2 + 2*np.cos(2*t)*h2
    


n_strokes = 1
T = np.linspace(0, n_strokes*2*np.pi, 1000*n_strokes, endpoint=True)

hist_xi_opt= np.array([xi_opt(t) for t in T])
hist_xi_4terms = np.array([xi_4terms(t) for t in T])
hist_xi_3terms = np.array([xi_3terms(t) for t in T])
hist_xi_2terms = np.array([xi_2terms(t) for t in T])

hists = [ hist_xi_opt, hist_xi_2terms, hist_xi_3terms, hist_xi_4terms]
colors = ["b", "r", "g", "c"]

fig, axs = plt.subplots(4, 4, figsize=(9, 3), sharey=True, sharex=True)

for hist, color, j in zip(hists, colors, range(4)):
    for i in range(4):
        axs[j, i].plot(T, hist[:, i], color = color)
plt.show()


print("=========== NET DISPLACEMENTS ===============")
print("theoretical net displacement:", dp)
print("net displacement opt:", swimmer.net_displacement(xi_opt, dxidt_opt) - dp)
print("net displacmenent 2 terms:", swimmer.net_displacement(xi_2terms, dxidt_2terms)- dp)
print("net displacmenent 3 terms:", swimmer.net_displacement(xi_3terms, dxidt_3terms)- dp)
print("net displacmenent 4 terms:", swimmer.net_displacement(xi_3terms, dxidt_4terms)- dp)
print("---------------------------------------------")




R0 = np.eye(3)
c0 = np.array([0,0,0])
p0 = np.vstack((c0, R0))
