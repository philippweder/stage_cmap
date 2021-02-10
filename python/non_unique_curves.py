#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 16:07:37 2020

@author: philipp
"""
import numpy as np
from odeintw import odeintw
from scipy.integrate import solve_ivp
from numpy.linalg import norm, det
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


from spr4 import spr4, convergence_plot, rel_convergence_plot


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
a = 0.001
xi0 = 100

swimmer = spr4(a, xi0)
U = swimmer.U
Lg12inv = swimmer.Lg12inv
Lg = swimmer.Lg
Lh = swimmer.Lh
ULginv12 = np.matmul(U, Lg12inv)
tildeLg = swimmer.tildeLg

# =============================================================================
# Let us note that the decomposition of the bivector w = e1^e2 + e^3^e4 is not 
# unique since we also have
#       w = e1^e2 + e3^e4 = 1/2(e1 + e3)^(e2 + e4) + 1/2(e1 - e3)^(e2 - e4)
# So to have exactly this bivector in the construction of the optimal control 
# curve, we set dp = 1/sqrt(det Lg)(Lh.tildeLg)w
# =============================================================================

# construction of the first curve

w = np.array([0,0,1,0,0,1])
dp = 1/np.sqrt(np.linalg.det(Lg))*np.matmul(Lh, tildeLg).dot(w)
                                                           

amp = 1
v2 = e1*amp
u2 = e2*amp

v1 = e3*amp
u1 = e4*amp

a1 = (1/np.sqrt(2*np.pi))*np.matmul(ULginv12, u1)
b1 = (1/np.sqrt(2*np.pi))*np.matmul(ULginv12, v1)

a2 = (1/np.sqrt(4*np.pi))*np.matmul(ULginv12, u2)
b2 = (1/np.sqrt(4*np.pi))*np.matmul(ULginv12, v2)


xi1 = lambda t: np.cos(t)*a1 + np.sin(t)*b1 + np.cos(2*t)*a2 + np.sin(2*t)*b2
dxi1dt = lambda t: -np.sin(t)*a1 + np.cos(t)*b1 -2*np.sin(2*t)*a2 + 2*np.cos(2*t)*b2
coeffs1 = (a1, b1, a2, b2)


# construction of the second curve

y1 = 1/np.sqrt(2)*(e1 + e3)*amp
x1 = 1/np.sqrt(2)*(e2 + e4)*amp

y2 = 1/np.sqrt(2)*(e1 - e3)*amp
x2 = 1/np.sqrt(2)*(e2 - e4)*amp

c1 = (1/np.sqrt(2*np.pi))*np.matmul(ULginv12, x2)
d1 = (1/np.sqrt(2*np.pi))*np.matmul(ULginv12, y2)

c2 = (1/np.sqrt(4*np.pi))*np.matmul(ULginv12, x1)
d2 = (1/np.sqrt(4*np.pi))*np.matmul(ULginv12, y1)

xi2 = lambda t: np.cos(t)*c1 + np.sin(t)*d1 + np.cos(2*t)*c2 + np.sin(2*t)*d2
dxi2dt = lambda t: -np.sin(t)*c1 + np.cos(t)*d1 -2*np.sin(2*t)*c2 + 2*np.cos(2*t)*d2
coeffs2 = (c1, d1, c2, d2)


# construction of the third curve

yy1 = -1/np.sqrt(2)*(e1 + e4)*amp
xx1 = -1/np.sqrt(2)*(e2 - e3)*amp

yy2 = -1/np.sqrt(2)*(e1 - e4)*amp
xx2 = -1/np.sqrt(2)*(e2 + e3)*amp

cc1 = (1/np.sqrt(2*np.pi))*np.matmul(ULginv12, xx2)
dd1 = (1/np.sqrt(2*np.pi))*np.matmul(ULginv12, yy2)

cc2 = (1/np.sqrt(4*np.pi))*np.matmul(ULginv12, xx1)
dd2 = (1/np.sqrt(4*np.pi))*np.matmul(ULginv12, yy1)

xi3 = lambda t: np.cos(t)*cc1 + np.sin(t)*dd1 + np.cos(2*t)*cc2 + np.sin(2*t)*dd2
dxi3dt = lambda t: -np.sin(t)*cc1 + np.cos(t)*dd1 -2*np.sin(2*t)*cc2 + 2*np.cos(2*t)*dd2
coeffs3 = (cc1, dd1, cc2, dd2)





n_strokes = 1
T = np.linspace(0, n_strokes*2*np.pi, 5000*n_strokes, endpoint=True)

hist_xi1 = np.array([xi1(t) for t in T])
hist_xi2 = np.array([xi2(t) for t in T])
hist_xi3 = np.array([xi3(t) for t in T])


fig, axs = plt.subplots(3, 4, figsize=(9, 3), sharey=True, sharex=True)
for i in range(4):
    axs[0, i].plot(T, hist_xi1[:, i], color="b")
for i in range(4):
    axs[1, i].plot(T, hist_xi2[:, i], color="r")
for i in range(4):
    axs[2, i].plot(T, hist_xi3[:, i], color="g")
plt.show()


eps = 1/1000
dp_eps = eps**2*dp

xi1_eps = lambda t: eps*xi1(t)
dxi1dt_eps = lambda t: eps*dxi1dt(t)

xi2_eps = lambda t: eps*xi2(t)
dxi2dt_eps = lambda t: eps*dxi2dt(t)

xi3_eps = lambda t: eps*xi3(t)
dxi3dt_eps = lambda t: eps*dxi3dt(t)



t_span = [0, 2*np.pi*n_strokes]
R0 = np.eye(3)
c0 = np.array([0,0,0])

p0 = swimmer.build_ic(c0, R0)

sol1 = solve_ivp(swimmer.rhs, t_span,p0, t_eval=T, method="DOP853",atol=10E-16, rtol=10E-16, args=(xi1_eps, dxi1dt_eps))
sol1_c = sol1.y[:3,:]

sol2 = solve_ivp(swimmer.rhs, t_span,p0, t_eval=T, method="DOP853", atol=10E-16, rtol=10E-16, args=(xi2_eps, dxi2dt_eps))
sol2_c = sol2.y[:3,:]

sol3 = solve_ivp(swimmer.rhs, t_span,p0, t_eval=T, method="DOP853", atol=10E-16, rtol=10E-16, args=(xi3_eps, dxi3dt_eps))
sol3_c = sol3.y[:3,:]

    

dc1_eps, dR1_eps = swimmer.net_displacement(xi1_eps, dxi1dt_eps, split=True)
dc2_eps, dR2_eps = swimmer.net_displacement(xi2_eps, dxi2dt_eps, split=True)
dc3_eps, dR3_eps = swimmer.net_displacement(xi3_eps, dxi3dt_eps, split=True)

diff1 = sol1_c[:, -1] - sol1_c[:, 0]
diff2 = sol2_c[:, -1] - sol2_c[:, 0]
diff3 = sol3_c[:, -1] - sol3_c[:, 0]

dc1_plot = np.array([n_strokes*dc1_eps/len(T)*k for k in range(len(T))])
dc2_plot = np.array([n_strokes*dc2_eps/len(T)*k for k in range(len(T))])

dc1_plot[:,:2] = np.zeros((len(T), 2))
dc2_plot[:,:2] = np.zeros((len(T), 2))



#3D Plotting
fig = plt.figure(figsize=(24, 6))

#frontal view
ax1 = fig.add_subplot(131, projection = "3d")
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.set_xticks([])
ax1.ticklabel_format(style = "sci", scilimits = (0,0))
ax1.plot3D(sol1_c[0,:], sol1_c[1,:], sol1_c[2,:])
ax1.plot3D(sol2_c[0,:], sol2_c[1,:], sol2_c[2,:], color="r")
# ax1.plot3D(sol3_c[0,:], sol3_c[1,:], sol3_c[2,:], color="g")
ax1.tick_params(which='major', pad=5)
# ax1.yaxis.labelpad= 15
# ax1.zaxis.labelpad= 15
# ax1.plot3D(highlights[:,0], highlights[:,1], highlights[:,2], "*",  color = "#e31c23")
ax1.view_init(0.01, 0)

#side view
ax2 = fig.add_subplot(132, projection = "3d")
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
ax2.set_yticks([])
ax2.ticklabel_format(style = "sci", scilimits = (0,0))
ax2.plot3D(sol1_c[0,:], sol1_c[1,:], sol1_c[2,:])
ax2.plot3D(sol2_c[0,:], sol2_c[1,:], sol2_c[2,:], color="r")
# ax2.plot3D(sol3_c[0,:], sol3_c[1,:], sol3_c[2,:], color="g")
ax2.tick_params(which='major', pad=5)
# ax2.plot3D(highlights[:,0], highlights[:,1], highlights[:,2], "*", color="#e31c23")
# ax2.xaxis.labelpad= 15
# ax2.zaxis.labelpad= 25
ax2.view_init(0, 90)

#top view
ax3 = fig.add_subplot(133, projection = "3d")
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_zlabel('z')
ax3.set_zticks([])
ax3.ticklabel_format(style = "sci", scilimits = (0,0))
ax3.plot3D(sol1_c[0,:], sol1_c[1,:], sol1_c[2,:])
ax3.plot3D(sol2_c[0,:], sol2_c[1,:], sol2_c[2,:], color="r")
# ax3.plot3D(sol3_c[0,:], sol3_c[1,:], sol3_c[2,:], color="g")
ax3.tick_params(which='major', pad=5)
# ax3.plot3D(highlights[:,0], highlights[:,1], highlights[:,2], "*", color="#e31c23")
ax3.yaxis.labelpad= 15
# ax3.xaxis.labelpad= 25
ax3.view_init(90, 0)

plt.tight_layout()
plt.show()


print("====THEORETICAL NET DISPLACEMENT=======")
print(dp_eps)
print("------------------------------")


print("=======CURVE 1============")
print("difference curve 1: ", diff1)
print("nstrokes * exp net displacement", n_strokes*dc1_eps)
print("------------------------------")


print("=======CURVE 2============")
print("difference curve 2: ", diff2)
print("nstrokes * exp net displacement", n_strokes*dc2_eps)
print("------------------------------")


print("=======CURVE 3============")
print("difference curve 3: ", diff3)
print("nstrokes * exp net displacement", n_strokes*dc3_eps)
print("------------------------------")



fig, axs = plt.subplots(3, 3, figsize=(9, 3), sharey=True, sharex=True)
for i in range(3):
    axs[0, i].plot(T, sol1_c[i, :], color="b")
for i in range(3):
    axs[1, i].plot(T, sol2_c[i, :], color="r")
for i in range(3):
    axs[2, i].plot(T, sol3_c[i, :], color="g")
fig.suptitle('Plot of coordinates')
plt.show

print("========ENERGIES==============")
print(swimmer.energy(dxi1dt_eps))
print(swimmer.energy(dxi2dt_eps))
print(swimmer.energy(dxi3dt_eps))
print("------------------------------")



# =============================================================================
# Convergence experiment
# =============================================================================

# labels = ["$xi1$", "$xi2$"]

# rel_errors = []
# abs_errors = []

# corr_rel_errors = []
# corr_abs_errors = []

# # grid for different values of scaling
# epsilons = np.linspace(1/100000, 1, 500)


# curves = [(xi1, dxi1dt, coeffs1), (xi2, dxi2dt, coeffs2)]

# for curve in curves:
#         new_rel_errs = []
#         new_abs_errs = []
#         new_corr_rel_errs = []
#         new_corr_abs_errs = []
#         for eps in epsilons:
            
#             xi, dxidt, coeffs = curve
            
#             #rescaling of curves and coefficients
#             xi_eps = lambda t: eps*xi(t)
#             dxidt_eps = lambda t: eps*dxidt(t)
#             a1, b1, a2, b2 = coeffs
#             coeffs_eps =(eps*a1, eps*b1, eps*a2, eps*b2)

#             #calculating the experimental net displacement due to the rescaled
#             #curve
#             dp_exp = swimmer.net_displacement(xi_eps, dxidt_eps)
#             dR_exp = dp_exp[3:]
            
#             #add correction term of order 3 (optional)
#             dp_corr = dp_exp + correction(swimmer, dR_exp, coeffs_eps)
#             # dp_corr = dp_exp
            
#             #integration of ODE
#             n_strokes = 1
#             T = np.linspace(0, n_strokes*2*np.pi, 1000*n_strokes, endpoint=True)
            
            
#             t_span = [0, 2*np.pi*n_strokes]
#             R0 = np.eye(3)
#             c0 = np.array([0,0,0])
#             p0 = np.hstack((c0, R0[0,:], R0[1,:], R0[2,:]))

#             sol = solve_ivp(swimmer.rhs, t_span,p0, t_eval=T, method="Radau ", atol=10E-16, rtol=10E-16, args=(xi_eps, dxidt_eps))
#             sol_c = sol.y[:3,:]
                    
#             diff = sol_c[:, -1] - sol_c[:, 0]
            
#             new_rel_errs.append( norm(diff - dp_exp[:3])/norm(dp_exp[:3]))
#             new_abs_errs.append( norm(diff - dp_exp[:3]))
            
#             new_corr_rel_errs.append( norm(diff - dp_corr[:3])/norm(dp_corr[:3]))
#             new_corr_abs_errs.append( norm(diff - dp_corr[:3]))
            
            
            
#         new_rel_errs = np.array(new_rel_errs)
#         new_abs_errs = np.array(new_abs_errs)
#         new_corr_abs_errs = np.array(new_corr_abs_errs)
#         new_corr_rel_errs = np.array(new_corr_rel_errs)
        
#         rel_errors.append(new_rel_errs)
#         abs_errors.append(new_abs_errs)
#         corr_rel_errors.append(new_corr_rel_errs)
#         corr_abs_errors.append(new_corr_abs_errs)
        
        

# #convergence plots
# fig, axs = plt.subplots(nrows=2, ncols=2, figsize = (12,8), sharey=False)
# for errors, label, ax in zip(abs_errors, labels, axs[0,:]):
#     convergence_plot(ax, epsilons, errors, "$\epsilon$", "$||\Delta c - \delta c||$", label)
# for errors, label, ax in zip(rel_errors, labels, axs[1,:]):
#     rel_convergence_plot(ax, epsilons, errors, "$\epsilon$", "||\Delta c - \delta c||/||\delta c||", label)
# plt.tight_layout()
# plt.suptitle("uncorrected absolute and relative errors")


# fig, axs = plt.subplots(nrows=2, ncols=2, figsize = (12,8), sharey=False)
# for errors, label, ax in zip(corr_abs_errors, labels, axs[0,:]):
#     convergence_plot(ax, epsilons, errors, "$\epsilon$", "$||\Delta c - \delta c||$", label)
# for errors, label, ax in zip(corr_rel_errors, labels, axs[1,:]):
#     rel_convergence_plot(ax, epsilons, errors, "$\epsilon$", "$||\Delta c- \delta c||/||\delta c||$", label)
# plt.tight_layout()
# plt.suptitle("corrected absolute and relative errors")


