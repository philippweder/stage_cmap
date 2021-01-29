#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 13:34:31 2021

@author: philipp
"""

import numpy as np

import matplotlib.pyplot as plt
from scipy.linalg import norm
from spr4 import spr4, convergence_plot, rel_convergence_plot
from scipy.integrate import solve_ivp

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
z2 = np.array([-np.sqrt(2/9), -np.sqrt(2/3), - 1.0/3.0])
z3 = np.array([-np.sqrt(2/9), np.sqrt(2/3), - 1.0/3.0])
z4 = np.array([0,0,1])




a = 0.01
xi0 = 10

swimmer = spr4(a, xi0)


dp_simple_spat = np.array([1 ,0 ,0 ,0 ,0 ,0])
dp_simple_rot = np.array([0 ,0 ,0 ,0 ,0 ,1])
dp_nonsimple = np.array([0, 0, 1, 0, 0, 1])
dp_full = np.array([1, 1, 1, 1, 1, 1])

dps = [dp_simple_spat, dp_simple_rot, dp_nonsimple, dp_full]

label_simple_spat = "$\delta p = (1, 0, 0, 0, 0, 0)$"
label_simple_rot = "$\delta p = (0, 0, 0, 0, 0, 1)$"
label_nonsimple = "$\delta p = (0, 0, 1, 0, 0, 1)$"
label_full = "$\delta p = (1, 1, 1, 1, 1, 1)$"

labels = [label_simple_spat, label_simple_rot, label_nonsimple, label_full]

rel_errors = []
abs_errors = []
corr_rel_errors = []
corr_abs_errors = []
corrs = []

epsilons = np.linspace(1/1000000, 1, 300)


for dp in dps:
    new_rel_errs = []
    new_abs_errs = []
    new_corr_rel_errs = []
    new_corr_abs_errs = []
    new_corr = []
    for eps in epsilons:
        
        xi, dxidt, coeffs = swimmer.optimal_curve(dp, full=True)
        xi_eps = lambda t: eps*xi(t)
        dxidt_eps = lambda t: eps*dxidt(t)
        a1, b1, a2, b2 = coeffs
        coeffs_eps =(eps*a1, eps*b1, eps*a2, eps*b2)

        
        dp_exp = swimmer.net_displacement(xi_eps, dxidt_eps)
        dR_exp = dp_exp[3:]
        
        
        dp_corr = dp_exp + swimmer.correction( dR_exp, coeffs_eps)
        # dp_corr = dp_exp
        new_corr.append(norm(swimmer.correction(dR_exp, coeffs_eps)))
        
        n_strokes = 1
        T = np.linspace(0, n_strokes*2*np.pi, 600*n_strokes, endpoint=True)
        
        # R0 = np.eye(3)
        # c0 = np.array([0,0,0])
        # p0 = np.vstack((c0, R0))
        
        t_span = [0, 2*np.pi*n_strokes]
        R0 = np.eye(3)
        c0 = np.array([0,0,0])
        p0 = np.hstack((c0, R0[0,:], R0[1,:], R0[2,:]))
        
        # sol = odeintw(swimmer.rhs, p0, T, args=(xi_eps, dxidt_eps))
        sol = solve_ivp(swimmer.rhs, t_span, p0, t_eval = T, method="RK45", atol=10E-16, rtol=10E-16, args=(xi_eps, dxidt_eps))
        sol_c = sol.y[:3,:]
        
        # sol_c = sol[:,0,:]
        # diff = sol_c[-1, :] - sol_c[0, :]
        diff = sol_c[:, -1] - sol_c[:, 0]
        
    
    
        new_rel_errs.append( norm(diff - dp_exp[:3])/norm(dp_exp[:3]))
        new_abs_errs.append( norm(diff - dp_exp[:3]))
        
        new_corr_rel_errs.append( norm(diff - dp_corr[:3])/norm(dp_corr[:3]))
        new_corr_abs_errs.append( norm(diff - dp_corr[:3]))
        
        
    new_corr = np.array(new_corr)
    new_rel_errs = np.array(new_rel_errs)
    new_abs_errs = np.array(new_abs_errs)
    new_corr_abs_errs = np.array(new_corr_abs_errs)
    new_corr_rel_errs = np.array(new_corr_rel_errs)
    
    corrs.append(new_corr)
    rel_errors.append(new_rel_errs)
    abs_errors.append(new_abs_errs)
    corr_rel_errors.append(new_corr_rel_errs)
    corr_abs_errors.append(new_corr_abs_errs)
    
    

fig, axs = plt.subplots(nrows=2, ncols=4, figsize = (18,8), sharey=False, sharex=True)
for errors, label, ax in zip(abs_errors, labels, axs[0,:]):
    
    convergence_plot(ax, epsilons, errors, "$\epsilon$", "$||\Delta c- \delta c||$", label)
    
for errors, label, ax in zip(rel_errors, labels, axs[1,:]):
    rel_convergence_plot(ax, epsilons, errors, "$\epsilon$", "$||\Delta c- \delta c||/||\delta c||$")

plt.tight_layout()
# plt.suptitle("uncorrected absolute and relative errors")


fig, axs = plt.subplots(nrows=3, ncols=4, figsize = (18,12), sharey=False, sharex=True)
for errors, label, ax in zip(corr_abs_errors, labels, axs[0,:]):
    convergence_plot(ax, epsilons, errors,  "$\epsilon$", "$||\Delta c- \delta c||$", label )
for errors, label, ax in zip(corr_rel_errors, labels, axs[1,:]):
    rel_convergence_plot(ax, epsilons, errors, "$\epsilon$", "$||\Delta c- \delta c||/||\delta c||$")
for corr, ax in zip(corrs, axs[2,:]):
    ax.loglog(epsilons, corr, "g", label = "$||\zeta||$")
    ax.loglog(epsilons, (1.0*epsilons)**(3), "r-.", label = "O(3)")
    ax.legend(loc = "lower right")
    ax.set_xlabel("$\epsilon$")
plt.tight_layout()
# plt.suptitle("corrected absolute and relative errors")


    

