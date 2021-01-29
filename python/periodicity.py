#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 14:21:13 2021

@author: philipp
"""

import numpy as np
from scipy.linalg import norm
from scipy.integrate import quad, solve_ivp
import matplotlib.pyplot as plt


from spr4 import spr4, convergence_plot


def rhs(t, c, dxidt):
    
    dcdt = np.dot(Fc0, dxidt(t))
    
    return dcdt

a = 1
xi0 = 1

swimmer = spr4(a, xi0)
Fc0 = swimmer.Fc0

dp = np.array([1,0,0,1,0,0])
dc = dp[:3]
xi, dxidt, coeffs = swimmer.optimal_curve(dp, full=True)


epsilons = np.linspace(1/100000, 1, 300)

errs = []

for eps in epsilons:
    xi_eps = lambda t: eps*xi(t)
    dxidt_eps = lambda t: eps*dxidt(t)
        
    c0 = np.array([0, 0, 0])
    t_span = [0, 2*np.pi]
    sol_c = solve_ivp(rhs, t_span, c0, method="Radau", args = (dxidt_eps,))
    Dc = sol_c.y[:,-1] - sol_c.y[:,0]
    new_err = norm(Dc)
    errs.append(new_err)
    
errs = np.array(errs)


plt.figure()
plt.loglog(epsilons, errs)
plt.loglog(epsilons, (1.0*epsilons)**1, "k--", label = "O(1)")
plt.loglog(epsilons, (1.0*epsilons)**2, "k-.", label = "O(2)")
plt.loglog(epsilons, (1.0*epsilons)**(3), "k.-", label = "O(3)")
plt.legend(loc = "lower right")
plt.show()





