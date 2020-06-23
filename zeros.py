#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 15:58:15 2020

@author: philipp
"""

import numpy as np
from scipy.optimize import fsolve, root


def f(x, alpha, beta):
    res = np.zeros((2,))
    res[0] = x[0] + x[1] + x[2] + x[3] - alpha - beta
    res[1] = x[0]*x[2] + 1/4. * x[1]* x[3] - alpha * beta

    
    return res


alph = 10
bet = 5
x0 = np.ones((4,))
print(x0)
F = lambda x: f(x, alph, bet)

x_star = root(F, x0)