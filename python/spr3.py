#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 15:47:37 2021

@author: philipp
"""

import numpy as np
from numpy.linalg import norm, det
from scipy.integrate import quad, solve_ivp

import matplotlib.pyplot as plt

class spr3:
        
    def get_radius(self):
        return self.a
        
    def set_radius(self, a):
        self.a = a
        
    def get_init_length(self):
        return self.xi0
        
    def set_init_length(self, xi0):
        self.xi0 = xi0
        
    def build_params(self):
        
        self.kappa = 2/3 + 1/np.sqrt(3)*(self.a/self.xi0)
        
        self.h = 1/6 + (7/(16*np.sqrt(3)))*(self.a/self.xi0)
        
        self.gc = 1/2 + (3*np.sqrt(3))/16*(self.a/self.xi0)
        
        self.gt = 1 + (5*np.sqrt(3))/8*(self.a/self.xi0)
        
        self.aa = 1/6 - 1/(16*np.sqrt(3))*(self.a/self.xi0)
        
        self.alpha = (1/(32*np.sqrt(3))*(self.a/self.xi0**2))
        
        self.beta = 1/(16*np.sqrt(3))*(self.a/self.xi0**2)
    
        self.lab = 5/(48*np.sqrt(3))*(self.a/self.xi0**2)
        
        self.gamma = 1/(6*np.sqrt(3))*(1/self.xi0**2)
        
        self.tau1 = np.array([0, -1, 1])
        self.tau2 = (1/np.sqrt(3))*np.array([-2, 1, 1])
        self.tau3 = np.array([1, 1, 1])
        
        self.hc = self.alpha*norm(self.tau1)
        self.ht = self.gamma*norm(self.tau3)
        
        
    def build_energy_matrices(self):
        
        self.G = np.array([
            [self.kappa, self.h, self.h],
            [self.h, self.kappa, self.h],
            [self.h, self.h, self.kappa]
            ])
        
        
        self.Lg = np.diag([self.gc, self.gc, self.gt])
        
        self.U = np.array([self.tau1/norm(self.tau1), self.tau2/norm(self.tau2), self.tau3/norm(self.tau3)]).T
        
        self.Lg12inv = np.diag([1/np.sqrt(self.gc), 1/np.sqrt(self.gc), 1/np.sqrt(self.gt)])
        
        self.LL = np.diag([np.sqrt(self.gc*self.gt)/(np.sqrt(2)*self.alpha), np.sqrt(self.gc*self.gt)/(np.sqrt(2)*self.alpha), self.gc/(np.sqrt(3)*self.gamma)])
        
        
    def build_control_matrices(self):
        
        self.A1 = np.array([
            [-self.lab, self.alpha + self.beta/3, self.alpha + self.beta/3],
            [-self.alpha + self.beta/3, self.lab/2, -(2/3.0)*self.beta],
            [-self.alpha + self.beta/3, -(2/3.0)*self.beta, self.lab/2]
            ])
        
        self.A2 = np.sqrt(3)*np.array([
            [0, (self.alpha - self.beta)/3, (self.beta - self.alpha)/3],
            [-(self.beta + self.alpha)/3, self.lab/2, -(2/3.0)*self.alpha],
            [(self.beta + self.alpha)/2, (2/3.0)*self.alpha, -self.lab/2]
            ])
        
        self.A3 = np.array([
            [0, -self.gamma, self.gamma],
            [self.gamma, 0, -self.gamma],
            [-self.gamma, self.gamma, 0]
            ])
        
        self.M1 = 2*self.alpha*np.array([
            [0, 0, 1],
            [-1, 0, 0],
            [.1, 0, 0]
            ])
        
        self.M2 = (2 * self.alpha/np.sqrt(3))*np.array([
            [0, 1, -1],
            [-1, 0, -2],
            [1, 2, 0]
            ])
        
        self.M3 = self.A3
        
        self.F0 = self.aa*np.array([
            [-2, 1, 1],
            [0, np.sqrt(3), - np.sqrt(3)]
            ])
        
        
    def __init__(self, a, xi0):
        self.a = a
        self.xi0 = xi0   
        
  
        # standard basis of R^3
        self.e1 = np.array([1,0,0])
        self.e2 = np.array([0,1,0])
        self.e3 = np.array([0,0,1])
        
        self.f1 = np.array([1,0])
        self.f2 = np.array([0,1])
        
        self.basis_r3 = (self.e1, self.e2, self.e3)
        
        
        self.build_params()
        
        self.build_energy_matrices()
        
        self.build_control_matrices()
        
        
    def net_displacement(self, xi, dxidt, split=False):
        
        integrand1 = lambda t: np.dot(np.dot(self.M1, dxidt(t)), xi(t))
        integrand2 = lambda t: np.dot(np.dot(self.M2, dxidt(t)), xi(t))
        integrand3 = lambda t: np.dot(np.dot(self.M3, dxidt(t)), xi(t))     
        
        int1 = quad(integrand1, 0, 2*np.pi)[0]
        int2 = quad(integrand2, 0, 2*np.pi)[0]
        int3 = quad(integrand3, 0, 2*np.pi)[0]
        
        if split:
            return (int1*self.e1 + int2*self.e2), int3*self.e3
        
        else:
            return int1*self.e1 + int2*self.e2 + int3*self.e3
        
    def check_lin_dep(self, u, v, tol = 10E-13):
        """
    
        Parameters
        ----------
        u : numpy array nx1
            DESCRIPTION.
        v : numpy array nx1
            DESCRIPTION.
        tol : tolerance, optional
            tolerance for being zero. The default is 10E-13.
    
        Returns
        -------
        True if u and v are linearly dependent and False otherwise.
    
        """
        
        if np.abs(np.linalg.norm(u)*np.linalg.norm(v) - np.dot(u,v)) < tol:
            return True
        else:
            return False
        
    def optimal_curve(self, dp, full=False):
   
        ULginv12 = np.matmul(self.U, self.Lg12inv)
        
        w = np.dot(self.LL, dp)
        # print("w : ", w)
        
        if not self.check_lin_dep(w, self.e1):
            sigma = np.sqrt(norm(w))*np.cross(self.e1, w)/norm(np.cross(self.e1, w))
            
        elif not self.check_lin_dep(w, self.e2):         
            sigma = np.sqrt(norm(w))*np.cross(self.e2, w)/norm(np.cross(self.e2, w))

        else:
            sigma = np.sqrt(norm(w))*np.cross(self.e3, w)/norm(np.cross(self.e3, w))
            

            
        what = w/norm(w)
        
        # print("sigma : ", sigma, np.cross(sigma, what))
        
        u = (1/np.sqrt(2*np.pi))*np.dot(ULginv12, sigma)
        v = (1/np.sqrt(2*np.pi))*np.dot(ULginv12, np.cross(sigma, what))
        
        xi = lambda t: np.cos(t)*u + np.sin(t)*v
        dxidt = lambda t: -np.sin(t)*u + np.cos(t)*v
        coeffs = (u, v)
        
        if full:
            return (xi, dxidt, coeffs)
        
        else:
            return xi
        
    def R(self, theta):
        
        return np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
            ])
        
    def rhs(self, t, u, xi, dxidt):
        theta = u[-1]
        
        dthetadt = -np.dot(np.dot(self.M3, dxidt(t)), xi(t)) 
        dcdt =  np.dot(self.R(theta), np.dot(self.F0, dxidt(t))) + np.dot(self.R(theta), (np.dot(np.dot(self.A1, xi(t)), dxidt(t))\
                *self.f1 + np.dot(np.dot(self.A2, xi(t)),  dxidt(t))*self.f2 ))
            
        dudt = np.array([0,0,0])
        
        dudt[:2] = dcdt
        dudt[-1] = dthetadt
        
        return dudt
    
    
    def rhs_c(self, t, x, xi, dxidt, coeffs):
        u,v = coeffs
        sigma = np.dot(self.gamma*np.cross(u, v), self.tau3)
        
        R = self.R(sigma*t)

        # dcdt = np.dot(R, (np.dot(np.dot(self.A1, xi(t)),\
        #     dxidt(t))*self.f1 + np.dot(np.dot(self.A2, xi(t)),  dxidt(t))*self.f2 ))
            
        dcdt = np.dot(R, np.dot(self.F0, dxidt(t))) + np.dot(R, (np.dot(np.dot(self.A1, dxidt(t)),\
            xi(t))*self.f1 + np.dot(np.dot(self.A2, dxidt(t)),  xi(t))*self.f2 ))

        return dcdt
            
def convergence_plot(ax, epsilons, errors, xlab, ylab, title):
    ax.loglog(epsilons, errors)
    ax.loglog(epsilons, (1.0*epsilons)**1, "k--", label = "O(1)")
    ax.loglog(epsilons, (1.0*epsilons)**2, "k-.", label = "O(2)")
    ax.loglog(epsilons, (1.0*epsilons)**(3), "r-.", label = "O(3)")
    ax.legend(loc = "lower right")
    ax.set_ylabel(ylab)
    ax.set_xlabel(xlab)
    ax.set_title(title)
    
def rel_convergence_plot(ax, epsilons, errors, xlab, ylab, title):
    ax.plot(epsilons, errors)
    ax.set_ylabel(ylab)
    ax.set_xlabel(xlab)
    ax.set_title(title)
    
    
    
def main():
    a = 1/100
    xi0 = 10
    swimmer = spr3(a, xi0)
    # eps = 1/100

    # dp1 = np.array([0,0,1])*eps
    
    # xi, dxidt, coeffs = swimmer.optimal_curve(dp1, full=True)
    # u, v = coeffs
    # print("th net disp : ", dp1)
    # # print("u = ", u)
    # # print("v = ", v)
    # # print("v x u : ", np.cross(v,u))
    # # print("w : ", np.dot(swimmer.LL, dp1))
    # dp2 = swimmer.net_displacement(xi, dxidt)
    
    # print("exp net disp : ", dp2)
    
    
    # y0= np.array([0,0,0])
    # t_span = [0, 2*np.pi]
    # sol = solve_ivp(swimmer.rhs, t_span, y0, args=(xi, dxidt))
    # print("int syst : ", sol.y[:2,-1])
    
    
    # y0 = np.array([0, 0])
    # sol_c = solve_ivp(swimmer.rhs_c, t_span, y0, args=(xi, dxidt, coeffs))
    # print("int sys c : ", sol_c.y[:,-1])
    
    
    # print("||int sys c - exp net disp||/eps**2", norm(sol_c.y[:,-1] - dp2[:2])/eps**2)
    
    
    # convergence in eps
    

    
    
    dp_simple_spat = np.array([1 ,0 ,0])
    dp_simple_rot = np.array([0 ,0 ,1])
    dp_nonsimple = np.array([1, 0, 1])
    dp_full = np.array([1, 1, 1])
    
    dps = [dp_simple_spat, dp_simple_rot, dp_nonsimple, dp_full]
    
    label_simple_spat = "$ \delta p = (1, 0, 0)$"
    label_simple_rot = "$ \delta p = (0, 0, 1)$"
    label_nonsimple = "$ \delta p = (1, 0, 1)$"
    label_full = "$ \delta p = (1, 1, 1)$"
    
    labels = [label_simple_spat, label_simple_rot, label_nonsimple, label_full]
    
    rel_errors = []
    abs_errors = []
    
    epsilons = np.linspace(1/100000, 1, 300)

    for dp in dps:
        new_rel_errs = []
        new_abs_errs = []
        for eps in epsilons:
        
            xi, dxidt, coeffs = swimmer.optimal_curve(dp, full=True)
            xi_eps = lambda t: eps*xi(t)
            dxidt_eps = lambda t: eps*dxidt(t)
            u, v = coeffs
            u_eps = eps*u
            v_eps = eps*v
            coeffs_eps = (u_eps, v_eps)
            
            dp_exp = swimmer.net_displacement(xi_eps, dxidt_eps)
            
            t_span = [0, 2*np.pi]
            y0 = np.array([0, 0])
            sol_c = solve_ivp(swimmer.rhs_c, y0, t_span, method="RK45", atol=10E-6, rtol=10E-6, args=(xi_eps, dxidt_eps, coeffs_eps))
            diff = sol_c.y[:,-1] - sol_c.y[:,0]
    
        
            new_rel_errs.append( norm(diff- dp_exp[:2])/norm(dp_exp[:2]))
            new_abs_errs.append( norm(diff - dp_exp[:2]))
            
            
        new_rel_errs = np.array(new_rel_errs)
        new_abs_errs = np.array(new_abs_errs)
        rel_errors.append(new_rel_errs)
        abs_errors.append(new_abs_errs)
        
        
    
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize = (18,8))
    for errors, label, ax in zip(abs_errors, labels, axs[0,:]):
        convergence_plot(ax, epsilons, errors, "$\epsilon$", "$||\Delta c - \delta c||$", label)
    for errors, label, ax in zip(rel_errors, labels, axs[1,:]):
        rel_convergence_plot(ax, epsilons, errors, "$\epsilon$", "$||\Delta c - \delta c||/||\delta c||$", label)
    plt.tight_layout()
    

main()