#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 09:26:05 2020

@author: philipp
"""
import numpy as np
import clifford as cf
from numpy.linalg import norm, det
from scipy.integrate import quad, solve_ivp
from odeintw import odeintw
from scipy.linalg import expm
from myodeintw import myodeintw

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




class spr4:
        
    def get_radius(self):
        return self.a
        
    def set_radius(self, a):
        self.a = a
        
    def get_init_length(self):
        return self.xi0
        
    def set_init_length(self, xi0):
        self.xi0 = xi0
        
    
    def decompose3d(self, coeffs, basis, tol = 10E-17):
        """
    
        Parameters
        ----------
        coeffs : numpy array 3x1 [a12, a13, a23]
        basis : tuple containing 3 vectors (u1,u2,u3) of size nx1
        
        The parameters coeffs and basis vectors represent the bivector
            w = a12(u1^u2) + a13(u1^u3) + a23(u2 ^u3)
    
        Returns
        -------
        a tuple of vectors (u,v) such that w = u^v
    
        """
        u1, u2, u3 = basis
        a12 = coeffs[0]
        a13 = coeffs[1]
        a23 = coeffs[2]
        
        if np.abs(a13) < tol:
            u = a12*u1 - a23*u3
            v = u2
            
            return (u,v)
        
        else:
            u = (u1 + (a23/a13)*u2)
            v = (a12 * u2 + a13*u3)
            
            return (u,v)


    def check_is_simple(self, coeffs, tol = 10E-17):
        """
        
    
        Parameters
        ----------
        coeffs : numpy array 6x1 [a14, a24, a34, a23, a31, a12] containing floats
    
        tol : float, optional
            tolerance for being zero. The default is 10E-13.
            
        The coefficients represent the coefficients of a bivector w of R^4 
        exressed in the ordered basis (e14, e24, e34, e23, e31, e12) with eij = ei^ej, i.e.
        w = a14 * e14 + a24 * e12 + a34 * e34 + a23 * e23 + a31 * e31 + a12 * e12,
        cf. report p. 18, (5.15)
        
        Returns 
        -------
        True if w^w = 0, False otherwise.
    
        """
        
        layout, blades = cf.Cl(4)
        
        
        e12 = blades['e12']
        e31 = -blades['e13']
        e14 = blades['e14']
        e23 = blades['e23']
        e24 = blades['e24']
        e34 = blades['e34']
        
        a14 = coeffs[0]
        a24 = coeffs[1]
        a34 = coeffs[2]
        a23 = coeffs[3]
        a31 = coeffs[4]
        a12 = coeffs[5]
        
        w = a14 * e14 + a24 * e24 + a34 * e34 + a23 * e23 + a31 * e31 + a12 * e12
        
    
        
        if abs(w^w) < tol:
            return True
        
        else:
            return False
    
    
    def decompose4d(self, coeffs, basis, tol = 10E-17, trace=False):
        """
        Parameters
        ----------
        coeffs : numpy array 6x1 a14, a24, a34, a23, a31, a12] containing floats
    
        basis : tuple of vectors (e1, e2, e3, e4)
            ordered basis of R^4.
        tol : float, optional
            tolerance for being zero. The default is 10E-13.
        The coeffs give the coefficients of a bivector w of R^4 expressed in the 
        ordered basis (e14, e24, e34, e23, e31, e12) with eij = ei^ej 
        Returns
        -------
        tuple of vectors (u1, v1, u2, v2) which satisfy w = u1 ^v1 + u2^v2.
    
        """
        e1, e2, e3, e4 = basis
        
        a14 = coeffs[0]
        a24 = coeffs[1]
        a34 = coeffs[2]
        a23 = coeffs[3]
        a31 = coeffs[4]
        a12 = coeffs[5]
        
        u1 = (a14*e1 + a24*e2 + a34*e3)
        v1 = e4
        
        if trace:
            print("a31= " , a31)
        
        if np.abs(a31) < tol:
            u2 = a12*e1 - a23*e3
            v2 = e2
            
        else:
            u2 = a23*e2 - a31*e1
            v2 = e3 - (a12/a31)*e2
            
        return (u1, v1, u2, v2)
    
    def check_lin_dep(self, u, v, tol = 10E-17):
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
    
    def check_decomposition(self, u1, v1, u2, v2, w, tol=10E-17):
        """
        

        Parameters
        ----------
        u1 : numpy array 4x1, float
            vector in R^4.
        v1 : numpy array 4x1, float
            ditto.
        u2 : numpy array 4x1, float
            ditto.
        v2 : numpy array 4x1, float
            ditto.
        w : numpy array 6x1, float
            bivector of R^4 expressed as a vector of R^6 in the special basis.
        tol : float, optional
            Error tolerance. The default is 10E-17.

        Returns
        -------
        bool
            Returns True if w = u1^v1 + u2^v2, False otherwise.

        """
        
        
        layout, blades = cf.Cl(4)
        
        
        ee1 = blades['e1']
        ee2 = blades['e2']
        ee3 = blades['e3']
        ee4 = blades['e4']
        
        ee12 = blades['e12']
        ee31 = -blades['e13']
        ee14 = blades['e14']
        ee23 = blades['e23']
        ee24 = blades['e24']
        ee34 = blades['e34']
        
        ww = w[0]*ee14 + w[1]*ee24 + w[2]*ee34 + \
            w[3]*ee23 + w[4]*ee31 + w[5]*ee12
    
            
        uu1 = u1[0]*ee1 + u1[1]*ee2 + u1[2]*ee3 + u1[3]*ee4
        uu2 = u2[0]*ee1 + u2[1]*ee2 + u2[2]*ee3 + u2[3]*ee4
        
        vv1 = v1[0]*ee1 + v1[1]*ee2 + v1[2]*ee3 + v1[3]*ee4
        vv2 = v2[0]*ee1 + v2[1]*ee2 + v2[2]*ee3 + v2[3]*ee4
        
        if abs(ww - (uu1^vv1) - (uu2^vv2)) <= tol:
            return True
        
        else:
            # print("u1^v1 = ", uu1^vv1)
            # print("u2^v2 = ", uu2^vv2)
            print("w = ", ww)
            print("u1^v1 + u2^v2 = ", (uu1^vv1) + (uu2^vv2))
            print("error", abs(ww - (uu1^vv1) - (uu2^vv2)))
    
            return False
    
    
    
    
    def decompose4d_simple(self, coeffs, basis, tol=10E-13, trace = False):
        """
        Parameters
        ----------
        coeffs : numpy array 6x1 a14, a24, a34, a23, a31, a12] containing floats
    
        basis : tuple of vectors (e1, e2, e3, e4)
            ordered basis of R^4.
        tol : float, optional
            tolerance for being zero. The default is 10E-10.
        The coeffs give the coefficients of a bivector w of R^4 expressed in the 
        ordered basis (e14, e24, e34, e23, e31, e12) with eij = ei^ej
        The bivector w has to satisfy w^w = 0. Otherwise an error is raised.
        Returns
        -------
        tuple of vectors (u,v) such that w = u^v
    
        """
        
        e1, e2, e3, e4 = basis
        
        a14 = coeffs[0]
        a24 = coeffs[1]
        a34 = coeffs[2]
        a23 = coeffs[3]
        a31 = coeffs[4]
        a12 = coeffs[5]
        
        u = (a14*e1 + a24*e2 + a34*e3)
        if trace:
            print("u = ",u)
        
        if np.abs(a31) < tol:
            u1 = a12*e1 - a23*e3
            u2 = e2
            
        else:
            u1 = a23*e2 - a31*e1 
            u2 = e3 - (a12/a31)*e2
            
        if trace:
            print("u1 = ", u1)
            print("u2 = ", u2)
            
        if self.check_lin_dep(u1, u2):
            if trace:
                print("u1, u2 are linearly dependent")
            return (u,e4)
        
        else:
            if trace:
                print("u1, u2 are linearly independent")
            l1 = np.dot(u1, u)/norm(u1)**2
            l2 = np.dot(u2, u)/norm(u2)**2
            
            basis3d = (u1, u2, e4)
            coeffs3d = np.array([1.0, l1, l2])
            
            if trace:
                print("basis3d = ", u1, u2, e4)
                print("coeffs3d = ", coeffs3d)
            
            
            (u,v) = self.decompose3d(coeffs3d, basis3d)
            
            return (u,v)
        
    def post_processor(self, u,v):
        """
        
    
        Parameters
        ----------
        u : numpy array nx1
            vector in R^n.
        v : numpy array nx1
            vector in R^n.
            
        It is assumed that u and v are linearly independent since we only use the
        post processing step for u^v nonzero.
    
        Returns
        -------
        tuple of vectors (u_new, v_new) such that u_new^v_new = u^v, |u_new| = |v_new|
        and u_new.v_new = 0.
        """
        # TODO: Raise error if u,v are linearly dependent
        
        # step 1: orthogonalize u,v with Gram-Schmidt
        u_temp = u
        v_temp = v - (np.dot(u_temp, v)/norm(u_temp)**2)*u_temp
        
        # step 2: balance the lengths
        # Recall that since u_temp and v_temp are orthogonal, we have |u_temp^v_temp| = |u_temp||v_temp|
        
        u_new = u_temp*np.sqrt(norm(v_temp)/norm(u_temp))
        v_new = v_temp*np.sqrt(norm(u_temp)/norm(v_temp))
        
        return (u_new, v_new)
        
        
        
    def build_params(self):
        """
        

        Returns
        -------
        None. This function initializes all the parameters (order 1 approximation),
        see report p.28 - p.29

        """
        self. kappa = 3/4 + (9/16)*np.sqrt(3/2)*(self.a/self.xi0)
        
        self.h = 1/12 + (3/16)*np.sqrt(3/2)*(self.a/self.xi0)
        
        self.gc = 2/3 + (3/8)*np.sqrt(3/2)*(self.a/self.xi0)
        
        self.gt = 1 + np.sqrt(3/2)*(9/8)*(self.a/self.xi0)
        
        self.aa = -1/4 + (3/32)*np.sqrt(3/2)*(self.a/self.xi0)
        
        self.alpha = np.sqrt(3)/256*(self.a/self.xi0**2)
        
        self.beta = -3/128*np.sqrt(3/2)*(self.a/self.xi0**2)   
        
        self.lam = -9/128*np.sqrt(3/2)*(self.a/self.xi0**2)
        
        self.delta = (1/self.xi0**2)*(-1/(16*np.sqrt(6)) + (9/512)*(self.a/self.xi0))
        
        self.hc = -2*np.sqrt(6)*self.alpha
        
        self.ht = -2*np.sqrt(6)*self.delta
        
        self.energy_params = (self.kappa, self.h, self.gc, self.gt, self.hc, self.ht)
        
        self.control_params = (self.aa, self.alpha, self.beta, self.lam, self.delta)
        
        self.tau1 = (1/np.sqrt(6))*np.array([-2, 1, 1, 0])
        
        self.tau2 = (1/np.sqrt(2))*np.array([0, 1, -1, 0])
        
        self.tau3 = (1/(2*np.sqrt(3)))*np.array([1, 1, 1, -3])
        
        self.tau4 = (1/2)*np.array([1, 1, 1, 1])
        
        self.tau_basis = (self.tau1, self.tau2, self.tau3, self.tau4)
    
    
    
    def build_energy_matrices(self):
        """
        

        Returns
        -------
        None. This function initializes all the matrices related to the energy
        functional. See report p. 16, p. 18

        """

        self.G = np.array([[self.kappa,self.h,self.h,self.h],[self.h,self.kappa,self.h,self.h],[self.h,self.h,self.kappa,self.h],[self.h,self.h,self.h,self.kappa]])
        
        self.U = np.array([self.tau1, self.tau2, self.tau3, self.tau4]).T
        
        self.Lg = np.diag([self.gc, self.gc, self.gc, self.gt])
        
        self.Lg12 = np.diag([np.sqrt(self.gc), np.sqrt(self.gc), np.sqrt(self.gc), np.sqrt(self.gt)])
        
        self.Lg12inv = np.diag([1/np.sqrt(self.gc), 1/np.sqrt(self.gc), 1/np.sqrt(self.gc), 1/np.sqrt(self.gt)])
         
        self.tildeLg = np.diag([self.gc, self.gc, self.gc, np.sqrt(self.gc*self.gt), np.sqrt(self.gc*self.gt), np.sqrt(self.gc*self.gt)])
        
        self.Lh = np.diag([self.hc, self.hc, self.hc, self.ht, self.ht, self.ht])
        
        
    
    def build_control_matrices(self):
        """
        

        Returns
        -------
        None. This function initializes all the matrices related to the first order
        approximation of the control system, cf. report p. 10 eq. (4.2), (4.2). 
        The matrices are on p. 12-13 and p.31

        """
        self.M1 = self.alpha*np.array([
            [0, 3, 3, 2],
            [-3, 0, 0, -1],
            [-3, 0, 0, -1],
            [-2, 1, 1, 0]
            ])
        
        self.M2 = np.sqrt(3)*self.alpha*np.array([
            [0, 1, -1, 0],
            [-1, 0, -2, -1],
            [1, 2, 0, 1],
            [0, 1, -1, 0]
            ])
        
        self.M3 = 2*np.sqrt(2)*self.alpha*np.array([
            [0, 0, 0, -1],
            [0, 0, 0, -1],
            [0, 0, 0, -1],
            [1, 1, 1, 0]
            ])
        
        self.M4 = self.delta*np.array([
            [0, 1, -1, 0],
            [-1, 0, -2, 3],
            [1, 2, 0, -3],
            [0, -3, 3, 0]
            ])
    
        self.M5 = np.sqrt(3)*self.delta*np.array([
            [0, -1, -1, 2],
            [1, 0, 0, -1],
            [1, 0, 0, -1],
            [-2, 1, 1, 0]
            ])
        
        self.M6 = 2*np.sqrt(2)*self.delta*np.array([
            [0, 1, -1, 0],
            [-1, 0, 1, 0],
            [1, -1, 0, 0],
            [0, 0, 0, 0]
            ])
        
      
        self.Fc0 = -3*np.sqrt(3)*self.aa*np.array([self.tau1, self.tau2, self.tau3])
        
        self.Ft0 = np.zeros((3, 4))
        
        self.F0 = np.vstack((self.Fc0, self.Ft0))
        
        self.N1 = (np.sqrt(2)/3.0)*np.array([
            [2*self.lam, -self.beta, -self.beta, -2*self.beta],
            [-self.beta, -self.lam, 2*self.beta, self.beta],
            [-self.beta, 2*self.beta, -self.lam, self.beta],
            [-2*self.beta, self.beta, self.beta, 0]
            ])
        
        self.N2 = np.sqrt(2/3.0)*np.array([
            [0, self.beta, -self.beta, 0],
            [self.beta, -self.lam, 0, self.beta],
            [-self.beta, 0, self.lam, -self.beta],
            [0, self.beta, -self.beta, 0]
            ])
        
        self.N3 = (1/3.0)*np.array([
            [-self.lam, 2*self.beta, 2*self.beta, -2*self.beta],
            [2*self.beta, -self.lam, 2*self.beta, -2*self.beta],
            [2*self.beta, 2*self.beta, -self.lam, -2*self.beta],
            [-2*self.beta, -2*self.beta, -2*self.beta, 3*self.lam]
            ])
        
        self.A1 = self.M1 + self.N1
        self.A2 = self.M2 + self.N2
        self.A3 = self.M3 + self.N3
            
          
    def __init__(self, a, xi0):
        self.a = a
        self.xi0 = xi0   
        
  
        # standard basis of R^4
        self.e1 = np.array([1,0,0,0])
        self.e2 = np.array([0,1,0,0])
        self.e3 = np.array([0,0,1,0])
        self.e4 = np.array([0,0,0,1])
        
        self.basis_r4 = (self.e1, self.e2, self.e3, self.e4)
        
        # standard basis of so(3), i.e. infinitesimal rotations
        self.L1 = np.array([[0,0,0],[0,0,-1],[0,1,0]])
        self.L2 = np.array([[0,0,1],[0,0,0],[-1,0,0]])
        self.L3 = np.array([[0,-1,0 ],[1,0,0],[0,0,0]])
        
        # standard basis of R^3
        self.f1 = np.array([1,0,0])
        self.f2 = np.array([0,1,0])
        self.f3 = np.array([0,0,1])
        
        self.build_params()
        
        self.build_energy_matrices()
        
        self.build_control_matrices()
    
    
    def net_displacement(self, xi, dxidt, split=False):
        """
        

        Parameters
        ----------
        xi : vector valued function returning numpy arrays (4,)
            control curve J -> R^4.
        dxidt : vector valued function returning numpy arrays (4,)
            time derivative of xi.
        split : bool, optional
            If False, the net displacement is returned as a (6,) array. If True,
            then the net displacement is SPLIT into spatial and rotational parts.
            The default is False.

        Returns
        -------
        numpy array (6,) or tuple of two arrays (3,)
            This function returns the net displacement due to a control curve, 
            see equation (5.6) on p. 16 in the report.

        """
        
        
        integrand1 = lambda t: self.hc * det(np.array([xi(t), dxidt(t), self.tau2, self.tau3]))
        integrand2 = lambda t: self.hc * det(np.array([xi(t), dxidt(t), self.tau3, self.tau1]))
        integrand3 = lambda t: self.hc * det(np.array([xi(t), dxidt(t), self.tau1, self.tau2]))
        
        integrand4 = lambda t: self.ht * det(np.array([xi(t), dxidt(t), self.tau1, self.tau4]))
        integrand5 = lambda t: self.ht * det(np.array([xi(t), dxidt(t), self.tau2, self.tau4]))
        integrand6 = lambda t: self.ht * det(np.array([xi(t), dxidt(t), self.tau3, self.tau4]))
        
        position_integrands = [integrand1, integrand2, integrand3]
        orientation_integrands = [integrand4, integrand5, integrand6]
        integrands = [integrand1, integrand2, integrand3, integrand4, integrand5, integrand6]
    
        if split:
    
            dc = [ quad(integrand, 0, 2*np.pi)[0] for integrand in position_integrands]
            dR = [ quad(integrand, 0, 2*np.pi)[0] for integrand in orientation_integrands]
        
            return  (np.array(dc), np.array(dR))
    
        else:
            
            dp = [quad(integrand, 0, 2*np.pi)[0] for integrand in integrands]
            
            return np.array(dp)
        
        
        
        
    def optimal_curve(self, dp, full=False):
        """
        

        Parameters
        ----------
        dp : numpy array 6x1
            prescribed net displacement in R^3 x so(3) expressed in the standard
            basis (defined above).
        full : bool, optional
            Controls the output format. The default is False.

        Returns
        -------
        If full: tuple (xi, dxidt, coeffs)
            xi: optimal control curve that realizes dp
            dxidt: time derivative of xi
            coeffs: tuple (a1, b1, a2, b2) containing the Fourier coefficients of xi
        If not full: only returns xi

        """
        
        # rescaling of the net displacement, see Prop. 10., eq. (5.25)
        ULginv12 = np.matmul(self.U, self.Lg12inv)
        LLinv = np.linalg.inv(np.matmul(self.Lh, self.tildeLg))
        w = np.sqrt(det(self.Lg))*np.dot(LLinv, dp)
        
        # distinction of simple and general cases
        # if simple, then the curve is constructed according to Thm. 14.
        # if not simple, then the curve is constructed according to Thm. 17.
        if self.check_is_simple(w):
            
            vraw, uraw = self.decompose4d_simple(w, self.basis_r4)
            v, u = self.post_processor(vraw, uraw)

            a = (1/np.sqrt(2*np.pi))*np.matmul(ULginv12, u)
            b = (1/np.sqrt(2*np.pi))*np.matmul(ULginv12, v)
            
            xi = lambda t: np.cos(t)*a + np.sin(t)*b
            coeffs = (a, b, np.zeros((4,)), np.zeros((4,)))
            dxidt = lambda t: -np.sin(t)*a + np.cos(t)*b
               
            if full:
                return (xi, dxidt, coeffs)
            
            else:
                return xi
            
        else:
            
            v1raw, u1raw, v2raw, u2raw = self.decompose4d(w, self.basis_r4)
            v1, u1 = self.post_processor(v1raw, u1raw)
            v2, u2 = self.post_processor(v2raw, u2raw)
     
            
            if norm(u2)**2 + norm(v2)**2 <= norm(u1)**2 + norm(v1)**2:
                a1 = (1/np.sqrt(2*np.pi))*np.matmul(ULginv12, u1)
                b1 = (1/np.sqrt(2*np.pi))*np.matmul(ULginv12, v1)
                
                a2 = (1/np.sqrt(4*np.pi))*np.matmul(ULginv12, u2)
                b2 = (1/np.sqrt(4*np.pi))*np.matmul(ULginv12, v2)
        
            else:
                a1 = (1/np.sqrt(2*np.pi))*np.matmul(ULginv12, u2)
                b1 = (1/np.sqrt(2*np.pi))*np.matmul(ULginv12, v2)
                
                a2 = (1/np.sqrt(4*np.pi))*np.matmul(ULginv12, u1)
                b2 = (1/np.sqrt(4*np.pi))*np.matmul(ULginv12, v1)
            
            xi = lambda t: np.cos(t)*a1 + np.sin(t)*b1 + np.cos(2*t)*a2 + np.sin(2*t)*b2
            dxidt = lambda t: -np.sin(t)*a1 + np.cos(t)*b1 - 2*np.sin(2*t)*a2 + 2*np.cos(2*t)*b2
            coeffs = (a1, b1, a2, b2)
            
            
            if full:
                return (xi, dxidt, coeffs)
            
            else:
                return xi   
            
            
    def energy(self, dxidt):
        """
        

        Parameters
        ----------
        dxidt : vector valued function returning numpy arrays 4x1
            time derivative of a control curve xi: J -> R^4.

        Returns
        -------
        float
            energy consumed by the control curve xi during one period.

        """
        integrand = lambda t: np.dot(self.G.dot(dxidt(t)), dxidt(t))
            
        return quad(integrand, 0, 2*np.pi)[0]
    
    
    
    def rhs(self, t, u, xi, dxidt):
        """
        

        Parameters
        ----------
        u : numpy array 12x1
            the first row contains the current position, the remaining entries
            contain the current orientation as a rotation matrix.
        t : float
            current time.
        xi : vector valued function returning 4x1 arrays
            control curve .
        dxidt : vector valued function returning 4x1 arrays
            time derivative of the control curve.

        Returns
        -------
        dudt : numpy array 4x3
            current time derivative.

        """
                
        R1 = u[3:6]
        R2 = u[6:9]
        R3 = u[9:]
        
        R = np.vstack((R1, R2, R3))
        
        dcdt = np.dot(R,(np.dot(self.Fc0, dxidt(t)) + np.dot(np.dot(self.A1, xi(t)), dxidt(t))\
            *self.f1 + np.dot(np.dot(self.A2, xi(t)),  dxidt(t))*self.f2 + np.dot(np.dot(self.A3,\
                                                        xi(t)), dxidt(t))*self.f3))

        dRdt = np.dot(R,(np.dot(np.dot(self.M4, xi(t)), dxidt(t))*self.L1 + np.dot(np.dot(self.M5, \
            xi(t)), dxidt(t))*self.L2 + np.dot(np.dot(self.M6, xi(t)), dxidt(t))*self.L3))
        # I = np.eye(3)
        
        # dcdt = np.dot(R, (np.dot(np.dot(self.A1, xi(t)), dxidt(t))\
        #     *self.f1 + np.dot(np.dot(self.A2, xi(t)),  dxidt(t))*self.f2 + np.dot(np.dot(self.A3,\
        #                                                 xi(t)), dxidt(t))*self.f3))

        # dRdt = np.dot(R,(np.dot(np.dot(self.M4, xi(t)), dxidt(t))*self.L1 + np.dot(np.dot(self.M5, \
        #     xi(t)), dxidt(t))*self.L2 + np.dot(np.dot(self.M6, xi(t)), dxidt(t))*self.L3))
        

        

        # dudt = np.zeros((4,3))
        # dudt[0,:] = dcdt
        # dudt[1:,:] = dRdt
        
        dudt = np.hstack((dcdt, dRdt[0,:], dRdt[1,:], dRdt[2,:]))
        
        return dudt
            
            
            
            
def pos_array_to_pos_tuples(pos):
    n = len(pos)
    
    tuples = []
    
    for i in range(n):
        new_tuple = (pos[i, 0], pos[i, 1], pos[i, 2])
        tuples.append(new_tuple)
        
        
    return tuples


def calculate_sphere_positions(swimmer, xi, dxidt, n_strokes, fps, amplification, filepath = "/Users/philipp/Documents/GitHub/stage_cmap/python/"):
    """
    

    Parameters
    ----------
    swimmer : spr4 object
    xi : vector valued function returning 4x1 arrays
        control curve .
    dxidt : vector valued function returning 4x1 arrays
        time derivative of the control curve.
    n_strokes : int
        number of strokes.
    fps : int
        frames/s.
    amplification : float
        amplification of the results.
    filepath : string, optional
        filepath to where one wants to save the histories.
        The default is "/Users/philipp/Documents/GitHub/stage_cmap/python/".

    Returns
    -------
    None. Writes the positions of the 4 spheres and the center into csv files.

    """
    xi0 = swimmer.get_init_length()
    
    # discretization
    T = n_strokes*2*np.pi
    n_keyframes = int(T*fps)
    dt = 1/fps
    
    
    R0 = np.eye(3)
    c0 = np.array([0,0,0])
    p0 = np.vstack((c0, R0))
    tspan = np.linspace(0, T, n_keyframes)
    
    sol = odeintw(swimmer.rhs, p0, tspan, args= (xi, dxidt))
    solc = sol[:, 0, :]*amplification
    solR = sol[:, 1:, :]
    
    hist1 = []
    hist2 = []
    hist3 = []
    hist4 = []
    histc = []
    
    for k in range(len(sol)):
        newpos1 = solc[k,:] + np.dot(solR[k,:], (xi0 + xi(k*dt)[0]) *z1 )
        hist1.append(newpos1)
        
        newpos2 = solc[k,:] + np.dot(solR[k,:], (xi0 + xi(k*dt)[1]) *z2 )
        hist2.append(newpos2)
        
        newpos3 = solc[k,:] + np.dot(solR[k,:], (xi0 + xi(k*dt)[2]) *z3 )
        hist3.append(newpos3)
        
        newpos4 = solc[k,:] + np.dot(solR[k,:], (xi0 + xi(k*dt)[3]) *z4 )
        hist4.append(newpos4)
        
        newposc = solc[k,:]
        histc.append(newposc)
        
        
    
    hist1 = np.array(hist1)  
    np.savetxt(filepath + "loc1.csv",  hist1, delimiter =", ")
    
    hist2 = np.array(hist2)
    np.savetxt(filepath + "loc2.csv",  hist2, delimiter =", ")
    
    hist3 = np.array(hist3)
    np.savetxt(filepath + "loc3.csv",  hist3, delimiter =", ")
    
    hist4 = np.array(hist4)
    np.savetxt(filepath + "loc4.csv",  hist4, delimiter =", ")
    
    histc = np.array(histc)
    np.savetxt(filepath + "locc.csv",  histc, delimiter =", ")
    
    print("# of keyframes: ", n_keyframes)
    
def correction(swimmer, dR, coeffs):
    """
    

    Parameters
    ----------
    swimmer : spr4 object
    dR : numpy array 3x1
        rotational net displacement in the std basis of so(3).
    coeffs : tuple containing 4 numpy arrays 4x1
        Fourier coefficients of a control curve.

    Returns
    -------
    numpy array 6x1
        Correction term of order 3 in eq. (4.35) on p. 14.

    """
    a1, b1, a2, b2 = coeffs
    
    dR_mat = dR[0]*L1 + dR[1]*L2 + dR[2]*L3
    xi_end = a1 + a2
    

    ht = swimmer.ht
    
    def phi1(u, v):
        C = np.array([u, v, swimmer.tau1, swimmer.tau4])
        return det(C)
    
    def phi2(u, v):
        C = np.array([u, v, swimmer.tau2, swimmer.tau4])
        return det(C)
    
    def phi3(u, v):
        C = np.array([u, v, swimmer.tau3, swimmer.tau4])
        return det(C)
    
    def Phi(u, v):
        
        return phi1(u, v)*L1 + phi2(u, v)*L2 + phi3(u, v)*L3
    

    Fc0 = swimmer.Fc0
    dRFc0 = np.dot(dR_mat, Fc0)
    
    
    A = np.dot((3*ht*np.pi/2)*(Phi(a1, b2) + Phi(a2, b1)), Fc0)
    B = np.dot((3*ht*np.pi/2)*(Phi(a2, a1) + Phi(b2, b1)), Fc0)
    
    
    corr_c = np.dot(dRFc0, xi_end)  - np.dot(A, a1) - np.dot(B, b1)
    
    corr = np.zeros((6,))
    
    corr[:3] = corr_c
    
    return corr
    

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
    # ax.set_xlim(1/5000, 1)
    ax.set_ylabel(ylab)
    ax.set_xlabel(xlab)
    ax.set_title(title)
    
    


def main():

    a = 0.01
    xi0 = 10
    
    swimmer = spr4(a, xi0)
    eps = 1/10000
    dp = np.array([0, 0, 1, 0, 0, 0])*eps
    dR = dp[3:]
    
    xi, dxidt, coeffs = swimmer.optimal_curve(dp, full=True)
    
    n_strokes = 1
    T = np.linspace(0, n_strokes*2*np.pi, 1000*n_strokes, endpoint=True)
    t_span = [0, 2*np.pi*n_strokes]
    
    
    R0 = np.eye(3)
    c0 = np.array([0,0,0])
    p0 = np.hstack((c0, R0[0,:], R0[1,:], R0[2,:]))

    
    # sol = odeintw(swimmer.rhs, p0, T, args=(xi, dxidt))
    sol = solve_ivp(swimmer.rhs, t_span,p0, t_eval=T,  method="Radau", args=(xi, dxidt))
    sol_c = sol.y[:3,:]
    sol_R = sol.y[3:, :]
    print(sol_c.shape)
    print(sol_R.shape)


    dp_raw = swimmer.net_displacement(xi, dxidt)
    dp_corr = dp_raw + correction(swimmer, dR, coeffs)
    dc_corr = dp_corr[:3]
    
    diff = sol_c[:, -1] - sol_c[:, 0]
    print("th net displacement: ", dp)
    print("exp net displacement without correction: ", dp_raw)
    print("corrected exp net displacement: ", dp_corr)
    print("diff: ", diff)
    print("||dc_corr - diff||/eps**2", norm(dc_corr - diff))

    
    print("====THEORETICAL NET DISPLACEMENT=======")
    print(dp)
    print("------------------------------")
    
    
    print("=======CURVE ============")
    print("difference curve: ", diff)
    print("exp net displacement: ", dp_raw)
    print("corrected exp net displacement: ", dp_corr)
    print("------------------------------")
    

        

    

# main()

