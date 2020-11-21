#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 21:07:26 2020

@author: philipp
"""

import numpy as np
import clifford as cf
from numpy.linalg import norm
from scipy.integrate import quad




"""
CONSTANTS
"""

# standard basis of so(3)
L1 = np.array([[0,0,0],[0,0,-1],[0,1,0]])
L2 = np.array([[0,0,1],[0,0,0],[-1,0,0]])
L3 = np.array([[0,-1,0 ],[1,0,0],[0,0,0]])

basis_so3 = (L1, L2, L3)


# standard basis of R^4
e1 = np.array([1,0,0,0])
e2 = np.array([0,1,0,0])
e3 = np.array([0,0,1,0])
e4 = np.array([0,0,0,1])

basis_r4 = (e1, e2, e3, e4)

# standard basis of R^3
ehat1 = np.array([1,0,0])
ehat2 = np.array([0,1,0])
ehat3 = np.array([0,0,1])

basis_r3 = (ehat1, ehat2, ehat3)




def decompose3d(coeffs, basis, tol = 10E-13):
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


def check_is_simple(coeffs, tol = 10E-13):
    """
    

    Parameters
    ----------
    coeffs : numpy array 6x1 [a14, a24, a34, a23, a31, a12] containing floats

    tol : float, optional
        tolerance for being zero. The default is 10E-13.
        
    The coefficients represent the coefficients of a bivector w of R^4 
    exressed in the ordered basis (e14, e24, e34, e23, e31, e12) with eij = ei^ej, i.e.
    w = a14 * e14 + a24 * e12 + a34 * e34 + a23 * e23 + a31 * e31 + a12 * e12
    
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
    
    
def decompose4d(coeffs, basis, tol = 10E-13):
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
    
    if np.abs(a31) < tol:
        u2 = a12*e1 - a23*e3
        v2 = e2
        
    else:
        u2 = a23*e2 - a31*e1
        v2 = e3 - (a12/a31)*e2
        
    return (u1, v1, u2, v2)

def check_lin_dep(u,v, tol = 10E-13):
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





def decompose4d_simple(coeffs, basis, tol=10E-13, trace = False):
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
    
    # TODO: Raise error in the non-simple case
    
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
        u2 = e3 - (a12/a34)*e2
        
    if trace:
        print("u1 = ", u1)
        print("u2 = ", u2)
        
    if check_lin_dep(u1, u2):
        if trace:
            print("u1, u2 are linearly dependent")
        return (u,e4)
    
    else:
        if trace:
            print("u1, u2 are linearly independent")
        l1 = np.dot(u1, u)
        l2 = np.dot(u2, u)
        
        basis3d = (u1, u2, e4)
        coeffs3d = np.array([1.0, l1, l2])
        
        if trace:
            print("basis3d = ", u1, u2, e4)
            print("coeffs3d = ", coeffs3d)
        
        
        (u,v) = decompose3d(coeffs3d, basis3d)
        
        return (u,v)
    
def post_processor(u,v):
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
    
    
    
def build_params(a, xi0):
    """
    

    Parameters
    ----------
    a : float
        radius of the four balls.
    xi0 : float
        initial lenth of all the four arms .

    Returns
    -------
    Tuple of all relevant parameters of the problem and the basis (tau1, tau2, tau3, tau4).
    energy_params, control_params, tau_basis = (kappa, h, gc, gt, hc, ht), ( aa, alpha, beta, lam, delta ), (tau1, tau2, tau3, tau4)

    """
    kappa = (3/256)*(64 + (25*np.sqrt(6)*a/xi0) + 12*a/(9*a - 2*np.sqrt(6)*xi0))
    
    h = (1/768)*(60 + 69*np.sqrt(6)*a/xi0 - 16*xi0/(3*np.sqrt(6)*a - 4*xi0))
    
    gc = (1/192)*(129 + 39*np.sqrt(6)*a/xi0 + 4*xi0/(3*np.sqrt(6)*a - 4*xi0) + 27*a/(9*a - 2*np.sqrt(6)*xi0))
    
    gt = 1 + np.sqrt(3/2)*9*a/(8*xi0)
    
    aa = -5/16 + xi0/(16*xi0 - 12*np.sqrt(6)*a)
    
    alpha = -(np.sqrt(3)/8)*a/(3 *np.sqrt(6) * a - 4* xi0)**2
    
    beta = -3*a*(81*np.sqrt(2)*a - 16*np.sqrt(3)*xi0)/(128*np.sqrt(2)*xi0*(3*np.sqrt(6)*a - 4*xi0)**2)
    
    lam = 9*a*(27*a - 8*np.sqrt(6)*xi0)/(128*xi0*(3*np.sqrt(6)*a - 4*xi0)**2)
    
    delta = (81*a**2 - 240*np.sqrt(6)*a*xi0 + 256*xi0**2)/(64*xi0**2*(-27*np.sqrt(6)*a**2 - 252*a*xi0 + 64*np.sqrt(6)*xi0**2))
    
    energy_params = (kappa, h, gc, gt)
    
    control_params = (aa, alpha, beta, lam, delta)
    
    tau1 = (1/np.sqrt(6))*np.array([-2, 1, 1, 0])
    
    tau2 = (1/np.sqrt(2))*np.array([0, 1, -1, 0])
    
    tau3 = (1/2*np.sqrt(3))*np.array([1, 1, 1, -3])
    
    tau4 = (1/2)*np.array([1, 1, 1, 1])
    
    tau_basis = (tau1, tau2, tau3, tau4)

    return energy_params, control_params, tau_basis


def build_energy_matrices(energy_params, tau_basis):
    """
    

    Parameters
    ----------
    energy_params : tuple of floats (kappa, h, gc, gt, hc, ht) containing the energy parameters of the problem

    tau_basis : tuple of vectors 4x1  (tau1, tau2, tau3, tau4)
        eigenbasis of the energy functional
    Returns
    -------
    tuple of matrices (G, U, Lg, tildeLg, Lh) where
        G : matrix of the energy functional
        G = U Lg U^T
        Lg = diag(gc, gc, gc, gt)

    """
    kappa, h, gc, gt, hc, ht = energy_params
    tau1, tau2, tau3, tau4 = tau_basis
    
    G = np.array([[kappa,h,h,h],[h,kappa,h,h],[h,h,kappa,h],[h,h,h,kappa]])
    
    U = np.array([tau1, tau2, tau3, tau4]).T
    
    Lg = np.diag([kappa-h, kappa-h, kappa-h, kappa + 3*h])
    
    tildeLg = np.diag([gc, gc, gc, np.sqrt(gc*gt), np.sqrt(gc*gt), np.sqrt(gc*gt)])
    
    Lh = np.diag([hc, hc, hc, ht, ht, ht])
    
    return (G, U, Lg, tildeLg, Lh)
    

def build_control_matrices(control_params, tau_basis, full=False):
    """
    

    Parameters
    ----------
    control_params : tuple of control parameters ( aa, alpha, beta, lam, delta )
    
    tau_basis : tuple of vectors 4x1  (tau1, tau2, tau3, tau4)
        eigenbasis of the energy functional.
    full : boolean, optional
        If TRUE, the full control matrices F0, A1, A2, A3, B1, B2, B3 are returned. Otherwise, only the skew-symmetric parts M1, ..., M6 are returned. 
        The default is False.

    Returns
    -------
    Tuple of matrices.
    if full, then (F0, A1, A2, A3, B1, B2, B3)
    if not full, then (M1, M2, M3, M4, M5, M6)
    """
    
    aa, alpha, beta, lam, delta = control_params
    
    tau1, tau2, tau3, tau4 = tau_basis
    
    M1 = alpha*np.array([
        [0, 3, 3, 2],
        [-3, 0, 0, -1],
        [-3, 0, 0, -1],
        [-2, 1, 1, 0]
        ])
    
    M2 = np.sqrt(3)*alpha*np.array([
        [0, 1, -1, 0],
        [-1, 0, -2, -1],
        [1, 2, 0, 1],
        [0, 1, -1, 0]
        ])
    
    M3 = 2*np.sqrt(2)*alpha*np.array([
        [0, 0, 0, -1],
        [0, 0, 0, -1],
        [0, 0, 0, -1],
        [1, 1, 1, 0]
        ])
    
    M4 = delta*np.array([
        [0, 1, -1, 0],
        [-1, 0, -2, 3],
        [1, 2, 0, -3],
        [0, -3, 3, 0]
        ])

    M5 = np.sqrt(3)*np.array([
        [0, -1, -1, 2],
        [1, 0, 0, -1],
        [1, 0, 0, -1],
        [-2, 1, 1, 0]
        ])
    
    M6 = 2*np.sqrt(2)*delta*np.array([
        [0, 1, -1, 0],
        [-1, 0, 1, 0],
        [1, -1, 0, 0],
        [0, 0, 0, 0]
        ])
    
    if full:
        Fc0 = -3*np.sqrt(3)*aa*np.array([tau1, tau2, tau3])
        
        Ft0 = np.zeros((3, 4))
        
        F0 = np.vstack((Fc0, Ft0))
        
        N1 = (np.sqrt(2)/3)*np.array([
            [0, beta, -beta, 0],
            [beta, -lam, 0, beta],
            [-beta, 0, lam, -beta],
            [0, beta, -beta, 0]
            ])
        
        N2 = np.sqrt(2/3)*np.array([
            [0, beta, -beta, 0],
            [beta, -lam, 0, beta],
            [-beta, 0, lam, -beta],
            [0, beta, -beta, 0]
            ])
        
        N3 = 3*np.array([
            [-lam, 2*beta, 2*beta, -2*beta],
            [2*beta, -lam, 2*beta, -2*beta],
            [2*beta, 2*beta, -lam, -2*beta],
            [-2*beta, -2*beta, -2*beta, 3*lam]
            ])
        
        return (F0, M1 + N1, M2 + N2, M3 + N3, M4, M5, M6)
    

    else:
        
        return(0, M1, M2, M3, M4, M5, M6)
    
    


def simulate_net_displacement(xi,dxidt, xi0, a):
    
    
    energy_params, _ , tau_basis = build_params(a, xi0)
    
        
    _, _, _, _, hc, ht = energy_params
    
    
    integrand1 = lambda t: np.det(np.array([xi(t), dxidt(t), tau_basis[1], tau_basis[2]]))
    integrand2 = lambda t: np.det(np.array([xi(t), dxidt(t), tau_basis[2], tau_basis[3]]))
    integrand3 = lambda t: np.det(np.array([xi(t), dxidt(t), tau_basis[3], tau_basis[0]]))
    
    integrand4 = lambda t: np.det(np.array([xi(t), dxidt(t), tau_basis[0], tau_basis[3]]))
    integrand5 = lambda t: np.det(np.array([xi(t), dxidt(t), tau_basis[1], tau_basis[3]]))
    integrand6 = lambda t: np.det(np.array([xi(t), dxidt(t), tau_basis[2], tau_basis[3]]))
    
    position_integrands = [integrand1, integrand2, integrand3]
    orientation_integrands = [integrand4, integrand5, integrand6]

    
    dc = [hc * quad(integrand, 0, 2*np.pi)*basis_vector for (integrand, basis_vector) in zip(position_integrands, basis_r3)]
    dR = [ht * quad(integrand, 0, 2*np.pi)*basis_vector for (integrand, basis_vector) in zip(orientation_integrands, basis_so3)]
    
    return (dc, dR)
    
    
            
def compute_optimal_curve(net_displacement):
    pass
    

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


test_coeffs = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 1.0])

w = test_coeffs[0]*ee14 + test_coeffs[1]*ee24 + test_coeffs[2]*ee34 + \
        test_coeffs[3]*ee23 + test_coeffs[4]*ee31 + test_coeffs[5]*ee12
        





# print(check_is_simple(test_coeffs))

# if check_is_simple(test_coeffs):
#     u,v = decompose4d_simple(test_coeffs, (e1, e2, e3, e4), trace=True)
#     print(u,v)

#     uu = u[0]*ee1 + u[1]*ee2 + u[2]*ee3 + u[3]*ee4
#     vv = v[0]*ee1 + v[1]*ee2 + v[2]*ee3 + v[3]*ee4
    
#     print(uu^vv)

#     print((uu^vv) - w)
    
    
# else:
#     u1, v1, u2, v2 = decompose4d(test_coeffs, (e1, e2, e3, e4))
#     print(u1, v1)
#     print(u2, v2)
          
          
# rand_coeffs = 2* np.random.randint(low = 0, high = 10, size = (6,))

# rand_coeffs = 2*np.random.rand(6)
# print(rand_coeffs)


# print(check_is_simple(rand_coeffs))
# w = rand_coeffs[0]*ee14 + rand_coeffs[1]*ee24 + rand_coeffs[2]*ee34 + \
#         rand_coeffs[3]*ee23 + rand_coeffs[4]*ee31 + rand_coeffs[5]*ee12

# if check_is_simple(rand_coeffs):
#     u,v = decompose4d_simple(rand_coeffs, (e1, e2, e3, e4), trace=True)
#     print(u,v)
#     uu = u[0]*ee1 + u[1]*ee2 + u[2]*ee3 + u[3]*ee4
#     vv = v[0]*ee1 + v[1]*ee2 + v[2]*ee3 + v[3]*ee4

#     print(abs((uu^vv) - w))
    
# else:
#     u1, v1, u2, v2 = decompose4d(rand_coeffs, (e1, e2, e3, e4))
#     print(u1, v1)
#     print(u2, v2)
#     uu1 = u1[0]*ee1 + u1[1]*ee2 + u1[2]*ee3 + u1[3]*ee4
#     vv1 = v1[0]*ee1 + v1[1]*ee2 + v1[2]*ee3 + v1[3]*ee4
    
#     uu2 = u2[0]*ee1 + u2[1]*ee2 + u2[2]*ee3 + u2[3]*ee4
#     vv2 = v2[0]*ee1 + v2[1]*ee2 + v2[2]*ee3 + v2[3]*ee4
    
#     print(abs((uu1^vv1) + (uu2^vv2) - w))



    
    

    