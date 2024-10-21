#Analytical models implemented to estimate the behaviour of different kind of porous materials

import numpy as np

def bulk_HS(sigma, phi, alpha_inf, gamma, Np, P0, rho0, r_por, omega):
    xi = (r_por * np.log(2)) ** 2
    ep = np.sqrt(1j * omega * rho0 * alpha_inf / (sigma * phi))
    tita1 = (4/3) * np.exp(4 * xi) - 1
    tita2 = np.exp(3 * xi / 2) / np.sqrt(2)
    a1 = tita1 / tita2
    a2 = tita1
    b1 = a1
    F = (1 + a1 * ep + a2 * ep ** 2) / (1 + b1 * ep)

    B_HS = (phi / (gamma * P0)) * (gamma - (rho0 * (gamma - 1) / 
                (rho0 - 1j * sigma * phi / (omega * alpha_inf * Np) * F)))
    
    return B_HS

def dens_HS(sigma, phi, alpha_inf, rho0, r_por, omega):
    xi = (r_por * np.log(2)) ** 2
    ep = np.sqrt(1j * omega * rho0 * alpha_inf / (sigma * phi))
    tita1 = (4/3) * np.exp(4 * xi) - 1
    tita2 = np.exp(3 * xi / 2) / np.sqrt(2)

    a1 = tita1 / tita2
    a2 = tita1
    b1 = a1

    F = (1 + a1 * ep + a2 * ep ** 2) / (1 + b1 * ep)
    d_HS = (alpha_inf / phi) * (rho0 - (1j * phi * sigma) / (omega * alpha_inf) * F)
    
    return d_HS

def dens_JCA(phi, alpha_inf, sigma, lamb, nu, rho0, omega):
    G2 = ((4 * 1j * (alpha_inf ** 2) * nu * rho0) /
           ((sigma ** 2) * (lamb ** 2) * (phi ** 2)))
    d_JCA = (alpha_inf * rho0 / phi) + (sigma / (1j * omega)) * np.sqrt(1 + G2)
    
    return d_JCA

def bulk_JCA(phi, lamb_prima, P0, Np, gamma, nu, rho0, omega):
    G1prima = 8 * nu / (rho0 * omega * Np * lamb_prima ** 2)
    G2prima = rho0 * omega * Np * lamb_prima ** 2 / (16 * nu)
    B_JCA = (gamma * P0 / phi) / (gamma - 
                (gamma - 1) * (1 + (G1prima / 1j) * 
                np.sqrt(1 + (1j * G2prima))) ** -1)
    
    return B_JCA

def bulk_JCAL(phi, lamb_prima, P0, Np, gamma, sigma, nu, rho0, omega):
    k0 = nu / sigma
    G1prima = phi * nu / (rho0 * omega * Np * k0)
    G2prima = (4 * Np * rho0 * k0 ** 2 * omega) / (nu * phi ** 2 * lamb_prima ** 2)
    B_JCAL = (gamma * P0 / phi) / (gamma - 
                (gamma - 1) * (1 + (G1prima / 1j) * 
                np.sqrt(1 + (1j * G2prima))) ** -1)
    
    return B_JCAL