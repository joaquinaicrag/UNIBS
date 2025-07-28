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
    
    # Calculate Gj
    #Gj = np.sqrt(1 + 4 * 1j * alpha_inf**2 * nu * rho0 * omega / (sigma**2 * lamb**2 * phi**2))
    # Calculate rho_eq
    #d_JCA = (1/phi) * alpha_inf * rho0 * (1 + sigma * phi * Gj / (1j * omega * rho0 * alpha_inf))
    
    return d_JCA

def bulk_JCA(phi, lamb_prima, P0, Np, gamma, nu, rho0, omega):
    G1prima = 8 * nu / (rho0 * omega * Np * lamb_prima ** 2)
    G2prima = rho0 * omega * Np * lamb_prima ** 2 / (16 * nu)
    B_JCA = (gamma * P0 / phi) / (gamma - 
                (gamma - 1) * (1 + (G1prima / 1j) * 
                np.sqrt(1 + (1j * G2prima))) ** -1)
    
    return B_JCA

def bulk_JCAL(phi, lamb_prima, P0, Np, gamma, nu, k0_prima, rho0, omega, temp=293):
    #k0 = nu / sigma # static viscous perm. (not used here)
    
    Cp = 4168.8 * (
    0.249679
    - 7.55179e-5 * temp
    + 1.69194e-7 * temp**2
    - 6.46128e-11 * temp**3)
    kappa = 2.624e-02 * ( (temp / 300)**(3/2) * (300 + 245.4 * np.exp(-27.6 / 300)) / (temp + 245.4 * np.exp(-27.6 / temp)) )  #Thermal conductivity
    
    G1prima = phi * nu / (rho0 * omega * Np * k0_prima)
    
    G2prima = (4 * Np * rho0 * k0_prima ** 2 * omega) / (nu * phi ** 2 * lamb_prima ** 2)
    
    b_jcal = (gamma * P0 / phi) / (gamma - 
                (gamma - 1) * (1 + (G1prima / 1j) * 
                np.sqrt(1 + (1j * G2prima))) ** -1)
    
    bulk_jcal = (gamma * P0 / phi) * (gamma - (gamma - 1) * (1 - 1j*((phi*kappa) / (k0_prima*Cp*rho0*omega)) * 
                np.sqrt(1 + 1j*((4*(k0_prima**2)*Cp*rho0*omega) / (kappa*(lamb_prima**2)*(phi**2))) ))**(-1) )
    
    
    #[m2] static thermal permeability (k0')
    
    # Compute the denominator term inside the sqrt
    sqrt_term = np.sqrt(1 + 1j * 4 * rho0 * omega * Cp * k0_prima**2 / (kappa * lamb_prima**2 * phi**2))

    # Compute the entire denominator
    denominator = gamma - (gamma - 1) / (1 - 1j * phi * kappa / (k0_prima * Cp * omega * rho0) * sqrt_term)

    # Compute K_eq
    K_eq = (1 / phi) * gamma * P0 / denominator
    
    return b_jcal