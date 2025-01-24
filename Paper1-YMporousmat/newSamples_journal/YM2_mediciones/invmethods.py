#Implementation of various methods for inverse estimation of the macroscopical parameters of porous materials

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from analmodels import *



# DIFFERENT ANALYTICAL MODELS

def jca_model(f, phi, alpha_inf, sigma, lamb, lamb_prima, d): 
    """
    Absorption equation based on the JCA model.
    """
    w = 2 * np.pi * f  # Angular frequency
    #phi, alpha_inf, sigma, lamb, lamb_prima = params
    # phi = x[0]  # Porosity
    # alpha_inf = x[1]  # Infinite frequency absorption coefficient
    # sigma = x[2]  # Flow resistance
    # lamb = x[3]  # Pore size parameter
    # lamb_prima = x[4]  # Another pore size parameter
    
    Np = 0.71  # Prandtl number for air
    p0 = 101325  # Standard air pressure [Pa]
    gamma = 1.4  # Heat capacity ratio of air
    nu = 1.95e-5  # [Pa.s] Dynamic viscosity of air
    rho0 = 1.204  # [kg/m^3] Density of air
    c0 = 343  # Velocity of sound in air [m/s]
    z0 = rho0 * c0  # Characteristic impedance of air
    #d = 40e-3  # [m] Sample thickness                        
    
    # Calculate the complex wave number K (Bonfiglio et al.)
    b_jca = (gamma * p0 / phi) / (
        gamma - (gamma - 1) * (1 + 
        (8 * nu / (1j * rho0 * w * Np * lamb_prima**2)) * 
        np.sqrt(1 + (1j * rho0 * w * Np * lamb_prima**2 / (16 * nu))))**(-1)
    )
    
    # Calculate the complex density rho (Bonfiglio et al.)
    d_jca = (alpha_inf * rho0 / phi) + (sigma / (1j * w)) * np.sqrt(1 + 
        (4j * (alpha_inf**2) * nu * rho0 * w / ((sigma**2) * (lamb**2) * (phi**2))))
    
    # Wavenumber in the porous material
    kc = w * np.sqrt(d_jca / b_jca)  
    # Characteristic impedance of the porous material
    Zc = np.sqrt(d_jca * b_jca)  
    # Surface acoustic impedance of the porous layer
    z = -1j * Zc * 1/np.tan(kc * d)
    # Normalized surface impedance
    ep = z / z0  
    # Calculate the final absorption response
    abs_jca = 4 * np.real(ep) / (np.abs(ep)**2 + 2 * np.real(ep) + 1)
    
    return abs_jca



def jcal_model(f, phi, alpha_inf, sigma, lamb, lamb_prima, d):
    """
    Absorption equation based on the JCAL model.
    """
    omega = 2 * np.pi * f  # Angular frequency
    #phi, alpha_inf, sigma, lamb, lamb_prima = params
    # phi = x[0]  # Porosity
    # alpha_inf = x[1]  # Infinite frequency absorption coefficient
    # sigma = x[2]  # Flow resistance
    # lamb = x[3]  # Pore size parameter
    # lamb_prima = x[4]  # Another pore size parameter
    
    Np = 0.71  # Prandtl number for air
    p0 = 101325  # Standard air pressure [Pa]
    gamma = 1.4  # Heat capacity ratio of air
    nu = 1.95e-5  # [Pa.s] Dynamic viscosity of air
    rho0 = 1.204  # [kg/m^3] Density of air
    c0 = 343  # Velocity of sound in air [m/s]
    z0 = rho0 * c0  # Characteristic impedance of air
    # d = 50e-3  # [m] Sample thickness 
    
    # Calculate density for the JCA model
    d_JCAL = dens_JCA(phi, alpha_inf, sigma, lamb, nu, rho0, omega) 
    # Calculate bulk modulus for the JCA model
    b_JCAL = bulk_JCAL(phi, lamb_prima, p0, Np, gamma, sigma, nu, rho0, omega)
    # Characteristic impedance
    Zc_JCAL = np.sqrt(d_JCAL * b_JCAL)   
    # Wavenumber for JCAL model
    k_JCAL = omega * np.sqrt(d_JCAL / b_JCAL)
    # Surface acoustic impedance of the porous layer for JCAL model
    z_JCAL = -1j * Zc_JCAL * (1/np.tan(k_JCAL * d))
    # Calculate the final absorption response for JCAL
    abs_JCAL = 1 - np.abs((z_JCAL - z0) / (z_JCAL + z0))**2

    return d_JCAL, b_JCAL

def horosh_model(f, phi, alpha_inf, sigma, r_por, d):
    """
    Absorption equation based on the Horoshenkov & Swift (HS) model.
    """
    omega = 2 * np.pi * f  # Angular frequency
    #phi, alpha_inf, sigma, r_por = params
    # phi = x[0]  # Porosity
    # alpha_inf = x[1]  # Infinite frequency absorption coefficient
    # sigma = x[2]  # Flow resistance
    # r_por = x[3]  # Pore radius
    
    Np = 0.71  # Prandtl number for air
    p0 = 101325  # Standard air pressure [Pa]
    gamma = 1.4  # Heat capacity ratio of air 
    rho0 = 1.204  # [kg/m^3] Density of air
    c0 = 343  # Velocity of sound in air [m/s]
    z0 = rho0 * c0  # Characteristic impedance of air
    #d = 40e-3  # [m] Sample thickness 

    xi = (r_por * np.log(2)) ** 2  # Parameter for absorption calculations
    ep = np.sqrt(1j * omega * rho0 * alpha_inf / (sigma * phi))  # Normalize absorption
    tita1 = (4 / 3) * np.exp(4 * xi) - 1  # Related to shear modulus
    tita2 = np.exp(3 * xi / 2) / np.sqrt(2)  # Another related term
    a1 = tita1 / tita2 
    a2 = tita1
    b1 = a1
    
    # Calculate the absorption response for HS model
    f = (1 + a1 * ep + a2 * ep**2) / (1 + b1 * ep) 
    # Compressibility (inverse of bulk modulus) for HS
    c_HS = (phi / (gamma * p0)) * (gamma - (rho0 * (gamma - 1) /
               (rho0 - 1j * sigma * phi / (omega * alpha_inf * Np) * f)))
    b_HS = 1/c_HS           
    # Density for HS model
    d_HS = (alpha_inf / phi) * (rho0 - ((1j * phi * sigma) / (omega * alpha_inf) * f))  
    #  Characteristic impedance for HS model
    Zc_HS = np.sqrt(d_HS / c_HS)  
    # Wavenumber for HS model
    k_HS = omega * np.sqrt(d_HS * c_HS)
    # Surface acoustic impedance of the porous layer for HS model
    z_HS = -1j * Zc_HS * (1/np.tan(k_HS * d))
    # Calculate the final absorption response for HS
    abs_HS = 1 - np.abs((z_HS - z0) / (z_HS + z0))**2
    
    return d_HS, b_HS





#%%

# ANALYTICAL MODELS ADAPTED FOR THE INVERSE METHOD IMPLEMENTATION

def test_jca_model(f, phi, alpha_inf, sigma, lamb, lamb_prima, d): 
    """
    Absorption equation based on the JCA model.
    """
    w = 2 * np.pi * f  # Angular frequency
    #phi, alpha_inf, sigma, lamb, lamb_prima = params
    # phi = x[0]  # Porosity
    # alpha_inf = x[1]  # Infinite frequency absorption coefficient
    # sigma = x[2]  # Flow resistance
    # lamb = x[3]  # Pore size parameter
    # lamb_prima = x[4]  # Another pore size parameter
    
    Np = 0.71  # Prandtl number for air
    p0 = 101325  # Standard air pressure [Pa]
    gamma = 1.4  # Heat capacity ratio of air
    nu = 1.95e-5  # [Pa.s] Dynamic viscosity of air
    rho0 = 1.204  # [kg/m^3] Density of air
    c0 = 343  # Velocity of sound in air [m/s]
    z0 = rho0 * c0  # Characteristic impedance of air
    #d = 40e-3  # [m] Sample thickness                        
    
    # Calculate the complex wave number K (Bonfiglio et al.)
    b_jca = (gamma * p0 / phi) / (
        gamma - (gamma - 1) * (1 + 
        (8 * nu / (1j * rho0 * w * Np * lamb_prima**2)) * 
        np.sqrt(1 + (1j * rho0 * w * Np * lamb_prima**2 / (16 * nu))))**(-1)
    )
    
    # Calculate the complex density rho (Bonfiglio et al.)
    d_jca = (alpha_inf * rho0 / phi) + (sigma / (1j * w)) * np.sqrt(1 + 
        (4j * (alpha_inf**2) * nu * rho0 * w / ((sigma**2) * (lamb**2) * (phi**2))))
    
    # Wavenumber in the porous material
    kc = w * np.sqrt(d_jca / b_jca)  
    # Characteristic impedance of the porous material
    Zc = np.sqrt(d_jca * b_jca)  
    # Surface acoustic impedance of the porous layer
    z = -1j * Zc * 1/np.tan(kc * d)
    # Normalized surface impedance
    ep = z / z0  
    # Calculate the final absorption response
    abs_jca = 4 * np.real(ep) / (np.abs(ep)**2 + 2 * np.real(ep) + 1)
    
    return abs_jca



def test_jcal_model(f, phi, alpha_inf, sigma, lamb, lamb_prima, d):
    """
    Absorption equation based on the JCAL model.
    """
    omega = 2 * np.pi * f  # Angular frequency
    #phi, alpha_inf, sigma, lamb, lamb_prima = params
    # phi = x[0]  # Porosity
    # alpha_inf = x[1]  # Infinite frequency absorption coefficient
    # sigma = x[2]  # Flow resistance
    # lamb = x[3]  # Pore size parameter
    # lamb_prima = x[4]  # Another pore size parameter
    
    Np = 0.71  # Prandtl number for air
    p0 = 101325  # Standard air pressure [Pa]
    gamma = 1.4  # Heat capacity ratio of air
    nu = 1.95e-5  # [Pa.s] Dynamic viscosity of air
    rho0 = 1.204  # [kg/m^3] Density of air
    c0 = 343  # Velocity of sound in air [m/s]
    z0 = rho0 * c0  # Characteristic impedance of air
    # d = 50e-3  # [m] Sample thickness 
    
    # Calculate density for the JCA model
    d_JCA = dens_JCA(phi, alpha_inf, sigma, lamb, nu, rho0, omega) 
    # Calculate bulk modulus for the JCA model
    b_JCAL = bulk_JCAL(phi, lamb_prima, p0, Np, gamma, sigma, nu, rho0, omega)
    # Characteristic impedance
    Zc_JCAL = np.sqrt(d_JCA * b_JCAL)   
    # Wavenumber for JCAL model
    k_JCAL = omega * np.sqrt(d_JCA / b_JCAL)
    # Surface acoustic impedance of the porous layer for JCAL model
    z_JCAL = -1j * Zc_JCAL * (1/np.tan(k_JCAL * d))
    # Calculate the final absorption response for JCAL
    abs_JCAL = 1 - np.abs((z_JCAL - z0) / (z_JCAL + z0))**2

    return abs_JCAL

def test_horosh_model(f, phi, alpha_inf, sigma, r_por, d):
    """
    Absorption equation based on the Horoshenkov & Swift (HS) model.
    """
    omega = 2 * np.pi * f  # Angular frequency
    #phi, alpha_inf, sigma, r_por = params
    # phi = x[0]  # Porosity
    # alpha_inf = x[1]  # Infinite frequency absorption coefficient
    # sigma = x[2]  # Flow resistance
    # r_por = x[3]  # Pore radius
    
    Np = 0.71  # Prandtl number for air
    p0 = 101325  # Standard air pressure [Pa]
    gamma = 1.4  # Heat capacity ratio of air 
    rho0 = 1.204  # [kg/m^3] Density of air
    c0 = 343  # Velocity of sound in air [m/s]
    z0 = rho0 * c0  # Characteristic impedance of air
    #d = 40e-3  # [m] Sample thickness 

    xi = (r_por * np.log(2)) ** 2  # Parameter for absorption calculations
    ep = np.sqrt(1j * omega * rho0 * alpha_inf / (sigma * phi))  # Normalize absorption
    tita1 = (4 / 3) * np.exp(4 * xi) - 1  # Related to shear modulus
    tita2 = np.exp(3 * xi / 2) / np.sqrt(2)  # Another related term
    a1 = tita1 / tita2 
    a2 = tita1
    b1 = a1
    
    # Calculate the absorption response for HS model
    f = (1 + a1 * ep + a2 * ep**2) / (1 + b1 * ep) 
    # Compressibility (inverse of bulk modulus) for HS
    c_HS = (phi / (gamma * p0)) * (gamma - (rho0 * (gamma - 1) /
               (rho0 - 1j * sigma * phi / (omega * alpha_inf * Np) * f)))
    b_HS = 1/c_HS           
    # Density for HS model
    d_HS = (alpha_inf / phi) * (rho0 - ((1j * phi * sigma) / (omega * alpha_inf) * f))  
    #  Characteristic impedance for HS model
    Zc_HS = np.sqrt(d_HS / c_HS)  
    # Wavenumber for HS model
    k_HS = omega * np.sqrt(d_HS * c_HS)
    # Surface acoustic impedance of the porous layer for HS model
    z_HS = -1j * Zc_HS * (1/np.tan(k_HS * d))
    # Calculate the final absorption response for HS
    abs_HS = 1 - np.abs((z_HS - z0) / (z_HS + z0))**2
    
    return abs_HS

# def nlcon(x):
#     """
#     Non-linear constraint function for the JCA model.
#     """
#     ceq = []  # No equality constraints
#     c = 2 * x[3] - x[4]  # Inequality constraint
#     return c, ceq

# def nlcon2(x):
#     """
#     Non-linear constraint function for the HS model.
#     """
#     ceq2 = []  # No equality constraints
#     c2 = []  # Inequality constraints (if any)
#     return c2, ceq2




# NON LINEAR LEAST SQUARE METHOD

def NonlinLS_inv(xdata, ydata, startpt, lb, ub, model, d):
    if model == 'JCA':
        def wrapper(f, *params):
            return test_jca_model(f, *params, d)
        #print(lb)
        lb_jca = lb[:-1]  # Lower bound of parameters
        ub_jca = ub[:-1]  # Upper bound of parameters
        startpt_jca = startpt[:-1]
        coef_JCA, cov = curve_fit(wrapper, xdata, ydata, p0=startpt_jca, bounds=(lb_jca, ub_jca))
        fitted_data, dens, bulk = jca_model(xdata, *coef_JCA, d)
        return fitted_data, dens, bulk, coef_JCA, cov
    
    elif model == 'HS':
        def wrapper(f, *params):
            return test_horosh_model(f, *params, d)
        
        lb_hs = lb[:-3] + lb[-1:]  # Lower bound of parameters
        ub_hs = ub[:-3] + ub[-1:]  # Upper bound of parameters
        startpt_hs = startpt[:-3] + startpt[-1:]
        coef_HS, cov = curve_fit(wrapper, xdata, ydata, p0=startpt_hs, bounds=(lb_hs, ub_hs))
        fitted_data, dens, bulk = horosh_model(xdata, *coef_HS, d)
        return fitted_data, dens, bulk, coef_HS, cov
    
    elif model == 'JCAL':
        def wrapper(f, *params):
            return test_jcal_model(f, *params, d)
        
        lb_hs = lb[:-3] + lb[-1:]  # Lower bound of parameters
        ub_hs = ub[:-3] + ub[-1:]  # Upper bound of parameters
        startpt_hs = startpt[:-3] + startpt[-1:]
        coef_HS, cov = curve_fit(wrapper, xdata, ydata, p0=startpt_hs, bounds=(lb_hs, ub_hs))
        fitted_data, dens, bulk = jcal_model(xdata, *coef_HS, d)
        return fitted_data, dens, bulk, coef_HS, cov



#%% GENETIC ALGORITHM METHOD







#%% NEURAL NETWORK METHOD



