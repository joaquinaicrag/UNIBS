#Implementation of various methods for inverse estimation of the macroscopical parameters of porous materials

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from analmodels import *



# DIFFERENT ANALYTICAL MODELS

def jca_model(f, phi, alpha_inf, sigma, lamb, lamb_prima, d, temp=20, p0=99000): 
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
    #p0 = 101325  # Standard air pressure [Pa]
    gamma = 1.4  # Heat capacity ratio of air
    nu = 1.95e-5  # [Pa.s] Dynamic viscosity of air

    tK = temp + 273.15
    c0 = 20.047 * np.sqrt( tK )     # m/s
    rho0 = 1.290 * (p0 / 101325) * (273.15 / tK)     # kg/m3
    z0 = rho0 * c0  # Characteristic impedance of air                 
    
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
    
    return abs_jca, d_jca, b_jca



def jcal_model(f, phi, alpha_inf, sigma, lamb, lamb_prima, k0_prima, d, temp=20, p0=99000):
    """
    Absorption equation based on the JCAL model.
    """
    omega = 2 * np.pi * f  # Angular frequency
   
    Np = 0.71  # Prandtl number for air  (cp*mu/kappa)
    #p0 = 101325  # Standard air pressure [Pa]
    gamma = 1.4  # Heat capacity ratio of air
    mu = 1.95e-5  # [Pa.s] Dynamic viscosity of air
    
    tK = temp + 273.15
    c0 = 20.047 * np.sqrt( tK )     # m/s
    rho0 = 1.290 * (p0 / 101325) * (273.15 / tK)     # kg/m3
    z0 = rho0 * c0  # Characteristic impedance of air 
    # kappa = 0.026 # Thermal conductivity of air  (W/m.K)
    # cp = 1005 # Specific heat for air    (J/kg.K)    
    
    # Calculate density for the JCA model
    d_JCAL = dens_JCA(phi, alpha_inf, sigma, lamb, mu, rho0, omega) 
    # Calculate bulk modulus for the JCA model
    b_JCAL = bulk_JCAL(phi, lamb_prima, p0, Np, gamma, mu, k0_prima, rho0, omega)
    # Characteristic impedance
    Zc_JCAL = np.sqrt(d_JCAL * b_JCAL)   
    # Wavenumber for JCAL model
    k_JCAL = omega * np.sqrt(d_JCAL / b_JCAL)
    # Surface acoustic impedance of the porous layer for JCAL model
    z_JCAL = -1j * Zc_JCAL * (1/np.tan(k_JCAL * d))
    # Calculate the final absorption response for JCAL
    abs_JCAL = 1 - np.abs((z_JCAL - z0) / (z_JCAL + z0))**2

    return abs_JCAL, d_JCAL, b_JCAL



def jcapl_model(f, phi, alpha_inf, sigma, lamb, lamb_prima, d, temp=20, p0=99000):
    """
    Absorption equation based on the JCAL model.
    """
    omega = 2 * np.pi * f  # Angular frequency
   
    Np = 0.71  # Prandtl number for air
    #p0 = 101325  # Standard air pressure [Pa]
    gamma = 1.4  # Heat capacity ratio of air
    eta = 1.95e-5  # [Pa.s] Dynamic viscosity of air
    cp = 1005  # Specific heat capacity of air [J/(kg*K)]

    
    tK = temp + 273.15
    c0 = 20.047 * np.sqrt( tK )     # m/s
    rho0 = 1.290 * (p0 / 101325) * (273.15 / tK)     # kg/m3
    z0 = rho0 * c0  # Characteristic impedance of air         
    
    alpha_zero = 111# Static viscous tortuosity (it was shown by Lafarge(2006) that alpha_zero >= alpha_inf)
    alpha_zero_prima = 111# Static thermal tortuosity
    k_zero = eta/sigma # Static viscous permeability air (m^2) 
    k_zero_prima = 111# Static thermal permeability air (m^2)
    kappa = 0.0257 # Static thermal conductivity air (W/(m*K))

    
    omega_line = omega * rho0 * k_zero * alpha_inf / (phi * eta)
    m = 8*k_zero*alpha_inf / (phi*lamb**2)
    p = m / 4*(alpha_zero/alpha_inf - 1)
    F_cup = 1 - p + p*np.sqrt(1 + (m/2*(p**2)) * 1j*omega_line)
    alpha_cup = alpha_inf * [1 + (1j*omega_line)**(-1) * F_cup]
    
    
    
    omega_line_prima = (omega * rho0 * cp * k_zero_prima) / (phi * kappa)
    m_prima = (8*k_zero_prima) / (phi*lamb_prima**2)  
    p_prima = m_prima / 4*(alpha_zero_prima - 1)
    F_cup_prima = 1 - p_prima + p_prima*np.sqrt(1 + (m_prima/2*(p_prima**2)) * 1j*omega_line_prima)
    B_cup = gamma - (gamma - 1) * [1 + (1j*omega_line_prima)**(-1) * F_cup_prima]**(-1)

    # Calculate density for the JCAPL model
    d_JCAPL = (rho0 * alpha_cup) / phi
    # Calculate bulk modulus for the JCAPL model
    b_JCAPL = (gamma * p0 / phi) * (B_cup)
    # Characteristic impedance
    Zc_JCAPL = np.sqrt(d_JCAPL * b_JCAPL)   
    # Wavenumber 
    k_JCAPL = omega * np.sqrt(d_JCAPL / b_JCAPL)
    # Surface acoustic impedance of the porous layer 
    z_JCAPL = -1j * Zc_JCAPL * (1/np.tan(k_JCAPL * d))
    # Calculate the final absorption response 
    abs_JCAPL = 1 - np.abs((z_JCAPL - z0) / (z_JCAPL + z0))**2

    return abs_JCAPL, d_JCAPL, b_JCAPL    





def horosh_model(f, phi, alpha_inf, sigma, r_por, d, temp=20, p0=99000):
    """
    Absorption equation based on the Horoshenkov & Swift (HS) model.
    """
    omega = 2 * np.pi * f  # Angular frequency
    
    Np = 0.71  # Prandtl number for air
    #p0 = 101325  # Standard air pressure [Pa]
    gamma = 1.4  # Heat capacity ratio of air 

    tK = temp + 273.15
    c0 = 20.047 * np.sqrt( tK )     # m/s
    rho0 = 1.290 * (p0 / 101325) * (273.15 / tK)     # kg/m3
    z0 = rho0 * c0  # Characteristic impedance of air       

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
    
    return abs_HS, d_HS, b_HS

#Attenborough & Swift model:
def attenborough_swift_model(f, phi, alpha_inf, sigma, lamb, d, temp=20, p0=99000):
    """
    Absorption equation based on the Attenborough & Swift model.
    """
    omega = 2 * np.pi * f  # Angular frequency

    tK = temp + 273.15
    c0 = 20.047 * np.sqrt( tK )     # m/s
    rho0 = 1.290 * (p0 / 101325) * (273.15 / tK)     # kg/m3
    z0 = rho0 * c0  # Characteristic impedance of air       

    # Effective density (CHECK THIS)
    d_AS = (alpha_inf * rho0 / phi) + (sigma / (1j * omega)) * np.sqrt(1 + 
        (4j * alpha_inf**2 * rho0 * omega / (sigma**2 * lamb**2 * phi**2)))

    # Effective bulk modulus
    b_AS = (1 / phi) * (1 / (1 - 1j * (sigma / (omega * rho0 * lamb))))

    # Characteristic impedance
    Zc_AS = np.sqrt(d_AS * b_AS)

    # Wavenumber
    k_AS = omega * np.sqrt(d_AS / b_AS)

    # Surface acoustic impedance
    z_AS = -1j * Zc_AS * (1 / np.tan(k_AS * d))

    # Absorption coefficient
    abs_AS = 1 - np.abs((z_AS - z0) / (z_AS + z0))**2

    return abs_AS, d_AS, b_AS

#Wilson & Stinson model:
#Absorption equation based on the Wilson & Stinson model.
def wilson_stinson_model(f, phi, alpha_inf, sigma, d, temp=20, p0=99000):
    """
    Absorption equation based on the Wilson & Stinson model.
    """
    omega = 2 * np.pi * f  # Angular frequency

    tK = temp + 273.15
    c0 = 20.047 * np.sqrt( tK )     # m/s
    rho0 = 1.290 * (p0 / 101325) * (273.15 / tK)     # kg/m3
    z0 = rho0 * c0  # Characteristic impedance of air       
    gamma = 1.4  # Adiabatic constant
    #l = 2e-3 # some dimension characteristic of the pore (Wilson) --> grain size (eg. 2 mm) [m]
    nu = 1.95e-5  # [Pa.s] Dynamic viscosity of air
    Np = 0.71 #Prandtl number
    
    
    #rho_inf = (rho0 * alpha_inf**2) / phi
    rho_inf = (rho0 * alpha_inf) / phi #from Luc page
    k_inf =  (p0 * gamma)/phi
    #tau_vor = 2*rho0*(alpha_inf**2) / (phi * sigma) # Relaxation time 
    tau_vor = 2*rho0*(alpha_inf) / (phi * sigma) # Relaxation time From Luc page
    tau_ent =  tau_vor * Np # Relaxation time
    
    # Effective density
    d_WS =  rho_inf * (np.sqrt(1 + 1j*omega*tau_vor) / (np.sqrt(1 + 1j*omega*tau_vor) - 1))

    # Effective bulk modulus
    b_WS =  k_inf * (np.sqrt(1 + 1j*omega*tau_ent) / (np.sqrt(1 + 1j*omega*tau_ent) + gamma - 1))  #phi * k_inf * (np.sqrt(1 - 1j*omega*tau_ent) / (np.sqrt(1 - 1j*omega*tau_ent) + gamma - 1))

    # Characteristic impedance
    Zc_WS = np.sqrt(d_WS * b_WS)

    # Wavenumber
    k_WS = omega * np.sqrt(d_WS / b_WS)

    # Surface acoustic impedance
    z_WS = -1j * Zc_WS * (1 / np.tan(k_WS * d))

    # Absorption coefficient
    #abs_WS = 1 - np.abs((z_WS - z0) / (z_WS + z0))**2

    # Normalized surface impedance
    ep = z_WS / z0  
    # Calculate the final absorption response
    abs_WS = 4 * np.real(ep) / (np.abs(ep)**2 + 2 * np.real(ep) + 1)

    return abs_WS, d_WS, b_WS




#%%

# ANALYTICAL MODELS ADAPTED FOR THE INVERSE METHOD IMPLEMENTATION
'''  
def test_jca_model(f, phi, alpha_inf, sigma, lamb, lamb_prima, d, temp=20, p0=99000): 
    """
    Absorption equation based on the JCA model.
    """
    w = 2 * np.pi * f  # Angular frequency
    
    Np = 0.71  # Prandtl number for air
    #p0 = 101325  # Standard air pressure [Pa]
    gamma = 1.4  # Heat capacity ratio of air
    nu = 1.95e-5  # [Pa.s] Dynamic viscosity of air
    tK = temp + 273.15
    c0 = 20.047 * np.sqrt( tK )     # m/s
    rho0 = 1.290 * (p0 / 101325) * (273.15 / tK)     # kg/m3
    z0 = rho0 * c0  # Characteristic impedance of air       

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



def test_jcal_model(f, phi, alpha_inf, sigma, lamb, lamb_prima, d, temp=20, p0=99000):
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
    #p0 = 101325  # Standard air pressure [Pa]
    gamma = 1.4  # Heat capacity ratio of air
    nu = 1.95e-5  # [Pa.s] Dynamic viscosity of air
    tK = temp + 273.15
    c0 = 20.047 * np.sqrt( tK )     # m/s
    rho0 = 1.290 * (p0 / 101325) * (273.15 / tK)     # kg/m3
    z0 = rho0 * c0  # Characteristic impedance of air       
 
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



def test_horosh_model(f, phi, alpha_inf, sigma, r_por, d, temp=20, p0=99000):
    """
    Absorption equation based on the Horoshenkov & Swift (HS) model.
    """
    omega = 2 * np.pi * f  # Angular frequency

    Np = 0.71  # Prandtl number for air
    gamma = 1.4  # Heat capacity ratio of air 
    
    tK = temp + 273.15
    c0 = 20.047 * np.sqrt( tK )     # m/s
    rho0 = 1.290 * (p0 / 101325) * (273.15 / tK)     # kg/m3
    z0 = rho0 * c0  # Characteristic impedance of air       

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

def test_wilson_stinson_model(f, phi, alpha_inf, sigma, d, temp=20, p0=99000):
    """
    Absorption equation based on the Wilson & Stinson model.
    """
    omega = 2 * np.pi * f  # Angular frequency

    tK = temp + 273.15
    c0 = 20.047 * np.sqrt( tK )     # m/s
    rho0 = 1.290 * (p0 / 101325) * (273.15 / tK)     # kg/m3
    z0 = rho0 * c0  # Characteristic impedance of air       
    gamma = 1.4  # Adiabatic constant
    # l = 2e-3 # some dimension characteristic of the pore (Wilson) --> grain size (eg. 2 mm) [m]
    nu = 1.95e-5  # [Pa.s] Dynamic viscosity of air
    Np = 0.71 #Prandtl number
    
    
    rho_inf = (rho0 * alpha_inf**2) / phi
    k_inf =  (p0 * gamma)/phi
    #tau_vor = (rho0 * l**2) / (2*nu) # Relaxation time
    tau_vor = 2*rho0*(alpha_inf**2) / (phi * sigma) # Relaxation time 
    tau_ent =  tau_vor * Np # Relaxation time
    
    # Effective density
    d_WS = rho_inf * (np.sqrt(1 + 1j*omega*tau_vor) / (np.sqrt(1 + 1j*omega*tau_vor) - 1))

    # Effective bulk modulus
    b_WS = k_inf * (np.sqrt(1 + 1j*omega*tau_ent) / (np.sqrt(1 + 1j*omega*tau_ent) + gamma - 1))

    # Characteristic impedance
    Zc_WS = np.sqrt(d_WS * b_WS)

    # Wavenumber
    k_WS = omega * np.sqrt(d_WS / b_WS)

    # Surface acoustic impedance
    z_WS = -1j * Zc_WS * (1 / np.tan(k_WS * d))

    # Absorption coefficient
    abs_WS = 1 - np.abs((z_WS - z0) / (z_WS + z0))**2

    return abs_WS

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

'''


# NON LINEAR LEAST SQUARE METHOD

def NonlinLS_inv(xdata, ydata, startpt, lb, ub, model, d):
    if model == 'JCA':
        def wrapper(f, *params):
            abs_jca, d_jca, b_jca = jca_model(f, *params, d)
            return abs_jca
        #print(lb)
    
        lb_jca = list([lb['phi'], lb['alpha_inf'], lb['sigma'], lb['lamb'], lb['lamb_prima']])  # Lower bound of parameters
        ub_jca = list([ub['phi'], ub['alpha_inf'], ub['sigma'], ub['lamb'], ub['lamb_prima']])  # Upper bound of parameters
        startpt_jca = list([startpt['phi'], startpt['alpha_inf'], startpt['sigma'], startpt['lamb'], startpt['lamb_prima']])
        coef_JCA, cov = curve_fit(wrapper, xdata, ydata, p0=startpt_jca, bounds=(lb_jca, ub_jca))
        fitted_data, dens, bulk = jca_model(xdata, *coef_JCA, d)
        return fitted_data, dens, bulk, coef_JCA, cov
    
    elif model == 'HS':
        def wrapper(f, *params):
            abs_hs, d_hs, b_hs = horosh_model(f, *params, d)
            return abs_hs
        
        lb_hs = list([lb['phi'], lb['alpha_inf'], lb['sigma'], lb['dev_por']])  # Lower bound of parameters
        ub_hs = list([ub['phi'], ub['alpha_inf'], ub['sigma'], ub['dev_por']])  # Upper bound of parameters
        startpt_hs = list([startpt['phi'], startpt['alpha_inf'], startpt['sigma'], startpt['dev_por']])
        coef_HS, cov = curve_fit(wrapper, xdata, ydata, p0=startpt_hs, bounds=(lb_hs, ub_hs))
        fitted_data, dens, bulk = horosh_model(xdata, *coef_HS, d)
        return fitted_data, dens, bulk, coef_HS, cov
    
    elif model == 'JCAL':
        def wrapper(f, *params):
            abs_jcal, d_jcal, b_jcal = jcal_model(f, *params, d)
            return abs_jcal
        
        lb_jcal = list([lb['phi'], lb['alpha_inf'], lb['sigma'], lb['lamb'], lb['lamb_prima'], lb['k0_prima']])  # Lower bound of parameters
        ub_jcal = list([ub['phi'], ub['alpha_inf'], ub['sigma'], ub['lamb'], ub['lamb_prima'], ub['k0_prima']])  # Upper bound of parameters
        startpt_jcal = list([startpt['phi'], startpt['alpha_inf'], startpt['sigma'], startpt['lamb'], startpt['lamb_prima'], startpt['k0_prima']])
        coef_JCAL, cov = curve_fit(wrapper, xdata, ydata, p0=startpt_jcal, bounds=(lb_jcal, ub_jcal))
        fitted_data, dens, bulk = jcal_model(xdata, *coef_JCAL, d)
        return fitted_data, dens, bulk, coef_JCAL, cov
    
    elif model == 'WS':
        def wrapper(f, *params):
            abs_ws, d_ws, b_ws = wilson_stinson_model(f, *params, d)
            return abs_ws
        
        lb_ws = list([lb['phi'], lb['alpha_inf'], lb['sigma']])  # Lower bound of parameters
        ub_ws = list([ub['phi'], ub['alpha_inf'], ub['sigma']])  # Upper bound of parameters
        startpt_ws = list([startpt['phi'], startpt['alpha_inf'], startpt['sigma']])
        coef_WS, cov = curve_fit(wrapper, xdata, ydata, p0=startpt_ws, bounds=(lb_ws, ub_ws))
        fitted_data, dens, bulk = wilson_stinson_model(xdata, *coef_WS, d)
        return fitted_data, dens, bulk, coef_WS, cov


#%% GENETIC ALGORITHM METHOD







#%% NEURAL NETWORK METHOD



