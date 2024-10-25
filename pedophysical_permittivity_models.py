"""
Pedophysical Permittivity models
======================
Soil dielectric permittivity modelling for low frequency instrumentation.    
...

:AUTHOR: Gaston Mendoza Veirana
:CONTACT: gaston.mendozaveirana@ugent.be

:REQUIRES: numpy
"""

# Import
import numpy as np

def topp(vwc):
    """
    Topp et al. (1980).
    
    Parameters
    ----------
    vwc: float
        Soil volumetric water content [-]
        
    Returns
    -------
    eps_ap: float
        Soil bulk apparent relative dielectric permittivity [-]       
    """
    p = [4.3e-6, -5.5e-4, 2.92e-2, -5.3e-2 - vwc]
    roots = np.roots(p)
    roots = roots[roots.imag == 0 ]
    perm_rel = roots[roots > 0]
    eps_ap = perm_rel[0].real
    return eps_ap


def jacandschjB(vwc, bd, cc, org):
    """
    Jacobsen and Schjonning (1993) (Equation 4)
    
    Parameters
    ----------
    vwc: float
        Soil volumetric water content [-]

    bd: float
        Soil dry bulk density [g/cm3]
        
    cc: float
        Soil clay content [g/100g]
        
    org: float
        organic matter content (%)
        
    Returns
    -------
    eps_ap: float
        Soil bulk apparent relative dielectric permittivity [-]  
    """
    p = [17.1e-6, -11.4e-4, 3.45e-2, 
         -3.41e-2 - vwc -3.7e-2 * bd + 7.36e-4 * cc + 47.7e-4 * org]
    roots = np.roots(p)
    roots = roots[roots.imag == 0 ]
    perm_rel = roots[roots > 0]
    eps_ap = (perm_rel[0].real)
    return eps_ap


def LR(vwc, bd, pd, eps_a, eps_s, eps_w, alpha): 
    """
    Lichtenecker and LRer (1931)
    
    Parameters
    ----------
    vwc: float
        Soil volumetric water content [-]
    
    bd: float
        Soil dry bulk density [g/cm3]
    
    pd: float
        Soil particle (solid phase) density [g/cm3]
        
    eps_a: float
        Soil air phase real relative dielectric permittivity [-]
        
    eps_s: float
        Soil solid phase real relative dielectric permittivity [-]
        
    eps_w: float
        Soil water phase real relative dielectric permittivity [-]
        
    alpha: float
        alpha exponent [-]
        
    Returns
    -------
    eps_b: float
        Soil bulk real relative dielectric permittivity [-]
    """
    por = 1 - bd/pd    # Eq. 3
    eps_b = (( vwc*eps_w**alpha + (1-por)*eps_s**alpha + (por-vwc)*eps_a**(alpha))**(1/alpha))
    return eps_b


def CRIM(vwc, bd, pd, eps_a, eps_s, eps_w, alpha = 0.5):
    """
    Birchak et.al 1974
    
    Parameters
    ----------
    vwc: float
        Soil volumetric water content [-]
    
    bd: float
        Soil dry bulk density [g/cm3]
    
    pd: float
        Soil particle (solid phase) density [g/cm3]
        
    eps_a: float
        Soil air phase real relative dielectric permittivity [-]
        
    eps_s: float
        Soil solid phase real relative dielectric permittivity [-]
        
    eps_w: float
        Soil water phase real relative dielectric permittivity [-]
        
    Returns
    -------
    eps_b: float
        Soil bulk real relative dielectric permittivity [-]
    """
    por = 1 - bd/pd        # Eq. 3
    eps_b = (( vwc*eps_w**alpha + (1-por)*eps_s**alpha + (por-vwc)*eps_a**(alpha))**(1/alpha))
    return eps_b 


def linde(vwc, bd, pd, eps_a, eps_s, eps_w, m, n):
    """
    Linde et al., 2006
    
    Parameters
    ----------
    vwc: float
        Soil volumetric water content [-]
    
    bd: float
        Soil dry bulk density [g/cm3]
    
    pd: float
        Soil particle (solid phase) density [g/cm3]
        
    eps_a: float
        Soil air phase real relative dielectric permittivity [-]
        
    eps_s: float
        Soil solid phase real relative dielectric permittivity [-]
        
    eps_w: float
        Soil water phase real relative dielectric permittivity [-]
        
    m: float
        cementation exponent/factor [-]
        
    n: float
        saturation exponent [-]
        
    Returns
    -------
    eps_b: float
        Soil bulk real relative dielectric permittivity [-]
    """
    por = 1 - bd/pd  # Eq. 3
    S = vwc / por     # Saturation
    eps_b  = (por**m) * ((S**n)*eps_w + ((por**-m) - 1)*eps_s +(1-S**n)*eps_a)
    return eps_b 


def wunderlich(vwc, perm_init, wat_init, eps_w, Lw): 
    """
    Wunderlich et.al 2013 
    
    Parameters
    ----------
    vwc: float
        Soil volumetric water content [-]
    
    perm_init: float
        lowest real permittivity [-]
        
    wat_init: float
        lowest volumetric water content [-]
        
    eps_w: float
        Soil water phase real relative dielectric permittivity [-]
        
    Lw: float
        Soil water phase depolarization factor [-]

    Returns
    -------
    eps_b: float
        Soil bulk real relative dielectric permittivity [-]   
    """ 
    diff = vwc - wat_init                                          # Diference utilized just for simplicity
    eps_b  = perm_init                                                  # Initial permitivity = epsilon sub 1  
    x = 0.001                                                      # Diferentiation from p = 0  
    dx = 0.05                                                     # Diferentiation step
                                                                   # Diferentiation until p = 1
    while x<1:                                                    
        dy = ((eps_b *diff)/(1-diff+x*diff)) * ((eps_w-eps_b )/(Lw*eps_w+(1-Lw)*eps_b ))
        x=x+dx
        eps_b =eps_b +dy*dx
        
    return eps_b 


def endres_redman(vwc, bd, pd, eps_a, eps_s, eps_w, L):   
    """
    Endres & Redman 1996
    
    Parameters
    ----------
    vwc: float
        Soil volumetric water content [-]
    
    bd: float
        Soil dry bulk density [g/cm3]
    
    pd: float
        Soil particle (solid phase) density [g/cm3]
        
    eps_a: float
        Soil air phase real relative dielectric permittivity [-]
        
    eps_s: float
        Soil solid phase real relative dielectric permittivity [-]
        
    eps_w: float
        Soil water phase real relative dielectric permittivity [-]
        
    L: float
        Soil solid phase depolarization factor [-]
        
    Returns
    -------
    eps_b: float
        Soil bulk real relative dielectric permittivity [-]
    """
    por = 1 - bd/pd                                              # Eq. 3
    S = vwc/por                                                   # Saturation
    y = eps_w                                                        # Initial permitivity = epsilon sub a  
    x = 0                                                         # Diferentiation from p = 0  
    dx = 0.05                                                    # Diferentiation step
    
    while x<1:                                                    # Diferentiation until p = 1
        dy = ((dx*y*(1-S))/(S+x*(1-S))) * ((eps_a-y)/(L*eps_a+(1-L)*y))  
        x = x + dx
        y = y + dy
                                                                  # Now y is equal to permitivity of pore(s)
    p = 0
    dp = 0.05
    z = y
    
    while p<1:    
        dz = (dp*z*(1-por))/(por+p*(1-por)) * ((eps_s-z)/(L*eps_s+(1-L)*z))
        p = p + dp
        z = z + dz
        
    return z


def hydraprobe(vwc):
    """
    HydraProbe default equation for water content (See Hydreps_arobe manual, equation A2 eps_aendix C).

    Parameters
    ----------
    vwc: float
        Soil volumetric water content [-]
        
    Returns
    -------
    eps_b: float
        Soil bulk real relative dielectric permittivity [-]
    """
    A = 0.109
    B = -0.179
    eps_b = ((vwc - B)/A)**2
    return eps_b
    
    
def nadler(vwc):
    """
    Nadler et al. (1991). 
    
    Parameters
    ----------
    vwc: float
        Soil volumetric water content [-]
        
    Returns
    -------
    eps_aarent bulk permittivity: float            
    """
    
    p = [15e-6, -12.3e-4, 3.67e-2, -7.25e-2 - vwc]
    roots = np.roots(p)
    roots = roots[roots.imag == 0 ]
    perm_rel = roots[roots > 0]
    eps_b = perm_rel[0].real
    return eps_b


def jacandschjA(vwc):
    """
    Jacobsen and Schjonning (1993) (Equation A2)
    
    Parameters
    ----------
    vwc: float
        Soil volumetric water content [-]  
        
    Returns
    -------
    eps_aarent bulk permittivity: float            
    """
    
    p = [18e-6, -11.6e-4, 3.47e-2, -7.01e-2 - vwc]
    roots = np.roots(p)
    roots = roots[roots.imag == 0 ]
    perm_rel = roots[roots > 0]
    eps_b = perm_rel[0].real
    return eps_b


def malicki(vwc, bd):
    """
    Malicki et al. 1996
    
    Parameters
    ----------
    vwc: float
        Soil volumetric water content [-]
    
    bd: float
        Soil dry bulk density [g/cm3]
        
    Returns
    -------
    eps_ap: float
        Soil bulk apparent relative dielectric permittivity [-]  
    """
    eps_ap = (vwc*(7.17 + 1.18*bd) + 0.809 + 0.168*bd + 0.159*bd**2)**2
    return eps_ap 


def steelman(vwc):
    """
    Steelman and Endres (2011) 
            
    Parameters
    ----------
    vwc: float
        Soil volumetric water content [-]
        
    Returns
    -------
    eps_ap: float
        Soil bulk apparent relative dielectric permittivity [-]  
    """
    p = [2.97e-5, -2.03e-3, 5.65e-2, -0.157 - vwc]
    roots = np.roots(p)
    roots = roots[roots.imag == 0 ]
    perm_rel = roots[roots > 0]
    eps_ap = perm_rel[0].real
    return eps_ap


def logsdonperm(vwc):
    """
    Logsdon 2010. 
    
    Parameters
    ----------
    vwc: float
        Soil volumetric water content [-]
        
    Returns
    -------
    eps_b: float
        Soil bulk real relative dielectric permittivity [-]            
    """
    p = [0.00000514, -0.00047, 0.022, 0-vwc]
    roots = np.roots(p)
    roots = roots[roots.imag == 0 ]
    perm_rel = roots[roots > 0]
    eps_b = perm_rel[0].real
    return eps_b


def peplinski(vwc, bd, pd, cc, sand, eps_s, eps_w, tau = 0.65):
    """
    Peplinski et al., 1995 
    
    Parameters
    ----------
    vwc: float
        Soil volumetric water content [-]
    
    bd: float
        Soil dry bulk density [g/cm3]
    
    pd: float
        Soil particle (solid phase) density [g/cm3]
        
    cc: float
        Soil clay content [g/100g]
        
    sand: float
        volumetric sand content (%)
        
    eps_s: float
        Soil solid phase real relative dielectric permittivity [-]
        
    eps_w: float
        Soil water phase real relative dielectric permittivity [-]
        
    Returns
    -------
    eps_b: float
        Soil bulk real relative dielectric permittivity [-]   
    """
    beta1 = 1.2748 - 0.519*sand/100 - 0.152*cc/100
    eps_b = (1 + (bd/pd)*eps_s**tau + (vwc**beta1)*(eps_w**tau) - vwc)**(1/tau)
    return eps_b


def dobson(vwc, bw, bd, pd, eps_a, eps_s, eps_w, beps_w):
    """
    Dobson et al., 1985  

    Parameters
    ----------
    vwc: float
        Soil volumetric water content [-]
    
    bw: float
        volumetric bound water content [-]
        
    bd: float
        Soil dry bulk density [g/cm3]
    
    pd: float
        Soil particle (solid phase) density [g/cm3]
        
    eps_a: float
        Soil air phase real relative dielectric permittivity [-]
        
    eps_s: float
        Soil solid phase real relative dielectric permittivity [-]
        
    eps_w: float
        Soil water phase real relative dielectric permittivity [-]
        
    beps_w: float
        bound water permittivity [-]
        
    Returns
    -------
    eps_b: float
        Soil bulk real relative dielectric permittivity [-]
    """
    por = 1 - bd/pd
    num = 3*eps_s+2*vwc*(eps_w-eps_s) + 2*bw*(beps_w-eps_s)+2*(por-vwc)*(eps_a-eps_s)
    den = 3+vwc*((eps_s/eps_w)-1) + bw*((eps_s/beps_w)-1)+(por-vwc)*((eps_s/eps_a)-1)
    eps_b = num/den
    return eps_b


def sen(vwc, bd, pd, eps_a, eps_s, eps_w, L):        
    """
    Sen et al., 1981
    
    Parameters
    ----------
    vwc: float
        Soil volumetric water content [-]
    
    bd: float
        Soil dry bulk density [g/cm3]
    
    pd: float
        Soil particle (solid phase) density [g/cm3]
        
    eps_a: float
        Soil air phase real relative dielectric permittivity [-]
        
    eps_s: float
        Soil solid phase real relative dielectric permittivity [-]
        
    eps_w: float
        Soil water phase real relative dielectric permittivity [-]
        
    L: float
        Soil solid phase depolarization factor [-]
        
    Returns
    -------
    eps_b: float
        Soil bulk real relative dielectric permittivity [-]
    """
    por = 1 - bd/pd                                       # Eq. 3 
    cl = vwc*(L*eps_s + (1-L)*eps_w)                             # Calculation just for simplicity 
    wcg = eps_w*(((1-por)*eps_s+cl) / ((1-por)*eps_w+cl))           # water coated grains
    df = (por*-1) + vwc                                    # Diference utilized just for simplicity
    y = eps_a                                                 # Initial permitivity = epsilon sub a  
    x = 0.001                                              # Diferentiation from p = 0  
    dx = 0.05                                              # Diferentiation step
                                                           
    while x<1:                                             # Diferentiation until p = 1
        dy = ((y*(1+df))/(-df+x*(1+df))) * ((wcg-y)/(L*wcg+(1-L)*y))
        x=x+dx
        y=y+dy*dx
        
    return y 
 
    
def feng_sen(vwc, bd, pd, eps_a, eps_s, eps_w, L):
    """
    Feng & Sen 1985
    
    Parameters
    ----------
    vwc: float
        Soil volumetric water content [-]
    
    bd: float
        Soil dry bulk density [g/cm3]
    
    pd: float
        Soil particle (solid phase) density [g/cm3]
        
    eps_a: float
        Soil air phase real relative dielectric permittivity [-]
        
    eps_s: float
        Soil solid phase real relative dielectric permittivity [-]
        
    eps_w: float
        Soil water phase real relative dielectric permittivity [-]
        
    L: float
        Soil solid phase depolarization factor [-]
        
    Returns
    -------
    eps_b: float
        Soil bulk real relative dielectric permittivity [-]
    """
    por = 1 - bd/pd                                       # Eq. 3
    y = eps_w                                                 # Initial permitivity = epsilon sub a  
    x = 0                                                  # Diferentiation from p = 0  
    dx = 0.05                                             # Diferentiation step
    
    while x<1:                                             # Diferentiation until p = 1
        dy = (y/(vwc+x*(1-vwc))) * ((((1-por)*((eps_s-y))/(L*eps_s+(1-L)*y))) + ((por-vwc)*(eps_a-y))/(L*eps_a+(1-L)*y)) 
        x = x + dx
        y = y + dy*dx
        
    return y 
    

def hallikainen_1_4(vwc, cc, sand):
    """ 
    Empirical model for permittivity at 1.4 Ghz. Hallikainen et al., 1985
    
    Parameters
    ----------
    vwc: float
        Soil volumetric water content [-]
    
    cc: float
        Soil clay content [g/100g]
        
    sand: float
        volumetric sand content (%)
        
    Returns
    -------
    eps_b: float
        Soil bulk real relative dielectric permittivity [-]
    """
    a0 = 2.862 
    a1 = -0.012 
    a2 = 0.001 
    b0 = 3.803 
    b1 = 0.462 
    b2 = -0.341 
    c0 = 119.006 
    c1 = -0.500 
    c2 = 0.633 
    eps_b = (a0 + a1*sand + a2*cc) + (b0 + b1*sand + b2*cc)*vwc + (c0 + c1*sand + c2*cc)*vwc**2
    return eps_b


def hallikainen_4(vwc, cc, sand):
    """ 
    Empirical model for permittivity at 4 Ghz. Hallikainen et al., 1985
    
    Parameters
    ----------
    vwc: float
        Soil volumetric water content [-]
    
    cc: float
        Soil clay content [g/100g]
        
    sand: float
        volumetric sand content (%)
        
    Returns
    -------
    eps_b: float
        Soil bulk real relative dielectric permittivity [-]            
    """
    
    a0 = 2.927
    a1 = -0.012 
    a2 = -0.001 
    b0 = 5.505 
    b1 = 0.371 
    b2 = 0.062
    c0 = 114.826
    c1 = -0.389
    c2 = -0.547
    eps_b = (a0 + a1*sand + a2*cc) + (b0 + b1*sand + b2*cc)*vwc + (c0 + c1*sand + c2*cc)*vwc**2
    return eps_b
    

def hallikainen_18(vwc, cc, sand):
    """ 
    Empirical model for permittivity at 18 Ghz. Hallikainen et al., 1985
    
    Parameters
    ----------
    vwc: float
        Soil volumetric water content [-]
    
    cc: float
        volumetric cc content [-]
        
    sand: float
        volumetric sand content [-]
        
    Returns
    -------
    eps_b: float
        Soil bulk real relative dielectric permittivity [-]
    """
    a0 = 1.912
    a1 = 0.007
    a2 = 0.021
    b0 = 29.123 
    b1 = -0.190 
    b2 = -0.545
    c0 = 6.960
    c1 = 0.822
    c2 = 1.195
    eps_b = (a0 + a1*sand + a2*cc) + (b0 + b1*sand + b2*cc)*vwc + (c0 + c1*sand + c2*cc)*vwc**2
    return eps_b


################################ # # #  REAL WATER PERMITTIVITY  # # # #########################################


def malmberg_maryott(T):
    """
    Malmberg & Maryott 1956, see also Glover 2005

    Parameters
    ----------
    T: float
        Temperature (Celsius)

    Returns
    -------
    eps_w: float
        Soil water phase real relative dielectric permittivity [-]
    """
    eps_w = 87.740 - 0.40008*T + 9.398e-4*T**2 - 1.410e-6*T**3
    return eps_w

########################################## # # #   SOLID PHASE PERMITTIVITY   # # # #############################


def crim_es(bulkperm, bd, pd, eps_a):
    """
    Kameyama & Miyamoto, 2008 
    
    Parameters
    ----------
    bulkperm: float
        bulk real permittivity [-]
    
    bd: float
        Soil dry bulk density [g/cm3]
    
    pd: float
        Soil particle (solid phase) density [g/cm3]
        
    eps_a: float
        Soil air phase real relative dielectric permittivity [-]

    Returns
    -------
    eps_s: float
        Soil solid phase real relative dielectric permittivity [-]   
    """
    
    por = 1 - bd/pd                                           
    eps_s = ((bulkperm**0.5 - por*eps_a**0.5)/(1-por))**2         
    return eps_s


####################################### # # #  New PPMs high frequency range   # # # ##############################


def LR_high_freq(vwc, bd, pd, cc, eps_a, eps_s, eps_w): 
    """
    Lichtenecker and LRer (1931) and Mendoza Veirana
    
    Parameters
    ----------
    vwc: float
        Soil volumetric water content [-]
    
    bd: float
        Soil dry bulk density [g/cm3]
    
    pd: float
        Soil particle (solid phase) density [g/cm3]
        
    cc: float
        clay content [-]
        
    eps_a: float
        Soil air phase real relative dielectric permittivity [-]
        
    eps_s: float
        Soil solid phase real relative dielectric permittivity [-]
        
    eps_w: float
        Soil water phase real relative dielectric permittivity [-]
        
    Returns
    -------
    eps_b: float
        Soil bulk real relative dielectric permittivity [-]
    """
    por = 1 - bd/pd   
    alpha = -0.46 * cc/100 + 0.71                                
    eps_b = (vwc*eps_w**alpha + (1-por)*eps_s**alpha + (por-vwc)*eps_a**(alpha))**(1/alpha)
    return eps_b


def linde_high_freq(vwc, bd, pd, cc, eps_a, eps_s, eps_w, eps_offset):
    """
    Linde et al., 2006 and Mendoza Veirana
    
    Parameters
    ----------
    vwc: float
        Soil volumetric water content [-]
    
    bd: float
        Soil dry bulk density [g/cm3]
    
    pd: float
        Soil particle (solid phase) density [g/cm3]
        
    cc: float
        clay content [-]
        
    eps_a: float
        Soil air phase real relative dielectric permittivity [-]
        
    eps_s: float
        Soil solid phase real relative dielectric permittivity [-]
        
    eps_w: float
        Soil water phase real relative dielectric permittivity [-]

    eps_offset: float
        eps_offset [-]

    Returns
    -------
    eps_b: float
        Soil bulk real relative dielectric permittivity [-]
    """
    por = 1 - bd/pd                                                                         
    S = vwc / por   
    alpha = -0.46 * cc/100 + 0.71                                
    m = np.log((((1 - por)*(eps_s/eps_w)**alpha) + por)**(1/alpha) - (eps_offset/eps_w))/np.log(por)                                                                         
    n = m
    eps_b = (por**m) * ((S**n)*eps_w + ((por**-m) - 1)*eps_s +(1-S**n)*eps_a)
    return eps_b


def endres_redman_high_freq(vwc, bd, pd, cc, eps_a, eps_s, eps_w, eps_offset):
    """
    Endres & Redman 1996 and Mendoza Veirana
    
    Parameters
    ----------
    vwc: float
        Soil volumetric water content [-]
    
    bd: float
        Soil dry bulk density [g/cm3]
    
    pd: float
        Soil particle (solid phase) density [g/cm3]
        
    cc: float
        clay content [-]
        
    eps_a: float
        Soil air phase real relative dielectric permittivity [-]
        
    eps_s: float
        Soil solid phase real relative dielectric permittivity [-]
        
    eps_w: float
        Soil water phase real relative dielectric permittivity [-]
        
    eps_offset: float
        eps_offset [-]
        
    Returns
    -------
    eps_b: float
        Soil bulk real relative dielectric permittivity [-]
    """
    por = 1 - bd/pd                                                                        
    S = vwc/por                                                                              
    alpha = -0.46 * cc/100 + 0.71                                                              
    m = (np.log((((1 - por)*(eps_s/eps_w)**alpha) + por)**(1/alpha) - (eps_offset/eps_w))) / np.log(por)    
    L = (-1/m) + 1                                                       
                                                                  # Initializing diferenciation parameters
    y = eps_w                                                     # Initial permitivity = epsilon sub a  
    x = 0                                                         # Diferentiation from p = 0  
    dx = 0.05                                                     # Diferentiation step
    
    while x<1:                                                    # Diferentiation until p = 1
        dy = ((dx*y*(1-S))/(S+x*(1-S))) * ((eps_a-y)/(L*eps_a+(1-L)*y))  
        x = x + dx
        y = y + dy
                                                                  # Now y is equal to permitivity of pore(s)
    p = 0
    dp = 0.05
    z = y
    
    while p<1:    
        dz = (dp*z*(1-por))/(por+p*(1-por)) * ((eps_s-z)/(L*eps_s+(1-L)*z))
        p = p + dp
        z = z + dz
        
    return z


############################################### New PPMs low frequency range ##############################################


def LR_mv(vwc, bd, pd, eps_a, eps_s, eps_w, CEC): 
    """
    LR et al., 1990 and Mendoza Veirana
    
    Parameters
    ----------
    vwc: float
        Soil volumetric water content [-]
    
    bd: float
        Soil dry bulk density [g/cm3]
    
    pd: float
        Soil particle (solid phase) density [g/cm3]
        
    eps_a: float
        Soil air phase real relative dielectric permittivity [-]
        
    eps_s: float
        Soil solid phase real relative dielectric permittivity [-]
        
    eps_w: float
        Soil water phase real relative dielectric permittivity [-]
        
    CEC: float
        Cation exchange ceps_aacity [meq/100g]
        
    Returns
    -------
    eps_b: float
        Soil bulk real relative dielectric permittivity [-]
    """
    por =1 - bd/pd                                      
    alpha = 0.271*np.log(CEC) + 0.306
    eps_b = ( vwc*eps_w**alpha + (1-por)*eps_s**alpha + (por-vwc)*eps_a**(alpha))**(1/alpha)

    return eps_b


def linde_mv(vwc, bd, pd, eps_a, eps_s, eps_w, CEC):
    """
    Linde et al., 2006 and Mendoza Veirana
    
    Parameters
    ----------
    vwc: float
        Soil volumetric water content [-]
    
    bd: float
        Soil dry bulk density [g/cm3]
    
    pd: float
        Soil particle (solid phase) density [g/cm3]
        
    eps_a: float
        Soil air phase real relative dielectric permittivity [-]
        
    eps_s: float
        Soil solid phase real relative dielectric permittivity [-]
        
    eps_w: float
        Soil water phase real relative dielectric permittivity [-]
        
    CEC: float
        Cation exchange Ceps_aacity [meq/100g]
        
    Returns
    -------
    eps_b: float
        Soil bulk real relative dielectric permittivity [-]
    """
    por = 1 - bd/pd                                                                                
    m = -0.269*np.log(CEC) + 1.716
    n = m                                                            
    S = vwc / por
    eps_b = (por**m) * ((S**n)*eps_w + ((por**-m) - 1)*eps_s +(1-S**n)*eps_a)
    return eps_b


def sen_mv(vwc, bd, pd, eps_a, eps_s, eps_w, CEC):        
    """
    Sen et al., 1981 and Mendoza Veirana
    
    Parameters
    ----------
    vwc: float
        Soil volumetric water content [-]
    
    bd: float
        Soil dry bulk density [g/cm3]
    
    pd: float
        Soil particle (solid phase) density [g/cm3]
        
    eps_a: float
        Soil air phase real relative dielectric permittivity [-]
        
    eps_s: float
        Soil solid phase real relative dielectric permittivity [-]
        
    eps_w: float
        Soil water phase real relative dielectric permittivity [-]
        
    CEC: float
        Cation exchange Ceps_aacity [meq/100g]
        
    Returns
    -------
    eps_b: float
        Soil bulk real relative dielectric permittivity [-]
    """

    por = 1 - bd/pd                                      
    L =  -0.186 *np.log(CEC) +  0.565               
    if CEC> 25:
        L = -0.02
        
    cl = vwc*(L*eps_s + (1-L)*eps_w)                             # Calculation just for simplicity 
    wcg = eps_w*(((1-por)*eps_s+cl) / ((1-por)*eps_w+cl))           # wcg = wat coated grains
    df = (por*-1) + vwc                                    # Diference utilized just for simplicity

    y = eps_a                                                 # Initial permitivity = epsilon sub a  
    x = 0.001                                              # Diferentiation from p = 0  
    dx = 0.05       

    while x<1:                                             # Diferentiation until p = 1
        dy = ((y*(1+df))/(-df+x*(1+df))) * ((wcg-y)/(L*wcg+(1-L)*y))
        x=x+dx
        y=y+dy*dx
        
    return y 


def feng_sen_mv(vwc, bd, pd, eps_a, eps_s, eps_w, CEC):
    """
    Feng & Sen 1985 and Mendoza Veirana
    
    Parameters
    ----------
    vwc: float
        Soil volumetric water content [-]
    
    bd: float
        Soil dry bulk density [g/cm3]
    
    pd: float
        Soil particle (solid phase) density [g/cm3]
        
    eps_a: float
        Soil air phase real relative dielectric permittivity [-]
        
    eps_s: float
        Soil solid phase real relative dielectric permittivity [-]
        
    eps_w: float
        Soil water phase real relative dielectric permittivity [-]
        
    CEC: float
        Cation exchange Ceps_aacity [meq/100g]
        
    Returns
    -------
    eps_b: float
        Soil bulk real relative dielectric permittivity [-]
    """

    por = 1 - bd/pd                                                                    
    L = -0.193*np.log(CEC) +  0.44 
    y = eps_w                                                     # Initial permitivity = epsilon sub a  
    x = 0                                                         # Diferentiation from p = 0  
    dx = 0.05                                                     # Diferentiation step
    
    while x<1:                                                    # Diferentiation until p = 1
        dy = (y/(vwc+x*(1-vwc))) * ((((1-por)*((eps_s-y))/(L*eps_s+(1-L)*y))) + ((por-vwc)*(eps_a-y))/(L*eps_a+(1-L)*y)) 
        x = x + dx
        y = y + dy*dx
        
    return y  


def endres_redman_mv(vwc, bd, pd, eps_a, eps_s, eps_w, CEC):   
    """
    Endres & Redman 1996 and Mendoza Veirana
    
    Parameters
    ----------
    vwc: float
        Soil volumetric water content [-]
    
    bd: float
        Soil dry bulk density [g/cm3]
    
    pd: float
        Soil particle (solid phase) density [g/cm3]
        
    eps_a: float
        Soil air phase real relative dielectric permittivity [-]
        
    eps_s: float
        Soil solid phase real relative dielectric permittivity [-]
        
    eps_w: float
        Soil water phase real relative dielectric permittivity [-]
        
    CEC: float
        Cation exchange Ceps_aacity [meq/100g]
        
    Returns
    -------
    eps_b: float
        Soil bulk real relative dielectric permittivity [-]
    """
    
    por = 1 - bd/pd                                           
    S = vwc/por          
    L = -0.194*np.log(CEC) + 0.472 
    y = eps_w                                                         
    x = 0                                                         # Diferentiation from p = 0  
    dx = 0.05                                                     # Diferentiation step
    
    while x<1:                                                    # Diferentiation until p = 1
        dy = ((dx*y*(1-S))/(S+x*(1-S))) * ((eps_a-y)/(L*eps_a+(1-L)*y))  
        x = x + dx
        y = y + dy
    p = 0
    dp = 0.05
    z = y
    
    while p<1:    
        dz = (dp*z*(1-por))/(por+p*(1-por)) * ((eps_s-z)/(L*eps_s+(1-L)*z))
        p = p + dp
        z = z + dz
        
    return z


def wunderlich_mv(vwc, perm_init, water_init, eps_w, CEC): 
    """
    Wunderlich et.al 2013 and Mendoza Veirana
    
    Parameters
    ----------
    vwc: float
        Soil volumetric water content [-]
    
    perm_init: float
        lowest real permittivity [-]
        
    wat_init: float
        lowest volumetric water content [-]
        
    eps_w: float
        Soil water phase real relative dielectric permittivity [-]
        
    CEC: float
        Cation exchange Ceps_aacity [meq/100g]

    Returns
    -------
    eps_b: float
        Soil bulk real relative dielectric permittivity [-]   
    """                
    Lw =  -0.0493*np.log(CEC) + 0.1279 
    diff = vwc - water_init                                        # Diference utilized just for simplicity
    y = perm_init                                                  # Initial permitivity = epsilon sub 1  
    x = 0.001                                                      # Diferentiation from p = 0  
    dx = 0.05                                                      # Initial x
                                                                   # Diferentiation step until p = 1
    while x<1:                                                    
        dy = ((y*diff)/(1-diff+x*diff)) * ((eps_w-y)/(Lw*eps_w+(1-Lw)*y))
        x=x+dx
        y=y+dy*dx
        
    return y