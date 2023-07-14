"""
Pedotransfer functions
======================
Soil dielectric permittivity modelling for low frequency instrumentation.    
...

:AUTHOR: Gaston Mendoza Veirana
:CONTACT: gaston.mendozaveirana@ugent.be

:REQUIRES: numpy
"""

# Import
import numpy as np

################################# # # # # #    PARTICLE DENSITY    # # # # ######################################


def schjonnpd(clay, org, densorg = 1.4, denspart = 2.65, densclay = 2.86, a = 1.127, b = 0.373, c = 2.648, d = 0.209):
    """
    Schjonnen et al. 2017 

    Parameters
    ----------
    clay: float
        Soil clay content [g/100g]
    
    org: float
        volumetric organic matter content (%)
        
    Returns
    -------
    pd: float
        particle density [g/cm3]   
    """
    
    clay = clay/100
    org = org/100
    
    somr = (org*densorg)/(org*densorg + (1-org)*denspart)
    claymass = (clay*densclay)/(clay*densclay + (1-clay)*denspart)
    pd = ((somr/(a+b*somr)) + (1-somr)/(c+d*claymass))**-1
    return pd


################################# # # # # #    CEMENTATION EXPONENT    # # # # ######################################


def eq16(clay):
    """
    Shah and Singh (2005)
    
    Parameters
    ----------
        
    clay: float
        Soil clay content [g/100g]
        
    Returns
    -------
    m: float
        Cementation exponent [-]   
    """      
    if (clay*100 >= 5).any():                                          
        m = 0.92*(clay*100)**0.2

    else:
        m = 1.25

    return m


def eq18(alpha):
    """
    Brovelli & Cassiani (2008)
    
    Parameters
    ----------
        
    alpha: float
        alpha geometrical parameter [-]
        
    Returns
    -------
    m: float
        Cementation exponent [-]   
    """                                       
    m = 1/alpha    
    return m


def eq20(bd, pd, sp, wp, eps_offset, alpha):
    """
    Mendoza Veirana
    
    Parameters
    ----------
    bd: float
        bulk density [g/cm3]
    
    pd: float
        particle density [g/cm3]
        
    eps_s: float
        Soil solid phase real relative dielectric permittivity [-]
        
    eps_w: float
        Soil water phase real relative dielectric permittivity [-]
        
    eps_offset: float
        eps_offset [-]
        
    alpha: float
        Alpha geometrical parameter [-] 
        
    Returns
    -------
    m: float
        Cementation exponent [-]   
    """ 
    por = 1 - bd/pd                                                                             
    m = (np.log((((1 - por)*(sp/wp)**alpha) + por)**(1/alpha) - (eps_offset/wp))) / np.log(por)          
    return m


def eq21(clay, bd, pd, sp, wp, eps_offset):
    """
    Mendoza Veirana
    
    Parameters
    ----------

    clay: float
        Soil clay content

    bd: float
        bulk density [g/cm3]
    
    pd: float
        particle density [g/cm3]
        
    eps_s: float
        Soil solid phase real relative dielectric permittivity [-]
        
    eps_w: float
        Soil water phase real relative dielectric permittivity [-]
        
    eps_offset: float
        eps_offset [-]
        
    alpha: float
        Alpha geometrical parameter [-] 
        
    Returns
    -------
    m: float
        Cementation exponent [-]   
    """ 
    por = 1 - bd/pd                                                                             
    m = (np.log((((1 - por)*(sp/wp)**(-0.46*clay+0.71)) + por)**(1/(-0.46*clay+0.71)) - (eps_offset/wp))) / np.log(por)          
    return m


def eq22(clay):
    """
    Mendoza Veirana
    
    Parameters
    ----------
        
    clay: float
        Soil clay content
        
    Returns
    -------
    m: float
        Cementation exponent [-]  
    """                                                     
    m = 1/(-0.46*clay+0.71)
    return m


################################# # # # # Alpha exponent # # # # ######################################

def eq17(clay):
    """
    Wunderlich et al., (2013)
    
    Parameters
    ----------
        
    clay: float
        Soil clay content [-]
        
    Returns
    -------
    alpha: float
        Alpha geometrical parameter [-]   
    """                                                
    alpha = -0.46*clay+0.71
    return alpha

################################# # # # # Depolarisation factor solid particles # # # # ######################################


def eq23_16(clay):
    """
        Mendoza Veirana

        Parameters
        ----------
            
        clay: float
            Soil clay content
            
        Returns
        -------
        Depolarisation factor solid particles: float   
    """                                
    if (clay*100 >= 5).any():                                          
        m = 0.92*(clay*100)**0.2

    else:
        m = 1.25
        
    L = -1/m+1
    return L


def eq23_12(clay):
    """
        Mendoza Veirana

        Parameters
        ----------
            
        clay: float
            Soil clay content
            
        Returns
        -------
        Depolarisation factor solid particles: float   
    """ 
    m = 1/(-0.46*clay+0.71)
    L = (-1/m) + 1
    return L


def eq23_21(clay, bd, pd, sp, wp, eps_offset):
    """
    Mendoza Veirana
    
    Parameters
    ----------
    clay: float
        Soil clay content [-]

    bd: float
        bulk density [g/cm3]
    
    pd: float
        particle density [g/cm3]
        
    eps_s: float
        Soil solid phase real relative dielectric permittivity [-]
        
    eps_w: float
        Soil water phase real relative dielectric permittivity [-]
        
    eps_offset: float
        eps_offset [-]
        
    Returns
    -------
    L: float
        Soil solid phase depolarization factor [-]  
    """ 
    alpha = -0.46*clay+0.71
    por = 1 - bd/pd 
    m = (np.log((((1 - por)*(sp/wp)**alpha) + por)**(1/alpha) - (eps_offset/wp))) / np.log(por)  
    L = -1/m+1
    return L


def eq24(alpha):
    """
    Mendoza Veirana

    Parameters
    ----------
        
    alpha: float
        Alpha geometrical parameter [-] 
        
    Returns
    -------
    L: float
        Soil solid phase depolarization factor [-]   
    """ 
    L = -alpha + 1
    return L


def eq23_20(bd, pd, sp, wp, offset, alpha):
    """
    Mendoza Veirana
    
    Parameters
    ----------
    bd: float
        bulk density [g/cm3]
    
    pd: float
        particle density [g/cm3]
            
    eps_s: float
        Soil solid phase real relative dielectric permittivity [-]
            
    eps_w: float
        Soil water phase real relative dielectric permittivity [-]
        
    eps_offset: float
        eps_offset [-]
        
    alpha: float
        alpha exponent as in LR model [-]
        
    Returns
    -------
    L: float
        Soil solid phase depolarization factor [-] 
    """ 
    por = 1 - bd/pd                                                                             
    m = (np.log((((1 - por)*(sp/wp)**alpha) + por)**(1/alpha) - (offset/wp))) / np.log(por)   
    L = -1/m + 1
    return L