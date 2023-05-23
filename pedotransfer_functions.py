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
            Soil clay content
        
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
    return ((somr/(a+b*somr)) + (1-somr)/(c+d*claymass))**-1


################################# # # # # #    CEMENTATION EXPONENT    # # # # ######################################


def eq16(clay):
    """
        Shah and Singh (2005)
        
        Parameters
        ----------
            
        clay: float
            Soil clay content
            
        Returns
        -------
        Cementation exponent: float   
    """      
    if (clay >= 5).any():                                          
        m = 0.92*clay**0.2

    else:
        m = 1.25

    return m


def eq18(alpha):
    """
        Brovelli & Cassiani (2008)
        
        Parameters
        ----------
            
        alpha: float
            alpha exponent as in LR model [-]
            
        Returns
        -------
        Cementation exponent: float   
    """ 
                                                                    
    m = 1/alpha    
    return m


def eq20(bd, pd, sp, wp, offset, alpha):
    """
        Mendoza Veirana
        
        Parameters
        ----------
        bd: float
            bulk density [g/cm3]
        
        pd: float
            particle density [g/cm3]
            
        sp: float
            solid permittivity phase [-]
            
        wp: float
            water permittivity phase [-]
            
        offset: float
            offset as defined in Hilhorst (2000) [-]
            
        alpha: float
            alpha exponent as in LR model [-]
            
        Returns
        -------
        Cementation exponent: float   
    """ 
    por = 1 - bd/pd                                                                             
    m = (np.log((((1 - por)*(sp/wp)**alpha) + por)**(1/alpha) - (offset/wp))) / np.log(por)          
    return m


def eq21(clay):
    """
        Mendoza Veirana
        
        Parameters
        ----------
            
        clay: float
            Soil clay content
            
        Returns
        -------
        Cementation exponent: float   
    """                                                     
    m = 1/(-0.46*clay+0.71)
    return m


def eq20_17(clay, bd, pd, sp, wp, offset):
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
            
        sp: float
            solid permittivity phase [-]
            
        wp: float
            water permittivity phase [-]
            
        offset: float
            offset as defined in Hilhorst (2000) [-]
            
        Returns
        -------
        Cementation exponent: float   
    """ 
    por = 1 - bd/pd  
    alpha = -0.46*clay + 0.71                                                   # Eq. 17                                                                           
    m = (np.log((((1 - por)*(sp/wp)**alpha) + por)**(1/alpha) - (offset/wp))) / np.log(por)      #Eq. 20     
    return m


################################# # # # # Alpha exponent # # # # ######################################

def eq17(clay):
    """
        Wunderlich et al., (2013)
        
        Parameters
        ----------
            
        clay: float
            Soil clay content
            
        Returns
        -------
        Alpha exponent: float   
    """                                                
    alpha = -0.4*clay+0.71
    return alpha

################################# # # # # Depolarisation factor solid particles # # # # ######################################


def eq23(clay):
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
    if (clay >= 5).any():                                          
        m = 0.92*clay**0.2

    else:
        m = 1.25

    L = -1/m+1
    return L


def eq22_21(clay):
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


def eq22_20_17(clay, bd, pd, sp, wp, offset):
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
    alpha = -0.4*clay+0.71
    por = 1 - bd/pd 
    m = (np.log((((1 - por)*(sp/wp)**alpha) + por)**(1/alpha) - (offset/wp))) / np.log(por)  
    L = -1/m+1
    return L


def eq24(alpha):
    """
        Mendoza Veirana

        Parameters
        ----------
            
        alpha: float
            alpha exponent as in LR model [-]
            
        Returns
        -------
        Depolarisation factor solid particles: float   
    """ 
    L = -alpha + 1
    return L


def eq20_22(bd, pd, sp, wp, offset, alpha):
    """
        Mendoza Veirana
        
        Parameters
        ----------
        bd: float
            bulk density [g/cm3]
        
        pd: float
            particle density [g/cm3]
            
        sp: float
            solid permittivity phase [-]
            
        wp: float
            water permittivity phase [-]
            
        offset: float
            offset as defined in Hilhorst (2000) [-]
            
        alpha: float
            alpha exponent as in LR model [-]
            
        Returns
        -------
        Depolarisation factor solid particles: float   
    """ 
    por = 1 - bd/pd                                                                             
    m = (np.log((((1 - por)*(sp/wp)**alpha) + por)**(1/alpha) - (offset/wp))) / np.log(por)   
    L = -1/m + 1
    return L