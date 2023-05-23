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

def topp(vmc):
    """
        Topp et al. (1980).
        
        Parameters
        ----------
        vmc: float
            volumetric moisture content [-]
            
        Returns
        -------
        Aparent bulk permittivity: float        
    """
    p = [4.3e-6, -5.5e-4, 2.92e-2, -5.3e-2 - vmc]
    roots = np.roots(p)
    roots = roots[roots.imag == 0 ]
    perm_rel = roots[roots > 0]
    
    return (perm_rel[0].real)


def jacandschjB(vmc, bd, cc, org):
    """
    Jacobsen and Schjonning (1993) (Equation 4)
        
        Parameters
        ----------
        vmc: float
            volumetric moisture content [-]
            
        cc: float
            volumetric clay content (%)
            
        org: float
            organic matter content (%)
            
        Returns
        -------
        Aparent bulk permittivity: float
    """
    
    p = [17.1e-6, -11.4e-4, 3.45e-2, 
         -3.41e-2 - vmc -3.7e-2 * bd + 7.36e-4 * cc + 47.7e-4 * org]
    roots = np.roots(p)
    roots = roots[roots.imag == 0 ]
    perm_rel = roots[roots > 0]
    
    return (perm_rel[0].real)


def LR(vmc, bd, pd, ap, sp, wp, alpha): 
    """
        Lichtenecker and LRer (1931)
        
        Parameters
        ----------
        vmc: float
            volumetric moisture content [-]
        
        bd: float
            bulk density [g/cm3]
        
        pd: float
            particle density [g/cm3]
            
        ap: float
            air permittivity phase [-]
            
        sp: float
            solid permittivity phase [-]
            
        wp: float
            water permittivity phase [-]
            
        alpha: float
            alpha exponent [-]
            
        Returns
        -------
        Real bulk permittivity: float
    """
    
    por = 1 - bd/pd    # Eq. 3
    
    return (( vmc*wp**alpha + (1-por)*sp**alpha + (por-vmc)*ap**(alpha))**(1/alpha))


def CRIM(vmc, bd, pd, ap, sp, wp, alpha = 0.5):
    """
        Birchak et.al 1974
        
        Parameters
        ----------
        vmc: float
            volumetric moisture content [-]
        
        bd: float
            bulk density [g/cm3]
        
        pd: float
            particle density [g/cm3]
            
        ap: float
            air permittivity phase [-]
            
        sp: float
            solid permittivity phase [-]
            
        wp: float
            water permittivity phase [-]
            
        Returns
        -------
        Real bulk permittivity: float
    """
    
    por = 1 - bd/pd        # Eq. 3
    
    return (( vmc*wp**alpha + (1-por)*sp**alpha + (por-vmc)*ap**(alpha))**(1/alpha))


def linde(vmc, bd, pd, ap, sp, wp, m, n):
    """
        Linde et al., 2006
        
        Parameters
        ----------
        vmc: float
            volumetric moisture content [-]
        
        bd: float
            bulk density [g/cm3]
        
        pd: float
            particle density [g/cm3]
            
        ap: float
            air permittivity phase [-]
            
        sp: float
            solid permittivity phase [-]
            
        wp: float
            water permittivity phase [-]
            
        m: float
            cementation exponent/factor [-]
            
        n: float
            saturation exponent [-]
            
        Returns
        -------
        Real bulk permittivity: float
    """
    
    por = 1 - bd/pd  # Eq. 3
    S = vmc / por     # Saturation
    
    return ((por**m) * ((S**n)*wp + ((por**-m) - 1)*sp)+(1-S**n)*ap)


def wunderlich(vmc, perm_init, wat_init, wp, L): 
    """
        Wunderlich et.al 2013 
        
        Parameters
        ----------
        vmc: float
            volumetric moisture content [-]
        
        perm_init: float
            lowest real permittivity [-]
            
        wat_init: float
            lowest volumetric water content [-]
            
        wp: float
            water permittivity phase [-]
            
        L: float
            depolarization factor [-]
   
        Returns
        -------
        Real bulk permittivity: float   
    """ 
    
    diff = vmc - wat_init                                          # Diference utilized just for simplicity
                                                                   # Initializing diferenciation parameters
    y = perm_init                                                  # Initial permitivity = epsilon sub 1  
    x = 0.001                                                      # Diferentiation from p = 0  
    dx = 0.05                                                     # Diferentiation step
                                                                   # Diferentiation until p = 1
    while x<1:                                                    
        dy = ((y*diff)/(1-diff+x*diff)) * ((wp-y)/(L*wp+(1-L)*y))
        x=x+dx
        y=y+dy*dx
        
    return y


def endres_redman(vmc, bd, pd, ap, sp, wp, L):   
    """
        Endres & Redman 1996
        
        Parameters
        ----------
        vmc: float
            volumetric moisture content [-]
        
        bd: float
            bulk density [g/cm3]
        
        pd: float
            particle density [g/cm3]
            
        ap: float
            air permittivity phase [-]
            
        sp: float
            solid permittivity phase [-]
            
        wp: float
            water permittivity phase [-]
            
        L: float
            depolarization factor [-]
            
        Returns
        -------
        Real bulk permittivity: float
    """
    
    por = 1 - bd/pd                                              # Eq. 3
    S = vmc/por                                                   # Saturation
                                                                  # Initializing diferenciation parameters
    y = wp                                                        # Initial permitivity = epsilon sub a  
    x = 0                                                         # Diferentiation from p = 0  
    dx = 0.05                                                    # Diferentiation step
    
    while x<1:                                                    # Diferentiation until p = 1
        dy = ((dx*y*(1-S))/(S+x*(1-S))) * ((ap-y)/(L*ap+(1-L)*y))  
        x = x + dx
        y = y + dy
                                                                  # Now y is equal to permitivity of pore(s)
    p = 0
    dp = 0.05
    z = y
    
    while p<1:    
        dz = (dp*z*(1-por))/(por+p*(1-por)) * ((sp-z)/(L*sp+(1-L)*z))
        p = p + dp
        z = z + dz
        
    return z


def hydraprobe(vmc):
    """
        Hydraprobe default equation for water content (See Hydraprobe manual, equation A2 apendix C).

        Parameters
        ----------
        vmc: float
            volumetric moisture content [-]
            
        Returns
        -------
        Real bulk permittivity: float
    """
    A = 0.109
    B = -0.179
    
    return (((vmc - B)/A)**2)
    
    
def nadler(vmc):
    """
        Nadler et al. (1991). 
        
        Parameters
        ----------
        vmc: float
            volumetric moisture content [-]
            
        Returns
        -------
        Aparent bulk permittivity: float            
    """
    
    p = [15e-6, -12.3e-4, 3.67e-2, -7.25e-2 - vmc]
    roots = np.roots(p)
    roots = roots[roots.imag == 0 ]
    perm_rel = roots[roots > 0]
    
    return (perm_rel[0].real)


def jacandschjA(vmc):
    """
        Jacobsen and Schjonning (1993) (Equation A2)
        
        Parameters
        ----------
        vmc: float
            volumetric moisture content [-]  
            
        Returns
        -------
        Aparent bulk permittivity: float            
    """
    
    p = [18e-6, -11.6e-4, 3.47e-2, -7.01e-2 - vmc]
    roots = np.roots(p)
    roots = roots[roots.imag == 0 ]
    perm_rel = roots[roots > 0]
    
    return (perm_rel[0].real)


def malicki(vmc, bd):
    """
        Malicki et al. 1996
        
        Parameters
        ----------
        vmc: float
            volumetric moisture content [-]
        
        bd: float
            bulk density [g/cm3]
            
        Returns
        -------
        Aparent bulk permittivity: float
    """
    
    return((vmc*(7.17 + 1.18*bd) + 0.809 + 0.168*bd + 0.159*bd**2)**2)


def steelman(vmc):
    """
        Steelman and Endres (2011) 
              
        Parameters
        ----------
        vmc: float
            volumetric moisture content [-]
            
        Returns
        -------
        Aparent bulk permittivity: float            
    """
    
    p = [2.97e-5, -2.03e-3, 5.65e-2, -0.157 - vmc]
    roots = np.roots(p)
    roots = roots[roots.imag == 0 ]
    perm_rel = roots[roots > 0]
    return (perm_rel[0].real)


def logsdonperm(vmc):
    """
        Logsdon 2010. 
        
        Parameters
        ----------
        vmc: float
            volumetric moisture content [-]
            
        Returns
        -------
        Real bulk permittivity: float            
    """
    
    p = [0.00000514, -0.00047, 0.022, 0-vmc]
    roots = np.roots(p)
    roots = roots[roots.imag == 0 ]
    perm_rel = roots[roots > 0]
    
    return (perm_rel[0].real)


def peplinski(vmc, bd, pd, cc, sand, sp, wp, ewinf, tau = 0.65):
    """
        Peplinski et al., 1995 (Equations 1 to 6) 
        
        Parameters
        ----------
        vmc: float
            volumetric moisture content [-]
        
        bd: float
            bulk density [g/cm3]
        
        pd: float
            particle density [g/cm3]
         
        cc: float
            volumetric clay content (%)
           
        sand: float
            volumetric sand content (%)
            
        sp: float
            solid permittivity phase [-]
            
        wp: float
            water permittivity phase [-]
            
        ewinf: float
            permittivity at infinite frequency [-]
            
        Returns
        -------
        Real bulk permittivity: float   
    """
    
    beta1 = 1.2748 - 0.519*sand/100 - 0.152*cc/100
    
    return (1 + (bd/pd)*sp**tau + (vmc**beta1)*(wp**tau) - vmc)**(1/tau)


def dobson(vmc, bw, bd, pd, ap, sp, wp, bwp):
    """
        Dobson et al., 1985  

        Parameters
        ----------
        vmc: float
            volumetric moisture content [-]
        
        bw: float
            volumetric bound water content [-]
            
        bd: float
            bulk density [g/cm3]
        
        pd: float
            particle density [g/cm3]
            
        ap: float
            air permittivity phase [-]
            
        sp: float
            solid permittivity phase [-]
            
        wp: float
            water permittivity phase [-]
            
        bwp: float
            bound water permittivity [-]
            
        Returns
        -------
        Real bulk permittivity: float
    """
    
    por = 1 - bd/pd
    
    num = 3*sp+2*vmc*(wp-sp) + 2*bw*(bwp-sp)+2*(por-vmc)*(ap-sp)
    den = 3+vmc*((sp/wp)-1) + bw*((sp/bwp)-1)+(por-vmc)*((sp/ap)-1)
    
    return (num/den)


def sen(vmc, bd, pd, ap, sp, wp, L):        
    """
        Sen et al., 1981
        
        Parameters
        ----------
        vmc: float
            volumetric moisture content [-]
        
        bd: float
            bulk density [g/cm3]
        
        pd: float
            particle density [g/cm3]
            
        ap: float
            air permittivity phase [-]
            
        sp: float
            solid permittivity phase [-]
            
        wp: float
            water permittivity phase [-]
            
        L: float
            depolarization factor [-]
            
        Returns
        -------
        Real bulk permittivity: float
    """
    
    por = 1 - bd/pd                                       # Eq. 3 
    cl = vmc*(L*sp + (1-L)*wp)                             # Calculation just for simplicity 
    wcg = wp*(((1-por)*sp+cl) / ((1-por)*wp+cl))           # water coated grains
    df = (por*-1) + vmc                                    # Diference utilized just for simplicity
                                                           # Initializing diferenciation parameters
    y = ap                                                 # Initial permitivity = epsilon sub a  
    x = 0.001                                              # Diferentiation from p = 0  
    dx = 0.05                                              # Diferentiation step
                                                           
    while x<1:                                             # Diferentiation until p = 1
        dy = ((y*(1+df))/(-df+x*(1+df))) * ((wcg-y)/(L*wcg+(1-L)*y))
        x=x+dx
        y=y+dy*dx
        
    return y 
 
    
def feng_sen(vmc, bd, pd, ap, sp, wp, L):
    """
        Feng & Sen 1985
        
        Parameters
        ----------
        vmc: float
            volumetric moisture content [-]
        
        bd: float
            bulk density [g/cm3]
        
        pd: float
            particle density [g/cm3]
            
        ap: float
            air permittivity phase [-]
            
        sp: float
            solid permittivity phase [-]
            
        wp: float
            water permittivity phase [-]
            
        L: float
            depolarization factor [-]
            
        Returns
        -------
        Real bulk permittivity: float
    """
    
    por = 1 - bd/pd                                       # Eq. 3
                                                           # Initializing diferenciation parameters
    y = wp                                                 # Initial permitivity = epsilon sub a  
    x = 0                                                  # Diferentiation from p = 0  
    dx = 0.05                                             # Diferentiation step
    
    while x<1:                                             # Diferentiation until p = 1
        dy = (y/(vmc+x*(1-vmc))) * ((((1-por)*((sp-y))/(L*sp+(1-L)*y))) + ((por-vmc)*(ap-y))/(L*ap+(1-L)*y)) 
        x = x + dx
        y = y + dy*dx
        
    return y 
    

def hallikainen_1_4(vmc, cc, sand):
    """ 
        Empirical model for permittivity at 1.4 Ghz. Hallikainen et al., 1985
        
        Parameters
        ----------
        vmc: float
            volumetric moisture content [-]
        
        cc: float
            volumetric clay content (%)
            
        sand: float
            volumetric sand content (%)
            
        Returns
        -------
        Real bulk permittivity: float
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
    
    return ((a0 + a1*sand + a2*cc) + (b0 + b1*sand + b2*cc)*vmc + (c0 + c1*sand + c2*cc)*vmc**2)


def hallikainen_4(vmc, cc, sand):
    """ 
        Empirical model for permittivity at 4 Ghz. Hallikainen et al., 1985
        
        Parameters
        ----------
        vmc: float
            volumetric moisture content [-]
        
        cc: float
            volumetric clay content (%)
            
        sand: float
            volumetric sand content (%)
            
        Returns
        -------
        Real bulk permittivity: float            
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
    
    return ((a0 + a1*sand + a2*cc) + (b0 + b1*sand + b2*cc)*vmc + (c0 + c1*sand + c2*cc)*vmc**2)

    
def hallikainen_18(vmc, cc, sand):
    """ 
        Empirical model for permittivity at 18 Ghz. Hallikainen et al., 1985
        
        Parameters
        ----------
        vmc: float
            volumetric moisture content [-]
        
        cc: float
            volumetric cc content [-]
            
        sand: float
            volumetric sand content [-]
            
        Returns
        -------
        Real bulk permittivity: float
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
    
    return ((a0 + a1*sand + a2*cc) + (b0 + b1*sand + b2*cc)*vmc + (c0 + c1*sand + c2*cc)*vmc**2)


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
        water permittivity phase: float  
    """
    
    return 87.740 - 0.40008*T + 9.398e-4*T**2 - 1.410e-6*T**3


########################################## # # #   SOLID PHASE PERMITTIVITY   # # # #############################


def crim_es(bulkperm, bd, pd, ap):
    """
        Kameyama & Miyamoto, 2008 
        
        Parameters
        ----------
        bulkperm: float
            bulk real permittivity [-]
        
        bd: float
            bulk density [g/cm3]
        
        pd: float
            particle density [g/cm3]
            
        ap: float
            air permittivity phase [-]

        Returns
        -------
        Solid permittivity phase: float   
    """
    
    por = 1 - bd/pd                                           
    return ((bulkperm**0.5 - por*ap**0.5)/(1-por))**2         
    

####################################### # # #  New PPMs high frequency range   # # # ##############################


def LR_high_freq(vmc, bd, pd, cc, ap, sp, wp): 
    """
        Lichtenecker and LRer (1931) and Mendoza Veirana
        
        Parameters
        ----------
        vmc: float
            volumetric moisture content [-]
        
        bd: float
            bulk density [g/cm3]
        
        pd: float
            particle density [g/cm3]
            
        cc: float
            clay content [-]
            
        ap: float
            air permittivity phase [-]
            
        sp: float
            solid permittivity phase [-]
            
        wp: float
            water permittivity phase [-]
            
        Returns
        -------
        Real bulk permittivity: float
    """
    
    por = 1 - bd/pd    # Eq. 3
    alpha = -0.46 * cc/100 + 0.71                                

    return (( vmc*wp**alpha + (1-por)*sp**alpha + (por-vmc)*ap**(alpha))**(1/alpha))


def linde_high_freq(vmc, bd, pd, cc, ap, sp, wp, offset):
    """
        Linde et al., 2006 and Mendoza Veirana
        
        Parameters
        ----------
        vmc: float
            volumetric moisture content [-]
        
        bd: float
            bulk density [g/cm3]
        
        pd: float
            particle density [g/cm3]
            
        cc: float
            clay content [-]
            
        ap: float
            air permittivity phase [-]
            
        sp: float
            solid permittivity phase [-]
            
        wp: float
            water permittivity phase [-]
            
        offset: float
            offset [-]
            
        Returns
        -------
        Real bulk permittivity: float
    """
    
    por = 1 - bd/pd                                                                         
    S = vmc / por                                                                             
    alpha = -0.46 * cc/100 + 0.71                                                              # Eq. 7
    m = (np.log((((1 - por)*(sp/wp)**alpha) + por)**(1/alpha) - (offset/wp))) / np.log(por)    # Eq. 10
    n = m
    
    return ((por**m) * ((S**n)*wp + ((por**-m) - 1)*sp)+(1-S**n)*ap)


def endres_redman_high_freq(vmc, bd, pd, cc, ap, sp, wp, offset):
    """
        Endres & Redman 1996 and Mendoza Veirana
        
        Parameters
        ----------
        vmc: float
            volumetric moisture content [-]
        
        bd: float
            bulk density [g/cm3]
        
        pd: float
            particle density [g/cm3]
            
        cc: float
            clay content [-]
            
        ap: float
            air permittivity phase [-]
            
        sp: float
            solid permittivity phase [-]
            
        wp: float
            water permittivity phase [-]
            
        offset: float
            offset [-]
            
        Returns
        -------
        Real bulk permittivity: float
    """
    
    por = 1 - bd/pd                                                                            # Eq. 3
    S = vmc/por                                                                                # Saturation
    alpha = -0.46 * cc/100 + 0.71                                                              # Eq. 7
    m = (np.log((((1 - por)*(sp/wp)**alpha) + por)**(1/alpha) - (offset/wp))) / np.log(por)    # Eq. 10
    L = (-1/m) + 1                                        
                                                                  
                                                                  # Initializing diferenciation parameters
    y = wp                                                        # Initial permitivity = epsilon sub a  
    x = 0                                                         # Diferentiation from p = 0  
    dx = 0.05                                                     # Diferentiation step
    
    while x<1:                                                    # Diferentiation until p = 1
        dy = ((dx*y*(1-S))/(S+x*(1-S))) * ((ap-y)/(L*ap+(1-L)*y))  
        x = x + dx
        y = y + dy
                                                                  # Now y is equal to permitivity of pore(s)
    p = 0
    dp = 0.05
    z = y
    
    while p<1:    
        dz = (dp*z*(1-por))/(por+p*(1-por)) * ((sp-z)/(L*sp+(1-L)*z))
        p = p + dp
        z = z + dz
        
    return z


############################################### New PPMs low frequency range ##############################################


def LR_mv(vmc, bd, pd, ap, sp, wp, CEC): 
    """
        LR et al., 1990 and Mendoza Veirana
        
        Parameters
        ----------
        vmc: float
            volumetric moisture content [-]
        
        bd: float
            bulk density [g/cm3]
        
        pd: float
            particle density [g/cm3]
            
        ap: float
            air permittivity phase [-]
            
        sp: float
            solid permittivity phase [-]
            
        wp: float
            water permittivity phase [-]
            
        CEC: float
            Cation exchange capacity [meq/100g]
            
        Returns
        -------
        Real bulk permittivity: float
    """

    por =1 - bd/pd                                           
    alpha = 0.271*np.log(CEC) + 0.306                       

    return (( vmc*wp**alpha + (1-por)*sp**alpha + (por-vmc)*ap**(alpha))**(1/alpha))


def linde_mv(vmc, bd, pd, ap, sp, wp, CEC):
    """
        Linde et al., 2006 and Mendoza Veirana
        
        Parameters
        ----------
        vmc: float
            volumetric moisture content [-]
        
        bd: float
            bulk density [g/cm3]
        
        pd: float
            particle density [g/cm3]
            
        ap: float
            air permittivity phase [-]
            
        sp: float
            solid permittivity phase [-]
            
        wp: float
            water permittivity phase [-]
            
        CEC: float
            Cation exchange Capacity [meq/100g]
            
        Returns
        -------
        Real bulk permittivity: float
    """

    por = 1 - bd/pd                                                 
    m = -0.28*np.log(CEC) + 1.762                                  
    n = m                                                            
    S = vmc / por

    return ((por**m) * ((S**n)*wp + ((por**-m) - 1)*sp)+(1-S**n)*ap)    


def sen_mv(vmc, bd, pd, ap, sp, wp, CEC):        
    """
        Sen et al., 1981 and Mendoza Veirana
        
        Parameters
        ----------
        vmc: float
            volumetric moisture content [-]
        
        bd: float
            bulk density [g/cm3]
        
        pd: float
            particle density [g/cm3]
            
        ap: float
            air permittivity phase [-]
            
        sp: float
            solid permittivity phase [-]
            
        wp: float
            water permittivity phase [-]
            
        CEC: float
            Cation exchange Capacity [meq/100g]
            
        Returns
        -------
        Real bulk permittivity: float
    """

    por = 1 - bd/pd                                      
    L =  -0.186 *np.log(CEC) +  0.565                    

    if CEC> 25:
        L = -0.02
        
    cl = vmc*(L*sp + (1-L)*wp)                             # Calculation just for simplicity 
    wcg = wp*(((1-por)*sp+cl) / ((1-por)*wp+cl))           # wcg = wat coated grains
    df = (por*-1) + vmc                                    # Diference utilized just for simplicity

    y = ap                                                 # Initial permitivity = epsilon sub a  
    x = 0.001                                              # Diferentiation from p = 0  
    dx = 0.05       

    while x<1:                                             # Diferentiation until p = 1
        dy = ((y*(1+df))/(-df+x*(1+df))) * ((wcg-y)/(L*wcg+(1-L)*y))
        x=x+dx
        y=y+dy*dx
        
    return y 


def feng_sen_mv(vmc, bd, pd, ap, sp, wp, CEC):
    """
        Feng & Sen 1985 and Mendoza Veirana
        
        Parameters
        ----------
        vmc: float
            volumetric moisture content [-]
        
        bd: float
            bulk density [g/cm3]
        
        pd: float
            particle density [g/cm3]
            
        ap: float
            air permittivity phase [-]
            
        sp: float
            solid permittivity phase [-]
            
        wp: float
            water permittivity phase [-]
            
        CEC: float
            Cation exchange Capacity [meq/100g]
            
        Returns
        -------
        Real bulk permittivity: float
    """

    por = 1 - bd/pd                                              
    L = -0.193*np.log(CEC) +  0.44                           
    y = wp                                                        # Initial permitivity = epsilon sub a  
    x = 0                                                         # Diferentiation from p = 0  
    dx = 0.05                                                     # Diferentiation step
    
    while x<1:                                                    # Diferentiation until p = 1
        dy = (y/(vmc+x*(1-vmc))) * ((((1-por)*((sp-y))/(L*sp+(1-L)*y))) + ((por-vmc)*(ap-y))/(L*ap+(1-L)*y)) 
        x = x + dx
        y = y + dy*dx
        
    return y  


def endres_redman_mv(vmc, bd, pd, ap, sp, wp, CEC):   
    """
        Endres & Redman 1996 and Mendoza Veirana
        
        Parameters
        ----------
        vmc: float
            volumetric moisture content [-]
        
        bd: float
            bulk density [g/cm3]
        
        pd: float
            particle density [g/cm3]
            
        ap: float
            air permittivity phase [-]
            
        sp: float
            solid permittivity phase [-]
            
        wp: float
            water permittivity phase [-]
            
        CEC: float
            Cation exchange Capacity [meq/100g]
            
        Returns
        -------
        Real bulk permittivity: float
    """
    
    por = 1 - bd/pd                                           
    S = vmc/por          
    L = -0.194*np.log(CEC) + 0.472                             
    y = wp                                                         
    x = 0                                                         # Diferentiation from p = 0  
    dx = 0.05                                                     # Diferentiation step
    
    while x<1:                                                    # Diferentiation until p = 1
        dy = ((dx*y*(1-S))/(S+x*(1-S))) * ((ap-y)/(L*ap+(1-L)*y))  
        x = x + dx
        y = y + dy
                                                                  # Now y is equal to permitivity of pore(s)
    p = 0
    dp = 0.05
    z = y
    
    while p<1:    
        dz = (dp*z*(1-por))/(por+p*(1-por)) * ((sp-z)/(L*sp+(1-L)*z))
        p = p + dp
        z = z + dz
        
    return z


def wunderlich_mv(vmc, perm_init, water_init, wp, CEC): 
    """
        Wunderlich et.al 2013 and Mendoza Veirana
        
        Parameters
        ----------
        vmc: float
            volumetric moisture content [-]
        
        perm_init: float
            lowest real permittivity [-]
            
        wat_init: float
            lowest volumetric water content [-]
            
        wp: float
            water permittivity phase [-]
            
        CEC: float
            Cation exchange Capacity [meq/100g]
   
        Returns
        -------
        Real bulk permittivity: float   
    """ 

    L =  -0.0493*np.log(CEC) + 0.1279                     
    diff = vmc - water_init                                        # Diference utilized just for simplicity
    y = perm_init                                                  # Initial permitivity = epsilon sub 1  
    x = 0.001                                                      # Diferentiation from p = 0  
    dx = 0.05                                                      # Initial x
                                                                   # Diferentiation step until p = 1
    while x<1:                                                    
        dy = ((y*diff)/(1-diff+x*diff)) * ((wp-y)/(L*wp+(1-L)*y))
        x=x+dx
        y=y+dy*dx
        
    return y