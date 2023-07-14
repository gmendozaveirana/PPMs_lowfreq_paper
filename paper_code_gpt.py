# Import
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import random

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse

import pedophysical_permittivity_models as ppm
import pedotransfer_functions as ptf
import plots

pd.set_option('display.max_columns', None)
""" 
######################## Reading calibration curves laboratory data ############################
"""
dt = pd.read_csv("Calibration-curves.csv")
dt
""" 
################################ Reading selected sample analysis ############################
"""
ds = pd.read_csv('Samples_table2.csv')
ds = ds.astype({"Bulk_density_reached": 'float', "Clay": "float", "Silt": "float",
                "Sand": "float", "CEC": "float", "org": "float", "Water_Ph": "float", 
                "Bound_water": "float", "Solid_phase_permittivity": 'float'})

samples = ds['Sample_name'].to_list()[:]                   # List of sample's names

"""                       Geoemtric parameters empirical PTF for high frequency range                    """
dg = pd.read_csv('data_geometric_params.csv')

# Plot
fig, ((a, b, c)) = plt.subplots(3, 1, sharex=True, figsize=(12, 20))
fig.subplots_adjust(hspace=0.1)
lw = 5
s = 50

# Defining fixed soil parameters
es = 4
ew = 80
pdn = 2.65
offset = 3
bd = 1.4

# Defining soil variables
clay_ = np.arange(0, 0.7, 0.01)

######################################       Clay vs m       ##################################### 

m_eq16 = []
for i in clay_:
    m_eq16.append(ptf.eq16(i))

print(m_eq16)

m_eq22 = ptf.eq22(clay_)
m_eq21 = ptf.eq21(clay_, bd, pdn, es, ew, offset)

a.plot(clay_*100, m_eq16,     "-", c = "black",  linewidth=lw, label = "Eq. 16 (Shah and Singh, 2005)")
a.plot(clay_*100, m_eq21,     "-", c = "orange",   linewidth=lw, label = "Eq. 21 (This study)")
a.plot(clay_*100, m_eq22,      "-", c = "navy", linewidth=lw, label = "Eq. 22 (This study)")

a.scatter(dg.clay_shah, dg.m, c = "black", s=s , marker='D', label = "Shah and Singh (2005) DataSet")

##############################    Clay vs Alpha      ##################################### 

alpha_eq17 = ptf.eq17(clay_)

b.plot(clay_*100,  alpha_eq17, "-", c = "red", linewidth=lw, label = "Eq. 17 (Wunderlich et al., (2013)"  )
b.scatter(dg.clay_wund, dg.alpha, c = "black", s=s , marker='D', label = "Wunderlich et al. (2013) DataSet")

######################################      L   vs Clay Content       ##################################### 

L_eq23_16=[]
for i in clay_:
    L_eq23_16.append(ptf.eq23(i))

L_eq23_22 = ptf.eq23_22(clay_)
L_eq23_21 = ptf.eq23_21(clay_, bd, pdn, es, ew, offset)

c.plot(clay_*100, L_eq23_16,     "-", c = "black",       linewidth=lw, label = "Eq. 23 and 16 (This study)")
c.plot(clay_*100, L_eq23_22,      "-", c = "navy",      linewidth=lw, label = "Eq. 23 and 22 (This study)")
c.plot(clay_*100, L_eq23_21,     "-", c = "orange",        linewidth=lw, label = "Eq. 23, and 21 (This study)")

plots.fig2(a, b, c)

plt.savefig("fig2", dpi=300)

################################### ####################################################

df = pd.read_csv("Field_data.csv")                   # Read the field data

scale = [df.bound_water/np.max(df.bound_water)]
l1_1 =      np.arange(3.2   ,   4.7 , (4.7   - 3.2)    /300)

fig6, ((ax4)) = plt.subplots(1, 1, figsize=(9.5, 8))

pcm = ax4.pcolormesh(scale, vmin=np.min(df.bound_water), vmax=np.max(df.bound_water), cmap='Reds')

fig6, ((ax4)) = plt.subplots(1, 1, figsize=(9.5, 8))

#pcm = ax4.pcolormesh(scale, vmin=np.min(df.bound_water), vmax=np.max(df.bound_water), cmap='Reds')

a = 0.8
s = 140

#pcm = ax4.pcolormesh(scale, vmin=np.min(df.bound_water), vmax=np.max(df.bound_water), cmap='Reds')
ax = fig6.colorbar(pcm, extend="neither", ax=ax4)
ax.set_label('$ϴ_{bw}$ [%]', size='large')

ax4.scatter(df.perm_solid, df.perm_solid_105,              c=scale , cmap = "Reds", s=s)
ax4.plot(l1_1,     l1_1, c="black",  alpha=a, label='1:1 line')

plots.fig6(ax4)
plt.savefig("figure6", dpi=400)


#################################### FIGURE 7 ###########################################

fig7, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(20,6))
fig7.subplots_adjust(wspace=0.25)

lw = 0
a = 0.8
s = 140
ms = 10

corr1 = np.corrcoef(df.Clay, df.bound_water)[1,0]
corr2 = np.corrcoef(df.Clay, df.perm_solid)[1,0]
corr3 = np.corrcoef(df.bound_water, df.perm_solid)[1,0]
print('Pearson correlation coeficients:', corr1, corr2, corr3)

ax2.scatter(df.Clay[df['SAMPLE'].str.startswith("A")], df.perm_solid[df['SAMPLE'].str.startswith("A")],alpha=a, s=s, c = 'Navy', linewidth=lw, label= 'A')
ax2.scatter(df.Clay[df['SAMPLE'].str.startswith("DREN")], df.perm_solid[df['SAMPLE'].str.startswith("DREN")],alpha=a, s=s, c = 'Firebrick', linewidth=lw, label= 'DREN')
ax2.scatter(df.Clay[df['SAMPLE'].str.startswith("D34")], df.perm_solid[df['SAMPLE'].str.startswith("D34")],alpha=a, s=s, c = 'wheat', linewidth=lw, label= 'D34')
ax2.scatter(df.Clay[df['SAMPLE'].str.startswith("EH2")], df.perm_solid[df['SAMPLE'].str.startswith("EH2")],alpha=a, s=s, c = 'Indianred', linewidth=lw, label= 'EH2')
ax2.scatter(df.Clay[df['SAMPLE'].str.startswith("E_")], df.perm_solid[df['SAMPLE'].str.startswith("E_")],alpha=a, s=s, c = 'blue', linewidth=lw, label= 'E')
ax2.scatter(df.Clay[df['SAMPLE'].str.startswith("HOEKE")], df.perm_solid[df['SAMPLE'].str.startswith("HOEKE")],alpha=a, s=s, c = 'teal', linewidth=lw, label= 'HOEKE')
ax2.scatter(df.Clay[df['SAMPLE'].str.startswith("HULD")], df.perm_solid[df['SAMPLE'].str.startswith("HULD")], alpha=a, s=s,c = 'violet', linewidth=lw, label= 'HULD')
ax2.scatter(df.Clay[df['SAMPLE'].str.startswith("P")], df.perm_solid[df['SAMPLE'].str.startswith("P")], alpha=a, s=s,c = 'cornflowerblue', linewidth=lw, label= 'P')
ax2.scatter(df.Clay[df['SAMPLE'].str.startswith("S")], df.perm_solid[df['SAMPLE'].str.startswith("S")],alpha=a, s=s, c = 'Orange', linewidth=lw, label= 'S')
ax2.scatter(df.Clay[df['SAMPLE'].str.startswith("V")], df.perm_solid[df['SAMPLE'].str.startswith("V")], alpha=a, s=s,c = 'sandybrown', linewidth=lw, label= 'VALTHE')


ax3.scatter(df.perm_solid[df['SAMPLE'].str.startswith("A")], df.bound_water[df['SAMPLE'].str.startswith("A")], alpha=a, s=s, c = 'Navy', linewidth=lw, label= 'A')
ax3.scatter(df.perm_solid[df['SAMPLE'].str.startswith("DREN")], df.bound_water[df['SAMPLE'].str.startswith("DREN")],  alpha=a, s=s,c = 'Firebrick', linewidth=lw, label= 'DREN')
ax3.scatter(df.perm_solid[df['SAMPLE'].str.startswith("D34")], df.bound_water[df['SAMPLE'].str.startswith("D34")], alpha=a, s=s, c = 'wheat', linewidth=lw, label= 'D34')
ax3.scatter(df.perm_solid[df['SAMPLE'].str.startswith("EH2")], df.bound_water[df['SAMPLE'].str.startswith("EH2")], alpha=a, s=s, c = 'Indianred', linewidth=lw, label= 'EH2')
ax3.scatter(df.perm_solid[df['SAMPLE'].str.startswith("E_")], df.bound_water[df['SAMPLE'].str.startswith("E_")], alpha=a, s=s,c = 'blue', linewidth=lw, label= 'E')
ax3.scatter(df.perm_solid[df['SAMPLE'].str.startswith("HOEKE")], df.bound_water[df['SAMPLE'].str.startswith("HOEKE")], alpha=a, s=s,c = 'teal', linewidth=lw, label= 'HOEKE')
ax3.scatter(df.perm_solid[df['SAMPLE'].str.startswith("HULD")], df.bound_water[df['SAMPLE'].str.startswith("HULD")], alpha=a, s=s,c = 'violet', linewidth=lw, label= 'HULD')
ax3.scatter(df.perm_solid[df['SAMPLE'].str.startswith("P")], df.bound_water[df['SAMPLE'].str.startswith("P")], alpha=a, s=s,c = 'cornflowerblue', linewidth=lw, label= 'P')
ax3.scatter(df.perm_solid[df['SAMPLE'].str.startswith("S")], df.bound_water[df['SAMPLE'].str.startswith("S")], alpha=a, s=s, c = 'Orange', linewidth=lw, label= 'S')
ax3.scatter(df.perm_solid[df['SAMPLE'].str.startswith("V")], df.bound_water[df['SAMPLE'].str.startswith("V")], alpha=a, s=s, c = 'sandybrown', linewidth=lw, label= 'VALTHE')


ax1.scatter(df.Clay[df['SAMPLE'].str.startswith("A")], df.bound_water[df['SAMPLE'].str.startswith("A")],alpha=a, s=s, c = 'Navy', linewidth=lw, label= 'A')
ax1.scatter(df.Clay[df['SAMPLE'].str.startswith("DREN")], df.bound_water[df['SAMPLE'].str.startswith("DREN")],alpha=a, s=s, c = 'Firebrick', linewidth=lw, label= 'DREN')
ax1.scatter(df.Clay[df['SAMPLE'].str.startswith("D34")], df.bound_water[df['SAMPLE'].str.startswith("D34")],alpha=a, s=s, c = 'wheat', linewidth=lw, label= 'D34')
ax1.scatter(df.Clay[df['SAMPLE'].str.startswith("EH2")], df.bound_water[df['SAMPLE'].str.startswith("EH2")],alpha=a, s=s, c = 'Indianred', linewidth=lw, label= 'EH2')
ax1.scatter(df.Clay[df['SAMPLE'].str.startswith("E_")], df.bound_water[df['SAMPLE'].str.startswith("E_")],alpha=a, s=s, c = 'blue', linewidth=lw, label= 'E')
ax1.scatter(df.Clay[df['SAMPLE'].str.startswith("HOEKE")], df.bound_water[df['SAMPLE'].str.startswith("HOEKE")],alpha=a, s=s, c = 'teal', linewidth=lw, label= 'HOEKE')
ax1.scatter(df.Clay[df['SAMPLE'].str.startswith("HULD")], df.bound_water[df['SAMPLE'].str.startswith("HULD")], alpha=a, s=s,c = 'violet', linewidth=lw, label= 'HULD')
ax1.scatter(df.Clay[df['SAMPLE'].str.startswith("P")], df.bound_water[df['SAMPLE'].str.startswith("P")], alpha=a, s=s,c = 'cornflowerblue', linewidth=lw, label= 'P')
ax1.scatter(df.Clay[df['SAMPLE'].str.startswith("S")], df.bound_water[df['SAMPLE'].str.startswith("S")],alpha=a, s=s, c = 'Orange', linewidth=lw, label= 'S')
ax1.scatter(df.Clay[df['SAMPLE'].str.startswith("V")], df.bound_water[df['SAMPLE'].str.startswith("V")], alpha=a, s=s,c = 'sandybrown', linewidth=lw, label= 'VALTHE')

plots.fig7(ax2, ax3, ax1)

plt.savefig("figure7", dpi=400)

################################################ FIGURE 8 ######################################################
"""                       PPM's evaluation high frequency                   """
# Varying soil properties
vmc_ = np.arange(0.04, .5 ,  (.5 - 0.03) /20)  # Volumetric water content
cc_  = np.arange(0.1, 60 , (60 - 0.1)/20)    # Clay content

# Fixed soil properties
bd = 1.5
wp = 80
offset = 3
sp = 3.7
ap = 1
pdn = 2.65
vmc = 0.2
sand = 0
clay = 40
sc = 80
org = 1

# Evaluating PPMs for high frequency range
jacandschjB_cc =    [ppm.jacandschjB(vmc, bd, cc_[0] + (cc_[1]-cc_[0])*i, org) for i in range(len(cc_))]
hallikainen1_4_cc =  ppm.hallikainen_1_4(vmc,  cc_, sc                      )
hallikainen4_cc =    ppm.hallikainen_4(  vmc,  cc_, sc                      )
hallikainen18_cc =   ppm.hallikainen_18( vmc,  cc_, sc                      )
LR_eq25_cc =       ppm.LR_high_freq(vmc, bd, pdn, cc_, ap, sp, wp)
linde_eq30_cc =      ppm.linde_high_freq(vmc, bd, pdn, cc_, ap, sp, wp, offset)
endres_redman_eq41_cc = ppm.endres_redman_high_freq(vmc, bd, pdn, cc_, ap, sp, wp, offset)

LR_eq25_sand =          ppm.LR_high_freq(vmc_, bd, pdn, sand, ap, sp, wp)
linde_eq30_sand =         ppm.linde_high_freq(vmc_, bd, pdn, sand, ap, sp, wp, offset)
endres_redman_eq41_sand = ppm.endres_redman_high_freq(vmc_, bd, pdn, sand, ap, sp, wp, offset)

LR_eq25_clay =          ppm.LR_high_freq(vmc_, bd, pdn, clay, ap, sp, wp)
linde_eq30_clay =         ppm.linde_high_freq(vmc_, bd, pdn, clay, ap, sp, wp, offset)
endres_redman_eq41_clay = ppm.endres_redman_high_freq(vmc_, bd, pdn, clay, ap, sp, wp, offset)

"""                       Plotting Figure 8                     """

fig8, ((p1, p2)) = plt.subplots(1, 2, sharey= True, figsize=(20,8))
fig8.subplots_adjust(hspace=0.04)

lw = 2
aa = 0.5
ms= 10

p2.plot(vmc_*100, endres_redman_eq41_sand, "-" , color = "mediumvioletred", markersize=ms,marker='^', alpha = aa, linewidth=lw+1, label = "Eq. 29 with 0% of clay")
p2.plot(vmc_*100, linde_eq30_sand,         "c-",    marker='^',   markersize=ms,  alpha = aa, linewidth=lw+1, label = "Eq. 28 with 0% of clay")
p2.plot(vmc_*100, LR_eq25_sand,          "g-",    marker='^',   markersize=ms,  alpha = aa, linewidth=lw+1, label = "Eq. 27 with 0% of clay")

p2.plot(vmc_*100, endres_redman_eq41_clay, "-" , color = "mediumvioletred", markersize=ms, marker='o', alpha = aa, linewidth=lw+1, label = "Eq. 29 with 40% of clay")
p2.plot(vmc_*100, linde_eq30_clay,         "c-", markersize=ms,     marker='o', alpha = aa, linewidth=lw+1, label = "Eq. 28 with 40% of clay")
p2.plot(vmc_*100, LR_eq25_clay ,         "g-", markersize=ms,     marker='o', alpha = aa, linewidth=lw+1, label = "Eq. 27 with 40% of clay")

p1.plot(cc_, jacandschjB_cc,    "k-", linewidth=lw+1, label = "Jacob-Schj PPM(Eq. 4)")
p1.plot(cc_, hallikainen1_4_cc, "m-", linewidth=1, label = "Hallikainen PPM at 1.4 GHz")
p1.plot(cc_, hallikainen4_cc,   "m-", linewidth=2, label = "Hallikainen PPM at 4 GHz")
p1.plot(cc_, hallikainen18_cc,  "m-", linewidth=3, label = "Hallikainen PPM at 18 GHz")
p1.plot(cc_, LR_eq25_cc,      "g--", alpha = aa, linewidth=lw+1, label = "LR new PPM (Eq. 27)")
p1.plot(cc_, linde_eq30_cc,     "c--", alpha = aa, linewidth=lw+1, label = "Linde new PPM (Eq. 28)")
p1.plot(cc_, endres_redman_eq41_cc, "--" , color = "mediumvioletred",  alpha = aa, linewidth=lw+1, label = "Endr-Redm new PPM (Eq.29)")

plt.subplots_adjust(bottom=0.16)
plots.fig8(p1, p2)

plt.savefig("figure8", dpi=400)

########################################### FIGURE 9 ##########################################
""""Ploting calibration curves for visualization"""
lw = 4                         # Plot parameter: line width
aa = 0.8                      # Plot parameter: alpha 
fig9, ((ax9)) = plt.subplots(1, 1, figsize=(6, 6))

ax9.plot(dt.A_44_w*100,      dt.A_44_p,       c = "navy",           marker="o", alpha=aa, linewidth=lw, label= 'A_44')
ax9.plot(dt.DREN_8_w*100,    dt.DREN_8_p,     c = 'firebrick',      marker="o", alpha=aa, linewidth=lw, label= 'DREN_8')
ax9.plot(dt.D34_8_w*100,     dt.D34_8_p,      c = "wheat",          marker="o", alpha=aa, linewidth=lw, label= 'D34_8')
ax9.plot(dt.EH2_3_w*100,     dt.EH2_3_p,      c = 'indianred',      marker="o", alpha=aa, linewidth=lw, label= 'EH2_3')
ax9.plot(dt.EH2_6_w*100,     dt.EH2_6_p,      c = "darkorange",     marker="o", alpha=aa, linewidth=lw, label= 'EH2_6')
ax9.plot(dt.E_44_w*100,      dt.E_44_p,       c = "blue",           marker="o", alpha=aa, linewidth=lw, label= 'E_44')
ax9.plot(dt.HULD_586_w*100,   dt.HULD_586_p,  c = "violet",         marker="o", alpha=aa, linewidth=lw, label= 'HULD_586')
ax9.plot(dt.P_17_w*100,       dt.P_17_p,       c = "cornflowerblue", marker="o", alpha=aa, linewidth=lw, label= 'P_17')
ax9.plot(dt.VALTHE_N5_w*100,  dt.VALTHE_N5_p, c = "sandybrown",     marker="o", alpha=aa, linewidth=lw, label= 'VALTHE_N5')
ax9.plot(dt.VALTHE_A11_w*100, dt.VALTHE_A11_p,c = "sandybrown",     marker="o", alpha=aa, linewidth=lw, label= 'VALTHE_A11')

plots.fig9(ax9)
plt.savefig("fig9", dpi=400)

""" Calculating water permittivity phase based on Malmberg & Maryott model (Eq. 13) """
dt["EH2_6_wpc"]      = ppm.malmberg_maryott(dt.EH2_6_t      )
dt["A_44_wpc"]       = ppm.malmberg_maryott(dt.A_44_t       )
dt["VALTHE_N5_wpc"]   = ppm.malmberg_maryott(dt.VALTHE_N5_t   )
dt["EH2_3_wpc"]      = ppm.malmberg_maryott(dt.EH2_3_t      )
dt["P_17_wpc"]       = ppm.malmberg_maryott(dt.P_17_t       )
dt["DREN_8_wpc"]     = ppm.malmberg_maryott(dt.DREN_8_t     )
dt["E_44_wpc"]       = ppm.malmberg_maryott(dt.E_44_t       )
dt["D34_8_wpc"]      = ppm.malmberg_maryott(dt.D34_8_t      )
dt["HULD_586_wpc"]    = ppm.malmberg_maryott(dt.HULD_586_t    )
dt["VALTHE_A11_wpc"]  = ppm.malmberg_maryott(dt.VALTHE_A11_t  )


"""
################################## PERMITTIVITY PEDOPHYSICAL MODELS EVALUATION ##################################
"""

"""  Defining a DataFrame to fill out with model's errors, considering both RMSE and R2 score.  """ 

PPM_list = [## Empirical PPMs
             "Topp (Eq. 3)", "Jac-SchjA (Eq. 4)", "HydraProbe (Eq. 25)",  
             "Nadler (Eq. A1)",  "Jac-SchjB (Eq. A2)", "Malicki (Eq. A3)", "Steel-Endr (Eq. A4)", "Logsdon (Eq. A5)",  
            
            ## Fixed parameter PPMs
             "CRIM (Eq. 6)", "Linde (Eq. 8)", "Wunderlich (Eq. 10)", "Endr-Redm (Eq. 12)",  
             "Peplinski (Eq. A6)",  "Sen (Eq. A9)", "Feng-Sen (Eq. A11)", 
            
            ## Fitted parameter PPMs
           "LR (Eq. 5)", "Linde (Eq. 7)", "Wunderlich (Eq. 9)", "Endr-Redm (Eq. 11)",   
           'Dobson (Eq. A7)', "Sen (Eq. A8)", "Feng-Sen (Eq. A10)",

            ## New pre-fitted PPMs (this study)
             "LR (Eq. 36)", "Linde (Eq. 37)", "Wunderlich (Eq. 38)", 
             "Endr-Redm (Eq. 39)", "Sen (Eq. 40)", "Feng-Sen (Eq. 41)"    ]    

RMSE= pd.DataFrame( index = PPM_list )
                    
R2= pd.DataFrame( index = PPM_list )


""" 
Defining a few fix parameters 
"""
ap =       1.3                                              # Air permittivity phase
p_inf =    3.5                                              # Permittivity relative of the soil at infinity frecuency

"""
Defining varying soil properties

"""
vmc_ =           np.arange(0.0025, .5 , (.5 - 0.0025)/100)  # Volumetric water content
alpha_ =         np.arange(0.01,   1.5, (1.5-0.01 )  /100)  # All posible alpha values (geometric parameter)
m_ =             np.arange(0.5,   2,    (2 - 0.5)    /100)  # All posible cementation factor (m) values (geometric parameter)
bwp_ =           np.arange(-5000, 2000, 7000         /100)  # All posible bound water values 
L_ =             np.arange(-0.2,  0.8,  (1)          /50 )  # All posible depolarization factor (L) values (geometric parameter)
Lw_ =            np.arange(-0.1,  0.2,  (0.3)        /50 )  # All posible depolarization factor (Lw) values (geometric parameter)

s = 0
for sample in samples:                                      # This iterates over the ten samples       
    
    ############################## Reading some parameters for each soil sample ######################
    
    print("Sample:", sample)
    sc =      ds.at[s, "Sand"]                       # SAND content (63 - 2000 * e-6 meters) (%)
    silt =    ds.at[s, "Silt"]                       # SILT content (%)
    cc  =     ds.at[s, "Clay"]                       # CLAY content (0  - 2    * e-6 meters) %[m3/m3]
    bd =      ds.at[s, "Bulk_density_reached"]       # Bulk density reached during the laboratory sample preparation the same as measured in field conditions.
    ps =      ds.at[s, "Solid_phase_permittivity"]   # Solid phase permittivity calculated using Eq. 14, after experiment.
    org =     ds.at[s, "org"]                        # Organic matter content 
    bw =      ds.at[s, "Bound_water"]                # Bound water, after experiment [m3/m3]  
    CEC =     ds.at[s, "CEC"]                        # Cation Exchange Capacity [meq/100g]

    pdn =     ptf.schjonnpd(cc, org)                 # partile density of layer(s) (g/cm³)

    ################################  Fitting models evaluation ######################################
   
    etot =    np.sum((dt.loc[:, sample+"_p"]-np.mean(dt.loc[:, sample+"_p"]))**2)   # Total sum of squares
    LR_RMSE, LindeP_RMSE, EndresRedman_RMSE, FengSen_RMSE, Wund_RMSE, Sen_RMSE, Dobson_RMSE = [], [], [], [], [], [], [] # Definition of lists
    CRIM_pred_, LR_mv_ = [], []
    
    """ LR PPM evaluation """
    for i in range(len(alpha_)):
        
        LR_RMSE.append(np.sqrt(np.mean((ppm.LR( dt.loc[:, sample+"_w"], bd, pdn, ap, ps, dt.loc[:, sample+"_wpc"], alpha_[i])-dt.loc[:, sample+"_p"])**2)))    

    LR_RMSE_min = min(LR_RMSE)
    alpha_fitted = alpha_[LR_RMSE.index(LR_RMSE_min)]
    
    LR_fitted = ppm.LR( dt.loc[:, sample+"_w"], bd, pdn, ap, ps, dt.loc[:, sample+"_wpc"], alpha_fitted)
    RMSE.at["LR (Eq. 5)", "RMSE_"+sample] = LR_RMSE_min
    R2.at["LR (Eq. 5)", "R2_"+sample] =   1 - (np.sum((LR_fitted-dt.loc[:, sample+"_p"])**2)/etot)
    ds.at[s, "Alpha"] = alpha_fitted
    print("The optimum Alpha is:", alpha_fitted)
    
    
    """ Linde's model"""
    for i in range(len(m_)):
        LindeP_RMSE.append(np.sqrt(np.mean((ppm.linde(dt.loc[:, sample+"_w"], bd, pdn, ap, ps, dt.loc[:, sample+"_wpc"], m_[i], m_[i])-dt.loc[:, sample+"_p"])**2)))

    m_fitted = m_[LindeP_RMSE.index(min(LindeP_RMSE))]
    
    LindeP_fitted = ppm.linde(dt.loc[:, sample+"_w"], bd, pdn, ap, ps, dt.loc[:, sample+"_wpc"], m_fitted, m_fitted)
    RMSE.at["Linde (Eq. 7)", "RMSE_"+sample] = min(LindeP_RMSE)
    R2.at["Linde (Eq. 7)", "R2_"+sample]   = 1 - (np.sum((LindeP_fitted-dt.loc[:, sample+"_p"])**2)/etot)
    ds.at[s, "m_linde"] = m_fitted
    print("The optimum m is:", m_fitted)
    
    
    """ Dobson's model"""
    for i in range(len(bwp_)):
        
        Dobson_RMSE.append(np.sqrt(np.mean((ppm.dobson(dt.loc[:, sample+"_w"], bw, bd, pdn, ap, ps, dt.loc[:, sample+"_wpc"], bwp_[i]) -dt.loc[:, sample+"_p"])**2)))

    bwp_fitted = bwp_[Dobson_RMSE.index(min(Dobson_RMSE))]
    
    Dobson_fitted = ppm.dobson(dt.loc[:, sample+"_w"], bw, bd, pdn, ap, ps, dt.loc[:, sample+"_wpc"], bwp_fitted)
    RMSE.at["Dobson (Eq. A7)", "RMSE_"+sample] = min(Dobson_RMSE)
    R2.at["Dobson (Eq. A7)", "R2_"+sample] =   1 - (np.sum((Dobson_fitted-dt.loc[:, sample+"_p"])**2)/etot)
    ds.at[s, "bwp_dobson"] = bwp_fitted
    print("The optimum bwp is:", bwp_fitted)
    
    
    """ Embedding schemes evaluation"""
    for i in range(len(L_)):
        
        Sen_RMSE.append(np.sqrt(np.mean((ppm.sen(dt.loc[:, sample+"_w"], bd, pdn, ap, ps, dt.loc[:, sample+"_wpc"], L_[i] )-dt.loc[:, sample+"_p"])**2)))        
        FengSen_RMSE.append(np.sqrt(np.mean((ppm.feng_sen(dt.loc[:, sample+"_w"], bd, pdn, ap, ps, dt.loc[:, sample+"_wpc"], L_[i] )-dt.loc[:, sample+"_p"])**2)))
        EndresRedman_RMSE.append(np.sqrt(np.mean((ppm.endres_redman(dt.loc[:, sample+"_w"], bd, pdn, ap, ps, dt.loc[:, sample+"_wpc"], L_[i] )-dt.loc[:, sample+"_p"])**2)))
        Wund_RMSE.append(np.sqrt(np.mean((ppm.wunderlich(dt.loc[:, sample+"_w"], dt.loc[:, sample+"_p"].min(), dt.loc[:, sample+"_w"].min(), dt.loc[:, sample+"_wpc"], Lw_[i])-dt.loc[:, sample+"_p"])**2)))
    
    RMSE.at["Sen (Eq. A8)", "RMSE_"+sample]        = min(Sen_RMSE)  
    RMSE.at["Feng-Sen (Eq. A10)", "RMSE_"+sample]      = min(FengSen_RMSE)    
    RMSE.at["Endr-Redm (Eq. 11)", "RMSE_"+sample] = min(EndresRedman_RMSE)
    RMSE.at["Wunderlich (Eq. 9)", "RMSE_"+sample]       = min(Wund_RMSE)
    
    L_fitted_Sen        = L_[Sen_RMSE.index(min(Sen_RMSE))]
    L_fitted_FengSen      = L_[FengSen_RMSE.index(min(FengSen_RMSE))]
    L_fitted_EndresRedman = L_[EndresRedman_RMSE.index(min(EndresRedman_RMSE))]
    L_fitted_Wund       = Lw_[Wund_RMSE.index(min(Wund_RMSE))]

    Sen_fitted = ppm.sen(           dt.loc[:, sample+"_w"], bd, pdn, ap, ps, dt.loc[:, sample+"_wpc"], L_fitted_Sen)
    FengSen_fitted = ppm.feng_sen(       dt.loc[:, sample+"_w"], bd, pdn, ap, ps, dt.loc[:, sample+"_wpc"], L_fitted_FengSen)
    EndresRedman_fitted = ppm.endres_redman(  dt.loc[:, sample+"_w"], bd, pdn, ap, ps, dt.loc[:, sample+"_wpc"], L_fitted_EndresRedman)
    Wund_fitted = ppm.wunderlich(dt.loc[:, sample+"_w"], dt.loc[:, sample+"_p"].min(), dt.loc[:, sample+"_w"].min(), dt.loc[:, sample+"_wpc"], L_fitted_Wund)

    R2.at["Sen (Eq. A8)", "R2_"+sample]         = 1 - (np.sum((Sen_fitted       -dt.loc[:, sample+"_p"])**2)/etot)
    R2.at["Feng-Sen (Eq. A10)", "R2_"+sample]    = 1 - (np.sum((FengSen_fitted     -dt.loc[:, sample+"_p"])**2)/etot)
    R2.at["Endr-Redm (Eq. 11)", "R2_"+sample]   = 1 - (np.sum((EndresRedman_fitted-dt.loc[:, sample+"_p"])**2)/etot)
    R2.at["Wunderlich (Eq. 9)", "R2_"+sample]  = 1 - (np.sum((Wund_fitted      -dt.loc[:, sample+"_p"])**2)/etot)                

    ds.at[s, "L_Sen"] = L_fitted_Sen    
    ds.at[s, "L_FengSen"] = L_fitted_FengSen
    ds.at[s, "L_EndresRedman"] = L_fitted_EndresRedman
    ds.at[s, "L_Wund"] = L_fitted_Wund
    
    
    ####################################### Non-fitting models evaluation ################################

    length = pd.notna(dt.loc[:,sample+"_w"]).sum()
    
    CRIM_pred, Peplinski95_pred, Malicki_pred= [], [], [], 
    FengSen_Mod_pred, LR_Mod_pred, LindeP_Mod_pred, Sen_Mod_pred, Dobson_mod_pred, EndresRedman_mod_pred, Wund_mod_pred = [], [], [], [], [], [], []
    
    EndresRedman = ppm.endres_redman(  dt.loc[:, sample+"_w"], bd, pdn, ap, ps, dt.loc[:, sample+"_wpc"], 0.3)
    FengSen = ppm.feng_sen(       dt.loc[:, sample+"_w"], bd, pdn, ap, ps, dt.loc[:, sample+"_wpc"],  0.3)
    Wund = ppm.wunderlich(dt.loc[:, sample+"_w"], 4, 0.05, dt.loc[:, sample+"_wpc"],  0.01)
    Sen = ppm.sen(           dt.loc[:, sample+"_w"], bd, pdn, ap, ps, dt.loc[:, sample+"_wpc"],  0.3)
    LindeP = ppm.linde(dt.loc[:, sample+"_w"], bd, pdn, ap, ps, dt.loc[:, sample+"_wpc"], 1.5, 2)

    R2.at["Feng-Sen (Eq. A11)", "R2_"+sample] =      1 - (np.sum((FengSen-dt.loc[:, sample+"_p"])**2)/etot)
    R2.at["Linde (Eq. 8)", "R2_"+sample] =       1 - (np.sum((LindeP - dt.loc[:, sample+"_p"])**2)/etot)
    R2.at["Sen (Eq. A9)", "R2_"+sample] =        1 - (np.sum((Sen - dt.loc[:, sample+"_p"])**2)/etot)
    R2.at["Endr-Redm (Eq. 12)", "R2_"+sample] = 1 - (np.sum((EndresRedman-dt.loc[:, sample+"_p"])**2)/etot)
    R2.at["Wunderlich (Eq. 10)", "R2_"+sample] =       1 - (np.sum((Wund-dt.loc[:, sample+"_p"])**2)/etot)
    
    RMSE.at["Feng-Sen (Eq. A11)", "RMSE_"+sample] =      np.sqrt(np.mean((FengSen - dt.loc[:, sample+"_p"])**2))
    RMSE.at["Linde (Eq. 8)", "RMSE_"+sample] =       np.sqrt(np.mean((LindeP - dt.loc[:, sample+"_p"])**2))
    RMSE.at["Sen (Eq. A9)", "RMSE_"+sample] =        np.sqrt(np.mean((Sen - dt.loc[:, sample+"_p"])**2))
    RMSE.at["Endr-Redm (Eq. 12)", "RMSE_"+sample] = np.sqrt(np.mean((EndresRedman - dt.loc[:, sample+"_p"])**2))
    RMSE.at["Wunderlich (Eq. 10)", "RMSE_"+sample] =       np.sqrt(np.mean((Wund - dt.loc[:, sample+"_p"])**2))

    Hydra = ppm.hydraprobe(      dt.loc[:, sample+"_w"]  )
    Topp =  [ppm.topp(dt.loc[i, sample+"_w"]) for i in range(length)]
    Jac_SchjB =  [ppm.jacandschjB( dt.loc[i, sample+"_w"], bd, cc, org  )  for i in range(length)]
    Nadler = [ppm.nadler(  dt.loc[i, sample+"_w"]  ) for i in range(length)]
    Logsdon = [ppm.logsdonperm(  dt.loc[i, sample+"_w"] ) for i in range(length)]
    Steelman = [ppm.steelman(     dt.loc[i, sample+"_w"]  ) for i in range(length)]
    Jac_SchjA = [ppm.jacandschjA(     dt.loc[i, sample+"_w"] ) for i in range(length)]

    RMSE.at["HydraProbe (Eq. 25)", "RMSE_"+sample]  =          np.sqrt(np.mean((Hydra - dt.loc[:, sample+"_p"])**2))
    RMSE.at["Topp (Eq. 3)",  "RMSE_"+sample] =           np.sqrt(np.mean(( Topp - dt.loc[:length-1, sample+"_p"])**2))
    RMSE.at["Jac-SchjB (Eq. A2)", "RMSE_"+sample] = np.sqrt(np.mean(( Jac_SchjB - dt.loc[:length-1, sample+"_p"])**2))
    RMSE.at["Nadler (Eq. A1)", "RMSE_"+sample] =          np.sqrt(np.mean(( Nadler - dt.loc[:length-1, sample+"_p"])**2))
    RMSE.at["Logsdon (Eq. A5)", "RMSE_"+sample] =     np.sqrt(np.mean(( Logsdon - dt.loc[:length-1, sample+"_p"])**2))
    RMSE.at["Steel-Endr (Eq. A4)", "RMSE_"+sample] =        np.sqrt(np.mean(( Steelman - dt.loc[:length-1, sample+"_p"])**2))
    RMSE.at["Jac-SchjA (Eq. 4)", "RMSE_"+sample] = np.sqrt(np.mean(( Jac_SchjA - dt.loc[:length-1, sample+"_p"])**2))
    
    R2.at["HydraProbe (Eq. 25)", "R2_"+sample]  =          1 - (np.sum((Hydra-dt.loc[:, sample+"_p"])**2)/etot)                                               
    R2.at["Topp (Eq. 3)",  "R2_"+sample] =           1 - (np.sum((Topp-dt.loc[:length-1, sample+"_p"])**2)/etot)
    R2.at["Jac-SchjB (Eq. A2)", "R2_"+sample] = 1 - (np.sum((Jac_SchjB-dt.loc[:length-1, sample+"_p"])**2)/etot)
    R2.at["Nadler (Eq. A1)", "R2_"+sample] =          1 - (np.sum((Nadler-dt.loc[:length-1, sample+"_p"])**2)/etot)
    R2.at["Logsdon (Eq. A5)", "R2_"+sample] =     1 - (np.sum((Logsdon-dt.loc[:length-1, sample+"_p"])**2)/etot)
    R2.at["Steel-Endr (Eq. A4)", "R2_"+sample] =        1 - (np.sum((Steelman-dt.loc[:length-1, sample+"_p"])**2)/etot)
    R2.at["Jac-SchjA (Eq. 4)", "R2_"+sample] = 1 - (np.sum((Jac_SchjA-dt.loc[:length-1, sample+"_p"])**2)/etot)

    CRIM_pred = ppm.CRIM(            dt.loc[:, sample+"_w"], bd, pdn,            ap, ps, dt.loc[:, sample+"_wpc"])
    dt['CRIM'+sample+'_p'] = CRIM_pred
    Peplinski95_pred = ppm.peplinski(       dt.loc[:, sample+"_w"], bd, pdn, cc, sc, ps, dt.loc[:, sample+"_wpc"], p_inf)
    Malicki_pred = ppm.malicki(         dt.loc[:, sample+"_w"], bd                                )
    
    RMSE.at["CRIM (Eq. 6)", "RMSE_"+sample] =            np.sqrt(np.mean((CRIM_pred - dt.loc[:, sample+"_p"])**2))
    RMSE.at["Peplinski (Eq. A6)", "RMSE_"+sample] =     np.sqrt(np.mean((Peplinski95_pred - dt.loc[:, sample+"_p"])**2))
    RMSE.at["Malicki (Eq. A3)", "RMSE_"+sample] =         np.sqrt(np.mean((Malicki_pred - dt.loc[:, sample+"_p"])**2))
    
    R2.at["CRIM (Eq. 6)", "R2_"+sample] =            1 - (np.sum((CRIM_pred-dt.loc[:, sample+"_p"])**2)/etot)
    R2.at["Peplinski (Eq. A6)", "R2_"+sample] =     1 - (np.sum((Peplinski95_pred-dt.loc[:, sample+"_p"])**2)/etot)
    R2.at["Malicki (Eq. A3)", "R2_"+sample] =         1 - (np.sum((Malicki_pred-dt.loc[:, sample+"_p"])**2)/etot)

    LR_mv =        ppm.LR_mv(dt.loc[:, sample+"_w"],      bd, pdn, ap, ps, dt.loc[:, sample+"_wpc"], CEC)
    dt['LR_mv'+sample+'_p'] = LR_mv 
    LindeP_mv =        ppm.linde_mv(dt.loc[:, sample+"_w"],        bd, pdn, ap, ps, dt.loc[:, sample+"_wpc"], CEC)
    Sen_mv =         ppm.sen_mv(dt.loc[:, sample+"_w"],         bd, pdn, ap, ps, dt.loc[:, sample+"_wpc"], CEC)
    endres_redman_mv = ppm.endres_redman_mv(dt.loc[:, sample+"_w"], bd, pdn, ap, ps, dt.loc[:, sample+"_wpc"], CEC)
    feng_sen_mv =    ppm.feng_sen_mv(dt.loc[:, sample+"_w"],    bd, pdn, ap, ps, dt.loc[:, sample+"_wpc"], CEC)
    Wunderlich_perm_mv =     ppm.wunderlich_mv(dt.loc[:, sample+"_w"], 4, 0.05,             dt.loc[:, sample+"_wpc"], CEC)
    
    RMSE.at["Feng-Sen (Eq. 41)",   "RMSE_"+sample] =  np.sqrt(np.mean((feng_sen_mv - dt.loc[:, sample+"_p"])**2))
    RMSE.at["LR (Eq. 36)",       "RMSE_"+sample] =  np.sqrt(np.mean((LR_mv - dt.loc[:, sample+"_p"])**2))
    RMSE.at["Linde (Eq. 37)",      "RMSE_"+sample] =  np.sqrt(np.mean((LindeP_mv - dt.loc[:, sample+"_p"])**2))
    RMSE.at["Sen (Eq. 40)",        "RMSE_"+sample] =  np.sqrt(np.mean((Sen_mv - dt.loc[:, sample+"_p"])**2))
    RMSE.at["Endr-Redm (Eq. 39)",  "RMSE_"+sample] =  np.sqrt(np.mean((endres_redman_mv - dt.loc[:, sample+"_p"])**2))
    RMSE.at["Wunderlich (Eq. 38)", "RMSE_"+sample] =  np.sqrt(np.mean((Wunderlich_perm_mv - dt.loc[:, sample+"_p"])**2))

    R2.at["Feng-Sen (Eq. 41)",   "R2_"+sample] =   1 - (np.sum((feng_sen_mv-dt.loc[:, sample+"_p"])**2)/etot)
    R2.at["LR (Eq. 36)",       "R2_"+sample] =   1 - (np.sum((LR_mv - dt.loc[:, sample+"_p"])**2)/etot)
    R2.at["Linde (Eq. 37)",      "R2_"+sample] =   1 - (np.sum((LindeP_mv - dt.loc[:, sample+"_p"])**2)/etot)
    R2.at["Sen (Eq. 40)",        "R2_"+sample] =   1 - (np.sum((Sen_mv  - dt.loc[:, sample+"_p"])**2)/etot)
    R2.at["Endr-Redm (Eq. 39)",  "R2_"+sample] =   1 - (np.sum((endres_redman_mv-dt.loc[:, sample+"_p"])**2)/etot)
    R2.at["Wunderlich (Eq. 38)", "R2_"+sample] =   1 - (np.sum((Wunderlich_perm_mv-dt.loc[:, sample+"_p"])**2)/etot)
    
    s += 1

########################################### APENDIX TABLE 1 ###########################################
RMSE['Mean'] = RMSE.mean(numeric_only=None, axis=1)
RMSE['Models'] = PPM_list
RMSE = RMSE.round(2)
RMSE.to_csv("RMSE.csv")

RMSE

R2['Mean'] = R2.mean(numeric_only=None, axis=1)
R2['Models'] = PPM_list
R2 = R2.round(2)
R2.to_csv("R2.csv")

R2

############################################# FIGURE 12 ###########################################
hue_colors = {'R2_A_44': 'navy',
              'R2_D34_8': 'wheat',
              'R2_DREN_8': 'firebrick',
              'R2_E_44': 'blue',
              'R2_EH2_3': 'indianred',
              'R2_EH2_6': 'darkorange',
              'R2_HULD_586': 'violet',
              'R2_P_17': 'cornflowerblue',
              'R2_VALTHE_A11': 'sandybrown',
              'R2_VALTHE_N5': 'sandybrown'    } 

er = R2.loc[:, ['Models',  'R2_A_44', 'R2_D34_8',   'R2_DREN_8',  'R2_E_44', 'R2_EH2_3', 'R2_EH2_6', 
                  'R2_HULD_586', 'R2_P_17', 'R2_VALTHE_A11' , 'R2_VALTHE_N5'        ]]

er.loc['Peplinski (Eq. A6)', 'R2_D34_8'] = np.nan         # Delete outlier just for visualization

dfm = er.melt('Models', var_name='cols', value_name='R2')
sns.set_theme(style="whitegrid")

g = sns.relplot(x="Models", y="R2", data=dfm, hue='cols', aspect = 1.6, height= 5, alpha=0.7, s=160, 
                palette=hue_colors)

g.set_xticklabels(rotation=80, size=13)
g.set_yticklabels(size=13)

g.set(ylim=(None, None))
plt.subplots_adjust(bottom=0.35)

sns.despine()
plt.savefig("fig12A", dpi=400)


hue_colors = {'RMSE_A_44': 'navy',
              'RMSE_D34_8': 'wheat',
              'RMSE_DREN_8': 'firebrick',
              'RMSE_E_44': 'blue',
              'RMSE_EH2_3': 'indianred',
              'RMSE_EH2_6': 'darkorange',
              'RMSE_HULD_586': 'violet',
              'RMSE_P_17': 'cornflowerblue',
              'RMSE_VALTHE_A11': 'sandybrown',
              'RMSE_VALTHE_N5': 'sandybrown'    } 

er = RMSE.loc[:, ['Models',  'RMSE_A_44', 'RMSE_D34_8',   'RMSE_DREN_8',  'RMSE_E_44', 'RMSE_EH2_3', 'RMSE_EH2_6', 
                  'RMSE_HULD_586', 'RMSE_P_17', 'RMSE_VALTHE_A11' , 'RMSE_VALTHE_N5'        ]]

dfm = er.melt('Models', var_name='cols', value_name='RMSE')
sns.set_theme(style="whitegrid")

g = sns.relplot(x="Models", y="RMSE", data=dfm, hue='cols', aspect = 1.6, height= 5, alpha=0.7, s=140, 
                palette=hue_colors)

g.set_xticklabels(rotation=80, size=13)
g.set_yticklabels(size=13)

g.set(ylim=(0, None))
plt.subplots_adjust(bottom=0.35)

sns.despine()
plt.savefig("fig12B", dpi=400)

################################################## FIGURE 10 #############################################
# Plot settings
fig10, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, figsize=(20, 8))
fig10.subplots_adjust(hspace=0.1)
fig10.subplots_adjust(wspace=0.1)
lw = 6
a = 0.7
ss = 180
ms = 12
yl=1.5
cc_ = np.arange(0, 40, 0.5)
cec_ = np.arange(0.1, 35, 0.5)

########################### Calculating mepirical pedotransfer functions #####################################

fit2_clay_alpha = np.polyfit(np.array(ds.Clay[:], dtype=float), np.array(ds.Alpha[:], dtype=float), 2)
pred2_clay_alpha = fit2_clay_alpha[-1] + fit2_clay_alpha[-2]*cc_ + fit2_clay_alpha[-3]*cc_*cc_
pred2r2_clay_alpha =       r2_score(ds.Alpha[:], fit2_clay_alpha[-1] + fit2_clay_alpha[-2]*ds.Clay[:] + fit2_clay_alpha[-3]*ds.Clay[:]*ds.Clay[:])

fit_cec_alpha_lg = np.polyfit(np.log(np.array(ds.CEC[:]).astype(float)), np.array(ds.Alpha[:], dtype=float), 1)
print("logarithmic coefficients for CEC vs alpha =", fit_cec_alpha_lg)
predlg_cec_alpha =     fit_cec_alpha_lg[-1] + fit_cec_alpha_lg[-2]*np.log(cec_)
r2_cec_alpha_lg =       r2_score(ds.Alpha[:], fit_cec_alpha_lg[-1] + fit_cec_alpha_lg[-2]*np.log(np.array(ds.CEC[:]).astype(float)))

fit2_clay_m = np.polyfit(np.array(ds.Clay[:], dtype=float), np.array(ds.m_linde[:], dtype=float), 2)
pred2_clay_m = fit2_clay_m[-1] + fit2_clay_m[-2]*cc_ + fit2_clay_m[-3]*cc_*cc_
pred2r2_clay_m =       r2_score(ds.m_linde[:], fit2_clay_m[-1] + fit2_clay_m[-2]*ds.Clay[:] + fit2_clay_m[-3]*ds.Clay[:]*ds.Clay[:])

fit_cec_m_lg = np.polyfit(np.log(np.array(ds.CEC[:]).astype(float)), np.array(ds.m_linde[:], dtype=float), 1)
print("logarithmic coefficients for CEC vs m =", fit_cec_m_lg)
predlg_cec_m =     fit_cec_m_lg[-1] + fit_cec_m_lg[-2]*np.log(cec_)
r2_cec_m_lg =       r2_score(ds.m_linde[:], fit_cec_m_lg[-1] + fit_cec_m_lg[-2]*np.log(np.array(ds.CEC[:]).astype(float)))


########################### Plotting data and empirical pedotransfer functions #####################################

colors = ['navy', 'firebrick', 'wheat', 'indianred', 'darkorange', 'blue', 'violet', 'cornflowerblue', 'sandybrown','sandybrown']

ax1.plot(   cc_,                 pred2_clay_alpha, c='g',  linewidth=lw,   alpha=a, label='square model, ${R^2}=$  '+str("{:.2f}".format(pred2r2_clay_alpha)))
ax1.scatter(ds.Clay[:], ds.Alpha[:], c=colors, s=ss)

ax2.plot(   cec_,                 predlg_cec_alpha, c='r', alpha=a,  linewidth=lw,   label='Eq. 30, ${R^2}=$  '+str("{:.2f}".format(r2_cec_alpha_lg)))
ax2.scatter(ds.CEC[:], ds.Alpha[:],  c=colors,    s=ss)

ax3.plot(   cc_,                 pred2_clay_m, c='g',  alpha=a,  linewidth=lw,  label='square model, ${R^2}=$  '+str("{:.2f}".format(pred2r2_clay_m)))
ax3.scatter(ds.Clay[:], ds.m_linde[:], c=colors, s=ss)

ax4.plot(   cec_,                 predlg_cec_m, c='r', alpha=a,  linewidth=lw,   label='Eq. 31, ${R^2}=$  '+str("{:.2f}".format(r2_cec_m_lg)))
ax4.scatter(ds.CEC[:], ds.m_linde[:],  c=colors,  s=ss)


plots.fig10(ax1, ax2, ax3, ax4)
plt.savefig("fig10", dpi=400)

#################################################### FIGURE 11 ##############################################
fig11, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(20, 8))

lw = 6
a = 0.7
ss = 180
fig11.subplots_adjust(hspace=0.1)
fig11.subplots_adjust(wspace=0.1)

fit_cec_Lsen = np.polyfit(np.log(np.array(ds.CEC[:]).astype(float)), np.array(ds.L_Sen[:], dtype=float), 1)
print("logarithmic coefficients for CEC vs L Sen =", fit_cec_Lsen)
pred_cec_Lsen =     fit_cec_Lsen[-1] + fit_cec_Lsen[-2]*np.log(cec_)
r2_cec_Lsen =       r2_score(ds.L_Sen[:], fit_cec_Lsen[-1] + fit_cec_Lsen[-2]*np.log(np.array(ds.CEC[:]).astype(float)))

fit_cec_Lfengsen = np.polyfit(np.log(np.array(ds.CEC[:]).astype(float)), np.array(ds.L_FengSen[:], dtype=float), 1)
print("logarithmic coefficients for CEC vs L Feng Sen =", fit_cec_Lfengsen)
pred_cec_Lfengsen =     fit_cec_Lfengsen[-1] + fit_cec_Lfengsen[-2]*np.log(cec_)
r2_cec_Lfengsen =       r2_score(ds.L_FengSen[:], fit_cec_Lfengsen[-1] + fit_cec_Lfengsen[-2]*np.log(np.array(ds.CEC[:]).astype(float)))

fit_cec_EndresRedman_lg = np.polyfit(np.log(np.array(ds.CEC[:]).astype(float)), np.array(ds.L_EndresRedman[:], dtype=float), 1)
print("logarithmic coefficients for CEC vs L Endres Redman =", fit_cec_EndresRedman_lg)
predlg_cec_EndresRedman =     fit_cec_EndresRedman_lg[-1] + fit_cec_EndresRedman_lg[-2]*np.log(cec_)
r2_cec_EndresRedman_lg =       r2_score(ds.L_EndresRedman[:], fit_cec_EndresRedman_lg[-1] + fit_cec_EndresRedman_lg[-2]*np.log(np.array(ds.CEC[:]).astype(float)))


fit_cec_Lw = np.polyfit(np.log(np.array(ds.CEC[:]).astype(float)), np.array(ds.L_Wund, dtype=float), 1)
print("logarithmic coefficients for CEC vs Lw =", fit_cec_Lw)
pred_cec_Lw =     fit_cec_Lw[-1] + fit_cec_Lw[-2]*np.log(cec_)
r2_cec_Lw =       r2_score(ds.L_Wund[:], fit_cec_Lw[-1] + fit_cec_Lw[-2]*np.log(np.array(ds.CEC[:]).astype(float)))


ax3.plot(   cec_,                 pred_cec_Lsen , c='r', alpha=a, linewidth=lw,    label='Eq. 34, ${R^2}=$  '+str("{:.2f}".format(r2_cec_Lsen)))
ax3.scatter(ds.CEC[:], ds.L_Sen[:], c=colors,  s=ss)

ax4.plot(   cec_,                 pred_cec_Lfengsen , c='r', alpha=a, linewidth=lw,    label='Eq. 35, ${R^2}=$  '+str("{:.2f}".format(r2_cec_Lfengsen)))
ax4.scatter(ds.CEC[:], ds.L_FengSen[:], c=colors,  s=ss)

ax2.plot(   cec_,                 predlg_cec_EndresRedman, c='r', linewidth=lw,  alpha=a,   label='Eq. 33, ${R^2}=$  '+str("{:.2f}".format(r2_cec_EndresRedman_lg)))
ax2.scatter(ds.CEC[:], ds.L_EndresRedman[:],  c=colors,  s=ss)

ax1.plot(   cec_,                 pred_cec_Lw , c='r', alpha=a, linewidth=lw,    label='Eq. 32, ${R^2}=$  '+str("{:.2f}".format(r2_cec_Lw)))
ax1.scatter(ds.CEC[:], ds.L_Wund[:], c=colors,  s=ss)

plots.fig11(ax1, ax2, ax3, ax4)
plt.savefig("fig11", dpi=400)

##################################################################################
'''                    Fieldwork results                 '''

# Defining a dataframe to save evaluation errors
Fits = pd.DataFrame(index=[ "CRIM" ,"Linde_fixed", "Endr_Redm_fixed",
                           "LR_mv", "Linde_mv", "Endr_Redm_mv"],
                    columns = ["RMSE", "R2"])

# Evaluating the models

df["partdens"] = ptf.schjonnpd(df.Clay, df.Humus)
df['WatpermT'] = ppm.malmberg_maryott(df.field_temp)

df["CRIM"] = ppm.CRIM(df.field_water/100, df.Bulk_density, df.partdens, 1.4, ps, df.WatpermT)
df["LR_mv"] = ppm.LR_mv(df.field_water/100, df.Bulk_density, df.partdens, 1.4, ps, df.WatpermT, df.CEC_meq100g)

df["linde"] = ppm.linde(df.field_water/100, df.Bulk_density, df.partdens, 1.4, ps, df.WatpermT, 1.5, 2)
df["linde_mv"] = ppm.linde_mv(df.field_water/100, df.Bulk_density, df.partdens, 1.4, ps, df.WatpermT, df.CEC_meq100g)

df["endresredman_mv"]   =  ppm.endres_redman_mv(df.field_water/100, df.Bulk_density, df.partdens, 1.4, ps, df.WatpermT, df.CEC_meq100g)
df["endresredman"]      =  ppm.endres_redman(df.field_water/100, df.Bulk_density, df.partdens, 1.4, ps, df.WatpermT, 0.3)
    
# Error evaluation
Fits.at["CRIM", "R2"] = r2_score(df.CRIM, df.field_realperm)
Fits.at["LR_mv", "R2"] = r2_score(df.LR_mv, df.field_realperm)

Fits.at["Linde_mv", "R2"] = r2_score(df.linde_mv, df.field_realperm)
Fits.at["Linde_fixed", "R2"] = r2_score(df.linde, df.field_realperm)

Fits.at["Endr_Redm_mv", "R2"] = r2_score(df.endresredman_mv, df.field_realperm)
Fits.at["Endr_Redm_fixed", "R2"] = r2_score(df.endresredman, df.field_realperm)

Fits.at["CRIM", "RMSE"] = math.sqrt(mse(df.field_realperm, df.CRIM))
Fits.at["LR_mv", "RMSE"] = math.sqrt(mse(df.field_realperm, df.LR_mv))

Fits.at["Linde_mv", "RMSE"] = math.sqrt(mse(df.field_realperm, df.linde_mv))
Fits.at["Linde_fixed", "RMSE"] = math.sqrt(mse(df.field_realperm, df.linde))

Fits.at["Endr_Redm_fixed", "RMSE"] = math.sqrt(mse(df.field_realperm, df.endresredman))
Fits.at["Endr_Redm_mv", "RMSE"] = math.sqrt(mse(df.field_realperm, df.endresredman_mv))


#################################### FIGURE 13 #########################################################
"""                                     Field data graph                             """

fig13, ((fx1, fx2), (fx3, fx4), (fx5, fx6)) = plt.subplots(3, 2, sharex=True, sharey=True,  figsize=(20, 22))
fig13.subplots_adjust(hspace=0.1)
fig13.subplots_adjust(wspace=0.1)

x = np.arange(df.field_realperm.min(), df.field_realperm.max(), df.field_realperm.max()/100)
ss = 330
s = 180
ft = 18
a = 0.7
hp = 30

fx3.plot(x, x, c = "black")
fx3.scatter(df.field_realperm[df['SAMPLE'].str.startswith("A")], df.linde[df['SAMPLE'].str.startswith("A")], alpha=a, s=s, c = "navy", label = 'A')
fx3.scatter(df.field_realperm[df['SAMPLE'].str.startswith("D34")], df.linde[df['SAMPLE'].str.startswith("D34")], alpha=a,  s=s, c = "wheat", label = 'D34')
fx3.scatter(df.field_realperm[df['SAMPLE'].str.startswith("DREN_")], df.linde[df['SAMPLE'].str.startswith("DREN_")],alpha=a,  s=s,  c = "firebrick", label = 'DREN')
fx3.scatter(df.field_realperm[df['SAMPLE'].str.startswith("E_")], df.linde[df['SAMPLE'].str.startswith("E_")], alpha=a, s=s, c = "blue", label = 'E')
fx3.scatter(df.field_realperm[df['SAMPLE'].str.startswith("EH2_")], df.linde[df['SAMPLE'].str.startswith("EH2_")], alpha=a,  s=s, c = "indianred", label = 'EH2')
fx3.scatter(df.field_realperm[df['SAMPLE'].str.startswith("HOEKE")], df.linde[df['SAMPLE'].str.startswith("HOEKE")], alpha=a, s=ss, c = "teal", label = 'HOEKE')
fx3.scatter(df.field_realperm[df['SAMPLE'].str.startswith("HULD_")], df.linde[df['SAMPLE'].str.startswith("HULD_")], alpha=a, s=s, c = 'violet', label = 'HULD')
fx3.scatter(df.field_realperm[df['SAMPLE'].str.startswith("P")], df.linde[df['SAMPLE'].str.startswith("P")], alpha=a, s=s, c = "cornflowerblue", label = 'P')
fx3.scatter(df.field_realperm[df['SAMPLE'].str.startswith("S")], df.linde[df['SAMPLE'].str.startswith("S")], alpha=a, s=ss, c = "darkorange", label = 'S')
fx3.scatter(df.field_realperm[df['SAMPLE'].str.startswith("V")], df.linde[df['SAMPLE'].str.startswith("V")], alpha=a, s=s, c = "sandybrown", label = 'VALTHE')
fx3.set_ylabel('${ε_b}$ by Eq. 8', fontsize = 24)
fx3.text(hp, 1, 'RMSE = '+str("{:.2f}".format(Fits.RMSE.Linde_fixed)), fontsize=ft)
fx3.text(hp, 5, '${R^2}=$'+str("{:.2f}".format(Fits.R2.Linde_fixed)), fontsize=ft)


fx1.plot(x, x, c = "black")
#fx1.set_title("Classic model" , fontweight='bold', fontsize=25) 
fx1.scatter(df.field_realperm[df['SAMPLE'].str.startswith("P")], df.CRIM[df['SAMPLE'].str.startswith("P")], alpha=a, s=s, c = "cornflowerblue", label = 'P')
fx1.scatter(df.field_realperm[df['SAMPLE'].str.startswith("S")], df.CRIM[df['SAMPLE'].str.startswith("S")],  alpha=a, s=ss,c = "darkorange", label = 'S')
fx1.scatter(df.field_realperm[df['SAMPLE'].str.startswith("E_")], df.CRIM[df['SAMPLE'].str.startswith("E_")], alpha=a, s=s, c = "blue", label = 'E')
fx1.scatter(df.field_realperm[df['SAMPLE'].str.startswith("A")], df.CRIM[df['SAMPLE'].str.startswith("A")], alpha=a, s=s, c = "navy", label = 'A')
fx1.scatter(df.field_realperm[df['SAMPLE'].str.startswith("D34")], df.CRIM[df['SAMPLE'].str.startswith("D34")],  alpha=a, s=s, c = "wheat", label = 'D34')
fx1.scatter(df.field_realperm[df['SAMPLE'].str.startswith("DREN_")], df.CRIM[df['SAMPLE'].str.startswith("DREN_")], alpha=a, s=s,  c = "firebrick", label = 'DREN')
fx1.scatter(df.field_realperm[df['SAMPLE'].str.startswith("V")], df.CRIM[df['SAMPLE'].str.startswith("V")], alpha=a, s=s, c = "sandybrown", label = 'VALTHE')
fx1.scatter(df.field_realperm[df['SAMPLE'].str.startswith("HOEKE")], df.CRIM[df['SAMPLE'].str.startswith("HOEKE")], alpha=a, s=ss, c = "teal", label = 'HOEKE')
fx1.scatter(df.field_realperm[df['SAMPLE'].str.startswith("HULD_")], df.CRIM[df['SAMPLE'].str.startswith("HULD_")], alpha=a, s=s, c = 'violet', label = 'HULD_d')
fx1.scatter(df.field_realperm[df['SAMPLE'].str.startswith("EH2_")], df.CRIM[df['SAMPLE'].str.startswith("EH2_")],  alpha=a, s=s, c = "indianred", label = 'EH2_')
fx1.set_ylabel('${ε_b}$ by Eq. 6', fontsize = 24)
fx1.text(hp, 1, 'RMSE = '+str("{:.2f}".format(Fits.RMSE.CRIM)), fontsize=ft)
fx1.text(hp, 5, '${R^2}=$'+str("{:.2f}".format(Fits.R2.CRIM)), fontsize=ft)


fx5.plot(x, x, c = "black")
#fx5.set_title("Sen" , fontweight='bold', fontsize=25) 
fx5.scatter(df.field_realperm[df['SAMPLE'].str.startswith("P")], df.endresredman[df['SAMPLE'].str.startswith("P")], alpha=a, s=s, c = "cornflowerblue", label = 'P')
fx5.scatter(df.field_realperm[df['SAMPLE'].str.startswith("S")], df.endresredman[df['SAMPLE'].str.startswith("S")], alpha=a, s=ss, c = "darkorange", label = 'S')
fx5.scatter(df.field_realperm[df['SAMPLE'].str.startswith("E_")], df.endresredman[df['SAMPLE'].str.startswith("E_")],alpha=a, s=s,  c = "blue", label = 'E')
fx5.scatter(df.field_realperm[df['SAMPLE'].str.startswith("A")], df.endresredman[df['SAMPLE'].str.startswith("A")], alpha=a, s=s, c = "navy", label = 'A')
fx5.scatter(df.field_realperm[df['SAMPLE'].str.startswith("D34")], df.endresredman[df['SAMPLE'].str.startswith("D34")],  alpha=a, s=s, c = "wheat", label = 'D34')
fx5.scatter(df.field_realperm[df['SAMPLE'].str.startswith("DREN_")], df.endresredman[df['SAMPLE'].str.startswith("DREN_")],  alpha=a, s=s, c = "firebrick", label = 'DREN')
fx5.scatter(df.field_realperm[df['SAMPLE'].str.startswith("V")], df.endresredman[df['SAMPLE'].str.startswith("V")], alpha=a, s=s, c = "sandybrown", label = 'VALTHE')
fx5.scatter(df.field_realperm[df['SAMPLE'].str.startswith("HOEKE")], df.endresredman[df['SAMPLE'].str.startswith("HOEKE")], alpha=a, s=ss, c = "teal", label = 'HOEKE')
fx5.scatter(df.field_realperm[df['SAMPLE'].str.startswith("HULD_")], df.endresredman[df['SAMPLE'].str.startswith("HULD_")], alpha=a, s=s, c = 'violet', label = 'HULD_d')
fx5.scatter(df.field_realperm[df['SAMPLE'].str.startswith("EH2_")], df.endresredman[df['SAMPLE'].str.startswith("EH2_")], alpha=a, s=s,  c = "indianred", label = 'EH2_')
fx5.set_ylabel('${ε_b}$ by Eq. 12', fontsize = 24)
fx5.text(hp, 1, 'RMSE = '+str("{:.2f}".format(Fits.RMSE.Endr_Redm_fixed)), fontsize=ft)
fx5.text(hp, 5, '${R^2}=$'+str("{:.2f}".format(Fits.R2.Endr_Redm_fixed)), fontsize=ft)

fx2.plot(x, x, c = "black")
fx4.plot(x, x, c = "black")
fx6.plot(x, x, c = "black")

fx6.scatter(df.field_realperm[df['SAMPLE'].str.startswith("P")], df.endresredman_mv[df['SAMPLE'].str.startswith("P")], alpha=a, s=s, c = "cornflowerblue", label = 'P')
fx6.scatter(df.field_realperm[df['SAMPLE'].str.startswith("S")], df.endresredman_mv[df['SAMPLE'].str.startswith("S")], alpha=a, s=ss, c = "darkorange", label = 'S')
fx6.scatter(df.field_realperm[df['SAMPLE'].str.startswith("E_")], df.endresredman_mv[df['SAMPLE'].str.startswith("E_")], alpha=a, s=s, c = "blue", label = 'E')
fx6.scatter(df.field_realperm[df['SAMPLE'].str.startswith("A")], df.endresredman_mv[df['SAMPLE'].str.startswith("A")], alpha=a, s=s, c = "navy", label = 'A')
fx6.scatter(df.field_realperm[df['SAMPLE'].str.startswith("D34")], df.endresredman_mv[df['SAMPLE'].str.startswith("D34")], alpha=a, s=s,  c = "wheat", label = 'D34')
fx6.scatter(df.field_realperm[df['SAMPLE'].str.startswith("DREN_")], df.endresredman_mv[df['SAMPLE'].str.startswith("DREN_")],  alpha=a, s=s, c = "firebrick", label = 'DREN')
fx6.scatter(df.field_realperm[df['SAMPLE'].str.startswith("V")], df.endresredman_mv[df['SAMPLE'].str.startswith("V")], alpha=a, s=s, c = "sandybrown", label = 'VALTHE')
fx6.scatter(df.field_realperm[df['SAMPLE'].str.startswith("HOEKE")], df.endresredman_mv[df['SAMPLE'].str.startswith("HOEKE")], alpha=a, s=ss, c = "teal", label = 'HOEKE')
fx6.scatter(df.field_realperm[df['SAMPLE'].str.startswith("HULD_")], df.endresredman_mv[df['SAMPLE'].str.startswith("HULD_")], alpha=a, s=s, c = 'violet', label = 'HULD_d')
fx6.scatter(df.field_realperm[df['SAMPLE'].str.startswith("EH2_")], df.endresredman_mv[df['SAMPLE'].str.startswith("EH2_")], alpha=a, s=s,  c = "indianred", label = 'EH2_')
fx6.text(hp, 1, 'RMSE = '+str("{:.2f}".format(Fits.RMSE.Endr_Redm_mv)), fontsize=ft)
fx6.text(hp, 5, '${R^2}=$'+str("{:.2f}".format(Fits.R2.Endr_Redm_mv)), fontsize=ft)

fx2.scatter(df.field_realperm[df['SAMPLE'].str.startswith("P")], df.LR_mv[df['SAMPLE'].str.startswith("P")],alpha=a, s=s,  c = "cornflowerblue", label = 'P')
fx2.scatter(df.field_realperm[df['SAMPLE'].str.startswith("S")], df.LR_mv[df['SAMPLE'].str.startswith("S")],alpha=a,  s=ss, c = "darkorange", label = 'S')
fx2.scatter(df.field_realperm[df['SAMPLE'].str.startswith("E_")], df.LR_mv[df['SAMPLE'].str.startswith("E_")], alpha=a, s=s, c = "blue", label = 'E')
fx2.scatter(df.field_realperm[df['SAMPLE'].str.startswith("A")], df.LR_mv[df['SAMPLE'].str.startswith("A")], alpha=a, s=s, c = "navy", label = 'A')
fx2.scatter(df.field_realperm[df['SAMPLE'].str.startswith("D34")], df.LR_mv[df['SAMPLE'].str.startswith("D34")],  alpha=a, s=s, c = "wheat", label = 'D34')
fx2.scatter(df.field_realperm[df['SAMPLE'].str.startswith("DREN_")], df.LR_mv[df['SAMPLE'].str.startswith("DREN_")], alpha=a,  s=s, c = "firebrick", label = 'DREN')
fx2.scatter(df.field_realperm[df['SAMPLE'].str.startswith("V")], df.LR_mv[df['SAMPLE'].str.startswith("V")],alpha=a,  s=s, c = "sandybrown", label = 'VALTHE')
fx2.scatter(df.field_realperm[df['SAMPLE'].str.startswith("HOEKE")], df.LR_mv[df['SAMPLE'].str.startswith("HOEKE")], alpha=a, s=ss, c = "teal", label = 'HOEKE')
fx2.scatter(df.field_realperm[df['SAMPLE'].str.startswith("HULD_")], df.LR_mv[df['SAMPLE'].str.startswith("HULD_")], alpha=a, s=s, c = 'violet', label = 'HULD_d')
fx2.scatter(df.field_realperm[df['SAMPLE'].str.startswith("EH2_")], df.LR_mv[df['SAMPLE'].str.startswith("EH2_")], alpha=a,  s=s, c = "indianred", label = 'EH2_')
fx2.text(hp, 1, 'RMSE = '+str("{:.2f}".format(Fits.RMSE.LR_mv)), fontsize=ft)
fx2.text(hp, 5, '${R^2}=$'+str("{:.2f}".format(Fits.R2.LR_mv)), fontsize=ft)

fx4.scatter(df.field_realperm[df['SAMPLE'].str.startswith("P")], df.linde_mv[df['SAMPLE'].str.startswith("P")],alpha=a, s=s,  c = "cornflowerblue", label = 'P')
fx4.scatter(df.field_realperm[df['SAMPLE'].str.startswith("S")], df.linde_mv[df['SAMPLE'].str.startswith("S")], alpha=a, s=ss, c = "darkorange", label = 'S')
fx4.scatter(df.field_realperm[df['SAMPLE'].str.startswith("E_")], df.linde_mv[df['SAMPLE'].str.startswith("E_")], alpha=a, s=s, c = "blue", label = 'E')
fx4.scatter(df.field_realperm[df['SAMPLE'].str.startswith("A")], df.linde_mv[df['SAMPLE'].str.startswith("A")],alpha=a,  s=s, c = "navy", label = 'A')
fx4.scatter(df.field_realperm[df['SAMPLE'].str.startswith("D34")], df.linde_mv[df['SAMPLE'].str.startswith("D34")], alpha=a, s=s,  c = "wheat", label = 'D34')
fx4.scatter(df.field_realperm[df['SAMPLE'].str.startswith("DREN_")], df.linde_mv[df['SAMPLE'].str.startswith("DREN_")], alpha=a, s=s,  c = "firebrick", label = 'DREN')
fx4.scatter(df.field_realperm[df['SAMPLE'].str.startswith("V")], df.linde_mv[df['SAMPLE'].str.startswith("V")], alpha=a, s=s, c = "sandybrown", label = 'VALTHE')
fx4.scatter(df.field_realperm[df['SAMPLE'].str.startswith("HOEKE")], df.linde_mv[df['SAMPLE'].str.startswith("HOEKE")], alpha=a, s=ss, c = "teal", label = 'HOEKE')
fx4.scatter(df.field_realperm[df['SAMPLE'].str.startswith("HULD_")], df.linde_mv[df['SAMPLE'].str.startswith("HULD_")], alpha=a, s=s, c = 'violet', label = 'HULD')
fx4.scatter(df.field_realperm[df['SAMPLE'].str.startswith("EH2_")], df.linde_mv[df['SAMPLE'].str.startswith("EH2_")], alpha=a, s=s,  c = "indianred", label = 'EH2')
fx4.text(hp, 1, 'RMSE = '+str("{:.2f}".format(Fits.RMSE.Linde_mv)), fontsize=ft)
fx4.text(hp, 5, '${R^2}=$'+str("{:.2f}".format(Fits.R2.Linde_mv)), fontsize=ft)

fx2.set_ylabel('${ε_b}$ by new PPM Eq. 36', fontsize = 24)
fx4.set_ylabel('${ε_b}$ by new PPM Eq. 37', fontsize = 24)
fx6.set_ylabel('${ε_b}$ by new PPM Eq. 39', fontsize = 24)

plots.fig13(fx1, fx2, fx3, fx4, fx5, fx6)
plt.savefig("fig13", dpi=400)

##################################################### FIGURE 14 ###################################################
fig14, ((x1, x2)) = plt.subplots(2, 1,  sharex=True, figsize=(12, 12))
fig14.subplots_adjust(hspace=0.10)
a = 0.8
s = 150                                                                # marker size
lw = 5

bd = 1.4
wp = 80
offset = 3
sp = 3.7
pdn = 2.65

m_eq20 = ptf.eq20(bd, pdn, sp, wp, offset, ds.Alpha[:].values)
m_eq18 = ptf.eq18(ds.Alpha[:].values)

r2_m_eq18    = r2_score(ds.m_linde[:], m_eq18)
r2_m_eq20    = r2_score(ds.m_linde[:], m_eq20)

L_eq24 = ptf.eq24(ds.Alpha[:].values)
L_eq20_22 = ptf.eq20_22(bd, pdn, sp, wp, offset, ds.Alpha[:].values)

r2_L_eq24    = r2_score(ds.L_EndresRedman[:], L_eq24)
r2_L_eq20_22 = r2_score(ds.L_EndresRedman[:], L_eq20_22)

i = np.argsort(ds.Alpha[:])

x1.scatter(ds.Alpha[:], ds.m_linde[:] ,   s=s , marker='D',  c='black', alpha=a, label="Obtained data")
x1.plot(   np.sort(ds.Alpha[:]), m_eq18[i], c='navy', alpha=a, linewidth=lw, marker ='o', label="Eq. 18     ${R^2}=$"+str("{:.3f}".format(r2_m_eq18)))
x1.plot(   np.sort(ds.Alpha[:]), m_eq20[i] , c='darkorange', alpha=a, linewidth=lw, marker ='o', label="Eq. 20     ${R^2}=$"+str("{:.3f}".format(r2_m_eq20)))

x2.scatter( ds.Alpha[:], ds.L_EndresRedman[:] ,  s=s, marker='D', c='black', alpha=a, label='Obtained data')
x2.plot(    np.sort(ds.Alpha[:]), L_eq24[i], c='navy', alpha=a, linewidth=lw, marker ='o', label="Eq. 24          ${R^2}=$"+str("{:.3f}".format(r2_L_eq24)))
x2.plot(    np.sort(ds.Alpha[:]), L_eq20_22[i], c='darkorange', alpha=a, linewidth=lw, marker ='o', label="Eq. 20 & 22  ${R^2}=$"+str("{:.3f}".format(r2_L_eq20_22)))

plots.fig14(x1, x2)
plt.savefig("fig14", dpi=400)

###################################################### FIGURE 15 #############################################
'''         Lists for synthetic evaluation of pedophysical models         '''
step = 10000    # Increasing to 100000 may require runtime

vmc_ = np.arange(np.min(df.field_water)/100,  np.max(df.field_water)/100 , (np.max(df.field_water)/100  - np.min(df.field_water)/100) /step)
bd_ =  np.arange(np.min(df.Bulk_density), np.max(df.Bulk_density), (np.max(df.Bulk_density) - np.min(df.Bulk_density))/step)
pd_ =  np.arange(np.min(df.partdens) ,    np.max(df.partdens) ,    (np.max(df.partdens)  -    np.min(df.partdens) )   /step)
cec_ = np.arange(np.min(df.CEC_meq100g),  np.max(df.CEC_meq100g),  (np.max(df.CEC_meq100g) -  np.min(df.CEC_meq100g) )/step)
wp_ =  np.arange(np.min(df.WatpermT) ,    np.max(df.WatpermT) ,    (np.max(df.WatpermT)  -    np.min(df.WatpermT) )   /step)
sp_ =  np.arange(np.min(df.perm_solid) ,  np.max(df.perm_solid) ,  (np.max(df.perm_solid)  -  np.min(df.perm_solid) ) /step)
ap_ =  np.ones(len(sp_))*1.3

rvmc_ = random.sample(list(vmc_),  len(vmc_))
rbd_  = random.sample(list(bd_ ),  len(bd_ ))
rpd_  = random.sample(list(pd_ ),  len(pd_ ))
rsp_  = random.sample(list(sp_ ),  len(sp_ ))
rwp_  = random.sample(list(wp_ ),  len(wp_ ))
rcec_ = random.sample(list(cec_),  len(cec_))

"""                          Figure generation                   """
fig15, ((r1, r2), (r3, r4), (r5, r6)) = plt.subplots(3, 2, sharey=True, figsize=(20, 22))
fig15.subplots_adjust(hspace=0.22)
fig15.subplots_adjust(wspace=0.08)

'''         Syntrthic evaluation         '''

lw = 2
yp = 45
a  = 0.15     # Increase for darker points 
ss = 14       # Increase for bigger points
lv = 3 
ts = 0.015

r_ = list(map(ppm.LR_mv, rvmc_, rbd_, rpd_, ap_, rsp_, rwp_, rcec_))

sns.kdeplot(x=rvmc_, y=r_, levels=lv, color="black", thresh=ts, ax=r1)
sns.kdeplot(x=rbd_,  y=r_, levels=lv, color="black", thresh=ts, ax=r2)
sns.kdeplot(x=rpd_,  y=r_, levels=lv, color="black", thresh=ts, ax=r3)
sns.kdeplot(x=rsp_,  y=r_, levels=lv, color="black", thresh=ts, ax=r4)
sns.kdeplot(x=rwp_,  y=r_, levels=lv, color="black", thresh=ts, ax=r5)
sns.kdeplot(x=rcec_, y=r_, levels=lv, color="black", thresh=ts, ax=r6)

r1.scatter(rvmc_, r_, color="blue", s=ss, alpha=a)
r2.scatter(rbd_,  r_, color="blue", s=ss, alpha=a)
r3.scatter(rpd_,  r_, color="blue", s=ss, alpha=a)
r4.scatter(rsp_,  r_, color="blue", s=ss, alpha=a)
r5.scatter(rwp_,  r_, color="blue", s=ss, alpha=a)
r6.scatter(rcec_, r_, color="blue", s=ss, alpha=a)


plots.fig15(r1, r2, r3, r4, r5, r6, yp)
plt.savefig("fig15", dpi=400)

print("Standard deviation = ", np.std(r_))
print("Mean =", np.mean(r_))

############################################### ABSTRACT ################################################
abst, ((x1, x2)) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(14, 6))
#abst.subplots_adjust(hspace=0.15)
abst.subplots_adjust(wspace=0.15)

a = 0.8
s = 100
hp = 31

x = np.arange(0, 50, 50/100)

x1.plot(x, x, c = "black")

x1.scatter(dt.A_44_p, dt.CRIMA_44_p, alpha=a, s=s, c = "navy")
x1.scatter(dt.D34_8_p, dt.CRIMD34_8_p, alpha=a, s=s,  c = "wheat")
x1.scatter(dt.DREN_8_p, dt.CRIMDREN_8_p,  alpha=a, s=s, c = "firebrick")
x1.scatter(dt.E_44_p, dt.CRIME_44_p, alpha=a, s=s, c = "blue")
x1.scatter(dt.EH2_3_p, dt.CRIMEH2_3_p, alpha=a, s=s,  c = "indianred")
x1.scatter(dt.EH2_6_p, dt.CRIMEH2_6_p, alpha=a, s=s,  c ='darkorange')
x1.scatter(dt.HULD_586_p, dt.CRIMHULD_586_p, alpha=a, s=s, c = 'violet')
x1.scatter(dt.P_17_p,dt.CRIMP_17_p, alpha=a, s=s, c = "cornflowerblue")
x1.scatter(dt.VALTHE_N5_p, dt.CRIMVALTHE_N5_p, alpha=a, s=s, c = "sandybrown")
x1.scatter(dt.VALTHE_A11_p, dt.CRIMVALTHE_A11_p, alpha=a, s=s, c = "sandybrown")

x1.text(1, hp, 'RMSE = '+str("{:.2f}".format(RMSE.at['CRIM (Eq. 6)', 'Mean'])), fontsize=ft)
x1.text(1, hp+4, '${R^2}=$'+str("{:.2f}".format(R2.at['CRIM (Eq. 6)', 'Mean'])), fontsize=ft)

x2.plot(x, x, c = "black")

x2.scatter(dt.A_44_p, dt.LR_mvA_44_p, alpha=a, s=s, c = "navy")
x2.scatter(dt.D34_8_p, dt.LR_mvD34_8_p, alpha=a, s=s,  c = "wheat")
x2.scatter(dt.DREN_8_p, dt.LR_mvDREN_8_p,  alpha=a, s=s, c = "firebrick")
x2.scatter(dt.E_44_p, dt.LR_mvE_44_p, alpha=a, s=s, c = "blue")
x2.scatter(dt.EH2_6_p, dt.LR_mvEH2_6_p, alpha=a, s=s,  c = 'darkorange')
x2.scatter(dt.EH2_3_p, dt.LR_mvEH2_3_p, alpha=a, s=s,  c = "indianred")
x2.scatter(dt.HULD_586_p, dt.LR_mvHULD_586_p, alpha=a, s=s, c = 'violet')
x2.scatter(dt.P_17_p,dt.LR_mvP_17_p, alpha=a, s=s, c = "cornflowerblue")
x2.scatter(dt.VALTHE_N5_p, dt.LR_mvVALTHE_N5_p, alpha=a, s=s, c = "sandybrown")
x2.scatter(dt.VALTHE_A11_p, dt.LR_mvVALTHE_A11_p, alpha=a, s=s, c = "sandybrown")


x2.text(1, hp, 'RMSE = '+str("{:.2f}".format(RMSE.at['LR (Eq. 36)', 'Mean'])), fontsize=ft)
x2.text(1, hp+4, '${R^2}=$'+str("{:.2f}".format(R2.at['LR (Eq. 36)', 'Mean'])), fontsize=ft)

plt.subplots_adjust(bottom=0.16)

#x1.legend(loc='upper right', fontsize = 16) 
x1.tick_params(axis='y', labelsize=20) 
x1.tick_params(axis='x', labelsize=20) 
x1.set_xlabel('${ε_b}$ [-] observed', fontsize = 22) 
x1.set_ylabel('${ε_b}$ by CRIM', fontsize = 22) 
x1.grid(True) 
#x1.set_ylim(3.3, 4) 
#x1.set_xlim(3.25, 4.2)

#x2.legend(loc='upper left', fontsize = 16)
x2.tick_params(axis='y', labelsize=20) 
x2.tick_params(axis='x', labelsize=20) 
x2.set_xlabel('${ε_b}$ [-] observed', fontsize = 22) 
x2.set_ylabel('${ε_b}$ by this study', fontsize = 22) 
#x2.set_yticks(np.arange(0, -120, -3))
x2.grid(True) 
#x2.set_ylim(3.2, 4.5)  
#x2.set_xlim(3.2, 4.5)
plt.savefig("graph_abstr", dpi=400)