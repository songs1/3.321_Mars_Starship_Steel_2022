# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 19:31:01 2022

@author: Samuel Song (ssong64@mit.edu)
"""

#-------------------------------------------------------------------------------
# 3.321 SpaceX Mars Starship Stainless Steel Project
# Austenite Stability Calculations
#-------------------------------------------------------------------------------

################################################################################
#-------------------------------------------------------------------------------
# USER-INPUT SECTION:
#-------------------------------------------------------------------------------

# Select alloy: 
SELECTED_ALLOY_CODE = 'CH4-v03a' # (see alloys.csv)
COLD_ROLL_PERCENT = 40

# Calculations to perform:
CALC_MARTENSITE_VOL_FRAC = False
OVERLAY_OLSON_COHEN_MODELS = True
COMPARE_NARUTANI_DeltaG_ch = False

CALC_ANNEALED_Ms_sigma = False
CALC_ANNEALED_TOUGHNESS = True
LABEL_ASP_THRESHOLD_CONTOURS = False
ANNEALED_CRYO_CONTOURS = ('Mn', 'Si'), # ('Ni', 'Cr'), ('C', 'Cu'), # ('N', 'Si')
MYLES_COMBINE = False

CALC_COLDROLLED_Ms_sigma = False
CALC_COLDROLLED_TOUGHNESS = True
COLDROLLED_CRYO_CONTOURS = [] # ('Ni', 'Cr'), ('C', 'Cu'), ('N', 'Si')

PLOT_ASP_vs_COLDROLL = False

#-------------------------------------------------------------------------------
################################################################################


#-------------------------------------------------------------------------------
# System Setup
#-------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Custom Functions
import asp
import coldroll
print('modules loaded')

alphaStr = '\u03B1'
betaStr = '\u03B2'
DeltaStr = '\u0394'
sigmaStr = '\u03C3'
epsilonStr = '\u03B5'
degreeStr = '\u00B0'


# Import 301 Stainless Alloys Compositions from alloys.csv
alloy_db = pd.read_csv('alloys.csv')
alloy_names = np.array(alloy_db['Alloy'], dtype=str)
alloy_op_temps = np.array(alloy_db['Op Temp'], dtype=float)
alloy_comp_data = alloy_db.drop(columns=['Alloy', 'Op Temp', 'Comments'])
alloy_elements = list(alloy_comp_data.columns)
alloy_comp, alloy_str, alloy_temp = {}, {}, {}
for i, row in enumerate(np.array(alloy_comp_data, dtype=float)):
    name = alloy_names[i]
    op_temp = alloy_op_temps[i]
    mass_fraction_dict = asp.make_comp(alloy_elements, row)
    alloy_comp[name] = mass_fraction_dict
    alloy_str[name] = asp.comp_str(mass_fraction_dict)
    alloy_temp[name] = op_temp


# Set up selected alloy
comp = alloy_comp[SELECTED_ALLOY_CODE]
comp_str = alloy_str[SELECTED_ALLOY_CODE]
alloy_str_code = f'Alloy 30X-{SELECTED_ALLOY_CODE}'
CRP_str = f'Cold Rolled ({COLD_ROLL_PERCENT}% Reduction)'
anneal_str = 'Welded Zone (0% Reduction)'
CRP_ep = coldroll.equiv_strain(COLD_ROLL_PERCENT/100)
print(comp_str)
print('equiv. C content:', round(asp.equivalent_carbon_content(comp),3),'wt%')


# Temperatures [Kelvin]
Celsius = 273.15
T_room = 25 + Celsius
T_cryo = alloy_temp[SELECTED_ALLOY_CODE] + Celsius
print(f'Operating Temperature: {T_cryo-Celsius}{degreeStr}C')
print('#----------------------------------------------------------------')

# Room Temp Strengths [MPa]
sy_fullhard = 1330

# sy_mart = 2000 #1960
# sy_aust = 515
# youngs_modulus = 200000
# poisson = 0.28
# JIC_aust_max = 260
# JIC_mart_max = 500


# Contour Plot Windows

element_info = {
    
    'Ni' : {'plotMin': 5, 'plotMax': 9, 
            'plotStep': 0.1,
            'boxMin': 6, 'boxMax': 8},
    
    'Cr' : {'plotMin': 15, 'plotMax': 19,
            'plotStep': 0.1,
            'boxMin': 16, 'boxMax': 18},
    
    #-------------------------------------

    'C' : {'plotMin': 0, 'plotMax': 0.15, 
            'plotStep': 0.0025,
            'boxMin': 0, 'boxMax': 0.15},
    
    'Cu' : {'plotMin': 0, 'plotMax': 4,
            'plotStep': 0.1,
            'boxMin': 0, 'boxMax': 4},
    
    #-------------------------------------
    
    'N' : {'plotMin': 0, 'plotMax': 0.1,
            'plotStep': 0.002,
            'boxMin': 0, 'boxMax': 0.1},
    
    'Mn' : {'plotMin': 0, 'plotMax': 2,
            'plotStep': 0.1,
            'boxMin': 0, 'boxMax': 2},
    
    'Si' : {'plotMin': 0, 'plotMax': 1,
            'plotStep': 0.05,
            'boxMin': 0, 'boxMax': 1},
    
    #-------------------------------------
    
    'Mo' : {'plotMin': 0, 'plotMax': 1.5,
            'plotStep': 0.05,
            'boxMin': 0, 'boxMax': 1},
    
    'V' : {'plotMin': 0, 'plotMax': 1.5,
            'plotStep': 0.05,
            'boxMin': 0, 'boxMax': 1},
    }


def get_e_tuple(e):
    info = element_info[e]
    return (e, info['plotMin'], info['plotMax'], info['plotStep'])


def plot_ASP_contours(e1, e2, T, epsilon, roll_str,
                      specialContours = None,
                      defaultContours = (10, 'gray')):
    
    e1_tuple = get_e_tuple(e1)
    e2_tuple = get_e_tuple(e2)
    X, Y, Z = asp.get_ASP_compMesh(comp, T, epsilon, e1_tuple, e2_tuple)  
    
    bigSubtitle = f'ASP [J/mol] at T = {T - Celsius}{degreeStr}C'
    
    copy_comp = comp.copy() # creates clone, not alias
    if e1 in comp:
        starX = copy_comp.pop(e1) * 100
    else:
        starX = 0
    if e2 in comp:
        starY = copy_comp.pop(e2) * 100
    else:
        starY = 0
        
    plotTitle = '(mart5.tbd) ' + alloy_str_code + '\n' +\
        asp.comp_str(copy_comp)[:-6] + f'-x{e1}-y{e2} (wt%)' + '\n' + roll_str
        
    boxLeft, boxRight = element_info[e1]['boxMin'], element_info[e1]['boxMax']
    boxDown, boxUp = element_info[e2]['boxMin'], element_info[e2]['boxMax']
    
    asp.matrix_colorplot(Z, plotTitle, bigSubtitle,
                         X, f'wt% {e1}', boxLeft, boxRight,
                         Y, f'wt% {e2}', boxDown, boxUp,
                         specialContours = specialContours,
                         specialLabel = LABEL_ASP_THRESHOLD_CONTOURS,
                         defaultContours = defaultContours,
                         boxColor = 'green',
                         cmap = 'coolwarm',
                         star_xy_and_color = ((starX, starY), 'magenta'),
                         )


#--------------------------------------------------------------------
# ESTIMATE MARTENSITE VOLUME FRACTION
#--------------------------------------------------------------------

# Literature 301LN Data = [CR, f, YS, TS, EL]
Huang_data = [[0.0, 0.0, 292, 697, 0.54],
              [0.1, 0.034, 533, 862,0.362],
              [0.2, 0.18, 750, 1034,0.243],
              [0.3, 0.37, 1069, 1175,0.129],
              [0.4, 0.465, 1239, 1321,0.0597],
              [0.5, 0.554, 1413, 1488,0.0193],
              [0.6, 0.624, 1529, 1602,0.021],
              [0.7, 0.633, 1678, 1749,0.0177],
              [0.8, 0.666, 1900, 1925,0.016]]
Huang_data = np.array(Huang_data)
Huang_CR = Huang_data.T[0]
Huang_vol = Huang_data.T[1]
Huang_sy = Huang_data.T[2]
    
if CALC_MARTENSITE_VOL_FRAC:
    
    # Get Olson-Cohen parameters from fitting literature data
    oc_params = coldroll.get_olson_cohen_fit_params(Huang_CR, Huang_vol,
                                                    showReport = True,
                                                    fix_alpha = False,
                                                    fix_n = False)
    
    # Martensite Volume Fraction
    MVF = coldroll.get_mart_vol_frac(COLD_ROLL_PERCENT,
                oc_params['alpha'], oc_params['beta'], oc_params['n'],
                'Cold-Rolling transformation of 301/301LN based on\n' +\
                    alloy_str['Huang']) # Huang 301LN composition
        
    print('Martensite Volume Fraction:', MVF)


if OVERLAY_OLSON_COHEN_MODELS:
    fix_none = [2.968, 1.076, 2.649]
    fix_alpha = [6, 0.942, 7.441]
    fix_n = [4.315, 1.008, 4.5]
    fix_both = [6, 0.886, 4.5]
    oc_naturani = [2.7, 2.6, 4.0]
    
    plt.figure()
    plt.title('Cold-Rolling transformation of 301/301LN based on\n' +\
        alloy_str['Huang'] + '\nOlson-Cohen Model Prediction', size=16)
    plt.xlabel('Cold Rolling Reduction [%]', size=20)
    plt.ylabel('Martensite Volume Fraction [%]', size=14)
    x = np.arange(0, 0.81, 0.01)
    # ep = coldroll.equiv_strain(x)
    xp = x*100
    
    def fMart(oc_params, xStep=0.1, percentage=True):
        alpha, beta, n = oc_params
        ep = coldroll.equiv_strain(np.arange(0, 0.8001, xStep))
        scale = 100 if percentage else 1
        return scale * coldroll.olson_cohen(ep, alpha, beta, n)
    
    plt.plot(xp, fMart(fix_none, 0.01), lw=3,
             label=f'Fitted {alphaStr}=2.97, {betaStr}=1.08, n=2.65'+\
                 f'\nR^2 = {round(r2_score(fMart(fix_none), 100*Huang_vol),3)}')
        
    plt.plot(xp, fMart(fix_n, 0.01), lw=3,
             label=f'\nFixed n=4.5, Fitted {alphaStr}=4.3, {betaStr}=1.01'+\
                 f'\nR^2 = {round(r2_score(fMart(fix_n), 100*Huang_vol),3)}')
    
    plt.plot(xp, fMart(fix_alpha, 0.01), lw=3,
             label=f'\nFixed {alphaStr}=6, Fitted {betaStr}=0.94, n=7.44'+\
                 f'\nR^2 = {round(r2_score(fMart(fix_alpha), 100*Huang_vol),3)}')
    
    plt.plot(xp, fMart(fix_both, 0.01), lw=3,
             label=f'\nFixed {alphaStr}=6, n=4.5, Fitted {betaStr}=0.886'+\
                 f'\nR^2 = {round(r2_score(fMart(fix_both), 100*Huang_vol),3)}')
    
    # plt.plot(xp, fMart(oc_naturani), lw=3,
    #          label='Naturani 304+Cu alloy')
    
    plt.plot(100*Huang_CR, 100*Huang_vol, 'ko', ms=8,
             label='\nAISI 301LN data (Huang et al.)\n')
    
    plt.ylim(0,100)
    plt.axvline(x=40, c='k', ls='--', lw=2,
                label='Cold Rolling 40% (3/4 hard)')
    plt.axhline(y=50, c='tab:purple', ls='--', lw=2,
                label='50% vol. Martensite')
    plt.grid()
    plt.legend(bbox_to_anchor=(1.01,1), shadow=True)
    plt.show()
    plt.close()

MVF = 0.5

if COMPARE_NARUTANI_DeltaG_ch:
    comp_huang = alloy_comp['Huang']
    comp_naru = alloy_comp['Narutani']
    comp_LO2 = alloy_comp['LO2-v03a']
    comp_CH4 = alloy_comp['CH4-v03a']
    
    T, DG_huang = asp.get_DeltaG_ch_tempAxis(comp_huang)
    _, DG_naru = asp.get_DeltaG_ch_tempAxis(comp_naru)
    _, DG_LO2 = asp.get_DeltaG_ch_tempAxis(comp_LO2)
    _, DG_CH4 = asp.get_DeltaG_ch_tempAxis(comp_CH4)
    
    plt.figure()
    plt.title('Martenstitic Transformation Chemical Driving Force')
    plt.xlabel(f'Temperature [{degreeStr}C]')
    plt.ylabel('DeltaG_ch [J/mol]')
    plt.plot(T-Celsius, DG_huang, label='Huang 301LN alloy')
    plt.plot(T-Celsius, DG_naru, label='Narutani 304 alloy')
    plt.plot(T-Celsius, DG_LO2, label='30X-LO2-v03a alloy')
    plt.plot(T-Celsius, DG_CH4, label='30X-CH4-v03a alloy')
    plt.legend(bbox_to_anchor=(1.01,1), shadow=True)
    plt.grid()
    plt.show()
    plt.close()

#-----------------------------------------------------------------------
# ANNEALED ASP
#-----------------------------------------------------------------------

# Fully Annealed - Predict Ms_sigma temperatures
if CALC_ANNEALED_Ms_sigma:
    T, ASP_anneal = asp.get_ASP_tempAxis(comp, 0)
    plotTitle = '(mart5.tbd) ' + alloy_str_code
    plotTitle += '\n' + comp_str + '\n' + anneal_str
    Ms_sigma, ASP_thresh = asp.get_Ms_sigma(T, ASP_anneal,sy_fullhard, T_cryo,
                                            plotTitle, showPlot=True,
                                            lowest_Ms_sigma_only=False,
                                            colors = ['tab:blue', 'tab:red',
                                                      'tab:green', 'tab:purple'])
    cryo_thresh_anneal = []
    cryo_missed_anneal = []
    for state, thresh in ASP_thresh.items():
        T_index = np.argmin(abs(T-T_cryo))
        printable = f'ASP threshold at {round(T[T_index]-Celsius)}{degreeStr}C:'
        print(state, printable, thresh[T_index])
        cryo_thresh_anneal.append((state, thresh[T_index]))
        cryo_missed_anneal.append((state, ASP_anneal[T_index]-thresh[T_index]))
    cryo_thresh_anneal.sort(key = lambda x: x[1])


    # Stavehaug Toughness Peak
    if CALC_ANNEALED_TOUGHNESS:
        bigSubtitle = f'TRIP Toughening at T = {T_cryo-Celsius}{degreeStr}C'
        plotTitle = '(mart5.tbd) ' + alloy_str_code
        plotTitle += '\n' + comp_str + '\n' + anneal_str
        plotTitle += ', 100% vol. Austenite'
        asp.toughness_peak_plot(cryo_missed_anneal, 1,
                                     plotTitle, bigSubtitle)
    
    # Fully Annealed - Cryo Composition Contours
    for e1,e2 in ANNEALED_CRYO_CONTOURS:
        print('\nCOMPUTE ASP CONTOURS FOR:', e1, e2, '\n')
        plot_ASP_contours(e1, e2, T_cryo, 0, anneal_str,
                          specialContours = cryo_thresh_anneal)
        
        
    # # Fully Annealed - Room Temp Composition Contours
    # for e1,e2 in ANNEALED_ROOM_CONTOURS:
    #     plot_ASP_contours(e1, e2, T_room, 0, anneal_str,
    #                       specialContours = None)
    

    if MYLES_COMBINE:
        pass
#-----------------------------------------------------------------------
# COLDROLLED ASP
#-----------------------------------------------------------------------

# Cold-Rolled 40% - Predict Ms_sigma temperatures
if CALC_COLDROLLED_Ms_sigma:
    T, ASP_CRP = asp.get_ASP_tempAxis(comp, CRP_ep)
    plotTitle = '(mart5.tbd) ' + alloy_str_code
    plotTitle += '\n' + comp_str + '\n' + CRP_str
    Ms_sigma, ASP_thresh = asp.get_Ms_sigma(T, ASP_CRP, sy_fullhard, T_cryo,
                                            plotTitle, showPlot=True,
                                            lowest_Ms_sigma_only=False,
                                            colors = ['tab:blue', 'tab:red',
                                                      'tab:green','tab:purple'])
    cryo_thresh_CRP = []
    cryo_missed_CRP = []
    for state, thresh in ASP_thresh.items():
        T_index = np.argmin(abs(T-T_cryo))
        printable = f'ASP threshold at {round(T[T_index]-Celsius)}{degreeStr}C:'
        print(state, printable, thresh[T_index])
        cryo_thresh_CRP.append((state, thresh[T_index]))
        cryo_missed_CRP.append((state, thresh[T_index] - ASP_CRP[T_index]))
    cryo_thresh_CRP.sort(key = lambda x: x[1])
    

    # Stavehaug Toughness Peak
    if CALC_COLDROLLED_TOUGHNESS:
        bigSubtitle = f'TRIP Toughening at T = {T_cryo-Celsius}{degreeStr}C'
        plotTitle = '(mart5.tbd) ' + alloy_str_code
        plotTitle += '\n' + comp_str + '\n' + CRP_str
        plotTitle += f', {round((1-MVF)*100, -1)}% vol. Austenite'
        asp.toughness_peak_plot(cryo_missed_CRP, 1-MVF,
                                     plotTitle, bigSubtitle)
            
    # Cold-Rolled 40% - Cryo Composition Contours
    for e1,e2 in COLDROLLED_CRYO_CONTOURS:
        plot_ASP_contours(e1, e2, T_cryo, CRP_ep, CRP_str,
                          specialContours = cryo_thresh_CRP)
        

    # # Cold-Rolled 40% - Room Temp Composition Contours
    # for e1,e2 in COLDROLLED_ROOM_CONTOURS:
    #     plot_ASP_contours(e1, e2, T_room, CRP_ep, CRP_str,
    #                       specialContours = None)


#-------------------------------------------------------------------------------
# ASP curves for different CR%
#-------------------------------------------------------------------------------

if PLOT_ASP_vs_COLDROLL:
    CRP_str_i, CRP_ep_i, T_i, ASP_i, = [], [], [], []
    for CRP_i in reversed([0, 20, 40, 60]):
        CRP_str_i.append(f'{CRP_i}% Reduction')
        ep_j = coldroll.equiv_strain(CRP_i/100)
        CRP_ep_i.append(ep_j)
        T_j, ASP_j = asp.get_ASP_tempAxis(comp, ep_j)
        T_i.append(T_j)
        ASP_i.append(ASP_j)
        
    blues = reversed(['deepskyblue','dodgerblue','cornflowerblue',
                      'royalblue','blue','navy','k'])
    
    greens = ['darkgreen','forestgreen','mediumseagreen','mediumaquamarine']
        
    plotTitle = 'ASP at various Cold Rolling Reductions'
    alloy_subtitle = '(mart5.tbd) ' + alloy_str_code + '\n' + comp_str 
    asp.plot_ASPs(T_i, ASP_i, plotTitle, alloy_subtitle,
                  CRP_str_i, greens)
    
#-------------------------------------------------------------------------------
print('DONE')
