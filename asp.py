# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 13:25:40 2022

@author: Samuel
"""

import numpy as np
import matplotlib.pyplot as plt

Celsius = 273.15
DeltaStr = '\u0394'
sigmaStr = '\u03C3'
degreeStr = '\u00B0'
epsilonStr = '\u03B5'

# Composition ------------------------------------------------------------------
def make_comp(alloy_elements, percent_dict):
    ''' Return a dictionary mapping element symbols to mass fraction '''
    return {e:w/100 for e,w in zip(alloy_elements, percent_dict) if w > 0}


def comp_str(comp):
    ''' Return alloy composition in wt% string format '''
    out = 'Fe'
    for elem, mass in comp.items(): # zip(alloy_elements, percent_dict):
        out += f'-{round(mass*100,3)}{elem}' if mass > 0 else ''
    out += ' (wt%)'
    return out


def equivalent_carbon_content(comp):
    t6 = comp.get('Mn',0) + comp.get('Si',0)
    t5 = comp.get('Cr',0) + comp.get('Mo',0) + comp.get('V',0)
    t15= comp.get('Cu',0) + comp.get('Ni',0)
    return comp.get('C',0) + t6/6 + t5/5 + t15/15


# Chemical Driving Force (TC-python) -------------------------------------------
from my_tc_python import get_DeltaG_ch_compMesh
'''
INPUTS:
    - Composition [mass fraction]
    - Temperature [K]
    - Composition tuple 1
    - Composition tuple 2
        - element_symbol [str], min [wt%], max [wt%], step [wt%]
        
OUTPUTS:
    - Composition range 1 [mass fraction],
    - Composition range 2 [mass fraction] (larger than input range),
    - DeltaG_ch matrix [J/mol] (rows:e1, cols:e2)
'''


from my_tc_python import get_DeltaG_ch_tempAxis
'''
INPUTS:
    - Composition [mass fraction]
    - Optional:
        - T_min [K], T_max [K], T_step [K]
        
OUTPUT:
    - Temperature axis (1D-array) [K]
    - DeltaG_ch (1D-array) [J/mol]
'''


# Mechanical Driving Force -----------------------------------------------------
def DeltaG_sigma(sigma, hydrostatic_coeff, DeltaV_V=0.04):
    term1 = 0.7183 * sigma
    term2 = 6.85 * DeltaV_V * hydrostatic_coeff * sigma
    term3 = -185.3*(1-np.exp(-0.003043*sigma))
    return -(term1 + term2 + term3)


hydrostatic_coeffs = {f'{sigmaStr}_h = 2.60{sigmaStr}' : 2.6,
                      f'{sigmaStr}_h = 1.00{sigmaStr}' : 1.0,
                      f'{sigmaStr}_h = 0.67{sigmaStr}' : 2/3}

ASP_color_order = ['tab:blue', 'tab:red', 'tab:green', 'tab:purple']
ASP_star_label = lambda state: f'ASP* ({state})'

# Nucleation Potency Constant --------------------------------------------------
G_n = 837.5 # Wengronovich


# Slip Yield Stress ------------------------------------------------------------
def sigma_y(T, sigma_y_298):
    ''' sigma_y_298 = Yield Strength of full-hard 301 at T_room '''
    return -1.425*(T-298) + sigma_y_298


# Ghosh Model ------------------------------------------------------------------
elements = ['C','N','Mn','Si','Cr','V','Ti','Mo','Cu','Al','Ni']
ki_values = [3807,3084,1980,1879,1868,1618,1473,1418,752,280,172]
ki0_values = [21216,16986,4107,3867,3923,3330,3031,2918,1548,576,345]
ki = {e:k for e,k in zip(elements, ki_values)}
ki0 = {e:k for e,k in zip(elements, ki0_values)}

elements_i = ['C','N']
elements_j = ['Mn','Si','Cr','V','Ti','Mo']
elements_k = ['Cu','Al','Ni']

def W_mu(comp): # W_f athermal component
    sum_i = sum(comp.get(e,0) * ki[e]**2 for e in elements_i)
    sum_j = sum(comp.get(e,0) * ki[e]**2 for e in elements_j)
    sum_k = sum(comp.get(e,0) * ki[e]**2 for e in elements_k)    
    return np.sqrt(sum_i) + np.sqrt(sum_j) + np.sqrt(sum_k)
    
def W_th(comp, T, p=1/2, q=3/2): #W_f thermal component
    sum_i = sum(comp.get(e,0) * ki0[e]**2 for e in elements_i)
    sum_j = sum(comp.get(e,0) * ki0[e]**2 for e in elements_j)
    sum_k = sum(comp.get(e,0) * ki0[e]**2 for e in elements_k)
    W0 = 836 + np.sqrt(sum_i) + np.sqrt(sum_j) + np.sqrt(sum_k)
    tempTerm = (1 - (T/510) ** (1/q)) ** (1/p)
    return W0 * tempTerm

def W_rho(epsilon): # W_f forest hardening component
    return 710.4 * np.sqrt(epsilon)

def W_f(comp, T, epsilon, p=1/2, q=3/2):
    return W_mu(comp) + W_th(comp, T, p, q) + W_rho(epsilon)


def get_W_f_compMesh(comp, T, epsilon,
                     e1, e1_range, e2, e2_range, p=1/2, q=3/2):
    
    W_f_matrix = np.zeros((e1_range.size, e2_range.size))
    for row, e1_mass in enumerate(e1_range):
        for col, e2_mass in enumerate(e2_range):
            iter_comp = comp.copy() # creates clone, not alias
            iter_comp[e1] = e1_mass
            iter_comp[e2] = e2_mass
            W_f_matrix[row][col] = W_f(iter_comp, T, epsilon, p, q)
    return e1_range, e2_range, W_f_matrix


def get_W_f_tempAxis(comp, T_axis, epsilon, p=1/2, q=3/2):
    return np.array([W_f(comp, temp, epsilon, p, q) for temp in T_axis])


#-------------------------------------------------------------------------------
# Austenite Stability Parameter
#-------------------------------------------------------------------------------

def get_ASP_compMesh(comp, T, epsilon, e1_tuple, e2_tuple, p=1/2, q=3/2):
    ''' Returns:
            - array of first element range [mass fraction], length m
            - array of second element range [mass fraction], length n
            - matrix of ASP values [J/mol], shape (m,n)
    '''
    
    g = get_DeltaG_ch_compMesh
    e1_range, e2_range, DeltaG_ch_matrix = g(comp, T, e1_tuple, e2_tuple)
    
    #-------------------------------------------------------------------
    '''For some reason, ThermoCalc sometimes lets composition axes 
        go way past the specified maximum, e.g. I'll set it to range
        from 15% to 19%, and TC-Python will calculate from 15% to 46%
        
        This seems to be a ThermoCalc since performing the same calcs
        in the graphical interface should also behave similarly
        
        I also tried self-specifying both compositional axes using a
        Batch Equilibrium Calculation, but the answers disagreed with
        my TempAxis calculation results
        
        So for now, we'll just have to manually correct the range here'''
    
    if True:
        if len(e2_range) > 1.5*len(e1_range):
            print('\nTC-PYTHON EXTRANEOUS COMP RANGE')
            print(e2_tuple[0], 'RANGE:', min(e2_range*100), max(e2_range),
                  '\nwith matrix shape:', DeltaG_ch_matrix.shape)
            print('\nReducing to fit within range...')
            
            e2_min, e2_max = e2_tuple[1]/100, e2_tuple[2]/100
            keep_e2_range, keep_cols = [], []
            for col ,e2_mass in enumerate(e2_range):
                if (e2_min < e2_mass < e2_max):
                    keep_cols.append(col)
                    keep_e2_range.append(e2_mass)
            
            print('Keep columns:', keep_cols)
            e2_range = np.array(keep_e2_range)
            DeltaG_ch_matrix = DeltaG_ch_matrix[:, keep_cols]
    #-------------------------------------------------------------------
    
    print('\nCalculating W_f...')
    e1, e2 = e1_tuple[0], e2_tuple[0]
    _, _, W_f_matrix = get_W_f_compMesh(comp, T, epsilon,
                                        e1, e1_range, e2, e2_range, p, q)
    
    out = DeltaG_ch_matrix + W_f_matrix
    print('ASP matrix shape:', out.shape)
    return e1_range, e2_range, out


def get_ASP_tempAxis(comp, epsilon,
                     T_min=-250+Celsius, T_max=50+Celsius, p=1/2, q=3/2):
    ''' Returns:
            - Array of Temperature axis [K]
            - Array of ASP values [J/mol]
    '''
    T_axis, DeltaG_ch = get_DeltaG_ch_tempAxis(comp, T_min, T_max)
    W_f_array = get_W_f_tempAxis(comp, T_axis, epsilon, p, q)    
    return T_axis, DeltaG_ch + W_f_array
    

# Plot Ms_sigma vs. Temperature ------------------------------------------------

def get_x_at_intersect(x, y1, y2, yTol = 0.001):
    ''' Given np.arrays x, y1(x), y2(x),
        return the list of the "distinct intercepts",
        i.e. the x values where |y2(x)-y1(x)| < yTol
                            and |y2(x)-y1(x)| is a local min
    '''
    dy = abs(y2-y1)
    x_index = np.where(abs(y2-y1) < yTol) # returns a tuple?
    
    if isinstance(x_index, tuple):
        if len(x_index) == 1:
            x_index = x_index[0]
        else:
            print('x_index = np.where(abs(y2-y1) < yTol) =')
            print(x_index)
            raise Exception
    
    out = []
    for i in x_index:
        if i == 0 or i == len(x)-1: # endpoints count
            out.append(x[i])
        elif dy[i] < dy[i-1] and dy[i] < dy[i+1]:
            out.append(x[i])
    return out


def plot_ASPs(T_axes, ASP_arrays,
              plotMainTitle, alloyInfoTitle,
              ASP_labels, ASP_colors
              ):
    ''' Inputs:
            - T_axes: list of T_axis arrays
            - ASP_arrays: list of ASP arrays, (ordered highest to lowest)
            - ASP_labels: list of labels for ASP curves
            - ASP_colors: list of colors for ASP curves
    '''
    plt.figure()
    plt.suptitle(alloyInfoTitle, y=1.08, size=14)
    plt.title(plotMainTitle, size=20)
    plt.xlabel(f'T [{degreeStr}C]', size=20)
    plt.ylabel('ASP [J/mol]', size=20)
    
    for T,ASP,lab,c in zip(T_axes, ASP_arrays, ASP_labels, ASP_colors):
        plt.plot(T-Celsius, ASP, c=c, lw=5, label=lab)
        
    plt.legend(bbox_to_anchor=(1.01,1), shadow=True)
    plt.grid()
    # Assume ASP_arrays are ordred highest to lowest to match legend order
    plt.ylim(min(0, min(ASP_arrays[-1])-100), None)
    plt.show()
    plt.close()


def get_Ms_sigma(T_axis, ASP_array, sigma_y_298, T_oper, alloyInfoTitle,
                 hydrostatic_coeffs = hydrostatic_coeffs, DeltaV_V = 0.04,
                 lowest_Ms_sigma_only = False, showPlot = True,
                 colors = ASP_color_order,
                 ):
    ''' Return
            - a dictionary mapping stress state names (strings)
                to their predicted Ms_sigma temperature(s) (list)
                
            - a dictionary mapping states to ASP_threshold curves
    '''
    
    sy = sigma_y(T_axis, sigma_y_298)
    
    thresh_curve = {state: -(G_n + DeltaG_sigma(sy, hydro, DeltaV_V)) \
                    for state, hydro in hydrostatic_coeffs.items()}
    
    Ms_sigma = {state: get_x_at_intersect(T_axis, ASP_array, curve, 10) \
                for state, curve in thresh_curve.items()}
    
    if showPlot:
        plt.figure()
        plt.suptitle(alloyInfoTitle, y=1.13, size=14)
        plt.title(f'Ms_{sigmaStr} Temperature Prediction', size=20)
        plt.xlabel(f'T [{degreeStr}C]', size=20)
        plt.ylabel('ASP [J/mol]', size=20)
        plt.plot(T_axis-Celsius, ASP_array, c='k', lw=5, label='ASP')
        plt.axvline(x=T_oper-Celsius, c='k', lw=0.5,
                    label=f'Operating Temp: {round(T_oper-Celsius)}{degreeStr}C')
        
        vert_label = lambda state, Ms_sig:\
            f'Ms_{sigmaStr} ({state}) = {round(Ms_sig - Celsius)}{degreeStr}C'
        
        T_list = list(T_axis)
        
        for state, color in zip(hydrostatic_coeffs.keys(), colors):
            plt.plot(T_axis-Celsius, thresh_curve[state],
                     lw = 2, c=color, label='\n'+ASP_star_label(state))
            
            for Ms_sig in Ms_sigma[state]:
                T_index = T_list.index(Ms_sig)
                plt.vlines([Ms_sig-Celsius], -500, ASP_array[T_index],
                           ls='--', color = color,
                           label = vert_label(state, Ms_sig))
                
                plt.plot([Ms_sig-Celsius], [ASP_array[T_index]], 'o', 
                         ms=10, c=color)
                
                if lowest_Ms_sigma_only:
                    break
        
        plt.legend(bbox_to_anchor=(1.01,1), shadow=True)
        plt.grid()
        plt.ylim(min(100, min(ASP_array)-100), None)
        plt.show()
        plt.close()
        
    return Ms_sigma, thresh_curve


# Plot ASP mesh ----------------------------------------------------------------
def matrix_colorplot(e1_e2_matrix, alloyInfoTitle, plotMainTitle,
        e1_range, e1_label, boxLeft, boxRight,
        e2_range, e2_label, boxDown, boxUp,
        cmap='Spectral_r', showPercent=True,
        specialContours = None, # list of tuples: (countourName, contourValue)
        specialLabel = True, # label contours by name
        defaultContours = (9, 'gray'),
        boxColor = None,
        star_xy_and_color = None, # (starX, starY), color
        ):
    
    X = e1_range * 100 if showPercent else e1_range
    Y = e2_range * 100 if showPercent else e2_range
    Z = e1_e2_matrix.T # transpose because python plots rows -> y, cols -> x
    
    plt.figure(figsize=(8,6))
    plt.suptitle(alloyInfoTitle, y=1.05, size=14)
    plt.title(plotMainTitle, size=20)
    plt.xlabel(e1_label, size=20)
    plt.ylabel(e2_label, size=20)
    
    PCM = plt.pcolormesh(X, Y, Z, cmap=cmap)
    if specialContours:
        CL1 = plt.contour(X, Y, Z, [val for (state,val) in specialContours],
                          colors = 'k', linewidths=5)
        if specialLabel:
            fmt = {line:'  '+ASP_star_label(state)+'  ' for line,(state, val) \
                     in zip(CL1.levels, specialContours)}
            plt.clabel(CL1, CL1.levels, inline=True, fontsize=14, fmt=fmt)
                
    if defaultContours:
            CL2 = plt.contour(X, Y, Z, defaultContours[0],
                              colors = defaultContours[1], linewidths=1)
            plt.clabel(CL2, CL2.levels, inline=True, fontsize=14)
        
    if boxColor:
        plt.vlines([boxLeft,boxRight],boxDown,boxUp,color=boxColor,ls='--',lw=3)
        plt.hlines([boxDown,boxUp],boxLeft,boxRight,color=boxColor,ls='--',lw=3)
        
    if star_xy_and_color:
        (starX, starY), starColor = star_xy_and_color
        plt.plot([starX], [starY], '*', color=starColor, ms=40)
        
    
    plt.colorbar(PCM)
    plt.show()
    plt.close()
    

# Leal-Stavehaug TRIP toughening peak-------------------------------------------

def toughness_peak_piecewise(asp_missed, vol_aust):
    peak = 0.5*vol_aust
    if asp_missed <= 0: return peak + (peak/500)*asp_missed
    if 0 < asp_missed < 1000: return peak - (peak/1000)*asp_missed
    if 1000 < asp_missed: return 0
    
    
def toughness_peak_plot(ASP_Deltas, vol_aust,
                        alloyInfoTitle, plotMainTitle=None,
                        colors = ASP_color_order):
    plt.figure()
    plt.suptitle(alloyInfoTitle, y=1.13, size=14)
    plt.title(plotMainTitle, size=20)
    plt.xlabel('ASP - ASP* [J/mol]', fontsize=20)
    plt.ylabel(f'{DeltaStr}J / J_0', fontsize=20)
    x = np.linspace(-1000, 2001, 500)
    y = [toughness_peak_piecewise(i, vol_aust) for i in x]
    yTickVals = np.arange(-0.5, 0.51, 0.1)
    yTickLabs = [f'+{round(v*100)}%' if v>=0 \
                 else f'{round(v*100)}%' for v in yTickVals]
    plt.yticks(yTickVals, yTickLabs)
    plt.ylim(-0.57,0.57)
    plt.plot(x, y, c='k', lw=4)
    plt.axhline(y=0, ls='--', c='k')
    for (state, dasp), color in zip(ASP_Deltas, colors):
        plt.plot([dasp], [toughness_peak_piecewise(dasp, vol_aust)],
                 'o', ms=15, label=state, c=color)
    plt.grid()
    plt.show()
    plt.close()
        
