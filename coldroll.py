# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 11:33:17 2022

@author: Samuel
"""
import numpy as np
import matplotlib.pyplot as plt
from lmfit.models import Model

DeltaStr = '\u0394'

# Equivalent Plastic Strain (Von Mises) ----------------------------------------
def equiv_strain(cr):
    return np.sqrt(4/3) * np.log(1/(1-cr))


# Martensite Phase Fraction ----------------------------------------------------
def olson_cohen(epsilon, alpha, beta, n):
    ''' calculate volume fraction of alpha prime BCC martensite '''
    termA = 1-np.exp(-alpha * epsilon)
    return 1-np.exp(-beta * termA**n)


oc_model = Model(olson_cohen)
oc_model.set_param_hint(name='alpha', value=6, min=0.01, max=10)
oc_model.set_param_hint(name='beta', value=3.82/2, min=0.01, max=10)
oc_model.set_param_hint(name='n', value=4.5, min=0.01, max=10)


def get_olson_cohen_fit_params(CR_array, vol_array, showReport=False,
                               fix_alpha = False, fix_n = False):
    '''Return dictionary of curve-fit Olson-Cohen parameters alpha, beta, n'''
    
    if fix_alpha: oc_model.set_param_hint(name='alpha', value=6, min=5.79, max=6)
    if fix_n: oc_model.set_param_hint(name='n', value=4.5, min=4.5, max=4.51)
        
    oc_fit = oc_model.fit(vol_array, epsilon = equiv_strain(CR_array))
    if showReport:
        print(oc_fit.fit_report())
        oc_fit.plot()
    return oc_fit.params.valuesdict()


#-------------------------------------------------------------------------------
def get_mart_vol_frac(CRP,
                      alpha, beta, n, # Olson-Cohen model params
                      plotTitle = None, CR_max_percent = 70
                      ):

    CR_range = np.arange(0, CR_max_percent+1) / 100
    epsilon_range = equiv_strain(CR_range)
    vol_mart_range = olson_cohen(epsilon_range, alpha, beta, n)
    MVF = vol_mart_range[np.argmin(abs(CR_range - CRP/100))]
    
    if plotTitle:
        plt.figure()
        plt.title(plotTitle)
        plt.xlabel('Cold Rolling (%)')
        plt.ylabel('Martensite Volume Percent (%)')
        plt.plot(100*CR_range, 100*vol_mart_range, label='Olson-Cohen model', lw=3)
        plt.xlim(0, CR_max_percent)
        plt.ylim(0, 100)
        plt.axvline(x=CRP, c='k', ls='--', label=f'Cold Rolling {CRP}%')
        plt.axhline(y=MVF*100, c='r', ls='--',
                    label=f'{round(MVF*100)}% martensite')
        plt.legend(loc='upper left', shadow=True)
        plt.grid()
        plt.show()
        plt.close()
    
    return MVF

# Strain-Induced Plastic Flow --------------------------------------------------
def narutani():
    pass
