# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 16:30:00 2022

@author: Samuel Song (ssong64@mit.edu)
"""

import numpy as np
from tc_python import *

'''
https://thermocalc.com/content/uploads/Documentation/
Current_Static/tc-python-api-programmer-guide.pdf
'''

#--------------------------------------------------------------------------
TQ_temp = ThermodynamicQuantity.temperature
TQ_mass_frac = ThermodynamicQuantity.mass_fraction_of_a_component
TQ_user_func = ThermodynamicQuantity.user_defined_function

Celsius = 273.15

#--------------------------------------------------------------------------
def get_DeltaG_ch_compMesh(COMP, T_kelvin, e1_tuple, e2_tuple):
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
    
    e1, e1_min, e1_max, e1_step = e1_tuple
    e2, e2_min, e2_max, e2_step = e2_tuple                             
    
    DeltaG_ch_MATRIX = []
    
    e1_range = np.arange(e1_min, e1_max+0.0001, e1_step) / 100
    
    FIXED_ELEMENTS = list(COMP.keys())
    ALL_ELEMENTS = ['Fe'] + FIXED_ELEMENTS
    
    if e1 in FIXED_ELEMENTS:
        FIXED_ELEMENTS.remove(e1)
    else:
        ALL_ELEMENTS.append(e1)
        
    if e2 in FIXED_ELEMENTS:
        FIXED_ELEMENTS.remove(e2)
    else:
        ALL_ELEMENTS.append(e2)
    
    with TCPython() as start:
        calc = (
            start
                .set_ges_version(5)
                .select_user_database_and_elements("MART5.TDB", ALL_ELEMENTS)
                .without_default_phases()
                .select_phase("FCC_A1")
                .select_phase("BCC_A2")
                .get_system()
                
                .with_property_diagram_calculation()
                .set_condition(TQ_temp(), T_kelvin)
                .disable_global_minimization()
                .enable_step_separate_phases()
            )
                
        for elem in FIXED_ELEMENTS:
            calc.set_condition(TQ_mass_frac(elem), COMP[elem])      
            
        for e1_mass in e1_range:
            calc.set_condition(TQ_mass_frac(e1), e1_mass)
            calc.with_axis(
                CalculationAxis(TQ_mass_frac(e2))
                    .set_min(e2_min/100)
                    .set_max(e2_max/100)
                    .with_axis_type(Linear().set_max_step_size(e2_step/100))
                )
        
            result = calc.calculate()
            dog,e2_range = result.get_values_of(TQ_user_func('gm(bcc)-gm(fcc)'),
                                                 TQ_mass_frac(e2))
            DeltaG_ch_MATRIX.append(dog)
    
    return e1_range, np.array(e2_range), np.array(DeltaG_ch_MATRIX)


#-------------------------------------------------------------------------
def get_DeltaG_ch_tempAxis(COMP, 
                           T_min=-250+Celsius, T_max=50+Celsius, T_step=1):
    '''
    INPUTS:
        - Composition [mass fraction]
        - Optional:
            - T_min [K], T_max [K], T_step [K]
            
    OUTPUT:
        - Temperature axis (1D-array) [K]
        - DeltaG_ch (1D-array) [J/mol]
    '''
    
    FIXED_ELEMENTS = list(COMP.keys())
    ALL_ELEMENTS = ['Fe'] + FIXED_ELEMENTS
    
    with TCPython() as start:
        calc = (
            start
                .set_ges_version(5)
                .select_user_database_and_elements("MART5.TDB", ALL_ELEMENTS)
                .without_default_phases()
                .select_phase("FCC_A1")
                .select_phase("BCC_A2")
                .get_system()
                
                .with_property_diagram_calculation()
                .with_axis(CalculationAxis(TQ_temp())
                      .set_min(T_min)
                      .set_max(T_max)
                      .with_axis_type(Linear().set_max_step_size(T_step)))
                .disable_global_minimization()
                .enable_step_separate_phases()
            )
                
        for elem in FIXED_ELEMENTS:
            calc.set_condition(TQ_mass_frac(elem), COMP[elem])      
    
        result = calc.calculate()
        dog, T_axis = result.get_values_of(TQ_user_func('gm(bcc)-gm(fcc)'),
                                           TQ_temp())
    
    return np.array(T_axis), np.array(dog)

