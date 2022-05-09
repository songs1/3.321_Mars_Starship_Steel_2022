# -*- coding: utf-8 -*-
"""
Created on Wed May  4 16:36:55 2022

@author: Samuel
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import asp

Celsius = 273.15
degreeStr = '\u00B0'

#-------------------------------------------------------------------------------
# Naming convention for both files and variables: rowElement_colElement
# When plotting, let X = rowArray, Y = colArray, Z = transpose(matrix)
#-------------------------------------------------------------------------------
# USER_SELECTION
tank, version = 'LO2', 'v03a'
e1, e2 = 'Ni', 'Cr'
# e1, e2 = 'C', 'Cu'
#-------------------------------------------------------------------------------

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
    
ASP_star_labels = ['  ASP* (σ_h = 0.67σ)  ', '  ASP* (σ_h = 1.00σ)  ',
                    '  ASP* (σ_h = 2.60σ)  ']
ASP_star = {}
ASP_star['LO2'] = [474.3372835330399, 625.9858943663733, 1353.899226366374]
ASP_star['CH4'] = [439.8197309753032, 587.9542918086363, 1299.0001838086368]
ASP_diff_labels = ['90% TRIP', 'MAX TRIP', '90% TRIP']

alloy = f'{tank}-{version}'
T_op = alloy_temp[alloy] + Celsius
#-------------------------------------------------------------------------------
# a, b, c = asp.get_ASP_compMesh(alloy_comp[alloy], T_op+Celsius, epsilon=0,
#                                 e1_tuple = ('Ni', 5, 9, 0.1),
#                                 e2_tuple = ('Cr', 15, 19, 0.1)) 

# np.savetxt('30X-v03a_plot_data/ASP_LO2_Ni_Cr.txt', c)
#-------------------------------------------------------------------------------

small_range = {}
big_range = {}

small_range['Ni'] = np.linspace(6, 8, 5) / 100
big_range['Ni'] = np.arange(5, 9.01, 0.1) / 100

small_range['Cr'] = np.linspace(16,18,5) / 100
big_range['Cr'] =\
    np.array([0.15001, 0.15055158, 0.15155158, 0.15255158, 0.15355158,
           0.15455158, 0.15555158, 0.15655158, 0.15755158, 0.15855158,
           0.15955158, 0.16055158, 0.16155158, 0.16255158, 0.16355158,
           0.16455158, 0.16555158, 0.16655158, 0.16755158, 0.16855158,
           0.16955158, 0.17055158, 0.17155158, 0.17255158, 0.17355158,
           0.17455158, 0.17555158, 0.17655158, 0.17755158, 0.17855158,
           0.17955158, 0.18055158, 0.18155158, 0.18255158, 0.18355158,
           0.18455158, 0.18555158, 0.18655158, 0.18755158, 0.18855158,
           0.18955158, 0.18999])

small_range['C'] = np.linspace(0.01, 0.05, 5) / 100
big_range['C'] = np.arange(0.01, 0.050001, 0.001) / 100

small_range['Cu'] = np.linspace(1.8, 2.5, 5) / 100
big_range['Cu'] =\
    np.array([0.00575416, 0.00575416, 0.00625416, 0.00675416, 0.00725416,
              0.00775416, 0.00825416, 0.00875416, 0.00925416, 0.00975416,
              0.01025416, 0.01075416, 0.01125416, 0.01175416, 0.01225416,
              0.01275416, 0.01325416, 0.01375416, 0.01425416, 0.01475416,
              0.01525416, 0.01575416, 0.01625416, 0.01675416, 0.01725416,
              0.01775416, 0.018005  , 0.01825416, 0.01875416, 0.01925416,
              0.01975416, 0.02025416, 0.02075416, 0.02125416, 0.02175416,
              0.02225416, 0.02275416, 0.02325416, 0.02375416, 0.02425416,
              0.02475416, 0.024995])

xMin = small_range[e1][0]
xMax = small_range[e1][-1]
yMin = small_range[e2][0]
yMax = small_range[e2][-1]
bigFont = 20
medFont = 16

def read_matrix(filehandle, sep=' ', folder='30X-v03a_plot_data/', exten='.txt'):
    return np.array(pd.read_csv(folder+filehandle+exten,
                                header=None, delimiter=sep))

plt.figure(figsize=(8,6))
plt.suptitle(f'Alloy 30X-{alloy}', y=1.02, size=bigFont, weight='bold')
plotTitle = alloy_str[alloy]
plotTitle += f'\nat T = {round(T_op - Celsius)}{degreeStr}C'
plt.title(plotTitle, size=medFont)
handles = []

plt.xlabel(f'wt% {e1}', size=bigFont)
plt.xlim(xMin, xMax)
ticks = np.linspace(xMin, xMax, 9) #21) #9)
plt.xticks(ticks, [round(100*t,2) if i%2==0 else '' \
                   for i,t in enumerate(ticks)],size=medFont)

plt.ylabel(f'wt% {e2}', size=bigFont)
plt.ylim(yMin, yMax)
ticks = np.linspace(yMin, yMax, 8) #21) #8
plt.yticks(ticks,[round(t*100,2) if i%1==0 else '' \
                  for i,t in enumerate(ticks)], size=medFont)

strength = read_matrix(f'strength_{e2}-{e1}').T
strength_contours = [-10, 0, 10, 100, 150, 200, 225]
CL = plt.contour(small_range[e1], small_range[e2], strength.T,
                  strength_contours, colors='tab:red',
                  linewidths=[6 if s==150 else 2 for s in strength_contours])
L = Line2D([0], [0], color='tab:red',
            label='Strenthening (Δσ_y) [MPa]')
handles.append(L)
plt.clabel(CL, CL.levels, inline=True, fontsize=medFont)


freezeRange = read_matrix(f'fr_{tank}_{e2}-{e1}').T
CL = plt.contour(small_range[e1], small_range[e2], freezeRange.T,
                  [88.1, 94, 100], colors='tab:blue', linewidths=[6,2,2])
L = Line2D([0], [0], color='tab:blue',
            label=f'Freezing Range [{degreeStr}C]')
handles.append(L)
plt.clabel(CL, CL.levels, inline=True, fontsize=medFont)

ASP = read_matrix(f'ASP_{tank}_{e1}_{e2}')
CL = plt.contour(big_range[e1],
                 big_range[e2],
                 ASP.T - ASP_star[tank][0],
                 #a, b, c.T,
                 [-50, 0, 100], colors='tab:green', linewidths=[2,6,2])
L = Line2D([0], [0], color='tab:green',
            label='ASP - ASP* [J/mol]')
handles.append(L)
fmt = {level:label for level,label in zip(CL.levels, ASP_diff_labels)}
plt.clabel(CL, CL.levels, inline=True, fontsize=medFont, fmt=fmt,
           manual=[(6.5/100, 17.6/100), (6.6/100,17.7/100), (7.3/100,17.5/100)]
           #manual=[(0.02/100, 1.9/100), (0.02/100, 2.17/100), (0.04/100, 2.43/100)]
           #manual=[(7.3/100,17.4/100), (7.4/100,17.4/100), (7.5/100, 17.5/100)]
           #manual=[(0.02/100, 1.9/100), (0.03/100, 2/100), (0.035/100, 2.4/100)]
           )

# plt.plot([8/100], [16.6/100], '*', color='magenta', ms=40)
# shade = np.where(freezeRange < 88.1, 1, 0)
# plt.contourf(Ni_range_5, Cr_range_5, shade.T,
#              [-1, 0, 1], colors=['white','gold'])

plt.show()
plt.close()

plt.figure()
plt.legend(handles=handles, shadow=True, loc='center',
            bbox_to_anchor= (0.5, 0.5), fontsize=medFont)
plt.show()
plt.close()

