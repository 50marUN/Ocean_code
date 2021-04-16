# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 20:02:51 2020

@author: mofoko
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

Path= 'C:/Users/mofoko/Desktop/50%MAR/Antartida/'
Ruta_df_CTD = Path + 'CTD_Data.csv'

df_CTD = pd.read_csv(Ruta_df_CTD)

#Exploración de la informacion
df_CTD.info()

df_CTD.describe()

sns.pairplot(df_CTD);

fig, axes = plt.subplots(nrows=1, ncols=2)
ax0, ax1 = axes
ax0.scatter(df_CTD['Temperature(°C)'],df_CTD['Depth(m)'],c='b')
ax0.invert_yaxis()
ax1.scatter(df_CTD['Salinity(psu)'],df_CTD['Depth(m)'],c='b')
ax1.invert_yaxis()
ax0.set_title('Temperatura vs. Profundidad')
ax1.set_title('Salinidad vs. Profundidad');
ax0.set_ylabel('Profundidad (m)')
ax0.set_xlabel('Temperatura (°C)')
ax1.set_ylabel('Profundidad (m)')
ax1.set_xlabel('Salinidad (psu)')

df_CTD['Presión(bar)'] = list(map(lambda x: (x/10) + 1 , df_CTD['Depth(m)']))
df_CTD.head()

def Kt(T,s,p):
    # Función Módulo de Compresibilidad Secante
    # Calcula el polinomio usando los parámetros entregados, y los envía a la función rho(T,s,p)
    base2 = [T**(0), T**(1), T**(2), T**(3), T**(4), T**(5)]
    E = sum([19652.21, 148.4206, -2.3271, 1.3604e-2, -5.1552e-5, 0]*np.transpose(base2))
    F = sum([54.6746, -0.6034, 1.0998e-2, -6.1670e-5, 0, 0]*np.transpose(base2))
    G = sum([7.944e-2, 1.6483e-2, -5.3009e-4, 0, 0, 0]*np.transpose(base2))
    H = sum([3.2399, 1.4371e-3, 1.1609e-4, -5.7790e-7, 0, 0]*np.transpose(base2))
    I = sum([2.2838e-3, -1.0981e-5, -1.6078e-6, 0, 0, 0]*np.transpose(base2))
    J = sum([1.9107e-4, 0, 0, 0, 0, 0]*np.transpose(base2))
    M = sum([8.5093e-5, -6.1229e-6, 5.2787e-7, 0, 0, 0] *np.transpose(base2))
    N = sum([-9.9348e-7, 2.0816e-8, 9.1697e-10, 0, 0, 0] *np.transpose(base2))
    Kt = E + F*s + G*s**(1.5) + (H + I*s + J*s**(1.5))*p + (M + N*s)*p**(2)
    return Kt

# Función Densidad del océano, la cual la calcula a partir de T(°C), s(psu) y p(bar)
# La función rho(T,s,p) calcula la densidad del agua de mar
# a partir de la aproximación empírica de UNESCO del año 1981
# Utilice T(Celsius), s(psu), p(bar)
# Salida en unidades SI [kg/m^3]
def rho(T,s,p):
    base = [T**(0), T**(1), T**(2), T**(3), T**(4), T**(5)]
    A = sum([999.8425, 6.7939e-2, -9.0952e-3, 1.0016e-4, -1.12e-6, 6.53e-9] *np.transpose(base))
    B = sum([8.2449e-1, -4.0899e-3, 7.6438e-5, -8.2467e-7, 5.3875e-9, 0] *np.transpose(base))
    C = sum([-5.7246e-3, 1.0227e-4, -1.6546e-6, 0, 0, 0] *np.transpose(base))
    D = sum([4.8314e-4, 0, 0, 0, 0, 0] *np.transpose(base))
    if p == 0:
        rho = A + B*s + C*s**(1.5) + D*s**(2);
    else:
        rho = (A + B*s + C*s**(1.5) + D*s**(2))/(1-(p / Kt(T,s,p)));
    return rho

df_CTD['Compresibilidad']=list(map(lambda T,s,p:Kt(T,s,p), df_CTD['Temperature(°C)'], df_CTD['Salinity(psu)'], df_CTD['Presión(bar)']))
df_CTD['Compresibilidad']

df_CTD['Density(Kg/m3)']=list(map(lambda T,s,p:rho(T,s,p), df_CTD['Temperature(°C)'], df_CTD['Salinity(psu)'], df_CTD['Presión(bar)']))
df_CTD['Density(Kg/m3)']

#Convertimos la densidad kg/m^3 a g/cm^3
df_CTD['Density(g/cm3)'] = list(map(lambda D:D*(1000/(100**3)), df_CTD['Density(Kg/m3)']))

#Graficamos Densidad vs. Profundidad
plt.scatter(df_CTD['Density(Kg/m3)'], df_CTD['Depth(m)'], c='b')
ax = plt.gca()
ax.invert_yaxis()
ax.set_title('Densidad vs. Profundidad')
ax.set_ylabel('Profundidad (m)')
ax.set_xlabel('Density (Kg/m3)')


fig, axes = plt.subplots(nrows=1, ncols=3)
ax0, ax1, ax2 = axes
ax0.scatter(df_CTD['Temperature(°C)'],df_CTD['Depth(m)'],c='b')
ax0.invert_yaxis()
ax1.scatter(df_CTD['Salinity(psu)'],df_CTD['Depth(m)'],c='b')
ax1.invert_yaxis()
ax2.scatter(df_CTD['Density(Kg/m3)'],df_CTD['Depth(m)'],c='b')
ax2.invert_yaxis()
ax0.set_title('Temperatura vs. Profundidad')
ax1.set_title('Salinidad vs. Profundidad')
ax2.set_title('Densidad vs. Profundidad')
ax0.set_ylabel('Profundidad (m)')
ax0.set_xlabel('Temperatura (°C)')
ax1.set_ylabel('Profundidad (m)')
ax1.set_xlabel('Salinidad (psu)')
ax2.set_ylabel('Profundidad (m)')
ax2.set_xlabel('Densidad (Kg/m3)')
fig.tight_layout();


