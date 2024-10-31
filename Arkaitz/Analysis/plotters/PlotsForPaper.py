#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 18:28:26 2024

@author: cesar
Bibliography: 
    Garcia Perez, van Baal, Phys. Lett. B 392 (1997) 163--171
    Burgio, Quandt, and Reinhart, Phys. Rev. D 86 (2012)  045029

"""

###########################################################
#   Python libraries
###########################################################

import numpy as np
import scipy.stats as stat
#import matplotlib
#matplotlib.rcParams['text.usetex'] = True

import matplotlib.pyplot as plt
plt.style.use('coulomb')
import pandas as pd
from scipy.optimize import fsolve
from scipy.optimize import root
import sys
sys.path.append('../code')
from jpac_colors import *
from iminuit import Minuit
from iminuit.cost import LeastSquares

#plt.style.use("/Users/cesar/GitHub/Lattice/SU2/coulomb.mplstyle")
#plt.style.use('coulomb')

###########################################################
#   JPAC color style
###########################################################

jpac_blue   = "#1F77B4"; jpac_red    = "#D61D28";
jpac_green  = "#2CA02C"; jpac_orange = "#FF7F0E";
jpac_purple = "#9467BD"; jpac_brown  = "#8C564B";
jpac_pink   = "#E377C2"; jpac_gold   = "#BCBD22";
jpac_aqua   = "#17BECF"; jpac_grey   = "#7F7F7F";

jpac_color = [jpac_blue, jpac_red, jpac_green, 
              jpac_orange, jpac_purple, jpac_brown,
              jpac_pink, jpac_gold, jpac_aqua, jpac_grey ];

dashes = 60*'_'; label_size = 12

###########################################################
#   Fitting eta1 using Minuit
###########################################################

eta1data = np.array([
    [1.25, 0.0268994230, 0.9448516628, 0.072639575, 0.941156329],
    [1.50, 0.0438407406, 0.9030990984, 0.120815052, 0.895220389],
    [1.75, 0.0553875971, 0.8715577166, 0.154865904, 0.862024175],
    [2.00, 0.0637309147, 0.8473816328, 0.180064348, 0.838519956],
    [2.25, 0.0700334250, 0.8285038242, 0.199377634, 0.821947171],
    [3.00, 0.0821772181, 0.7913271531, 0.236952649, 0.796282892],
    [4.00, 0.0909349958, 0.7647074013, 0.263881756, 0.786799000],
    [5.00, 0.0960771084, 0.7495355534, 0.279376900, 0.786585170],
    [6.00, 0.0994693690, 0.7398121769, 0.289388945, 0.789176074],
    [7.00, 0.1018786317, 0.7330678148, 0.296372263, 0.792513704],
    [8.00, 0.1036794982, 0.7281198326, 0.301513492, 0.795887850],
    [9.00, 0.1050771985, 0.7243357505, 0.305453471, 0.799053830],
    [10.00, 0.1061937831, 0.7213480410, 0.308567683, 0.801940305],
    [20.00, 0.1112064198, 0.7083043501, 0.322153959, 0.819059857]
]);
                    
def eta1(c1,c2,c3,xiR): # eta1 function
    return c1 + c2/xiR + c3/(xiR**3);

def funtofiteta1_minuit(xiR,c1,c2,c3): # eta1 function for Minuit
    return  eta1(c1,c2,c3,xiR)

def LSQeta1_minuit(c1,c2,c3):
    return np.sum((yeta1- funtofiteta1_minuit(xeta1, c1, c2, c3)) ** 2)

xeta1, yeta1 = eta1data[:,0], eta1data[:,3];
etafit = Minuit(LSQeta1_minuit,          
           c1=0.344, c2=-0.377, c3 = 0.065);
etafit.errordef = Minuit.LEAST_SQUARES
etafit.migrad(); etafit.hesse();
print(etafit.params); print(etafit.covariance); print(etafit.covariance.correlation())
print(dashes);
print('chi2=',etafit.fval);
print(dashes); print(dashes)
etafit_c1, etafit_c2, etafit_c3 = etafit.values
print('eta1 fit to PLB392(1997)163'); print('Fit parameter=',etafit.values)
print(dashes); print(dashes)

fig = plt.figure(figsize=(5,5))
plt.xlim((0,21))
plt.ylim((0,0.35))
plt.scatter(xeta1,yeta1, marker="s", s=10, c=jpac_color[0], alpha=1)
etafit_x = np.linspace(1.,20,50)
plt.plot(etafit_x,eta1(etafit_c1,etafit_c2,etafit_c3,etafit_x),'-', c=jpac_color[3], alpha=1)
plt.xlabel(r'$\xi_R$',size=15)
plt.ylabel(r'$\eta_1(\xi_R)$',size=15)

#plt.show()
fig.savefig('../../final_plots/eta1.pdf', bbox_inches='tight')

###########################################################
#
#   Fitting xi0/gamma data using Minuit
#
###########################################################

def eta(a1,a2,c1,c2,c3,beta,xiR): # eta function
    return 1.+4./6.*(  (1.+a1/beta)/(1.+a2/beta)  )*eta1(c1,c2,c3,xiR)/beta;

def xi0fun(a1,a2,c1,c2,c3,beta,xiR): # xi0/gamma calculation
           return xiR/eta(a1,a2,c1,c2,c3,beta,xiR)

vxi0fun = np.vectorize(xi0fun); # vectorization of previous function

#   Fitting function to 1D the bare anisotropy xi0/gamma
#   beta: SU(2) coupling
#   xiR: True anisotropy
#   xi0/gamma: bare anisotropy
#   a1, a2: parameters
#   c1, c2, c3: parameters (fitted and used from pervious fit)

def funtofitxi0_1Dminuit(beta,a1,a2,c1,c2,c3):
    return  xiR_fixed/eta(a1,a2,c1,c2,c3,beta,xiR_fixed)

def LSQxi0_1Dminuit(a1,a2,c1,c2,c3): # function for MINUIT
    return np.sum(( (yxi0- funtofitxi0_1Dminuit(xbeta,a1,a2,c1,c2,c3))**2)/(sigma_yxi0**2))

#   Fitting function to 2D the bare anisotropy xi0/gamma
#   beta: SU(2) coupling
#   xiR: True anisotropy
#   xi0/gamma: bare anisotropy
#   a1, a2: parameters
#   c1, c2, c3: parameters (fitted and used from pervious fit)

def funtofitxi0_2Dminuit(i,a1,a2,c1,c2,c3):
    beta = xdata[i]; xiR = ydata[i];
    return  xiR/eta(a1,a2,c1,c2,c3,beta,xiR)

def LSQxi0_2Dminuit(a1,a2,c1,c2,c3): # function for MINUIT
    residue = [ (( y[i] - funtofitxi0_2Dminuit(i, a1,a2,c1,c2,c3) )/sigma_y[i] )**2
               for i in x]
    return np.sum(residue)

# Uploading data 
betatableI_input = np.array( [ 2.15, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7 ] );
xitableI_input  = np.array( [2 , 3, 4 ] );
tableI_input = np.array([
    [ 1.654, 1.672, 1.712, 1.754, 1.796, 1.835, 1.870 ],
    [ 2.375, 2.407, 2.474, 2.545, 2.608, 2.663, 2.710 ],
    [ 3.106, 3.151, 3.243, 3.331, 3.406, 3.466, 3.511 ]
    ]);
tableI_error_input = np.multiply( np.array([ 
    [ 3, 3, 3, 4, 5, 6, 9 ],
    [ 3, 3, 4, 4, 5, 7, 9 ],
    [ 4, 4, 5, 5, 6, 7, 9 ]
    ]),0.001);


##### Playing with data

index = [0,1]; 
betatableI = np.delete(betatableI_input,index)
index = []; xitableI = np.delete(xitableI_input,index)

tableI , tableI_error = tableI_input,  tableI_error_input;

# removing columns 1 and 2

tableI_v1 = np.delete(tableI_input,0,1)
tableI_error_v1 = np.delete(tableI_error_input,0,1)

tableI = np.delete(tableI_v1,0,1)
tableI_error = np.delete(tableI_error_v1,0,1)

##### Finished playing with data

dif_fixed, ratio_fixed = [], []
print(dashes); print(dashes)
print('Fit one xiR at  a time')
print(dashes)
fig = plt.figure(figsize=(5,5))
betafit_x = np.linspace(2.,3.1,50)
xilabel = [r'$\xi=2$',r'$\xi=3$',r'$\xi=4$']
a1_xi0fixed = np.zeros(len(xitableI))
a2_xi0fixed = np.zeros(len(xitableI))
for i in range(len(xitableI)):
    xiR_fixed = xitableI[i];
    xbeta, yxi0, sigma_yxi0 = betatableI, tableI[i,:], tableI_error[i,:];
    m_xi0fixed = Minuit(LSQxi0_1Dminuit, 
           a1=-0.640, a2=-1.738,
           c1=etafit_c1, c2=etafit_c2, c3=etafit_c3 )
    m_xi0fixed.errordef = Minuit.LEAST_SQUARES
    m_xi0fixed.fixed["c1"] = True
    m_xi0fixed.fixed["c2"] = True
    m_xi0fixed.fixed["c3"] = True
    m_xi0fixed.migrad(); m_xi0fixed.hesse();
    a1_xi0fixed[i], a2_xi0fixed[i], c1, c2, c3 = m_xi0fixed.values
    print('xiR',xiR_fixed,'a1=',m_xi0fixed.values[0],'a2=',m_xi0fixed.values[1])
    print('chi2=',m_xi0fixed.fval); 
    print('chi2/dof=',m_xi0fixed.fval/(len(yxi0)-2))
    print(dashes)
    plt.errorbar(xbeta, yxi0, yerr=sigma_yxi0, fmt="o", markersize=3,capsize=5., c=jpac_color[i], alpha=1,label=xilabel[i],zorder=3)
    n = i

plt.ylim((1.,4.)); plt.xlim((2.2,2.8))
#plt.title(r'Fits to BQR data')
plt.xlabel(r'$\beta$',size=15)
plt.ylabel(r'$\xi_0$',size=15)

print('Fit all xiR simultaneously')

print(dashes); print(dashes)
print('eta1 fixed')
print(dashes)

#   Preparing data for the 2D fit
tableI_1d = tableI.flatten()
tableI_1d_error = tableI_error.flatten();
xdata = []; ydata = [];
for i in range(len(xitableI)): 
    xdata = np.concatenate((xdata,betatableI),casting="same_kind")
    for j in range(len(betatableI)): 
        ydata = np.hstack((ydata,xitableI[i]))

x = np.arange(tableI.size); y = tableI_1d; sigma_y = tableI_1d_error;

m_xi0= Minuit(LSQxi0_2Dminuit, 
       a1=-0.640, a2=-1.738,
       c1=etafit_c1, c2=etafit_c2, c3=etafit_c3 )
m_xi0.errordef = Minuit.LEAST_SQUARES
m_xi0.fixed["c1"] = True
m_xi0.fixed["c2"] = True
m_xi0.fixed["c3"] = True
m_xi0.migrad(); m_xi0.hesse();
a1_xi0, a2_xi0, c1, c2, c3 = m_xi0.values
print(dashes);
print(m_xi0.params); print(m_xi0.covariance); print(m_xi0.covariance.correlation())
print(dashes);
print('Fixed cs are')
print('c1=',etafit_c1,'c2=',etafit_c2,'c3=',etafit_c3)
print(dashes)
print('a1=',m_xi0.values[0],'a2=',m_xi0.values[1])
print('chi2=',m_xi0.fval); 
print('chi2/dof=',m_xi0.fval/(len(ydata)-2))
print(dashes)

dif_cfixed, ratio_cfixed = [], []
for i in range(len(xitableI)):
    xiR_fixed = xitableI[i];
    if i==0:
        plt.plot(betafit_x,xi0fun(a1_xi0, a2_xi0, c1, c2, c3,betafit_x,xiR_fixed),'-',lw=1,c=jpac_color[3], alpha=1,zorder=1,label='Fit')
    else:
        plt.plot(betafit_x,xi0fun(a1_xi0, a2_xi0, c1, c2, c3,betafit_x,xiR_fixed),'-',lw=1,c=jpac_color[3], alpha=1,zorder=1)

    xbeta, yxi0 = betatableI, tableI[i,:];
    dif = yxi0 - xi0fun(a1_xi0, a2_xi0, c1, c2, c3,xbeta,xiR_fixed)     
    dif_cfixed.append(dif)
    ratio = np.divide(dif,yxi0)
    ratio_cfixed.append(ratio)
        

plt.legend(loc='upper left',ncol=2,frameon=True,fontsize=10)#,bbox_to_anchor=(0.95,0.02))

#plt.show()
fig.savefig('../../final_plots/xi0.pdf', bbox_inches='tight')


###########################################################
#
#   Fitting as lattice spacing data using Minuit
#
###########################################################

#   2D Fitting function to the lattice spacing
#   beta: SU(2) coupling
#   xiR: True anisotropy
#   as: lattice spacing
#   b0, b1, b2, b3, b4: parameters

def asfun(b0,b1,b2,b3,b4,beta,xiR):
    escala = (4.*np.pi**2)/beta0;
    logaritmo = 2.*beta1*np.log(escala*beta)/(beta0**2);
    exponente = b4 + b3*escala/beta - escala*beta + logaritmo;
    return eta1(b0,b1,b2,xiR)*np.sqrt(np.exp(exponente)/sigma);

def funtofitas_2Dminuit(i,b0,b1,b2,b3,b4):
    beta = xdata[i]; xiR = ydata[i];
    escala = (4.*np.pi**2)/beta0;
    logaritmo = 2.*beta1*np.log(escala*beta)/(beta0**2);
    exponente = b4 + b3*escala/beta - escala*beta + logaritmo;
    return eta1(b0,b1,b2,xiR)*np.sqrt(np.exp(exponente)/sigma);

def LSQas_2Dminuit(b0,b1,b2,b3,b4):
    residue = [ (( y[i] - funtofitas_2Dminuit(i, b0,b1,b2,b3,b4) )/sigma_y[i] )**2
               for i in x]
    return np.sum(residue)

# SU(2) and string tension
beta0 = 22./3.;  beta1 = 68./3.; sigma = 0.440**2;

# Uploading data 

xitableII_input = np.array( [ 2 , 3, 4 ] );
betatableII_input = np.array( [ 2.15, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7 ] );
tableII_input = np.array([
    [ 1.355, 1.206, 0.940, 0.711, 0.511, 0.338, 0.268 ],
    [ 1.391, 1.239, 0.968, 0.732, 0.526, 0.345, 0.270 ],
    [ 1.406, 1.254, 0.979, 0.739, 0.530, 0.346, 0.271 ]
    ]);
tableII_error_input = np.multiply( np.array([ 
    [ 7, 6, 6, 6, 5, 5, 5 ],
    [ 7, 7, 6, 6, 5, 5, 5 ],
    [ 8, 7, 6, 6, 5, 5, 5 ]
    ]),0.001);


##### Playing with data

index = []; 
betatableII = np.delete(betatableII_input,index)
index = []; xitableII = np.delete(xitableII_input,index)

tableII , tableII_error = tableII_input,  tableII_error_input;

xilabel = [r'$\xi=2$',r'$\xi=3$',r'$\xi=4$']

# Sebastian's

b0 =  7.057569782719919
b1 =  1.613897922881925
c1 =  0.33982850818945587
c2 = -0.047729010997329825
c3 =  0.0

fig = plt.figure(figsize=(8,5))
for i in range(len(xitableII)):
    xiR_fixed = xitableII[i];
    xasbeta, yas, sigma_yas = betatableII, tableII[i,:], tableII_error[i,:];
    plt.plot(betafit_x,asfun(c1, c2, c3, b1, b0, betafit_x,xiR_fixed),'-',lw=1,c=jpac_color_around[i], alpha=1,zorder=2)
    plt.errorbar(xasbeta, yas, fmt="o",c=jpac_color_around[i], alpha=1,label=xilabel[i],markerfacecolor='white',markeredgewidth=1.5)

#plt.yscale('log')
plt.xlim((2.1,2.8)); 
plt.ylim((0,2.)); 
plt.xlabel(r'$\beta$',fontsize=28)
plt.ylabel(r'$a_s$ \huge{{[GeV$^{-1}$]}}',fontsize=28)
plt.legend(loc='upper right',ncol=1,frameon=True,fontsize=22,fancybox=True,shadow=False,framealpha=1)
#plt.show()
fig.savefig('../../final_plots/as_fitallfixedeta1.pdf', bbox_inches='tight')
