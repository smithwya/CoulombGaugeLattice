#------------------------------------------------------------------------------------------------------------------------------------------
# basic setup of the notebook
#------------------------------------------------------------------------------------------------------------------------------------------
import numpy as np
import scipy as sp
import math
import pickle
from scipy.optimize import *
import numba as nb
from numba.experimental import jitclass
from numba import *
import numba_stats
import matplotlib.transforms as transforms
import multiprocess as mp
from matplotlib import pyplot as plt
plt.style.use('coulomb')

# everything in iminuit is done through the Minuit object, so we import it
from iminuit import Minuit
from iminuit.util import describe
from typing import Annotated
import iminuit


from jpac_colors import *

    

#------------------------------------------------------------------------------------------------------------------------------------------
# Fit functions
#------------------------------------------------------------------------------------------------------------------------------------------
    
# Polynomial function
def line_np_mod(t, *pars):
    t=np.asarray(t).reshape(1, -1)[0,:]
    total=np.zeros(len(t))
    for j in range(len(t)):
        for i in range(len(pars)):
            coef=pars[i]
            total[j]+= t[j]**(2*i)*coef
    return total  # for len(pars) == 2, this is a line


# Polynomial function
def line_np(t, *pars):
    total=0
    for i in range(len(pars)):
        coef=pars[i]
        #if (i == 1):
        #    coef=-np.abs(pars[i])                     # Force the slope to be always negative

        total+= t**(i)*coef
    return total  # for len(pars) == 2, this is a line


# Polynomial function
@nb.njit(fastmath=True,cache=True)
def nb_line_np(t,*pars):
    t=np.asarray(t).reshape(1, -1)[0,:]
    total=np.zeros(len(t))
    for j in range(len(t)):
        for i in range(len(pars)):
            coef=pars[i]
            #if (i == 1):
            #    coef=-np.abs(pars[i])                     # Force the slope to be always negative

            total[j]+= t[j]**(i)*coef
    return total  # for len(pars) == 2, this is a line


# Exponential of a polynomial
def exp_line_np(t,*pars):
    total=0
    for i in range(1,len(pars)):
        total+= t**(i)*pars[i]
    return np.exp(total)*abs(pars[0])


@nb.njit(fastmath=True,cache=True)
def nb_exp_line_np(t,*pars):
    t=np.asarray(t).reshape(1, -1)[0,:]
    total=np.zeros(len(t))
    for j in range(len(t)):
        for i in range(1,len(pars)):
            total[j]+= t[j]**(i)*pars[i]
    return np.exp(total)*np.abs(pars[0])  # for len(pars) == 2, this is a line

def line_np_pole(t, *pars):
    t=np.asarray(t).reshape(1, -1)[0,:]
    total=np.ones(len(t))
    for j in range(len(t)):
        for i in range(1,len(pars)):
            total[j]*= np.exp(-t[j]**(i-1)*pars[i])
        total[j]*=1+pars[0]**2/t[j]
    return total  # for len(pars) == 2, this is a line


@nb.njit(fastmath=True,cache=True)
def nb_exp_line_np_pole(t, *pars):
    t=np.asarray(t).reshape(1, -1)[0,:]
    total=np.ones(len(t))
    for j in range(len(t)):
        for i in range(1,len(pars)):
            total[j]*= np.exp(-t[j]**(i-1)*pars[i])

        total[j]*=1+pars[0]**2/t[j]
    return total  # for len(pars) == 2, this is a line


# Exponential function
def exp_np(t,*pars):
    t=np.asarray(t).reshape(1, -1)[0,:]
    total=np.zeros(len(t))
    for j in range(len(t)):
        mass=np.abs(pars[1])
        total[j]+=np.abs(pars[0])*np.exp(-mass*t[j])
        for i in range(1,np.int_((len(pars)+1)/2)):
            coef=10**(i+2)
            mass+=np.abs(pars[2*i+1])
            const=max(np.abs(pars[0])/coef,min(100*np.abs(pars[0]),np.abs(pars[2*i])))
            total[j]+=const*np.exp(-mass*t[j])
    return total

# Exponential function
@nb.njit(fastmath=True,cache=True)
def nb_exp_np(t,*pars):
    t=np.asarray(t).reshape(1, -1)[0,:]
    total=np.zeros(len(t))
    for j in range(len(t)):
        mass=np.abs(pars[1])
        total[j]+=np.abs(pars[0])*np.exp(-mass*t[j])
        for i in range(1,np.int_((len(pars)+1)/2)):
            coef=10**(i+2)
            mass+=np.abs(pars[2*i+1])
            const=max(np.abs(pars[0])/coef,min(coef*np.abs(pars[0]),np.abs(pars[2*i])))
            total[j]+=const*np.exp(-mass*t[j])
    return total



# Exponential function
@nb.njit(fastmath=True,cache=True)
def nb_exp_np_pole(t,*pars):
    t=np.asarray(t).reshape(1, -1)[0,:]
    total=np.zeros(len(t))
    for j in range(len(t)):
        mass=np.abs(pars[2])
        total[j]+=np.abs(pars[1])*np.exp(-mass*t[j])
        for i in range(1,np.int_((len(pars))/2)):
            coef=10**(i+2)
            mass+=np.abs(pars[2*i+2])
            const=max(np.abs(pars[1])/coef,min(coef*np.abs(pars[1]),np.abs(pars[2*i+1])))
            total[j]+=const*np.exp(-mass*t[j])
        total[j]*=(1+np.abs(pars[0])/t[j])
    return total


@nb.njit(fastmath=True,cache=True)
def nb_exp_line_np_pole(t,*pars):
    t=np.asarray(t).reshape(1, -1)[0,:]
    total=np.zeros(len(t))
    for j in range(len(t)):
        exponent=0
        for i in range(2,len(pars)):
            exponent+= t[j]**(i-1)*pars[i]
        total[j]=np.exp(-exponent)*np.abs(pars[1])*(1+np.abs(pars[0])/t[j])
    return total  # for len(pars) == 2, this is a line


# Geometric Exponential function (from https://arxiv.org/pdf/2208.03867.pdf)
def exp_np_geom(t,*pars):
    mass=abs(pars[1])
    total=abs(pars[0])*np.exp(-mass*t)
    den=0
    for i in range(1,np.int_((len(pars)+1)/2)):
        coef=10**(i+2)
        mass=abs(pars[2*i+1])
        const=min(coef*np.abs(pars[0]),np.abs(pars[2*i]))
        den+=const*np.exp(-mass*t)
    return total/(1-den)


# Geometric Exponential function (from https://arxiv.org/pdf/2208.03867.pdf)
@nb.njit(fastmath=True,cache=True)
def nb_exp_np_geom(t,*pars):
    t=np.asarray(t).reshape(1, -1)[0,:]
    total=np.zeros(len(t))
    den=np.zeros(len(t))
    for j in range(len(t)):
        mass=np.abs(pars[1])
        total[j]+=np.abs(pars[0])*np.exp(-mass*t[j])
        den[j]=0
        for i in range(1,np.int_((len(pars)+1)/2)):
            coef=10**(i+2)
            mass=np.abs(pars[2*i+1])
            const=min(coef*np.abs(pars[0]),np.abs(pars[2*i]))
            den[j]+=const*np.exp(-mass*t[j])
    return total/(1-den)


@nb.njit(fastmath=True,cache=True)
def nb_exp_np_geom_pole(t,*pars):
    t=np.asarray(t).reshape(1, -1)[0,:]
    total=np.zeros(len(t))
    den=np.zeros(len(t))
    for j in range(len(t)):
        mass=np.abs(pars[2])
        total[j]+=np.abs(pars[1])*np.exp(-mass*t[j])*(1+np.abs(pars[0])/t[j])
        den[j]=0
        for i in range(1,np.int_((len(pars))/2)):
            coef=10**(i+2)
            mass=np.abs(pars[2*i+2])
            const=min(coef*np.abs(pars[1]),np.abs(pars[2*i+1]))
            den[j]+=const*np.exp(-mass*t[j])
    return total/(1-den)


# Potential function
def VR_line(x, *par):
    #total=-np.pi/(12.*x)
    total=-np.abs(par[0])/x
    if (len(par)>3):
        total+=np.abs(par[3])/x**2
    for i in range(len(par)-1):
        total+= x**(i)*np.abs(par[i+1])
    return total  # for len(par) == 2, this is a line


# Potential function
@nb.njit(fastmath=True,cache=True)
def nb_VR_line(x, *par):
    x=np.asarray(x).reshape(1, -1)[0,:]
    total=np.zeros(len(x))
    for j in range(len(x)):
        total[j]=-np.abs(par[0])/x[j]
        if (len(par)>3):
            total[j]+=abs(par[3])/x[j]**2
        for i in range(len(par)-1):
            total[j]+= x[j]**(i)*np.abs(par[i+1])
    return total  # for len(par) == 2, this is a line


# Function used to extrapolate ratios to xi-->infinity
def extrapolate_xi(xi,*pars):
    xi=np.asarray(xi).reshape(1, -1)[0,:]
    total=np.zeros(len(xi))
    for j in range(len(xi)):
        for i in range(len(pars)):
            total[j]+=pars[i]*(1/xi[j])**i
    return total
