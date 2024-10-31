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

from fit_functions import *
from jpac_colors import *



#------------------------------------------------------------------------------------------------------------------------------------------
# Output functions and error propagation
#------------------------------------------------------------------------------------------------------------------------------------------

# Numerical derivative for the log function, given the params
def loglimG(t,*args):
    mass=abs(args[1])
    total=abs(args[0])*mass
    norm=abs(args[0])
    for i in range(1,np.int_((len(args)+1)/2)):
        mass+=abs(args[2*i+1])
        total+=abs(args[2*i])*mass
        norm+=abs(args[2*i])
    return total/norm


def loglimG_line(t,*args):
    return -args[1]


def loglimG_geom(t,*args):
    mass=abs(args[1])
    total=0
    den=0
    for i in range(1,np.int_((len(args)+1)/2)):
        mass=abs(args[2*i+1])
        total+=abs(args[2*i])*mass
        den+=abs(args[2*i])
    return mass+total/(1-den)


# Numerical derivative for the function, given the params
def limG(t,*args):
    return -args[1]


# Numerical derivative for the function, given the params
def limGp(t,*args):
    ep=10.**5
    args=args
    tup=t+1/ep
    tdo=t-1/ep
    return -(line_np_pole(tup,*args)-line_np_pole(tdo,*args))*ep/2.
    

# Effective mass exponential function
def eff_m_exp_np(t,*pars):
    t=np.asarray(t).reshape(1, -1)[0,:]
    ep=10.**5
    total  = []
    totalp = []
    for j in range(len(t)):
        total.append(nb_exp_np(t[j],*pars)[0])
        totalp.append(nb_exp_np(t[j]+1,*pars)[0])

    total=np.array(total)
    totalp=np.array(totalp)
    return np.log(total/totalp)


# Effective mass exponential function
def eff_m_exp_np_pole(t,*pars):
    t=np.asarray(t).reshape(1, -1)[0,:]
    ep=10.**5
    total  = []
    totalp = []
    for j in range(len(t)):
        total.append(nb_exp_np_pole(t[j],*pars)[0])
        totalp.append(nb_exp_np_pole(t[j]+1,*pars)[0])

    total=np.array(total)
    totalp=np.array(totalp)
    return np.log(total/totalp)



# Linearized, correlated error propagation, given the params and errors
def prop_err(t,fun,args,errargs,corrs):
    ep=10**5
    errj=np.array([])
    t=np.asarray(t).reshape(1, -1)[0,:]
    for j in range(len(t)):
        err=np.array([])
        for i in range(len(args)):
            up=args.copy()
            up[i]=args[i].copy()+errargs[i].copy()/ep
            do=args.copy()
            do[i]=args[i].copy()-errargs[i].copy()/ep
            err=np.append(err,(eval(fun)(t[j],*up)-eval(fun)(t[j],*do))*ep/2)
        errj=np.append(errj,math.sqrt(np.dot(err,np.dot(corrs,err))))
    return errj

