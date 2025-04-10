#!/usr/bin/env python
# coding: utf-8


#------------------------------------------------------------------------------------------------------------------------------------------
# basic setup of the notebook
#------------------------------------------------------------------------------------------------------------------------------------------
import numpy as np
import math
import timeit
import time
import os, shutil
from pip._vendor import tomli
from io import StringIO
import sys
sys.path.append('./code')
from fit_drivers import *
from minimizer import *
from fit_functions import *
from output_functions import *
from general_stats import *
from jpac_colors import *
import fit_drivers, minimizer, fit_functions, output_functions, general_stats 

# everything in iminuit is done through the Minuit object, so we import it
from iminuit import Minuit
from iminuit import minimize
from iminuit.util import describe
from typing import Annotated

# display iminuit version
import iminuit
print("iminuit version:", iminuit.__version__)

from matplotlib import pyplot as plt

#import tomllib
from pip._vendor import tomli
import sys, os


#------------------------------------------------------------------------------------------------------------------------------------------
# Importing fixed params for analysis
#------------------------------------------------------------------------------------------------------------------------------------------

with open(sys.argv[1], "rb") as f:
    params = tomli.load(f)

with open(sys.argv[2], "rb") as f2:
    params2 = tomli.load(f2)

for i in params:
     params['{}'.format(i)].update(params2['{}'.format(i)])


who            = params['creator']['who']
  
  
path           = params['paths_to_files']['base_path']
mainpath       = '{}{}'.format(path,params['paths_to_files']['mainpath'])
resultspath    = '{}{}'.format(path,params['paths_to_files']['resultspath'])
resultsdir     = '{}{}'.format(path,params['paths_to_files']['resultsdir'])
plotsdir       = '{}{}'.format(path,params['paths_to_files']['plotsdir'])
Gcplotsdir     = '{}{}'.format(path,params['paths_to_files']['Gcplotsdir'])
sizelabel      = params['paths_to_files']['sizelabel']      
  
  
corrtype       = params['correlators']['corrtype']  
xi             = params['correlators']['xi']
beta           = params['correlators']['beta']

Lextent        = params['correlators']['Lextent']
Textent        = params['correlators']['Textent']
size           = params['correlators']['size']
Ncfgs          = params['correlators']['Ncfgs']
  
  
dini_Gc        = params['minimization_parameters']['dini_Gc']
dstop_Gc       = params['minimization_parameters']['dstop_Gc']
dmindata_Gc    = params['minimization_parameters']['dmindata_Gc']
dini_Vr        = params['minimization_parameters']['dini_Vr']
dstop_Vr       = params['minimization_parameters']['dstop_Vr']
dmindata_Vr    = params['minimization_parameters']['dmindata_Vr']
dfin_Gc        = params['minimization_parameters']['dfin_Gc']
dfin_Vr        = params['minimization_parameters']['dfin_Vr']
reuse          = params['minimization_parameters']['reuse']
inv_first      = params['minimization_parameters']['inv_first']
mcalls         = params['minimization_parameters']['mcalls']
mtol           = params['minimization_parameters']['mtol']
inipars_Gc     = params['minimization_parameters']['inipars_GC']
variants_Gc    = params['minimization_parameters']['variants_GC']
jackkl         = params['minimization_parameters']['jackkl']
xiini          = params['minimization_parameters']['xiini']
xifin          = params['minimization_parameters']['xifin']   
fileini        = params['minimization_parameters']['fileini']                         
filefin        = params['minimization_parameters']['filefin']
datatype_Gc    = params['minimization_parameters']['datatype_Gc']
model_Gc       = params['minimization_parameters']['model_Gc']
model_Vr       = params['minimization_parameters']['model_Vr']
datatype_Vr    = params['minimization_parameters']['datatype_Vr']
inipars_Vr     = params['minimization_parameters']['inipars_Vr']
variants_Vr    = params['minimization_parameters']['variants_Vr']
multiprocess   = params['minimization_parameters']['multiprocess']
cov_freeze     = params['minimization_parameters']['cov_freeze']
improve        = params['minimization_parameters']['improve']
multistart     = params['minimization_parameters']['multistart']


clean          = params['extra']['clean']
cutoff_ma      = params['extra']['cutoff_ma']
norm           = params['extra']['norm']
no_corrs       = params['extra']['no_corrs']
no_valid_check = params['extra']['no_valid_check']



#------------------------------------------------------------------------------------------------------------------------------------------
#This is the only part that varies between Sebastian's and Wyatt's Lattices, the paths/readings
#------------------------------------------------------------------------------------------------------------------------------------------

Gcplotsdircompact='{}_compact'.format(Gcplotsdir)
# Create results folders if they do not exist
if os.path.exists(resultsdir)==False:
    os.mkdir(resultsdir)
if os.path.exists(plotsdir)==False:
    os.mkdir(plotsdir)
if os.path.exists(Gcplotsdir)==False:
    os.mkdir(Gcplotsdir)  
if os.path.exists(Gcplotsdircompact)==False:
    os.mkdir(Gcplotsdircompact)        


# Clean results folder?
if clean == 1:
    os.system('rm -rf {}/*'.format(plotsdir))
    os.system('rm -rf {}/*'.format(resultsdir))  
    os.system('rm -rf {}/*'.format(Gcplotsdir)) 
    os.system('rm -rf {}/*'.format(Gcplotsdircompact)) 


for k in range(len(xi)):
    if os.path.exists('{}{}'.format(resultspath,xi[k]))==False:
        os.mkdir('{}{}'.format(resultspath,xi[k]))
    if os.path.exists('{}/xi={}'.format(Gcplotsdir,xi[k]))==False:
        os.mkdir('{}/xi={}'.format(Gcplotsdir,xi[k]))
    if os.path.exists('{}/xi={}'.format(Gcplotsdircompact,xi[k]))==False:
        os.mkdir('{}/xi={}'.format(Gcplotsdircompact,xi[k]))    

        
#------------------------------------------------------------------------------------------------------------------------------------------
# Read raw data and prepare accordingly
#------------------------------------------------------------------------------------------------------------------------------------------
         
# Data used in the analysis
datam=[]
for k in range(len(xi)):
    dataint=[]
    for j in range(len(beta)):
        dataint.append(np.genfromtxt('{}{}/L{}_b{}_xi{}_{}.dat'.format(mainpath,xi[k],int(sizelabel),beta[j],xi[k],corrtype)))
    datam.append(dataint)
    
# Collect completed configurations from raw data    
data=[]
for k in range(len(xi)):
    dataint=[]
    for j in range(len(datam[k])):
        if (len(np.array(collect_configs(datam[k][j],Lextent,Textent).trans_S()))==0):
            dataint.append(np.array(collect_configs(datam[k][j],Lextent,Textent).trans_W()))
        else:
            dataint.append(np.array(collect_configs(datam[k][j],Lextent,Textent).trans_S()))
    data.append(dataint)

      

nconfigs=[]
for k in range(len(xi)):
    nint=[]
    for j in range(len(beta)):
        nint.append(int(len(data[k][j])/size[k]))
        print(int(len(data[k][j])/size[k]))
    nconfigs.append(nint)    


#------------------------------------------------------------------------------------------------------------------------------------------
# Select data files to be fitted
#------------------------------------------------------------------------------------------------------------------------------------------

datarun=[]
dataint  = data[xiini:xifin+1]
xirun    = xi[xiini:xifin+1]
for i in range(len(dataint)):
    datarun.append(dataint[i][fileini:filefin+1])

sizerun  = size[fileini:filefin+1]
betarun  = beta[fileini:filefin+1]


#------------------------------------------------------------------------------------------------------------------------------------------
#Prepare options for fitting
#------------------------------------------------------------------------------------------------------------------------------------------
if (dini_Gc==0):
    diini  = 0
    distop = 0
else:
    diini   = dini_Gc
    distop  = dini_Gc+dstop_Gc


#------------------------------------------------------------------------------------------------------------------------------------------
# Main fit to correlation functions. Based on previous params
#------------------------------------------------------------------------------------------------------------------------------------------

print('Attempting fits to L={}'.format(sizelabel))

AIC_list_final=[]
for k in range(len(datarun)):
    print('Attempting fits to xi={}'.format(xirun[k]))
    path=['{}/xi={}'.format(Gcplotsdir,xirun[k]),'{}/xi={}'.format(Gcplotsdircompact,xirun[k])]
    result=fitter(datarun[k], sizerun, dini_Gc, dstop_Gc, dmindata_Gc, dfin_Gc, datatype_Gc, model_Gc, inipars_Gc, variants_Gc, mcalls, mtol, reuse, inv_first, multiprocess, cutoff_ma, xirun[k], betarun, path, corrtype, norm, cov_freeze, improve, multistart, no_corrs, no_valid_check)
    Vrlistavg, Vrlistavg_rescaled, Vrlistavg_rescaled_gaussian, Vrlistsel, worstsel, listfinal , datayf=result.jackk_fit(jackkl)
    
    
    AIC_list_final.append(listfinal)
    #Store the V(R) data Jackknife results
    Vrf=[]
    for i in range(len(datarun[k])):
        Vrf.append(jackknife(Vrlistavg[i]).up())
        np.savetxt('{}{}/{}_{}_{}_VR_ti{}_{}_tfin{}_tmin{}_cut{}_beta={}_nocorrs={}_{}.dat'.format(resultspath,xirun[k],len(variants_Gc),model_Gc,datatype_Gc,diini,dstop_Gc,dfin_Gc,dmindata_Gc,cutoff_ma,betarun[i],no_corrs,corrtype), jackknife(Vrlistavg[i]).up())
        np.savetxt('{}{}/{}_{}_{}_VR_rescaled_ti{}_{}_tfin{}_tmin{}_cut{}_beta={}_nocorrs={}_{}.dat'.format(resultspath,xirun[k],len(variants_Gc),model_Gc,datatype_Gc,diini,dstop_Gc,dfin_Gc,dmindata_Gc,cutoff_ma,betarun[i],no_corrs,corrtype), jackknife(Vrlistavg_rescaled[i]).up())
        np.savetxt('{}{}/{}_{}_{}_VR_rescaled_gaussian_ti{}_{}_tfin{}_tmin{}_cut{}_beta={}_nocorrs={}_{}.dat'.format(resultspath,xirun[k],len(variants_Gc),model_Gc,datatype_Gc,diini,dstop_Gc,dfin_Gc,dmindata_Gc,cutoff_ma,betarun[i],no_corrs,corrtype), jackknife(Vrlistavg_rescaled_gaussian[i]).up())
        np.savetxt('{}{}/{}_{}_{}_worstsel_ti{}_{}_tfin{}_tmin{}_cut{}_beta={}_nocorrs={}_{}.dat'.format(resultspath,xirun[k],len(variants_Gc),model_Gc,datatype_Gc,diini,dstop_Gc,dfin_Gc,dmindata_Gc,cutoff_ma,betarun[i],no_corrs,corrtype), [worstsel[i]])
