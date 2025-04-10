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
    distop = dini_Gc+dstop_Gc


#------------------------------------------------------------------------------------------------------------------------------------------
# V(R) linear fits for different spacings
#------------------------------------------------------------------------------------------------------------------------------------------

VRchisq2dof = []
worstsel=[]

for k in range(len(xirun)):
    for i in range(len(betarun)): 
        print ("xi={}, beta={}".format(xirun[k],betarun[i]))
        worstsel.append(np.loadtxt('{}{}/{}_{}_{}_worstsel_ti{}_{}_tfin{}_tmin{}_cut{}_beta={}_nocorrs={}_{}.dat'.format(resultspath,xirun[k],len(variants_Gc),model_Gc,datatype_Gc,diini,dstop_Gc,dfin_Gc,dmindata_Gc,cutoff_ma,betarun[i],no_corrs,corrtype)))
        Vdat=np.loadtxt('{}{}/{}_{}_{}_VR_ti{}_{}_tfin{}_tmin{}_cut{}_beta={}_nocorrs={}_{}.dat'.format(resultspath,xirun[k],len(variants_Gc),model_Gc,datatype_Gc,diini,dstop_Gc,dfin_Gc,dmindata_Gc,cutoff_ma,betarun[i],no_corrs,corrtype))
        Vdat_rescaled=np.loadtxt('{}{}/{}_{}_{}_VR_rescaled_ti{}_{}_tfin{}_tmin{}_cut{}_beta={}_nocorrs={}_{}.dat'.format(resultspath,xirun[k],len(variants_Gc),model_Gc,datatype_Gc,diini,dstop_Gc,dfin_Gc,dmindata_Gc,cutoff_ma,betarun[i],no_corrs,corrtype))
        Vdat_rescaled_gaussian=np.loadtxt('{}{}/{}_{}_{}_VR_rescaled_gaussian_ti{}_{}_tfin{}_tmin{}_cut{}_beta={}_nocorrs={}_{}.dat'.format(resultspath,xirun[k],len(variants_Gc),model_Gc,datatype_Gc,diini,dstop_Gc,dfin_Gc,dmindata_Gc,cutoff_ma,betarun[i],no_corrs,corrtype))
        jackkl=len(Vdat)
        lt=len(Vdat[i])

        data_t              = np.linspace(1, lt, lt)
        data_y              = jackknife(Vdat,jackkl).sample()
        data_cov            = jackknife(Vdat,jackkl).scov()
        data_cov_rescaled   = jackknife(jackknife(Vdat_rescaled,jackkl).sample()).upcov()
        data_y_rescaled     = jackknife(Vdat_rescaled,jackkl).sample()

        m=Modelsmin( data_t, data_y, data_cov, dini_Vr, dstop_Vr, dmindata_Vr, dfin_Vr, inipars_Vr, model_Vr, variants_Vr, datatype_Vr, mcalls, mtol, reuse, inv_first, multiprocess, cov_freeze, improve, multistart, no_corrs, no_valid_check, data_cov_rescaled, data_y_rescaled)
        mf=m.jackk_minimize()
        AIC_list=Jackknife_AIClist(mf)
        VRchisq2dof.append(AIC_list.ordered()[0,8])
        dummy, sigma_rescaled, sigma_rescaled_gaussian=AIC_list.avgsample(cutoff_ma,3)
        np.savetxt('{}{}/{}_{}_{}_fits_VR_ti{}_{}_tfin{}_tmin{}_rmin{}_{}_rfin{}_beta={}_nocorrs={}_{}.dat'.format(resultspath,xirun[k],len(variants_Gc),model_Gc,datatype_Gc,diini,dstop_Gc,dfin_Gc,dmindata_Gc,dini_Vr,dstop_Vr,dfin_Vr,betarun[i],no_corrs,corrtype),np.append(AIC_list.ordered()[0,1:3],[AIC_list.selval()[0:2],AIC_list.avgval(cutoff_ma)[0:2]]))
        np.savetxt('{}{}/{}_{}_{}_sigma_ti{}_{}_tfin{}_tmin{}_rmin{}_{}_rfin{}_beta={}_nocorrs={}_{}.dat'.format(resultspath,xirun[k],len(variants_Gc),model_Gc,datatype_Gc,diini,dstop_Gc,dfin_Gc,dmindata_Gc,dini_Vr,dstop_Vr,dfin_Vr,betarun[i],no_corrs,corrtype),[AIC_list.avgval(cutoff_ma)[0][2],AIC_list.avgval(cutoff_ma)[1][2]])
        np.savetxt('{}{}/{}_{}_{}_sigma_ti{}_{}_tfin{}_tmin{}_rmin{}_{}_rfin{}_beta={}_jackknife_nocorrs={}_{}.dat'.format(resultspath,xirun[k],len(variants_Gc),model_Gc,datatype_Gc,diini,dstop_Gc,dfin_Gc,dmindata_Gc,dini_Vr,dstop_Vr,dfin_Vr,betarun[i],no_corrs,corrtype),jackknife(sigma_rescaled).up())
        np.savetxt('{}{}/{}_{}_{}_sigma_gaussian_ti{}_{}_tfin{}_tmin{}_rmin{}_{}_rfin{}_beta={}_jackknife_nocorrs={}_{}.dat'.format(resultspath,xirun[k],len(variants_Gc),model_Gc,datatype_Gc,diini,dstop_Gc,dfin_Gc,dmindata_Gc,dini_Vr,dstop_Vr,dfin_Vr,betarun[i],no_corrs,corrtype),jackknife(sigma_rescaled_gaussian).up())


        dataV=np.array([AIC_list.selval()[2],AIC_list.avgval(cutoff_ma)[2]])
        with open('{}{}/{}_{}_{}_corrs_VR_ti{}_{}_tfin{}_tmin{}_rmin{}_{}_rfin{}_beta={}_nocorrs={}_{}.dat'.format(resultspath,xirun[k],len(variants_Gc),model_Gc,datatype_Gc,diini,dstop_Gc,dfin_Gc,dmindata_Gc,dini_Vr,dstop_Vr,dfin_Vr,betarun[i],no_corrs,corrtype), 'w') as outfile:
            outfile.write('# Model selection/average corrs:\n')
            for data_slice in dataV:
                np.savetxt(outfile, data_slice)


for k in range(len(datarun)):
    with open('{}{}/{}_{}_{}_labels_ti{}_{}_tfin{}_tmin{}_rmin{}_{}_rfin{}_nocorrs={}_{}.dat'.format(resultspath,xirun[k],len(variants_Gc),model_Gc,datatype_Gc,diini,dstop_Gc,dfin_Gc,dmindata_Gc,dini_Vr,dstop_Vr,dfin_Vr,no_corrs,corrtype), 'w') as outfile:
        for i in range(len(datarun[0])):
            digits_beta = np.log10(betarun[i]).astype(int)
            betanorm = betarun[i]/10**(digits_beta)
            outfile.write("$\\beta$={:.2f} $\\chi^2/$dof={:.1f}, {:.1f}\n".format(betanorm,worstsel[i+k*len(datarun[0])],VRchisq2dof[i+k*len(datarun[0])]))

