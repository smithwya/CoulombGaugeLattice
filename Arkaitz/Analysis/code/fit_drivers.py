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
from output_functions import *
from minimizer import *
from general_stats import *
from jpac_colors import *




#------------------------------------------------------------------------------------------------------------------------------------------
# Create class for MA minimization, given data, main model, and variants of main model
#------------------------------------------------------------------------------------------------------------------------------------------
    

def statoffit(data_tt,data_yt,data_incovt,calln,tol,inip,model,jackk,datatype,inipars,ndata_full):
            m=Minuit_fit(data_tt, data_yt, data_incovt, calln, tol, inip, model,jackk,0)
            if (m.minimize().valid==False):
                ntries=500
                for k in range(ntries):
                    s = np.random.normal(0, 0.00001, len(inip))
                    inipars_list=(1+s)*inip
                    m2=Minuit_fit(data_tt, data_yt, data_incovt, calln, tol, inipars_list, model,jackk,0)
                    if (m2.minimize().fval < m.minimize().fval):
                        m=m2
                    if (m.minimize().valid):
                        break

            matest=m.minimize().fval/m.minimize().ndof
            ma=MA_fit(m.minimize())
            maavg=ma.AICp(ndata_full)
            return stats_of_fits(datatype,m.minimize(),ma,model,len(inipars[-1])).jackk_stats(), maavg, matest, np.array(m.minimize().values)


class stats_of_fits:

    def __init__(self, datatype, minimize, ma_fit, model, maxpars):
        self.datatype=datatype
        self.minimize=minimize
        self.ma_fit=ma_fit
        self.maxpars=maxpars
        self.model=model
        if ((self.model=='line_np' or self.model=='nb_line_np') and self.datatype=="dlog"):
            self.modelval=self.model
        elif (self.model=='line_np' or self.model=='nb_line_np'):
            self.modelval='limG'
        elif (self.model=='line_np_pole' or self.model=='nb_line_np_pole'):
            self.modelval='limGp'
        elif (self.model=='exp_np' or self.model=='nb_exp_np' or self.model=='exp_np_pole' or self.model=='nb_exp_np_pole'):
            self.modelval='loglimG'
        elif (self.model=='exp_np_geom' or self.model=='nb_exp_np_geom'):
            self.modelval='loglimG_geom'
        elif (self.model=='exp_np_line' or self.model=='nb_exp_np_line'):
            self.modelval='loglimG_line'
        
        self.fpars=self.ma_fit.pars()
        if (self.model=='exp_np_pole' or self.model=='nb_exp_np_pole'):
            self.fpars=self.ma_fit.pars()[1:]


    def stats(self):
        if (self.datatype=="dlog" or self.datatype=="log" or self.datatype=="log_pole" or self.datatype=="exp" or self.datatype=="exp_line"):
            modelval=self.modelval
            output=[self.minimize.fval/self.minimize.ndof,eval(modelval)(0,*self.fpars),prop_err(0.,modelval,self.fpars,self.ma_fit.errs(),self.ma_fit.corrs())[0]]   
        elif (self.datatype=="exp_WL"):
            output=[self.minimize.fval/self.minimize.ndof,abs(self.fpars[1]),self.ma_fit.errs()[1]]            
        elif (self.datatype=="Vr"):
            output=np.zeros(self.maxpars+1)
            output[0]=self.minimize.fval/self.minimize.ndof
            for i in range(len(self.minimize.values)):
                output[i+1]=abs(self.minimize.values[i])
            output=list(output)
        else:
            output=np.zeros(self.maxpars+1)
            output[0]=self.minimize.fval/self.minimize.ndof
            for i in range(len(self.minimize.values)):
                output[i+1]=self.minimize.values[i]
            output=list(output)
        return output


    def jackk_stats(self):
        # Check what data type are we fitting, in order to use the proper functions/parameters for observables
        if (self.datatype=="dlog" or self.datatype=="log" or self.datatype=="log_pole" or self.datatype=="exp" or self.datatype=="exp_line"):
            modelval=self.modelval
            output=np.zeros(self.maxpars+2)
            output[0]=self.minimize.fval/self.minimize.ndof
            output[1]=eval(modelval)(0,*self.fpars) 
            for i in range(len(self.minimize.values)):
                output[i+2]=np.abs(self.minimize.values[i])
            output=list(output) 
        elif (self.datatype=="exp_WL"):
            output=np.zeros(self.maxpars+2)
            output[0]=self.minimize.fval/self.minimize.ndof
            output[1]=np.abs(self.fpars[1])
            for i in range(len(self.minimize.values)):
                output[i+2]=np.abs(self.minimize.values[i])
            output=list(output)           
        elif (self.datatype=="Vr"):
            output=np.zeros(self.maxpars+1)
            output[0]=self.minimize.fval/self.minimize.ndof
            for i in range(len(self.minimize.values)):
                output[i+1]=np.abs(self.minimize.values[i])
            output=list(output)
        else:
            output=np.zeros(self.maxpars+1)
            output[0]=self.minimize.fval/self.minimize.ndof
            for i in range(len(self.minimize.values)):
                output[i+1]=self.minimize.values[i]
            output=list(output)
        return output



class Modelsmin:     # It is designed to work with nested models ONLY, run it various times for non-nested cases

    def __init__(self, data_t, data_y, data_cov, dini, dstop, dmindata, dfin, inipars, model, variants, datatype, calln, tol, reuse, inv_first, multiprocess, freeze, improve, nmultistart, no_corrs, no_valid_check, data_cov_rescaled=None, data_y_rescaled=None):
        self.data_t=data_t
        self.data_y=data_y
        self.data_cov=data_cov
        self.dini=dini
        self.dstop=dstop
        self.dmindata=dmindata
        self.dfin=dfin
        self.inipars=inipars
        self.model=model
        self.variants=variants
        self.datatype=datatype
        self.calln=calln
        self.tol=tol
        self.reuse=reuse
        self.inv_first=inv_first
        self.multiprocess=multiprocess
        self.freeze=freeze
        self.improve=improve
        self.data_cov_rescaled=data_cov_rescaled
        self.data_y_rescaled=data_y_rescaled
        self.nmultistart=nmultistart
        self.no_corrs=no_corrs
        self.no_valid_check=no_valid_check


    def minimize(self):
        listfits=[]
        trials=0
        success=0
        lt=len(self.data_t)
        ndata_full=len(self.data_t)
        dfin=min(self.dfin,lt)
        if (self.dmindata==0):
            mindata=len(self.inipars[0])+1       #Assumes first entry of inipars is the model with less params
        elif (self.dmindata=='half'):
            mindata=np.int_(lt/2)
        else:
            mindata=self.dmindata

        if (self.dini>0):
            rang=range(self.dini,min(dfin-mindata+1+1,self.dini+self.dstop+1))
        else:
            rang=([1])
        for i in rang:
            for j in range(mindata,dfin-i+1+1):
                data_tt=self.data_t[i-1:i+j-1]
                data_yt=self.data_y[i-1:i+j-1]

                if (self.no_corrs!=0):
                    self.data_cov=np.diag(np.diag(self.data_cov))

                if (self.inv_first==1):
                    data_incovt=improved_inverse_covariance(self.data_cov)[i-1:i+j-1,i-1:i+j-1]
                else:
                    data_incovt=improved_inverse_covariance(self.data_cov[i-1:i+j-1,i-1:i+j-1])

                finp=np.array([])      # Empty list of params to be reused in following model calls
                for k in range(len(self.variants)):      # Loop over nested model variants
                    if (self.reuse == 1):
                        inip=np.append(finp,np.array(self.inipars[k][len(finp):]))
                    else:
                        inip=self.inipars[k]
                    if (len(inip)<len(data_tt)):
                        m=multistart_reuse_pars(data_tt, data_yt, data_incovt, self.calln, self.tol, inip, inip, inip, self.model, self.improve,self.nmultistart,self.multiprocess).multistart_select()
                        #Minuit_fit(data_tt, data_yt, data_incovt, self.calln, self.tol, inip, self.model,0,self.improve)
                        ma=MA_fit(m.minimize())
                        finp=np.array(m.minimize().values)   # Now reuse the best params for next model fit
                        trials+=1
                        if (m.minimize().valid==True or self.no_valid_check==1):
                            success+=1
                            listfits.append([self.variants[k],i,i+j-1,ma.AICp(ndata_full),np.array(stats_of_fits(self.datatype,m.minimize(),ma,self.model,len(self.inipars[-1])).stats()),m.minimize()])  
        listfits=np.array(listfits, dtype=object)
        return [[trials,success],listfits]
    
    

    def jackk_minimize(self):
        listfits=[]
        lt=len(self.data_t)
        ndata_full=len(self.data_t)
        if (self.dmindata==0):
            mindata=len(self.inipars[0])+1       #Assumes first entry of inipars is the model with less params
        elif (self.dmindata=='half'):
            mindata=np.int_(lt/2)
        else:
            mindata=self.dmindata        
        dfin=min(self.dfin,lt)
        if (self.dini>0):
            rang=range(self.dini,min(dfin-mindata+1+1,self.dini+self.dstop+1))
        else:
            rang=([1])

        for i in rang:
            inp_previous_range=[]
            for j in reversed(range(mindata,dfin-i+1+1)):

                data_tt=self.data_t[i-1:i+j-1]
                if (self.data_y_rescaled is not None):
                    data_yt=self.data_y_rescaled[:,i-1:i+j-1]
                else:
                    data_yt=self.data_y[:,i-1:i+j-1]

                if (self.freeze==1):                                                      # Freeze the covariance matrix, aka use the full sample all the times for the Jackknife minimizations
                    data_covf=jackknife(self.data_y).upcov()
                    if (self.data_cov_rescaled is not None):
                        for kk in range(len(data_covf)):
                            data_covf[kk,kk]=self.data_cov_rescaled[kk,kk]

                    if (self.no_corrs!=0):
                        data_covf=np.diag(np.diag(data_covf))

                    if (self.inv_first==1):
                        data_incovt=np.zeros([len(self.data_cov),j,j])
                        for k in range(len(self.data_cov)):
                            data_incovt[k]=improved_inverse_covariance(data_covf)[i-1:i+j-1,i-1:i+j-1]
                    else:
                        data_incovt=np.zeros([len(self.data_cov),j,j])
                        data_covt=data_covf[i-1:i+j-1,i-1:i+j-1]
                        for k in range(len(self.data_cov)):
                            data_incovt[k]=improved_inverse_covariance(data_covt)
                else:
                    if (self.no_corrs!=0):
                        self.data_cov=np.diag(np.diag(self.data_cov))

                    if (self.inv_first==1):
                        data_incovt=np.zeros([len(self.data_cov),j,j])
                        for k in range(len(self.data_cov)):
                            data_incovt[k]=improved_inverse_covariance(self.data_cov[k])[i-1:i+j-1,i-1:i+j-1]
                    else:
                        data_covt=self.data_cov[:,i-1:i+j-1,i-1:i+j-1]
                        data_incovt=improved_inverse_covariance(data_covt)

                finp=np.array([])                                                       # Empty list of params to be reused in following model calls
                inp_previous_range_int=[]
                for k in range(len(self.variants)):                                     # Loop over nested model variants
                    statsfit  =[] 
                    maavg=0
                    if (len(self.inipars[k])<len(data_tt)):
                        if (self.reuse == 1):
                            if (j==dfin-i+1):
                                m=multistart_reuse_pars(data_tt, np.mean(data_yt,axis=0), np.mean(data_incovt,axis=0), self.calln, self.tol, self.inipars[k], np.append(finp,np.array(self.inipars[k][len(finp):])), self.inipars[k], self.model, self.improve,self.nmultistart,self.multiprocess).multistart_select()
                                #inipp=np.append(finp,np.array(self.inipars[k][len(finp):]))

                            else:
                                m=multistart_reuse_pars(data_tt, np.mean(data_yt,axis=0), np.mean(data_incovt,axis=0), self.calln, self.tol, self.inipars[k], np.append(finp,np.array(self.inipars[k][len(finp):])), inp_previous_range[k], self.model, self.improve,0,self.multiprocess).select()
                                #inipp=inp_previous_range[k]

                            chi2_orig=(m.minimize().fval)/(m.minimize().ndof)
                            inipp=finp=np.array(m.minimize().values)
                            inp_previous_range_int.append(np.array(m.minimize().values))
                            inip=np.append(finp,np.array(self.inipars[k][len(finp):]))
                            ma_orig=MA_fit(m.minimize()).AICp(ndata_full)
                        else:
                            inipp=inip=self.inipars[k]
                            m=Minuit_fit(data_tt, np.mean(data_yt,axis=0), np.mean(data_incovt,axis=0), self.calln, self.tol, inip, self.model,1,self.improve)
                            chi2_orig=(m.minimize().fval)/(m.minimize().ndof)
                            ma_orig=MA_fit(m.minimize()).AICp(ndata_full)
                        
                        flagbreak=False
                        flagrepeat=False
                        chi2best=chi2_orig
                        chi2worst=chi2_orig
                        location = 0
                        if (m.minimize().valid==True or self.no_valid_check==1):
                            args=[(data_tt,data_yt[jackk],data_incovt[jackk],self.calln,self.tol,inip,self.model,1,self.datatype,self.inipars,ndata_full) for jackk in range(len(data_yt))]
                            with mp.Pool(processes=min(mp.cpu_count(),self.multiprocess)) as pool:   
                                count = 0                                       
                                for fstats,ma,chi2,jackkpars in pool.starmap(statoffit, args):
                                    statsfit.append(fstats)
                                    maavg+=ma/len(data_yt)
                                    if ((chi2-chi2_orig)>max(1,chi2_orig)*25./np.sqrt(len(data_yt)) and chi2>chi2worst):              # Discard Jackknifes that are deviated over "5 sigmas"
                                        flagbreak = True
                                        chi2worst = chi2
                                        location  = count
                                    elif ((chi2-chi2_orig)<-max(1,chi2_orig)*9./np.sqrt(len(data_yt)) and chi2<chi2best):
                                        flagrepeat=True
                                        inip=jackkpars
                                        chi2best=chi2
                                    count+=1

                            if (flagbreak==True):
                                print("Warning: Jackkknife fit went bananas",chi2_orig,chi2worst,location)

                            if (flagrepeat==True):
                                print("Warning: Jackkknife found way better fit, repeating",chi2_orig,chi2best)


                            if (flagrepeat==True):
                                flagbreak=False
                                statsfit=[] 
                                maavg=0
                                m=Minuit_fit(data_tt, np.mean(data_yt,axis=0), np.mean(data_incovt,axis=0), self.calln, self.tol, inip, self.model,1,self.improve)
                                chi2_orig_v2=(m.minimize().fval)/(m.minimize().ndof)
                                chi2_orig=chi2_orig_v2
                                ma_orig=MA_fit(m.minimize()).AICp(ndata_full)
                                if ((chi2_orig_v2-chi2_orig)<-max(1,chi2_orig)*1./np.sqrt(len(data_yt))):
                                    print("Good: repetition found better fit",chi2_orig,chi2_orig_v2)
                                args=[(data_tt,data_yt[jackk],data_incovt[jackk],self.calln,self.tol,inip,self.model,1,self.datatype,self.inipars,ndata_full) for jackk in range(len(data_yt))]
                                with mp.Pool(processes=min(mp.cpu_count(),self.multiprocess)) as pool:
                                    count = 0
                                    for fstats,ma,chi2,jackkpars in pool.starmap(statoffit, args):
                                        statsfit.append(fstats)
                                        maavg+=ma/len(data_yt)
                                        if (np.abs(chi2-chi2_orig_v2)>max(1,chi2_orig_v2)*25./np.sqrt(len(data_yt))):              # Discard Jackknifes that are deviated over "5 sigmas"
                                            flagbreak=True
                                            print("Warning: Jackkknife fit DEFINITELY went bananas",chi2_orig_v2,chi2,count)
                                            break
                                        count+=1
                                    if (flagbreak==True):
                                        break

                            if (flagbreak==False):
                                listfits.append([self.variants[k],i,i+j-1,ma_orig,np.array(statsfit),len(data_yt),np.array(m.minimize().values),inipp,chi2_orig])  
                inp_previous_range=inp_previous_range_int
        listfits=np.array(listfits, dtype=object)
        return listfits
    


class multistart_reuse_pars:                   # This algorithm tests 3 different inputs looking for a best fit: 1- The vanila input string from the ini file, 2- The same string but with reused params from previous model fit, 3- Reusing params from same model, but fitted to 1 extra point in previous iteration (tfin+1). If multistart==1 then it will ran a parallelized multistart procedure to look for best fits more globally 
    def __init__(self, data_t, data_y, data_incov, calln, tol, params, reused_params_model, reused_params_data,model,improve,multistart,multiprocess):
        self.data_t=data_t
        self.data_y=data_y
        self.data_incov=data_incov
        self.params=params
        self.reused_params_model=reused_params_model
        self.reused_params_data=reused_params_data
        self.model=model 
        self.calln=calln
        self.tol=tol
        self.improve=improve   
        self.multistart=multistart
        self.multiprocess=multiprocess

    def select(self):
        m_ini=Minuit_fit(self.data_t, self.data_y, self.data_incov, self.calln, self.tol, self.params, self.model,1,self.improve)
        m_reused_model=Minuit_fit(self.data_t, self.data_y, self.data_incov, self.calln, self.tol, self.reused_params_model, self.model,1,self.improve)
        m_reused_data=Minuit_fit(self.data_t, self.data_y, self.data_incov, self.calln, self.tol, self.reused_params_data, self.model,1,self.improve)

        test_vector=[m_ini,m_reused_model,m_reused_data]
        test_values_vector=[m_ini.minimize().fval,m_reused_model.minimize().fval,m_reused_data.minimize().fval]

        index_vector=test_values_vector.index(min(test_values_vector))

        return test_vector[index_vector]
    


    def multistart_select(self):

        einipars=self.params.copy()
        einipars_reused=self.reused_params_model.copy()

        inipars_list=[]
        inipars_reused_list=[]

        for j in range(len(einipars)):
            einipars[j]=max(0.01,einipars[j])
            einipars_reused[j]=max(0.01,einipars_reused[j])

        for k in range(self.multistart):
            s = np.random.normal(0, 2, len(self.params))
            inipars_list.append(self.params+s*einipars)
            inipars_reused_list.append(self.reused_params_model+s*einipars_reused)

        chi_list=[]
        chi_reused_list=[]

        args=[(self.data_t, self.data_y, self.data_incov, self.calln, self.tol, inipars_list[sample], self.model,0,self.improve) for sample in range(self.multistart)]
        args_reused=[(self.data_t, self.data_y, self.data_incov, self.calln, self.tol, inipars_reused_list[sample], self.model,0,self.improve) for sample in range(self.multistart)]
        with mp.Pool(processes=min(mp.cpu_count(),self.multiprocess)) as pool:                                          
            for chis in pool.starmap(fit_int, args):
                chi_list.append(chis)
            for chis in pool.starmap(fit_int, args_reused):
                chi_reused_list.append(chis)
                

        index=chi_list.index(min(chi_list))
        index_reused=chi_reused_list.index(min(chi_reused_list))

        if (chi_list[index]<=chi_reused_list[index_reused]):
            final_best_fit=Minuit_fit(self.data_t, self.data_y, self.data_incov, self.calln, self.tol, inipars_list[index], self.model,1,self.improve)
        else:
            final_best_fit=Minuit_fit(self.data_t, self.data_y, self.data_incov, self.calln, self.tol, inipars_reused_list[index_reused], self.model,1,self.improve)

        return final_best_fit



def fit_int(data_tt,data_yt,data_incovt,calln,tol,inip,model,jackk,improve):
            m=Minuit_fit(data_tt, data_yt, data_incovt, calln, tol, inip, model,jackk,improve)
            chi=m.minimize().fval
            return chi



class RepeatSingleFit:

    def __init__(self, data_t, data_y, data_cov, dini, dfin, inipars, model, calln, tol, inv_first,jackk,improve,no_corrs,multiprocess=5,multistart=None):
        self.data_t=data_t
        self.data_y=data_y
        self.data_cov=data_cov
        self.dini=dini
        self.dfin=dfin
        self.inipars=inipars
        self.model=model 
        self.calln=calln
        self.tol=tol
        self.inv_first=inv_first
        self.jackk=jackk
        self.improve=improve
        self.multistart=multistart
        self.multiprocess=multiprocess
        self.no_corrs=no_corrs

    def minimize(self):
        data_tt=self.data_t[self.dini-1:self.dfin]
        data_yt=self.data_y[self.dini-1:self.dfin]

        if (self.no_corrs!=0):
            self.data_cov=np.diag(np.diag(self.data_cov))

        if (self.inv_first==1):
            data_incovt=improved_inverse_covariance(self.data_cov)[self.dini-1:self.dfin,self.dini-1:self.dfin]
        else:
            data_incovt=improved_inverse_covariance(self.data_cov[self.dini-1:self.dfin,self.dini-1:self.dfin]) 


        inip=np.array(self.inipars)
        if (self.multistart):
            m=multistart_reuse_pars(data_tt, data_yt, data_incovt, self.calln, self.tol, inip, inip, inip, self.model, self.improve,self.multistart,self.multiprocess).multistart_select()
        else:
            m=Minuit_fit(data_tt, data_yt, data_incovt, self.calln, self.tol, inip, self.model,self.jackk,self.improve)
        return m.minimize()    


class plotGc:

    def __init__(self, data_t, data_y, data_cov, AIC_list, datatype, model, calln, tol, reuse, inv_first, xi, beta, R, corrtype, path, norm, cutoff, improve, no_corrs):
        self.data_t=data_t
        self.data_y=data_y
        self.data_cov=data_cov
        self.AIC_list=AIC_list
        self.datatype=datatype
        self.model=model
        self.calln=calln
        self.tol=tol
        self.reuse=reuse
        self.inv_first=inv_first
        self.xi=xi
        self.beta=beta
        self.R=R
        self.corrtype=corrtype 
        self.path=path
        self.norm=norm
        self.cutoff=cutoff
        self.improve=improve
        self.no_corrs=no_corrs


    def plot(self):   
        data_t     = self.data_t
        data_y_f   = np.mean(self.data_y,axis=0)
        data_cov_f = np.mean(self.data_cov,axis=0)
        data_err   = np.sqrt(np.diagonal(data_cov_f))

        tini    = self.AIC_list.ordered()[0,1]
        tfin    = self.AIC_list.ordered()[0,2]

        pars=np.array(self.AIC_list.selval()[0][1:])
        epars=np.array(self.AIC_list.selval()[1][1:])
        cpars=self.AIC_list.selval()[2][1:,1:]

        rin=tini-1
        rfin=tfin

        data_tp   = data_t[rin:rfin]
        propdata=100
        data_tpf  = np.linspace(data_tp[0],data_tp[-1],propdata*(len(data_tp)))
        data_tf   = np.linspace(data_t[0],data_t[-1],propdata*(len(data_t)))

        if ((self.datatype=='exp' or self.datatype=='exp_WL') and self.norm==1):
            normdat    = eval(self.model)(data_t,*np.array(pars[0:2]))[rin:rfin]
            normdatall = eval(self.model)(data_t,*np.array(pars[0:2]))
            norm       = eval(self.model)(data_tpf,*np.array(pars[0:2]))
            normall    = eval(self.model)(data_tf,*np.array(pars[0:2]))
        else:
            normdat    = 1
            normdatall = 1
            norm       = 1
            normall    = 1

        fit_plot    = eval(self.model)(data_tpf,*pars)/norm
        efit_plot   = prop_err(data_tpf,self.model,pars,epars,cpars)/norm

        fitall_plot  = eval(self.model)(data_tf,*pars)/normall
        efitall_plot = prop_err(data_tf,self.model,pars,epars,cpars)/normall
                
        data_yp   = data_y_f[rin:rfin]/normdat
        data_errp = data_err[rin:rfin]/normdat

        data_y_f/=normdatall
        data_err/=normdatall

        if (self.corrtype=='g'):
            if (self.datatype=='exp_WL'):
                corrplot='G_phys'
            else:
                corrplot='G_0'
        elif (self.corrtype=='w'):
            corrplot='W_phys'

        chi_val_total=self.AIC_list.ordered()[0,8]

        fig = plt.figure(figsize=(16,9))
        np.set_printoptions(formatter={'float': '{: .4f}'.format})
        plt.fill_between(data_tf, fitall_plot+efitall_plot, fitall_plot-efitall_plot,color=jpac_red,alpha=0.1)
        #plt.fill_between(data_tpf, fit+efit, fit-efit,color=jpac_red,alpha=0.7, label='Range={}-{} Model={} Pars={} $\\chi^2/\\text{{ndof}}={:.1f}$'.format(self.AIC_list.ordered()[0,1],self.AIC_list.ordered()[0,2],self.model,chi_val_total))
        #plt.fill_between(data_tpf, fit_plot+efit_plot, fit_plot-efit_plot,color=jpac_red,alpha=0.7, label='Val={}, {} Pars={} $\\chi^2/\\text{{ndof}}={:.1f}$' "\n" "{} $\\xi={},\\,\\beta={},\\,R={}$".format(selval,maval,pars,chi_val_total,corrplot,self.xi,self.beta,self.R))
        plt.fill_between(data_tpf, fit_plot+efit_plot, fit_plot-efit_plot,color=jpac_red,alpha=0.7)
        #plt.plot(1,label='$\\chi^2/\\text{{ndof}}={:.1f}$'.format(chi_val_total),marker = '',ls ='')
        plt.errorbar(data_t, data_y_f, data_err, fmt="ok", alpha=0.3)
        plt.errorbar(data_tp, data_yp, data_errp, fmt="ok")
        plt.xlabel("$T$",fontsize=24)
        plt.ylabel("$G({},T)$".format(self.R),fontsize=24)
        plt.xlim(xmin=0)
        #plt.tick_params(length=5, width=1.8)
        plt.yticks(fontsize=20)
        plt.xticks(fontsize=20)
        plt.legend(loc='upper right',fontsize=20,frameon=False)
        #plt.title('$\\xi={},\\,\\beta={},\\,R={}$'.format(self.xi,self.beta,self.R),pad = 20,fontsize=20)
        plt.figtext(0.82, 0.85, 
                '$\\xi={},\\,\\beta={:.2f}$'.format(self.xi,self.beta/100), 
                horizontalalignment ="center",  
                verticalalignment ="center",  
                wrap = True, fontsize = 18,  
                color ="black")
        plt.figtext(0.82, 0.8, 
                '$\\chi^2/\\text{{ndof}}={:.1f}$'.format(chi_val_total), 
                horizontalalignment ="center",  
                verticalalignment ="center",  
                wrap = True, fontsize = 18,  
                color ="black")
        plt.savefig('{}.pdf'.format(self.path), format="pdf", bbox_inches='tight', pad_inches=0.2)
        plt.close(fig)


    def fit_plot(self):   
        data_t     = self.data_t
        data_y_f   = np.mean(self.data_y,axis=0)
        data_cov_f = np.mean(self.data_cov,axis=0)
        data_err   = np.sqrt(np.diagonal(data_cov_f))

        tini    = self.AIC_list.ordered()[0,1]
        tfin    = self.AIC_list.ordered()[0,2]
        inipars = self.AIC_list.ordered()[0,6]

        maval=np.array([self.AIC_list.avgval(self.cutoff)[0][0],self.AIC_list.avgval(self.cutoff)[1][0]])
        selval=np.array([self.AIC_list.selval()[0][0],self.AIC_list.selval()[1][0]])

        m=RepeatSingleFit(data_t, data_y_f, data_cov_f, tini, tfin, inipars, self.model, self.calln, self.tol,self.inv_first,0,self.improve,self.no_corrs)

        fit=m.minimize()
        ma=MA_fit(fit)

        data_fit_range=[tini,tfin]
        pars=np.array(fit.values)
        epars=np.array(fit.errors)
        cpars=np.array(fit.covariance.correlation())
        data_G=[data_fit_range,pars,epars,cpars]

        rin=tini-1
        rfin=tfin


        data_tp   = data_t[rin:rfin]
        propdata=100
        data_tpf  = np.linspace(data_tp[0],data_tp[-1],propdata*(len(data_tp)))
        data_tf   = np.linspace(data_t[0],data_t[-1],propdata*(len(data_t)))

        if ((self.datatype=='exp' or self.datatype=='exp_WL') and self.norm==1):
            normdat    = eval(self.model)(data_t,*np.array(pars[0:2]))[rin:rfin]
            normdatall = eval(self.model)(data_t,*np.array(pars[0:2]))
            norm       = eval(self.model)(data_tpf,*np.array(pars[0:2]))
            normall    = eval(self.model)(data_tf,*np.array(pars[0:2]))
        else:
            normdat    = 1
            normdatall = 1
            norm       = 1
            normall    = 1

        fit_plot    = eval(self.model)(data_tpf,*pars)/norm
        efit_plot   = prop_err(data_tpf,self.model,pars,epars,cpars)/norm

        fitall_plot  = eval(self.model)(data_tf,*pars)/normall
        efitall_plot = prop_err(data_tf,self.model,pars,epars,cpars)/normall
                
        data_yp   = data_y_f[rin:rfin]/normdat
        data_errp = data_err[rin:rfin]/normdat

        data_y_f/=normdatall
        data_err/=normdatall

        if (self.corrtype=='g'):
            if (self.datatype=='exp_WL'):
                corrplot='G_phys'
            else:
                corrplot='G_0'
        elif (self.corrtype=='w'):
            corrplot='W_phys'

        chi_val_total=self.AIC_list.ordered()[0,8]

        fig = plt.figure(figsize=(16,9))
        np.set_printoptions(formatter={'float': '{: .4f}'.format})
        plt.fill_between(data_tf, fitall_plot+efitall_plot, fitall_plot-efitall_plot,color=jpac_red,alpha=0.1)
        #plt.fill_between(data_tpf, fit+efit, fit-efit,color=jpac_red,alpha=0.7, label='Range={}-{} Model={} Pars={} $\\chi^2/\\text{{ndof}}={:.1f}$'.format(self.AIC_list.ordered()[0,1],self.AIC_list.ordered()[0,2],self.model,chi_val_total))
        #plt.fill_between(data_tpf, fit_plot+efit_plot, fit_plot-efit_plot,color=jpac_red,alpha=0.7, label='Val={}, {} Pars={} $\\chi^2/\\text{{ndof}}={:.1f}$' "\n" "{} $\\xi={},\\,\\beta={},\\,R={}$".format(selval,maval,pars,chi_val_total,corrplot,self.xi,self.beta,self.R))
        plt.fill_between(data_tpf, fit_plot+efit_plot, fit_plot-efit_plot,color=jpac_red,alpha=0.7)
        #plt.plot(1,label='$\\chi^2/\\text{{ndof}}={:.1f}$'.format(chi_val_total),marker = '',ls ='')
        plt.errorbar(data_t, data_y_f, data_err, fmt="ok", alpha=0.3)
        plt.errorbar(data_tp, data_yp, data_errp, fmt="ok")
        plt.xlabel("$T$",fontsize=24)
        plt.ylabel("$G({},T)$".format(self.R),fontsize=24)
        plt.xlim(xmin=0)
        #plt.tick_params(length=5, width=1.8)
        plt.yticks(fontsize=20)
        plt.xticks(fontsize=20)
        plt.legend(loc='upper right',fontsize=20,frameon=False)
        #plt.title('$\\xi={},\\,\\beta={},\\,R={}$'.format(self.xi,self.beta,self.R),pad = 20,fontsize=20)
        plt.figtext(0.82, 0.85, 
                '$\\xi={},\\,\\beta={:.2f}$'.format(self.xi,self.beta/100), 
                horizontalalignment ="center",  
                verticalalignment ="center",  
                wrap = True, fontsize = 18,  
                color ="black")
        plt.figtext(0.82, 0.8, 
                '$\\chi^2/\\text{{ndof}}={:.1f}$'.format(chi_val_total), 
                horizontalalignment ="center",  
                verticalalignment ="center",  
                wrap = True, fontsize = 18,  
                color ="black")
        plt.savefig('{}.pdf'.format(self.path), format="pdf", bbox_inches='tight', pad_inches=0.2)
        plt.close(fig)



    def plot_log(self):   
        data_t     = self.data_t
        data_y_f   = np.mean(self.data_y,axis=0)
        data_cov_f = np.mean(self.data_cov,axis=0)
        data_err   = np.sqrt(np.diagonal(data_cov_f))

        tini    = self.AIC_list.ordered()[0,1]
        tfin    = self.AIC_list.ordered()[0,2]
        inipars = self.AIC_list.ordered()[0,6]


        pars=np.array(self.AIC_list.selval()[0][1:])
        epars=np.array(self.AIC_list.selval()[1][1:])
        cpars=self.AIC_list.selval()[2][1:,1:]

        rin=tini-1
        rfin=tfin

        data_tp   = data_t[rin:rfin]
        propdata=100
        data_tpf  = np.linspace(data_tp[0],data_tp[-1],propdata*(len(data_tp)))
        data_tf   = np.linspace(data_t[0],data_t[-1],propdata*(len(data_t)))

        if ((self.datatype=='exp' or self.datatype=='exp_WL') and self.norm==1):
            normdat    = eval(self.model)(data_t,*np.array(pars[0:2]))[rin:rfin]
            normdatall = eval(self.model)(data_t,*np.array(pars[0:2]))
            norm       = eval(self.model)(data_tpf,*np.array(pars[0:2]))
            normall    = eval(self.model)(data_tf,*np.array(pars[0:2]))
        else:
            normdat    = 1
            normdatall = 1
            norm       = 1
            normall    = 1

        fit_plot    = eval(self.model)(data_tpf,*pars)/norm
        efit_plot   = prop_err(data_tpf,self.model,pars,epars,cpars)/norm

        fitall_plot  = eval(self.model)(data_tf,*pars)/normall
        efitall_plot = prop_err(data_tf,self.model,pars,epars,cpars)/normall
                
        data_yp   = data_y_f[rin:rfin]/normdat
        data_errp = data_err[rin:rfin]/normdat

        data_y_f/=normdatall
        data_err/=normdatall

        if (self.corrtype=='g'):
            if (self.datatype=='exp_WL'):
                corrplot='G_phys'
            else:
                corrplot='G_0'
        elif (self.corrtype=='w'):
            corrplot='W_phys'

        chi_val_total=self.AIC_list.ordered()[0,8]

        fig = plt.figure(figsize=(16,9))
        np.set_printoptions(formatter={'float': '{: .4f}'.format})
        plt.fill_between(data_tf, fitall_plot+efitall_plot, fitall_plot-efitall_plot,color=jpac_red,alpha=0.1)
        #plt.fill_between(data_tpf, fit+efit, fit-efit,color=jpac_red,alpha=0.7, label='Range={}-{} Model={} Pars={} $\\chi^2/\\text{{ndof}}={:.1f}$'.format(self.AIC_list.ordered()[0,1],self.AIC_list.ordered()[0,2],self.model,chi_val_total))
        #plt.fill_between(data_tpf, fit_plot+efit_plot, fit_plot-efit_plot,color=jpac_red,alpha=0.7, label='Val={}, {} Pars={} $\\chi^2/\\text{{ndof}}={:.1f}$' "\n" "{} $\\xi={},\\,\\beta={},\\,R={}$".format(selval,maval,pars,chi_val_total,corrplot,self.xi,self.beta,self.R))
        plt.fill_between(data_tpf, fit_plot+efit_plot, fit_plot-efit_plot,color=jpac_red,alpha=0.7)
        #plt.plot(1,label='$\\chi^2/\\text{{ndof}}={:.1f}$'.format(chi_val_total),marker = '',ls ='')
        plt.errorbar(data_t, data_y_f, data_err, fmt="ok", alpha=0.3)
        plt.errorbar(data_tp, data_yp, data_errp, fmt="ok")
        plt.xlabel("$T$",fontsize=24)
        plt.ylabel("$G({},T)$".format(self.R),fontsize=24)
        plt.yscale('log')
        plt.yticks(fontsize=20)
        plt.xticks(fontsize=20)
        plt.xlim(xmin=0)
        #plt.tick_params(length=5, width=1.8)
        plt.legend(loc='upper right',fontsize=20,frameon=False)
        #plt.title('$\\xi={},\\,\\beta={},\\,R={}$'.format(self.xi,self.beta,self.R),pad = 20,fontsize=20)
        plt.figtext(0.82, 0.85, 
                '$\\xi={},\\,\\beta={:.2f}$'.format(self.xi,self.beta/100), 
                horizontalalignment ="center",  
                verticalalignment ="center",  
                wrap = True, fontsize = 18,  
                color ="black")
        plt.figtext(0.82, 0.8, 
                '$\\chi^2/\\text{{ndof}}={:.1f}$'.format(chi_val_total), 
                horizontalalignment ="center",  
                verticalalignment ="center",  
                wrap = True, fontsize = 18,  
                color ="black")
        plt.savefig('{}_log.pdf'.format(self.path), format="pdf", bbox_inches='tight', pad_inches=0.2)
        plt.close(fig)        




    def fit_plot_log(self):   
        data_t     = self.data_t
        data_y_f   = np.mean(self.data_y,axis=0)
        data_cov_f = np.mean(self.data_cov,axis=0)
        data_err   = np.sqrt(np.diagonal(data_cov_f))

        tini    = self.AIC_list.ordered()[0,1]
        tfin    = self.AIC_list.ordered()[0,2]
        inipars = self.AIC_list.ordered()[0,6]

        maval=np.array([self.AIC_list.avgval(self.cutoff)[0][0],self.AIC_list.avgval(self.cutoff)[1][0]])
        selval=np.array([self.AIC_list.selval()[0][0],self.AIC_list.selval()[1][0]])

        m=RepeatSingleFit(data_t, data_y_f, data_cov_f, tini, tfin, inipars, self.model, self.calln, self.tol,self.inv_first,0,self.improve,self.no_corrs)

        fit=m.minimize()
        ma=MA_fit(fit)

        pars=np.array(fit.values)
        epars=np.array(fit.errors)
        cpars=np.array(fit.covariance.correlation())

        rin=tini-1
        rfin=tfin


        data_tp   = data_t[rin:rfin]
        propdata=100
        data_tpf  = np.linspace(data_tp[0],data_tp[-1],propdata*(len(data_tp)))
        data_tf   = np.linspace(data_t[0],data_t[-1],propdata*(len(data_t)))

        if ((self.datatype=='exp' or self.datatype=='exp_WL') and self.norm==1):
            normdat    = eval(self.model)(data_t,*np.array(pars[0:2]))[rin:rfin]
            normdatall = eval(self.model)(data_t,*np.array(pars[0:2]))
            norm       = eval(self.model)(data_tpf,*np.array(pars[0:2]))
            normall    = eval(self.model)(data_tf,*np.array(pars[0:2]))
        else:
            normdat    = 1
            normdatall = 1
            norm       = 1
            normall    = 1

        fit_plot    = eval(self.model)(data_tpf,*pars)/norm
        efit_plot   = prop_err(data_tpf,self.model,pars,epars,cpars)/norm

        fitall_plot  = eval(self.model)(data_tf,*pars)/normall
        efitall_plot = prop_err(data_tf,self.model,pars,epars,cpars)/normall
                
        data_yp   = data_y_f[rin:rfin]/normdat
        data_errp = data_err[rin:rfin]/normdat

        data_y_f/=normdatall
        data_err/=normdatall

        if (self.corrtype=='g'):
            if (self.datatype=='exp_WL'):
                corrplot='G_phys'
            else:
                corrplot='G_0'
        elif (self.corrtype=='w'):
            corrplot='W_phys'

        chi_val_total=self.AIC_list.ordered()[0,8]

        fig = plt.figure(figsize=(16,9))
        np.set_printoptions(formatter={'float': '{: .4f}'.format})
        plt.fill_between(data_tf, fitall_plot+efitall_plot, fitall_plot-efitall_plot,color=jpac_red,alpha=0.1)
        #plt.fill_between(data_tpf, fit+efit, fit-efit,color=jpac_red,alpha=0.7, label='Range={}-{} Model={} Pars={} $\\chi^2/\\text{{ndof}}={:.1f}$'.format(self.AIC_list.ordered()[0,1],self.AIC_list.ordered()[0,2],self.model,chi_val_total))
        #plt.fill_between(data_tpf, fit_plot+efit_plot, fit_plot-efit_plot,color=jpac_red,alpha=0.7, label='Val={}, {} Pars={} $\\chi^2/\\text{{ndof}}={:.1f}$' "\n" "{} $\\xi={},\\,\\beta={},\\,R={}$".format(selval,maval,pars,chi_val_total,corrplot,self.xi,self.beta,self.R))
        plt.fill_between(data_tpf, fit_plot+efit_plot, fit_plot-efit_plot,color=jpac_red,alpha=0.7)
        #plt.plot(1,label='$\\chi^2/\\text{{ndof}}={:.1f}$'.format(chi_val_total),marker = '',ls ='')
        plt.errorbar(data_t, data_y_f, data_err, fmt="ok", alpha=0.3)
        plt.errorbar(data_tp, data_yp, data_errp, fmt="ok")
        plt.xlabel("$T$",fontsize=24)
        plt.ylabel("$G({},T)$".format(self.R),fontsize=24)
        plt.yscale('log')
        plt.yticks(fontsize=20)
        plt.xticks(fontsize=20)
        plt.xlim(xmin=0)
        #plt.tick_params(length=5, width=1.8)
        plt.legend(loc='upper right',fontsize=20,frameon=False)
        #plt.title('$\\xi={},\\,\\beta={},\\,R={}$'.format(self.xi,self.beta,self.R),pad = 20,fontsize=20)
        plt.figtext(0.82, 0.85, 
                '$\\xi={},\\,\\beta={:.2f}$'.format(self.xi,self.beta/100), 
                horizontalalignment ="center",  
                verticalalignment ="center",  
                wrap = True, fontsize = 18,  
                color ="black")
        plt.figtext(0.82, 0.8, 
                '$\\chi^2/\\text{{ndof}}={:.1f}$'.format(chi_val_total), 
                horizontalalignment ="center",  
                verticalalignment ="center",  
                wrap = True, fontsize = 18,  
                color ="black")
        plt.savefig('{}_log.pdf'.format(self.path), format="pdf", bbox_inches='tight', pad_inches=0.2)
        plt.close(fig)        



    def plot_effm(self):   
        data_t     = self.data_t
        data_y_f   = np.mean(self.data_y,axis=0)
        data_cov_f = np.mean(self.data_cov,axis=0)

        gdata=self.data_y
        effm=[]
        ll=len(gdata[0])
        for i in range(len(gdata)):
            int_ratio=gdata[i][0:ll-1]/gdata[i][1:ll]
            for j in range(len(int_ratio)):
                if (int_ratio[j]<0):
                    int_ratio[j]=1
            int_data=np.log(int_ratio)
            effm.append(int_data)

        effm=np.array(effm)

        data_effm_err_o=np.sqrt(np.diagonal(ensemble_stat(jackknife(effm).up()).rcov()))
        data_effm_o=jackknife(np.array(effm)).fmean()

        maxp=len(data_effm_o)
        for i in range(len(data_effm_o)):
            if (data_effm_err_o[i]>data_effm_o[i]):
                maxp=i-1
                break


        tini    = self.AIC_list.ordered()[0,1]
        tfin    = self.AIC_list.ordered()[0,2]
        inipars = self.AIC_list.ordered()[0,6]


        pars=np.array(self.AIC_list.selval()[0][1:])
        epars=np.array(self.AIC_list.selval()[1][1:])
        cpars=self.AIC_list.selval()[2][1:,1:]


        rin=tini-1
        rfin=tfin-1
        rdfin=max(0,min(maxp,tfin-1))

        data_tp   = data_t[rin:rfin]
        propdata=100
        data_tpf  = np.linspace(data_tp[0],data_tp[-1],propdata*(len(data_tp)))
        data_tf   = np.linspace(data_t[0],data_t[-1],propdata*(len(data_t)))


        if (self.model=="nb_exp_np" or self.model=="exp_np"):
            fit_plot    = eff_m_exp_np(data_tpf,*pars)
            efit_plot   = prop_err(data_tpf,'eff_m_exp_np',pars,epars,cpars)

            fitall_plot = eff_m_exp_np(data_tf,*pars)
            efitall_plot= prop_err(data_tf,'eff_m_exp_np',pars,epars,cpars)
            
        elif (self.model=="nb_exp_np_pole" or self.model=="exp_np_pole"):
            fit_plot    = eff_m_exp_np_pole(data_tpf,*pars)
            efit_plot   = prop_err(data_tpf,'eff_m_exp_np_pole',pars,epars,cpars)

            fitall_plot = eff_m_exp_np_pole(data_tf,*pars)
            efitall_plot= prop_err(data_tf,'eff_m_exp_np_pole',pars,epars,cpars)
                

        data_effmp   = data_effm_o[rin:rdfin]
        data_effm_errp = data_effm_err_o[rin:rdfin]

        maxplen=max(0,min(maxp,len(data_t)-1))

        data_effm_f     = data_effm_o[0:maxplen]
        data_effm_err_f = data_effm_err_o[0:maxplen]
        
        data_tp   = data_t[rin:rdfin]
        data_tpa  = data_t[0:maxplen]


        if (self.corrtype=='g'):
            if (self.datatype=='exp_WL'):
                corrplot='G_phys'
            else:
                corrplot='G_0'
        elif (self.corrtype=='w'):
            corrplot='W_phys'
    
        chi_val_total=self.AIC_list.ordered()[0,8]

        fig = plt.figure(figsize=(16,9))
        np.set_printoptions(formatter={'float': '{: .4f}'.format})
        plt.fill_between(data_tf, fitall_plot+efitall_plot, fitall_plot-efitall_plot,color=jpac_red,alpha=0.1)
        #plt.fill_between(data_tpf, fit+efit, fit-efit,color=jpac_red,alpha=0.7, label='Range={}-{} Model={} Pars={} $\\chi^2/\\text{{ndof}}={:.1f}$'.format(self.AIC_list.ordered()[0,1],self.AIC_list.ordered()[0,2],self.model,pars,chi_val_total))
        #plt.fill_between(data_tpf, fit_plot+efit_plot, fit_plot-efit_plot,color=jpac_red,alpha=0.7, label='Val={}, {} Pars={} $\\chi^2/\\text{{ndof}}={:.1f}$' "\n" "{} $\\xi={},\\,\\beta={},\\,R={}$".format(selval,maval,pars,chi_val_total,corrplot,self.xi,self.beta,self.R))
        plt.fill_between(data_tpf, fit_plot+efit_plot, fit_plot-efit_plot,color=jpac_red,alpha=0.7)
        #plt.plot(data_tpf,fit_plot,label='$\\chi^2/\\text{{ndof}}={:.1f}$'.format(chi_val_total),marker = '',ls ='')
        #print('{}_effm.pdf'.format(self.path))
        #print(len(data_tpa),len(data_effm_f),len(data_effm_err_f))
        #print(len(data_effmp))

        if (len(data_tp)>=1):
            plt.errorbar(data_tpa, data_effm_f, data_effm_err_f, fmt="ok", alpha=0.3)
            plt.errorbar(data_tp, data_effmp, data_effm_errp, fmt="ok")

        plt.xlabel("$T$",fontsize=24)
        plt.ylabel("$\\log(\\frac{{G({},T)}}{{G({},T+1)}})$".format(self.R,self.R),fontsize=24)
        plt.yticks(fontsize=20)
        plt.xticks(fontsize=20)
        plt.xlim(xmin=0)
        #plt.tick_params(length=5, width=1.8)
        plt.legend(loc='upper right',fontsize=20,frameon=False)
        #plt.title('$\\xi={},\\,\\beta={},\\,R={}$'.format(self.xi,self.beta,self.R),pad = 20,fontsize=20)

        plt.figtext(0.82, 0.85, 
                '$\\xi={},\\,\\beta={:.2f}$'.format(self.xi,self.beta/100), 
                horizontalalignment ="center",  
                verticalalignment ="center",  
                wrap = True, fontsize = 18,  
                color ="black")
        plt.figtext(0.82, 0.8, 
                '$\\chi^2/\\text{{ndof}}={:.1f}$'.format(chi_val_total), 
                horizontalalignment ="center",  
                verticalalignment ="center",  
                wrap = True, fontsize = 18,  
                color ="black")

        plt.savefig('{}_effm.pdf'.format(self.path), format="pdf", bbox_inches='tight', pad_inches=0.2)
        plt.close(fig)




    def fit_plot_effm(self):   
        data_t     = self.data_t
        data_y_f   = np.mean(self.data_y,axis=0)
        data_cov_f = np.mean(self.data_cov,axis=0)

        gdata=self.data_y
        effm=[]
        ll=len(gdata[0])
        for i in range(len(gdata)):
            int_ratio=gdata[i][0:ll-1]/gdata[i][1:ll]
            for j in range(len(int_ratio)):
                if (int_ratio[j]<0):
                    int_ratio[j]=1
            int_data=np.log(int_ratio)
            effm.append(int_data)

        effm=np.array(effm)

        data_effm_err_o=np.sqrt(np.diagonal(ensemble_stat(jackknife(effm).up()).rcov()))
        data_effm_o=jackknife(np.array(effm)).fmean()

        maxp=len(data_effm_o)
        for i in range(len(data_effm_o)):
            if (data_effm_err_o[i]>data_effm_o[i]):
                maxp=i-1
                break


        tini    = self.AIC_list.ordered()[0,1]
        tfin    = self.AIC_list.ordered()[0,2]
        inipars = self.AIC_list.ordered()[0,6]

        maval=np.array([self.AIC_list.avgval(self.cutoff)[0][0],self.AIC_list.avgval(self.cutoff)[1][0]])
        selval=np.array([self.AIC_list.selval()[0][0],self.AIC_list.selval()[1][0]])

        m=RepeatSingleFit(data_t, data_y_f, data_cov_f, tini, tfin, inipars, self.model, self.calln, self.tol,self.inv_first,0,self.improve,self.no_corrs)

        fit=m.minimize()
        ma=MA_fit(fit)

        pars=np.array(fit.values)
        epars=np.array(fit.errors)
        cpars=np.array(fit.covariance.correlation())


        rin=tini-1
        rfin=tfin-1
        rdfin=max(0,min(maxp,tfin-1))


        data_tp   = data_t[rin:rfin]
        propdata=100
        data_tpf  = np.linspace(data_tp[0],data_tp[-1],propdata*(len(data_tp)))
        data_tf   = np.linspace(data_t[0],data_t[-1],propdata*(len(data_t)))


        if (self.model=="nb_exp_np" or self.model=="exp_np"):
            fit_plot    = eff_m_exp_np(data_tpf,*pars)
            efit_plot   = prop_err(data_tpf,'eff_m_exp_np',pars,epars,cpars)

            fitall_plot = eff_m_exp_np(data_tf,*pars)
            efitall_plot= prop_err(data_tf,'eff_m_exp_np',pars,epars,cpars)
            
        elif (self.model=="nb_exp_np_pole" or self.model=="exp_np_pole"):
            fit_plot    = eff_m_exp_np_pole(data_tpf,*pars)
            efit_plot   = prop_err(data_tpf,'eff_m_exp_np_pole',pars,epars,cpars)

            fitall_plot = eff_m_exp_np_pole(data_tf,*pars)
            efitall_plot= prop_err(data_tf,'eff_m_exp_np_pole',pars,epars,cpars)
                

        data_effmp   = data_effm_o[rin:rdfin]
        data_effm_errp = data_effm_err_o[rin:rdfin]

        maxplen=max(0,min(maxp,len(data_t)-1))

        data_effm_f     = data_effm_o[0:maxplen]
        data_effm_err_f = data_effm_err_o[0:maxplen]
        
        data_tp   = data_t[rin:rdfin]
        data_tpa  = data_t[0:maxplen]


        if (self.corrtype=='g'):
            if (self.datatype=='exp_WL'):
                corrplot='G_phys'
            else:
                corrplot='G_0'
        elif (self.corrtype=='w'):
            corrplot='W_phys'
    
        chi_val_total=self.AIC_list.ordered()[0,8]

        fig = plt.figure(figsize=(16,9))
        np.set_printoptions(formatter={'float': '{: .4f}'.format})
        plt.fill_between(data_tf, fitall_plot+efitall_plot, fitall_plot-efitall_plot,color=jpac_red,alpha=0.1)
        #plt.fill_between(data_tpf, fit+efit, fit-efit,color=jpac_red,alpha=0.7, label='Range={}-{} Model={} Pars={} $\\chi^2/\\text{{ndof}}={:.1f}$'.format(self.AIC_list.ordered()[0,1],self.AIC_list.ordered()[0,2],self.model,pars,chi_val_total))
        #plt.fill_between(data_tpf, fit_plot+efit_plot, fit_plot-efit_plot,color=jpac_red,alpha=0.7, label='Val={}, {} Pars={} $\\chi^2/\\text{{ndof}}={:.1f}$' "\n" "{} $\\xi={},\\,\\beta={},\\,R={}$".format(selval,maval,pars,chi_val_total,corrplot,self.xi,self.beta,self.R))
        plt.fill_between(data_tpf, fit_plot+efit_plot, fit_plot-efit_plot,color=jpac_red,alpha=0.7)
        #plt.plot(data_tpf,fit_plot,label='$\\chi^2/\\text{{ndof}}={:.1f}$'.format(chi_val_total),marker = '',ls ='')
        #print('{}_effm.pdf'.format(self.path))
        #print(len(data_tpa),len(data_effm_f),len(data_effm_err_f))
        #print(len(data_effmp))

        if (len(data_tp)>=1):
            plt.errorbar(data_tpa, data_effm_f, data_effm_err_f, fmt="ok", alpha=0.3)
            plt.errorbar(data_tp, data_effmp, data_effm_errp, fmt="ok")

        plt.xlabel("$T$",fontsize=24)
        plt.ylabel("$\\log(\\frac{{G({},T)}}{{G({},T+1)}})$".format(self.R,self.R),fontsize=24)
        plt.yticks(fontsize=20)
        plt.xticks(fontsize=20)
        plt.xlim(xmin=0)
        #plt.tick_params(length=5, width=1.8)
        plt.legend(loc='upper right',fontsize=20,frameon=False)
        #plt.title('$\\xi={},\\,\\beta={},\\,R={}$'.format(self.xi,self.beta,self.R),pad = 20,fontsize=20)

        plt.figtext(0.82, 0.85, 
                '$\\xi={},\\,\\beta={:.2f}$'.format(self.xi,self.beta/100), 
                horizontalalignment ="center",  
                verticalalignment ="center",  
                wrap = True, fontsize = 18,  
                color ="black")
        plt.figtext(0.82, 0.8, 
                '$\\chi^2/\\text{{ndof}}={:.1f}$'.format(chi_val_total), 
                horizontalalignment ="center",  
                verticalalignment ="center",  
                wrap = True, fontsize = 18,  
                color ="black")

        plt.savefig('{}_effm.pdf'.format(self.path), format="pdf", bbox_inches='tight', pad_inches=0.2)
        plt.close(fig)




    def prepare(self):   
        data_t     = self.data_t
        data_y_f   = np.mean(self.data_y,axis=0)
        data_cov_f = np.mean(self.data_cov,axis=0)

        gdata=self.data_y
        effm=[]
        ll=len(gdata[0])
        for i in range(len(gdata)):
            int_ratio=gdata[i][0:ll-1]/gdata[i][1:ll]
            for j in range(len(int_ratio)):
                if (int_ratio[j]<0):
                    int_ratio[j]=1
            int_data=np.log(int_ratio)
            effm.append(int_data)

        effm=np.array(effm)

        data_effm_err_o=np.sqrt(np.diagonal(ensemble_stat(jackknife(effm).up()).rcov()))
        data_effm_o=jackknife(np.array(effm)).fmean()

        maxp=len(data_effm_o)
        for i in range(len(data_effm_o)):
            if (data_effm_err_o[i]>0.5*data_effm_o[i] or data_effm_err_o[i]==0):
                maxp=i-1
                break


        tini    = self.AIC_list.ordered()[0,1]
        tfin    = self.AIC_list.ordered()[0,2]
        inipars = self.AIC_list.ordered()[0,6]

        m=RepeatSingleFit(data_t, data_y_f, data_cov_f, tini, tfin, inipars, self.model, self.calln, self.tol,self.inv_first,0,self.improve,self.no_corrs)

        m.minimize()
        ma=MA_fit(m.minimize())

        pars=np.array(m.minimize().values)
        epars=np.array(m.minimize().errors)
        cpars=np.array(m.minimize().covariance.correlation())

        rin=tini-1
        rfin=tfin-1
        rdfin=max(0,min(maxp,tfin-1))


        data_tp   = data_t[rin:rfin]
        data_tpf  = np.linspace(data_tp[0],data_tp[-1],10*len(data_tp))
        data_tf   = np.linspace(data_t[0],data_t[-2],10*len(data_t))


        if (self.model=="nb_exp_np" or self.model=="exp_np"):
            fit     = eff_m_exp_np(data_tpf,*pars)
            efit    = prop_err(data_tpf,'eff_m_exp_np',pars,epars,cpars)

            fitall  = eff_m_exp_np(data_tf,*pars)
            efitall = prop_err(data_tf,'eff_m_exp_np',pars,epars,cpars)
        elif (self.model=="nb_exp_np_pole" or self.model=="exp_np_pole"):
            fit     = eff_m_exp_np_pole(data_tpf,*pars)
            efit    = prop_err(data_tpf,'eff_m_exp_np_pole',pars,epars,cpars)

            fitall  = eff_m_exp_np_pole(data_tf,*pars)
            efitall = prop_err(data_tf,'eff_m_exp_np_pole',pars,epars,cpars)
                

        data_effmp   = data_effm_o[rin:rdfin]
        data_effm_errp = data_effm_err_o[rin:rdfin]

        maxplen=max(0,min(maxp,len(data_t)-1))

        data_effm_f     = data_effm_o[0:maxplen]
        data_effm_err_f = data_effm_err_o[0:maxplen]
        
        data_tp   = data_t[rin:rdfin]
        data_tpa  = data_t[0:maxplen]

        np.set_printoptions(formatter={'float': '{}'.format})

        return data_tpa, data_effm_f, data_effm_err_f, data_tp, data_effmp, data_effm_errp, data_tf, fitall, efitall, data_tpf, fit, efit
    


def compact_eff_m(list_data_tpa, list_data_effm_f, list_data_effm_err_f, list_data_tp, list_data_effmp, list_data_effm_errp, list_data_tf, list_fitall, list_efitall, list_data_tpf, list_fit, list_efit, path, xi, beta, corrtype, datatype, elim):   
    fig = plt.figure(figsize=(16,9))
    for i in range(len(list_data_tpa)):
        data_tf=list_data_tf[i];data_tpf=list_data_tpf[i];fitall=list_fitall[i];efitall=list_efitall[i];fit=list_fit[i];efit=list_efit[i];data_tpa=list_data_tpa[i];
        data_tp=list_data_tp[i];data_effm_f=list_data_effm_f[i];data_effmp=list_data_effmp[i];data_effm_err_f=list_data_effm_err_f[i];data_effm_errp=list_data_effm_errp[i]
        if (efitall[-1]<=elim*fitall[-1]):
            plt.fill_between(data_tf, fitall+efitall, fitall-efitall,color=jpac_color_around[i],alpha=0.1)
            plt.fill_between(data_tpf, fit+efit, fit-efit,color=jpac_color_around[i],alpha=0.7)
            plt.text(data_tf[-1]+0.75, fitall[-1],'$R={}$'.format(i+1), fontsize = 18,color ="black")
            if (len(data_tp)>=1):
                plt.errorbar(data_tpa, data_effm_f, data_effm_err_f,color=jpac_color_around[i], fmt="ok", alpha=1)
                #plt.errorbar(data_tp, data_effmp, data_effm_errp,color=jpac_color_around[i], fmt="ok",alpha=1)

    if (corrtype=='g'):
        if (datatype=='exp_WL'):
            corrplot='G_phys'
        else:
            corrplot='G_0'
    elif (corrtype=='w'):
        corrplot='W_phys'

    plt.xlabel("$T$",fontsize=24)
    plt.ylabel("{} $\\xi={},\\,\\beta={}$".format(corrplot,xi,beta),fontsize=24)
    plt.yticks(fontsize=20)
    #plt.ylim(None,y_lim)
    plt.xlim(0,data_tf[-1]+2)
    plt.xticks(fontsize=20)
    plt.legend(fontsize=16,frameon=False)
    #plt.show()
    plt.savefig('{}_compact.pdf'.format(path), format="pdf", bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)      


#------------------------------------------------------------------------------------------------------------------------------------------
# Final fitter class: Can do both simple and Jackknife fitting. Produces a final list of fit results, 
# which can be ordered and manipulated with the AIClist classes
#------------------------------------------------------------------------------------------------------------------------------------------
    
class fitter:

    def __init__(self, data, size, dini, dstop, dmindata, dfin, datatype, model, inipars, variants, calln, tol, reuse, inv_first, multiprocess, cutoff, xi, beta, path, corrtype, norm, freeze, improve, multistart, no_corrs, no_valid_check):
        self.data=data
        self.size=size
        self.inipars=inipars
        self.calln=calln
        self.tol=tol
        self.model=model
        self.dini=dini
        self.dstop=dstop
        self.dmindata=dmindata
        self.dfin=dfin
        self.datatype=datatype
        self.model=model
        self.variants=variants
        self.reuse=reuse
        self.inv_first=inv_first
        self.multiprocess=multiprocess
        self.cutoff=cutoff
        self.xi=xi
        self.path=path
        self.beta=beta
        self.norm=norm
        self.freeze=freeze
        self.corrtype=corrtype
        self.improve=improve
        self.multistart=multistart
        self.no_corrs=no_corrs
        self.no_valid_check=no_valid_check


    def fit(self):
        valfk  = []
        evalfk = []
        valtrials = np.zeros(2)
        for k in range(len(self.data)):
            totaltraj=np.int_(len(self.data[k])/self.size[k])
            datak=self.data[k]
            Nt        = self.size[k]
            Nl        = len(datak[0])
            valf      = np.zeros((Nl))
            evalf     = np.zeros((Nl))
            for r in range(Nl):
                Gc=np.zeros((totaltraj,Nt))
                for i in range(totaltraj):
                    Gc[[i]]=datak[range(i*Nt,(i+1)*Nt),[r]]
                if (self.datatype=="dlog"):
                    gdata=np.log(ratio(trim_negative(jackknife(Gc).sample()).trimmed()).val())
                elif (self.datatype=="log" or self.datatype=="log_pole"):
                    gdata=np.log(trim_negative(jackknife(Gc).sample()).trimmed())
                elif (self.datatype=="exp" or self.datatype=="exp_WL" or self.datatype=="exp_line"):
                    gdata=jackknife(Gc).sample()

                gdata=jackknife(Gc).up()

                lt=len(gdata[0])
                dfin=min(self.dfin,lt)            # We cannot use more data than we have!!
                data_t   = np.linspace(1, lt, lt)
                data_y   = ensemble_stat(gdata).mean()
                data_cov = ensemble_stat(gdata).rcov()
                inipp  = self.inipars[0:lt-1]
                varp   = self.variants[0:lt-1]
                m=Modelsmin( data_t, data_y, data_cov, self.dini, self.dstop, self.dmindata, dfin, inipp, self.model, varp, self.datatype, self.calln, self.tol, self.reuse, self.inv_first,1, 0, self.improve, self.multistart,self.no_corrs,self.no_valid_check)
                mf=m.minimize()
                AIC_list=AIClist(mf[1])
                valf[r]=AIC_list.avgval0(self.cutoff)[0]
                evalf[r]=AIC_list.avgval0(self.cutoff)[1]
                valtrials[0]+=mf[0][0]
                valtrials[1]+=mf[0][1]

            valfk.append(valf)
            evalfk.append(evalf)

        return valtrials, valfk, evalfk
    


    def jackk_fit(self,jackkflag=None):
        Vrlistavg                   = []
        Vrlistavg_rescaled          = []
        Vrlistavg_rescaled_gaussian = []
        Vrlistsel                   = []
        worstsel                    = []
        AIC_result_final            = []
        datayf                      = []
        
        for k in range(len(self.data)):   
            print('Attempting fits to beta={}'.format(self.beta[k]))  
            totaltraj = np.int_(len(self.data[k])/self.size[k])
            datak     = self.data[k]
            Nt        = self.size[k]
            Nl        = len(datak[0])

            if (jackkflag is None or jackkflag >= totaltraj):
                jackkl=totaltraj
            else:
                jackkl=jackkflag

            valkVravg                   = np.zeros((Nl,jackkl))
            valkVravg_rescaled          = np.zeros((Nl,jackkl))
            valkVravg_rescaled_gaussian = np.zeros((Nl,jackkl))
            valkVrsel  = np.zeros((Nl,jackkl))
            meanchisq2 = np.zeros(Nl)
            AIC_result = []
            datay=[]
            list_data_tpa=[];list_data_effm_f=[];list_data_effm_err_f=[];list_data_tp=[];list_data_effmp=[];list_data_effm_errp=[];list_data_tf=[];list_fitall=[];list_efitall=[];list_data_tpf=[];list_fit=[];list_efit=[]
            for r in range(Nl):
                print('Attempting fits to r={}'.format(r+1))
                Gc=np.zeros((totaltraj,Nt))
                for i in range(totaltraj):
                    Gc[[i]]=datak[range(i*Nt,(i+1)*Nt),[r]]

                if (self.datatype=="dlog"):
                    gdata=np.log(ratio(trim_negative(jackknife(Gc,jackkl).sample()).trimmed()).val())
                elif (self.datatype=="log" or self.datatype=="log_pole"):
                    gdata=np.log(trim_negative(jackknife(Gc,jackkl).sample()).trimmed())
                elif (self.datatype=="exp" or self.datatype=="exp_WL" or self.datatype=="exp_line"):
                    gdata=jackknife(Gc,jackkl).sample()

                gdata=jackknife(gdata,jackkl).up()

                lt=len(gdata[0])
                dfin              = min(self.dfin,lt)
                if (self.dini==0):
                    endl=np.int_(lt-self.dini)
                else:
                    endl=np.int_(lt-self.dini+1)

                dmindata          = min(self.dmindata,endl)
                data_t            = np.linspace(1, lt, lt)
                data_y            = jackknife(gdata,jackkl).sample()
                data_cov          = jackknife(gdata,jackkl).scov()
                inipp             = self.inipars[0:lt-1]
                varp              = self.variants[0:lt-1]
                m                 = Modelsmin( data_t, data_y, data_cov, self.dini, self.dstop, dmindata, dfin, inipp, self.model, varp, self.datatype, self.calln, self.tol, self.reuse, self.inv_first,self.multiprocess,self.freeze,self.improve,self.multistart, self.no_corrs, self.no_valid_check)
                mf                = m.jackk_minimize()
                AIC_list          = Jackknife_AIClist(mf)
                path='{}/{}_{}_{}_G_ti{}_{}_tfin{}_tmin{}_beta={}_R={}_nocorrs={}_{}'.format(self.path[0],len(varp),self.model,self.datatype,self.dini,self.dstop, dfin, dmindata,self.beta[k],r+1,self.no_corrs,self.corrtype)
                path_pickle='{}/{}_{}_{}_AIC_list_ti{}_{}_tfin{}_tmin{}_beta={}_R={}_nocorrs={}_{}'.format(self.path[0],len(varp),self.model,self.datatype,self.dini,self.dstop, dfin, dmindata,self.beta[k],r+1,self.no_corrs,self.corrtype)
                with open(path_pickle, "wb") as fp:   #Pickling
                    pickle.dump(AIC_list, fp)
                #path_txt='{}/{}_{}_{}_txt_ti{}_{}_tfin{}_tmin{}_beta={}_R={}_nocorrs={}_{}'.format(self.path[0],len(varp),self.model,self.datatype,self.dini,self.dstop, dfin, dmindata,self.beta[k],r+1,self.no_corrs,self.corrtype)
                #dataP=np.array([self.model,AIC_list.ordered()[0,6]],dtype=object)
                #print(dataP)
                #with open('{}.txt'.format(path_txt), 'w') as outfile:  
                #    outfile.write('# Txt input for best fits to G(R,T) data:\n')
                #    for item in AIC_list.ordered:
                #        outfile.write("%s\n" % item)

                plotGcorr=plotGc(data_t, data_y, data_cov, AIC_list, self.datatype, self.model, self.calln, self.tol, self.reuse, self.inv_first, self.xi,self.beta[k],r+1,self.corrtype, path, self.norm, self.cutoff, self.improve, self.no_corrs)
                plotGcorr.plot()  
                plotGcorr.plot_log()                
                plotGcorr.plot_effm()  
                data_tpa, data_effm_f, data_effm_err_f, data_tp, data_effmp, data_effm_errp, data_tf, fitall, efitall, data_tpf, fit, efit=plotGcorr.prepare() 
                list_data_tpa.append(data_tpa);list_data_effm_f.append(data_effm_f);list_data_effm_err_f.append(data_effm_err_f);list_data_tp.append(data_tp)
                list_data_effmp.append(data_effmp);list_data_effm_errp.append(data_effm_errp);list_data_tf.append(data_tf);list_fitall.append(fitall)
                list_efitall.append(efitall);list_data_tpf.append(data_tpf);list_fit.append(fit);list_efit.append(efit)
               
                meanchisq2[[r]] = AIC_list.ordered()[0,8]
                valkVravg[[r]], valkVravg_rescaled[[r]], valkVravg_rescaled_gaussian[[r]]  = AIC_list.avgsample(self.cutoff,1)
                valkVrsel[[r]]  = AIC_list.selsample(1)
                AIC_result.append(AIC_list.ordered())
                datay.append(mf)
            
            
            vlim=0.1
            pathcompact='{}/{}_{}_{}_G_ti{}_{}_tfin{}_tmin{}_beta={}_nocorrs={}_{}'.format(self.path[1],len(varp),self.model,self.datatype,self.dini,self.dstop,dfin,dmindata,self.beta[k],self.no_corrs,self.corrtype)
            compact_eff_m(list_data_tpa,list_data_effm_f,list_data_effm_err_f,list_data_tp,list_data_effmp,list_data_effm_errp,list_data_tf,list_fitall,list_efitall,list_data_tpf,list_fit,list_efit,pathcompact,self.xi,self.beta[k],self.corrtype,self.datatype, vlim)  

            print("Looped over R and Jackknife, fitting lattice is finished. Adding fit results to lists \n\n")
            Vrlistavg.append(np.transpose(valkVravg))               # Transposing the final data to order it as a Jackknife sample
            Vrlistavg_rescaled.append(np.transpose(valkVravg_rescaled))               # Transposing the final data to order it as a Jackknife sample
            Vrlistavg_rescaled_gaussian.append(np.transpose(valkVravg_rescaled_gaussian))               # Transposing the final data to order it as a Jackknife sample
            Vrlistsel.append(np.transpose(valkVrsel))               # Transposing the final data to order it as a Jackknife sample
            worstsel.append(np.max(meanchisq2))
            AIC_result_final.append(AIC_result)
            datayf.append(datay)

        return Vrlistavg, Vrlistavg_rescaled, Vrlistavg_rescaled_gaussian, Vrlistsel, worstsel, AIC_result_final, datayf
