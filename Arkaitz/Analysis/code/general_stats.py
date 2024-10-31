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
# Basic general code functions
#------------------------------------------------------------------------------------------------------------------------------------------


def mylen(x):
    return len(x) if isinstance(x, np.ndarray) or isinstance(x, list) else 1


def get_near_psd(A):   # As explained in N.J. Higham, "Computing a nearest symmetric positive semidefinite matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    if (len(np.shape(A))<2):
        result=A
    else:
        C=(A + np.transpose(A))/2
        eigval, eigvec = np.linalg.eig(C)
        if (np.min(eigval)<0):
            print('Negative eig found==',np.min(eigval))
        eigval[eigval < 0] = 0
        result=np.transpose(eigvec).dot(np.diag(eigval)).dot(eigvec)
        result=(result+np.transpose(result))/2
    return result

def improved_inverse_covariance(A):    # It computes the Moore-Penrose pseudo-inverse. It produces a SVD decomp and inverse, which resets small singular values via a condition related to machine precision. If no small singular values are found, it provides with the actual SVD method inversion
    rtol_reset=len(A)*np.finfo(np.float64).eps
    svd=sp.linalg.svd(A)
    eig_sign=np.round(np.real(sp.linalg.eig(A)[0])/np.abs(np.real(sp.linalg.eig(A)[0])))
    singular_value=svd[1][-1]/svd[1][0]
    #print(singular_value,rtol)
    if (singular_value>rtol_reset):
        UT=np.transpose(svd[0])
        inv_list=1/svd[1]*eig_sign
        inv_list[inv_list < 0]=0
        S_inv=np.diag(inv_list)
        V=np.transpose(svd[2])
        inv_A=np.dot(np.dot(V,S_inv),UT)
    else:
        inv_A=sp.linalg.pinvh(A,rtol=rtol_reset)
    return inv_A

def improved_inverse_covariance_no_near_psd(A):
    lA=len(A)
    I=np.diag(np.ones(lA))
    if (np.min(sp.linalg.eig(A)[0])>0):
        result=sp.linalg.svd(A)
        UT=np.transpose(result[0])
        S_inv=np.diag(1./result[1])
        V=np.transpose(result[2])
        inv_A=np.dot(np.dot(V,S_inv),UT)
    else:
        inv_A=sp.linalg.pinv(A)
    return inv_A


def add_gaussian_noise(list,final_var):        # Modifies an starting normal dist. of length=len(list) to have exact mean=0 and exact var=extra_var
    extra_var=final_var-np.cov(list)
    sig=np.sqrt(extra_var)
    white_noise=np.random.normal(0,sig,len(list))
    white_noise-=np.mean(white_noise)
    white_noise/=np.sqrt(np.cov(white_noise))/sig
    return list+white_noise


def add_gaussian_noise2(list,final_var):        # Adds normal dist. noise of length=len(list) to orig list to have final_var
    extra_var=final_var-np.cov(list)
    sig=np.sqrt(extra_var)
    white_noise=np.random.normal(0,sig,len(list))
    return list+white_noise



#------------------------------------------------------------------------------------------------------------------------------------------
# Code used to produce Jackknife ensembles for fitting out of raw data for correlators
#------------------------------------------------------------------------------------------------------------------------------------------


# Collecting completed configs from orig data files and transforming data ordering into my preferred choice
class collect_configs:

    def __init__(self, listd, Lextent, Textent):
        self.list=listd
        self.Lextent=Lextent
        self.Textent=Textent
        

    def trans_W(self):
        datafin=[]
        counter=0
        counterin=0
        while (counter < len(self.list)):
            #print(counter,counterin)
            flagbreak=False
            dataint=[]
            for i in range(np.int_(self.Textent)):
                datatrans=[]
                for j in range(np.int_(self.Lextent)):
                    index=counterin+self.Lextent*i+j
                    #print(counterin,index,i,j)
                    if (index < len(self.list)):
                        datatrans.append(self.list[index,2])
                        counter+= 1
                        if (np.int_(self.list[index,1])!=i+1 or np.int_(self.list[index,0])!=j+1):      #stop reading if internal i j do not correspond with files i j
                            #print(i, np.int_(self.list[index,1]), j, np.int_(self.list[index,0]),index,counterin)
                            counterin = index
                            flagbreak=True
                            break
                    else:
                        flagbreak=True
                        break
                if flagbreak==True:      #stop reading if we encounter a new config when i and j are not 0
                    break
                else:
                    dataint.append(datatrans)
            if flagbreak==False:
                counterin = index+1
                for inside in dataint:
                    datafin.append(inside)
            #print('working')
        return datafin   
    

    def trans_S(self):
        datafin=[]
        counter=0
        counterin=0
        while (counter < len(self.list)):
            #print(counter,counterin)
            flagbreak=False
            dataint=[]
            for j in range(np.int_(self.Lextent)):
                datatrans=[]
                for i in range(np.int_(self.Textent)):
                    index=counterin+i+self.Lextent*j
                    #print(counterin,index,i,j)
                    if (index < len(self.list)):
                        datatrans.append(self.list[index,2])
                        counter+= 1
                        if (np.int_(self.list[index,1])!=i+1 or np.int_(self.list[index,0])!=j+1):      #stop reading if internal i j do not correspond with files i j
                            #print(i, np.int_(self.list[index,1]), j, np.int_(self.list[index,0]),index,counterin)
                            counterin = index
                            flagbreak=True
                            break
                    else:
                        flagbreak=True
                        break
                if flagbreak==True:      #stop reading if we encounter a new config when i and j are not 0
                    break
                else:
                    dataint.append(datatrans)
            if flagbreak==False:
                counterin = index+1
                for inside in np.transpose(dataint):
                    datafin.append(inside)
            #print('working')
        return datafin    



# Just a trivial class to store the mean and correct covariance for a given sample
class ensemble_stat:

    def __init__(self, list):
        self.list=list

    def mean(self):
        mean=np.mean(self.list,axis=0)
        return mean

    # variance
    def cov(self):
        Nj=len(self.list)
        cov=np.cov(self.list, rowvar=False,bias=True)
        return cov    

    #reduced variance
    def rcov(self):
        Nj=len(self.list)
        cov=np.cov(self.list, rowvar=False,bias=True) / (Nj-1)
        return cov    
    

# Create Jackknife class. Produce N Jackknife samples out of N data points, store sample and final mean and covariance
class jackknife:

    def __init__(self, list, Nj=None):
        self.list=list
        self.Lj=len(list)
        if Nj is None:
            self.Nj=self.Lj
        else:
            self.Nj=min(Nj,self.Lj)
        Nt=mylen(self.list[0])
        jackk=np.zeros((self.Nj-1,Nt))
        self.jackkf=np.zeros((self.Nj,Nt))
        self.covjackk=np.zeros((self.Nj,Nt,Nt))
        trim=self.list[0:self.Nj]
        for i in range(self.Nj):
            jackk            = np.delete(trim,i,axis=0)
            self.jackkf[i]   = np.sum(jackk,axis=0)/(self.Nj-1)
            self.covjackk[i] = np.cov(jackk,rowvar=False,bias=False) / (self.Nj-1)

    #sample values: jackknife(original_ensemble).sample()=jackknife_ensemble
    def sample(self):
        return self.jackkf
    
    #sample covariances
    def scov(self):
        return self.covjackk     

    #final mean: jackknife(jackknife_ensemble).fmean()=jackknife(original_ensemble).fmean()=ensemble_stat(original_ensemble).mean()
    def fmean(self):
        return np.mean(self.jackkf,axis=0) 
    
    
    #final cov: jackknife(original_ensemble).fcov()=ensemble_stat(original_ensemble).rcov()
    def fcov(self):
        return np.cov(self.jackkf,rowvar=False,bias=True) * (self.Nj-1)  

    #augmented variance for jackknife: jackknife(jackknife_ensemble).upcov()=ensemble_stat(original_ensemble).rcov()
    def upcov(self):
        Nj=len(self.list)
        cov=np.cov(self.list, rowvar=False,bias=True) * (self.Nj-1)
        return cov      
    
    def cov(self):
        Nj=len(self.list)
        cov=np.cov(self.list, rowvar=False,bias=True)
        return cov       
    
    #augmented sample: jackknife(jackknife_ensemble).up()=original_ensemble
    def up(self):
        Lj=len(self.list)
        mean=np.mean(self.list,axis=0)
        ensem=self.list+(Lj)*(mean-self.list)
        return ensem   
    
    #augmented sample to sqrt: jackknife(jackknife(jackknife_ensemble).pup()).cov()=jackknife(jackknife_ensemble).upcov()
    def pup(self):
        Lj=len(self.list)
        mean=np.mean(self.list,axis=0)
        ensem=self.list+(math.sqrt(Lj-1)-1)*(self.list-mean)
        return ensem   


    #augmented sub sample to arbitrary variance: provide a variance to rescale a singled valued jackknife (a list where every entry is a Jackknife sample)
    # ensemble_stat(jackknife(jackknife_ensemble[:,0]).rup(factor)).cov()=factor
    def rup(self,val):
        Lj=len(self.list)
        mean=np.mean(self.list,axis=0)
        if (jackknife(self.list).cov()==0):         #Prevent fixed 0 param from creating nans
            valf= 1
        else:
            valf= val/jackknife(self.list).cov()
        ensem=self.list+(math.sqrt(valf)-1)*(self.list-mean)
        return ensem              


# We are fitting the log of the data, so we need to trim up to the first Nt for which the values are negative, for ANY gauge config
class trim_negative:

    def __init__(self, list):
        self.list=list

    def trimmed(self):
        Ntrim=len(self.list[0])
        trimf=np.array([])
        for i in range(len(self.list)):
            if (len(np.where(self.list[i] < 0)[0]) > 0):
                trimf=np.append(trimf,np.where(self.list[i] < 0)[0])

        if (len(trimf) > 0):
            Ntrim=np.min(trimf)
        flist=self.list[:,0:np.int_(Ntrim)]
        return flist
    

# Finally, we want the ratio between t and t+1 of this data, before computing the log function
class ratio:

    def __init__(self, list):
        self.list=list    
        self.flist=list[:,0:len(list[0])-1]/list[:,1:len(list[0])]

    def val(self):
        return self.flist
    


#------------------------------------------------------------------------------------------------------------------------------------------
# Class to extract params, errors, correlations and create AICperf from fit to data
#------------------------------------------------------------------------------------------------------------------------------------------
    
class MA_fit:

    def __init__(self, minimization):
        self.minimization=minimization   
    
    def pars(self):
        vals=np.array([])
        le=len(self.minimization.values)
        for i in range(le):
            vals=np.append(vals,self.minimization.values[i])
        return vals
    
    def errs(self):
        errs=np.array([])
        le=len(self.minimization.values)
        for i in range(le):
            errs=np.append(errs,self.minimization.errors[i])
        return errs  
      
    def corrs(self):
        le=len(self.minimization.values)
        corrs=np.zeros((le,le))
        for i in range(le):
            for j in range(le):
                corrs[i,j]=self.minimization.covariance.correlation()[i,j]
        return corrs      

    # I'm defining the AICp from https://arxiv.org/pdf/2008.01069.pdf, other criterions can be added here
    def AICp(self,ndata_full):
        fval=self.minimization.fval                            # get total chi2 value
        ndof=self.minimization.ndof                            # get number of degrees of freedom (ndata-npars)    
        nfit=self.minimization.nfit                           # Get number of fitted (non-fixed) params  
        ndata=self.minimization.ndof+nfit                         # Get number of data points    
        ndropped=ndata_full-ndata                                  # Number of dropped data points from fit
        kcorr=nfit+ndropped                                            # This is the corrected bias term (kcorr=npars when no data is dropped) for accounting when data is dropped
        return (fval+2.*kcorr)   
    
    def AICcp(self,ndata_full):
        fval=self.minimization.fval                            # get total chi2 value
        ndof=self.minimization.ndof                            # get number of degrees of freedom (ndata-npars)  
        nfit=self.minimization.nfit                           # Get number of fitted (non-fixed) params
        ndata=self.minimization.ndof+nfit                         # Get number of data points
        ndropped=ndata_full-ndata                                  # Number of dropped data points from fit
        kcorr=nfit+ndropped                                            # This is the corrected bias term (kcorr=npars when no data is dropped) for accounting when data is dropped
        return (fval+2.*kcorr)+2*kcorr*(kcorr+1)/(ndata_full-kcorr-1)  # Note that this value diverges when ndata_full-kcorr=0, remember this when fitting using this criterion to k+1 data points, with k as free params



#------------------------------------------------------------------------------------------------------------------------------------------
# Now classify the list by lowest AICp and get MA average values and errors
#------------------------------------------------------------------------------------------------------------------------------------------
    
class AIClist:
    
    def __init__(self, list):
        self.list=list
        np.amin(self.list[:,[3]])
        np.argmin(self.list[:,[3]])
        self.aic_listfits = self.list[self.list[:,3].argsort()[::1]]
        self.aic_final=self.aic_listfits[:]                             # Get AIC weights normalizing from lowest to greatest
        for i in range(len(self.aic_listfits)):
            self.aic_final[i,3]=np.exp(-(self.aic_listfits[i,3]-np.amin(self.list[:,[3]]))/2.)


    # Order the list by decreasing AIC weight
    def ordered(self):
        return self.aic_final

    # Final average, error taken from Eq. 18 of https://arxiv.org/pdf/2008.01069.pdf, where <a> represents the value given by a model, averaged over the jackknife sample, or simply the error given by the param errors
    def avgval0(self):         
        val0=0
        eval0=0
        norm=0
        for i in range(len(self.aic_listfits)):
            norm+=self.aic_final[i,3]
            val0+=self.aic_final[i,3]*self.aic_final[i,5].values[0]
            eval0+=self.aic_final[i,3]*(self.aic_final[i,5].values[0]**2+self.aic_final[i,5].errors[0]**2)
        val0/=norm
        eval0/=norm
        return(val0, np.sqrt(eval0-val0**2))    
    

    # Model selection  instead, it just propagates the best model according to the MA criterion chosen 
    def selval0(self):         
        val0=self.aic_final[0,5].values[0]
        eval0=(self.aic_final[0,5].values[0]**2+self.aic_final[0,5].errors[0]**2)
        return(val0, np.sqrt(eval0-val0**2))     



    # Final average, error taken from Eq. 18 of https://arxiv.org/pdf/2008.01069.pdf, where <a> represents the value given by a model, averaged over the jackknife sample, or simply the error given by the param errors
    def avgval(self):         
        lenpar=len(self.aic_final[0,4])-1
           
        valf = []
        errf = [] 
        for i in range(lenpar):
            lenl=len(self.aic_listfits)
            val=0
            eval=0
            norm=0
            for k in range(lenl):
                if (len(self.aic_final[k,5].values)>i):
                    a  = self.aic_final[k,5].values[i]
                    ea = self.aic_final[k,5].errors[i]
                    norm += self.aic_final[k,3]
                    val  += self.aic_final[k,3]*a
                    eval += self.aic_final[k,3]*(a**2+ea**2)
            val/=norm
            eval/=norm

            err   =np.sqrt((eval-val**2))

            valf.append(val)
            errf.append(err)

        corr = np.zeros([lenpar,lenpar])
        for i in range(lenpar):
              for j in range(lenpar):  
                cov=0
                norm=0
                vala=0
                evala=0
                valb=0
                evalb=0
                for k in range(lenl):
                    if (len(self.aic_final[k,5].values)>max(i,j)):
                        a  = self.aic_final[k,5].values[i]
                        b  = self.aic_final[k,5].values[j]
                        norm += self.aic_final[k,3]
                        cov  += self.aic_final[k,3]*(np.array(self.aic_final[k,5].covariance)[i,j]+a*b)
                        vala += self.aic_final[k,3]*a
                        ea = self.aic_final[k,5].errors[i]
                        evala += self.aic_final[k,3]*(a**2+ea**2)
                        valb += self.aic_final[k,3]*b
                        eb = self.aic_final[k,5].errors[j]
                        evalb += self.aic_final[k,3]*(b**2+eb**2)
                vala/=norm
                evala/=norm
                valb/=norm
                evalb/=norm
                erra  =np.sqrt((evala-vala**2))
                errb  =np.sqrt((evalb-valb**2))
                cov/=norm
                cov-=vala*valb
                corr[i,j]=cov/(erra*errb)
        
        return (valf[:],errf[:], corr)   


    # Model selection instead, it just propagates the best model according to the MA criterion chosen 
    def selval(self):
        lenpar=len(self.aic_final[0,4])-1      
  
        val=[]
        err=[]
        for i in range(lenpar):
            if (self.aic_final[0,4][i+1]!=0):
                val.append(self.aic_final[0,5].values[i])
                err.append(self.aic_final[0,5].errors[i])

        corr=np.array(self.aic_final[0,5].covariance)
        for i in range(len(corr)):
            for j in range(len(corr)):
                corr[i,j]/=(err[i]*err[j])
        return(val[:],err[:],corr) 
 


class Jackknife_AIClist:
    
    def __init__(self, list):
        self.list=list
        self.nl=list[0,5]
        self.aic_listfits = self.list[self.list[:,3].argsort()[::1]]
        self.aic_final=self.aic_listfits[:]                             # Get AIC weights normalizing from lowest to greatest


        for i in range(len(self.aic_listfits)):
            self.aic_final[i,3]=np.exp(-(self.aic_listfits[i,3]-np.amin(self.list[:,[3]]))/2.)


    # Order the list by decreasing AIC weight
    def ordered(self):
        return self.aic_final

    # Final average, error taken from Eq. 18 of https://arxiv.org/pdf/2008.01069.pdf, where <a> represents the value given by a model, averaged over the jackknife sample, or simply the error given by the param errors
    def avgval0(self,cutoff,pos=1):         
        val0=0
        eval0=0
        norm=0
        totalw=0
        index=[]
        for i in range(len(self.aic_listfits)):
            a=jackknife(self.aic_final[i,4][:,pos]).pup()
            if (np.sum(a)!=0):
                totalw+=self.aic_final[i,3]
                index.append(i)

        trimmed=np.array([self.aic_final[x] for x in index])
        for i in range(len(trimmed)):
            if (i==0 or np.sum(trimmed[:,3][0:i+1])/totalw<=1-cutoff):
                a=jackknife(trimmed[i,4][:,pos]).pup()
                val0+=trimmed[i,3]*jackknife(a).fmean()
                eval0+=trimmed[i,3]*jackknife(a**2).fmean()
                norm+=trimmed[i,3]

        val0/=norm
        eval0/=norm
        return (val0, np.sqrt(eval0-val0**2))    


    # Create two Jackknife samples from model averaging, one rescaled and one without rescaling
    def avgsample(self,cutoff,pos):  
        val0   = 0 
        eval0  = 0
        sample = 0
        Nj     = len(self.aic_final[0,4][:,pos])
        norm   = 0
        totalw = 0
        index=[]
        for i in range(len(self.aic_listfits)):
            a=jackknife(self.aic_final[i,4][:,pos]).pup()
            if (np.sum(a)!=0):
                totalw+=self.aic_final[i,3]
                index.append(i)

        trimmed=np.array([self.aic_final[x] for x in index])
        for i in range(len(trimmed)): 
            if (i==0 or np.sum(trimmed[:,3][0:i+1])/totalw<=1-cutoff):             
                a       = jackknife(trimmed[i,4][:,pos]).pup()
                val0   += trimmed[i,3]*jackknife(a).fmean() 
                eval0  += trimmed[i,3]*jackknife(a**2).fmean()
                sample += trimmed[i,3]*trimmed[i,4][:,pos]
                norm   += trimmed[i,3]

        val0/=norm
        eval0/=norm
        sample/=norm
        samplef          = sample    
        samplef_rescaled = jackknife(sample).rup((eval0-val0**2)/(Nj-1))    # Rescales the MA Jackknife to have the same variance as the MA formula from the paper above. The explanation will be outlined in the notes
        samplef_rescaled_gaussian = add_gaussian_noise(sample,(eval0-val0**2))
        return samplef, samplef_rescaled, samplef_rescaled_gaussian
        

    # Model selection  instead, it just propagates the best model according to the MA criterion chosen 
    def selval0(self):         
        val0=np.mean(self.aic_final[0,4][:,1])
        eval0=(np.mean(self.aic_final[0,4][:,1])**2+np.cov(self.aic_final[0,4][:,1])*(self.nl-1))
        return (val0, np.sqrt(eval0-val0**2))     
    

    # Create a Jackknife sample from model selection 
    def selsample(self,pos):         
        sample=self.aic_final[0,4][:,pos]
        return sample  


    # Final average, error taken from Eq. 18 of https://arxiv.org/pdf/2008.01069.pdf, where <a> represents the value given by a model, averaged over the jackknife sample, or simply the error given by the param errors
    def avgval(self,cutoff):  
        lenpar=len(self.aic_final[0,4][0])-1
        
        Nj   = len(self.aic_final[0,4][:,1]) 
        corr = np.zeros([lenpar,lenpar])
        cov  = np.zeros([lenpar,lenpar])   
        valf = []
        vals = []
        errf = [] 
        for i in range(lenpar):
            lenl=len(self.aic_listfits)
            val=0
            eval=0
            norm=0
            totalw=0
            index=[]
            for k in range(lenl):
                a=jackknife(self.aic_final[k,4][:,i+1]).pup()
                if (np.sum(a)!=0):
                    totalw += self.aic_final[k,3]
                    index.append(k)

            trimmed=np.array([self.aic_final[x] for x in index])
            lent=len(trimmed)
            for k in range(lent):
                a=jackknife(trimmed[k,4][:,i+1]).pup()
                cumu_sum=np.sum(trimmed[:,3][0:k+1])
                if (k==0 or cumu_sum/totalw<=1-cutoff):
                    norm += trimmed[k,3]
                    val  += trimmed[k,3]*a
                    eval += trimmed[k,3]*jackknife(a**2).fmean()             #Equal to (jackknife(a).fmean()**2+jackknife(a).cov())
            val/=norm
            eval/=norm

            err   = np.sqrt((eval-jackknife(val).fmean()**2))
            val_rescaled   = jackknife(val).rup(err**2/(Nj-1))
            vals.append(val_rescaled)
            valf.append(jackknife(val).fmean()[0])
            errf.append(err[0])

        for i in range(lenpar):
              for j in range(lenpar):  
                norm=0
                totalw=0
                index=[]
                for k in range(lenl):
                    a=jackknife(self.aic_final[k,4][:,i+1]).pup()
                    b=jackknife(self.aic_final[k,4][:,j+1]).pup()
                    if (np.sum(a)!=0 and np.sum(b)!=0):
                        totalw += self.aic_final[k,3]
                        index.append(k)
                        
                trimmed=np.array([self.aic_final[x] for x in index])
                lent=len(trimmed)
                # This should (provided no mistakes) implement the proper non-diagonal covariance using MA
                for k in range(lent):
                    a=jackknife(trimmed[k,4][:,i+1]).pup()
                    b=jackknife(trimmed[k,4][:,j+1]).pup()
                    cumu_sum=np.sum(trimmed[:,3][0:k+1])
                    if (k==0 or cumu_sum/totalw<=1-cutoff):
                        norm += trimmed[k,3]
                        cov[i,j]+=trimmed[k,3]*jackknife(a*b).fmean()[0]
                        
                cov[i,j]/=norm
                cov[i,j]-=valf[i]*valf[j]
                corr[i,j]=cov[i,j]/(errf[i]*errf[j])

                if (i==j):
                    corr[i,j]=1.

        corr=(corr+np.transpose(corr))/2.
        
        return (valf[:],errf[:], corr)    
    

    # Model selection  instead, it just propagates the best model according to the MA criterion chosen 
    def selval(self): 
        lenpar=len(self.aic_final[0,4][0])-1

        corr=np.zeros([lenpar,lenpar])  
        val=[]
        err=[]
        for i in range(lenpar):
            val.append(jackknife(self.aic_final[0,4][:,i+1]).fmean()[0])
            eval=(jackknife(self.aic_final[0,4][:,i+1]).upcov())
            err.append(math.sqrt(eval))

            for j in range(lenpar):  
                if (i==j):
                    corr[i,j]=1 
                elif(np.mean(self.aic_final[0,4][:,i+1])==0 or np.mean(self.aic_final[0,4][:,j+1])==0):
                    corr[i,j]=0
                else:
                    corr[i,j]=np.corrcoef(self.aic_final[0,4][:,i+1],self.aic_final[0,4][:,j+1])[0,1]

        corr=(corr+np.transpose(corr))/2.
         
        return (val[:],err[:], corr)      