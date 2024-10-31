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
# Defining Minuit Classes
#------------------------------------------------------------------------------------------------------------------------------------------
class LeastSquares:
    """
    Generic least-squares cost function with cov.
    """

    errordef = Minuit.LEAST_SQUARES  # for Minuit to compute errors correctly

    def __init__(self, model, x, y, incov, local_min_pars=None, local_min_deriv=None):
        self.model = model  # model predicts y for given x
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.invcov = np.asarray(incov)
        self.local_min_pars=local_min_pars
        self.local_min_deriv=local_min_deriv


    def __call__(self, *par):  # we must accept a variable number of model parameters
        ym  = self.model(self.x, *par)
        incov_mat=self.invcov

        fun = np.dot(np.dot((self.y - ym), incov_mat),(self.y - ym))

        fun_fin=0
        if (self.local_min_pars!=None and  self.local_min_deriv!=None):     # Iterate to remove the local minima, one by one, in order
            #print(len(self.local_min_pars))
            eval_deriv = []
            eval_pars=[]
            for i in range(len(self.local_min_pars)):
                eval_deriv.append(np.dot(np.dot((par-self.local_min_pars[i]),self.local_min_deriv[i]),(par-self.local_min_pars[i])))
                eval_pars.append(self.local_min_pars[i])
            eval_pars.append(par)

            #print(len(eval_pars))
            eval_fun   = []
            if (np.min(eval_deriv)>1.e-100):
                for j in range(len(eval_pars)):
                    eval_int=[]
                    #print(eval_fun,eval_deriv,j)
                    for i in range(len(eval_pars)-j):
                        if (j==0):
                            ymint  = self.model(self.x, *eval_pars[i])
                            funint = np.dot(np.dot((self.y - ymint), incov_mat),(self.y - ymint))
                            eval_int.append(funint)
                        else:
                            #print(j,i)
                            #print(eval_deriv[j-1])
                            funint=(eval_fun[j-1][i+1]-eval_fun[j-1][i])/(eval_deriv[j-1])
                            eval_int.append(funint)
                    eval_fun.append(eval_int)

                fun_fin=eval_fun[len(self.local_min_pars)][0]
            else:
                #print(par);print(self.local_min_pars[0]);print(par-self.local_min_pars[0]);print(eval_deriv);print(self.local_min_deriv[0])

                fun_fin=fun

            
        else:
            fun_fin=fun

        return fun_fin
        
class BetterLeastSquares(LeastSquares):

    def __init__(self, model, x, y, incov, local_min_pars=None, local_min_deriv=None):
        super().__init__(model, x, y, incov, local_min_pars, local_min_deriv)
        pars = describe(model, annotations=True)
        model_args = iter(pars)
        next(model_args)
        _parameters = {k: pars[k] for k in model_args}


class EvenBetterLeastSquares(BetterLeastSquares):
    @property
    def ndata(self):
        return len(self.x)
    




#------------------------------------------------------------------------------------------------------------------------------------------
# Creating a fitting class for iminuit
#------------------------------------------------------------------------------------------------------------------------------------------
    
class Minuit_fit:

    def __init__(self, data_t, data_y, data_incov , calln, tol, inipars, ini_fun, jackk, improve):
        self.data_t=data_t
        self.data_y=data_y
        self.data_incov=data_incov
        self.ini=inipars
        self.calln=calln
        self.tol=tol
        self.jackk=jackk
        self.ini_fun=ini_fun
        self.improve=improve

    def minimize(self):

        inifun=eval(self.ini_fun)

        least_squares_np = EvenBetterLeastSquares(inifun, self.data_t, self.data_y, self.data_incov)
        m=Minuit(least_squares_np,*self.ini)   # pass starting values as a sequence

        if (self.calln < 1000):      # Make sure there is a decent amount of calls to Minuit, per subroutine
            ncall=1000
        else:
            ncall=self.calln

        if (self.tol > 0.001):        # Make sure tol is small, to get good fits, max value will be Minuit's/100 default value
            tolf=0.001
        else:
            tolf=self.tol
        m.tol=tolf                                        # low tolerance for precise minimum

        m.strategy=2

        mcalls=self.calln

        #rfit=m.scipy('L-BFGS-B',mcalls).migrad(mcalls).migrad(mcalls)
        #if (rfit.valid==False):
        rfit=m.simplex(mcalls).scipy('L-BFGS-B',mcalls).scipy('L-BFGS-B',mcalls).migrad(mcalls).migrad(mcalls)

        #.simplex(mcalls).scipy('CG',mcalls).simplex(mcalls).scipy('L-BFGS-B',mcalls).migrad(mcalls).migrad(mcalls)

        ntrials=self.improve                                # If self.improve>0 then it will try to improve upon a local minima using the old Improve subroutine from Migrad: https://www.semanticscholar.org/paper/On-descent-from-local-minima-Goldstein-Price/faec2d650b61bb4ddda1cc43ca0c35bea0c341ad?sort=pub-date&page=2
        if (rfit.valid and ntrials>0):
            local_min_pars=[np.array(rfit.values)]
            local_min_deriv=[improved_inverse_covariance(np.array(rfit.covariance))]
            result_improve=improve_fit(rfit,self.data_t,self.data_y,self.data_incov,self.calln,self.tol,self.ini_fun,local_min_pars,local_min_deriv,ntrials).fit()
            if (result_improve!=0):
                rfit=result_improve
            
         
         
        rfit_fin=rfit.hesse(ncall)                        # get proper covariance (redundant if we use jackknife)
        return rfit_fin     
         
    
class improve_fit:
    def __init__(self, ini_fit,data_t, data_y, data_incov , calln, tol, ini_fun, inipars, local_min_deriv, ntrials):
        self.data_t=data_t
        self.data_y=data_y
        self.data_incov=data_incov
        self.inipars=inipars
        self.calln=calln
        self.tol=tol
        self.ini_fun=ini_fun
        self.local_min_deriv=local_min_deriv
        self.ntrials=ntrials
        self.ini_fit=ini_fit

    
    def fit(self):
        local_min_pars=self.inipars
        local_min_deriv=self.local_min_deriv
        rfit=self.ini_fit
        inifun=eval(self.ini_fun)
            
        for k in range(self.ntrials):
            if (len(local_min_pars)>5):                         # If after finding 5 diff local minima, all of them are worse than the original one, we stop the procedure
                break
            least_squares_np2= EvenBetterLeastSquares(inifun, self.data_t, self.data_y, self.data_incov,local_min_pars,local_min_deriv)
            int_pars_ini=local_min_pars[0]*1.001
            m_imp=Minuit(least_squares_np2,*int_pars_ini)   # pass starting values as a sequence
            m_imp.tol=self.tol
            
            rfit_imp=m_imp.simplex(self.calln).scipy('L-BFGS-B',self.calln).migrad(self.calln).migrad(self.calln)
            
            if (rfit_imp.fval>=0):
                if (rfit_imp.valid==True):
                    local_min_pars.append(np.array(rfit_imp.values))
                    local_min_deriv.append(improved_inverse_covariance(np.array(rfit_imp.covariance)))
                else:                                       # We stop again if no improvement or local minima is found 
                    break

            elif (rfit_imp.fval<0):
                least_squares_np= EvenBetterLeastSquares(inifun, self.data_t, self.data_y, self.data_incov)
                inipars2=np.array(rfit_imp.values)
                
                m_imp=Minuit(least_squares_np,*inipars2)   # pass starting values as a sequence
                m_imp.tol=self.tol
                rfit_imp=m_imp.simplex(self.calln).scipy('L-BFGS-B',self.calln).migrad(self.calln).migrad(self.calln)
                if (rfit_imp.valid==True and (rfit_imp.fval-rfit.fval)/rfit.fval<-1.e-10):
                    local_min_pars=[np.array(rfit_imp.values)]
                    local_min_deriv=[improved_inverse_covariance(np.array(rfit_imp.covariance))]
                    rfit=rfit_imp
                else:                                        # We stop once again if the improvement is too small (probably just numerical instability) or if the improvement did not find an actual minima
                    break  

        return rfit