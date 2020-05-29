# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np

        
class net(object):

    def __init__(self,N,pNI,mean_TranDelay,mean_Refractory,G=1,
                 p=0.1, mean_GE = 0.02, mean_GI = 0.16, ITonic=9,
                 mean_TauFall_I=0.02,tau=0.02,tr=0.006,td=0.06,GaussSD=0.02):
        self.N = N  #Number of neurons
        self.NI = int(self.N*pNI) #N inhibitory
        self.NE = self.N-self.NI #N excitatory
        self.E_idx = np.arange(0,self.NE)
        self.I_idx = np.arange(self.NE,self.N)
        self.tau = tau
        #SINGLE CELL PARAMETERS
        mean_EE = 0				    # reversal potential (mV) 
        mean_EI = -80 				#reversal potential (mV)
        self.mean_VRest = -60       #resting-state potential (mV)
        mean_Theta = -50	 		#spike threshold (mV) 
        mean_TauFall_E = 0.02	    #EPSP fall for excitatory cells (ms) 
        mean_TauFall_I = mean_TauFall_I #EPSP fall for inhibitory cells (ms)		
        GaussSD = GaussSD			    # Gaussian parameter (sigma)
        GaussTheta = 0.01
        
        
        self.td = td
        self.tr = tr
        
        self.delays = np.round(np.random.normal(mean_TranDelay,GaussSD*mean_TranDelay,(N))).astype(int) #Matrix of transmission delays from pre to post syn
        self.tau = abs(np.random.normal(self.tau,GaussSD*self.tau,(N)))                            #Set membrane capacitance                                               #Set membrane time constant
        self.GE = abs(np.random.normal(mean_GE,GaussSD*mean_GE,(N)))                          #Excitatory leaky conductance
        self.GI = abs(np.random.normal(mean_GI,GaussSD*mean_GI,(N)))                          #Inhibitory leaky conductance
        self.RE = np.ones(N) * mean_EE												          #Excitatory reversal potential (mV)
        self.RI = np.ones(N) * mean_EI           										       #Inhibitory reversal potential (mV)
        self.Theta = np.random.normal(mean_Theta,abs(GaussTheta*mean_Theta),(N))         #Spiking threshold
        self.VRest = np.random.normal(self.mean_VRest,abs(GaussSD*self.mean_VRest),(N))         #Resting potential
        self.TauE = np.random.normal(mean_TauFall_E,abs(GaussSD*mean_TauFall_E),(N))  #Time constant for exponential decay E
        self.TauI = np.random.normal(mean_TauFall_I,abs(GaussSD*mean_TauFall_I),(N))  #Time constant for exponential decay I
        self.ITonic = np.ones(N)*ITonic     #Constant input received by each cell
        self.Refractory = np.random.normal(mean_Refractory,abs(GaussSD*mean_Refractory),(N))      #Constant input received by each cell
        self.W = np.abs(G*np.multiply(np.random.normal(0,1,(N,N)),np.random.rand(N,N)<p)/(np.sqrt(N*p)))

