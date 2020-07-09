# -*- coding: utf-8 -*-
import numpy as np
import network
import matplotlib.pyplot as plt
import os
from scipy.signal import iirfilter, lfilter
from scipy import stats
import pickle
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

np.random.seed(1)

def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)
        
def lowpass_filter(data, cutoff, fs):
    order = 3
    nyq  = fs/2.0
    cutoff = cutoff/nyq
    b, a = iirfilter(order, cutoff, btype='lowpass',
                     analog=False, ftype='butter')
    filtered_data = lfilter(b, a, data)
    return filtered_data    

#Sim params
output_path = "outputs/supp_fig1/"
ensure_dir(output_path)
T = 1.5                     #Time in seconds
dt = 5e-05                  #Integration time step
nt = int(T/dt)
nb_epochs = 11              #Number of epochs (divide by two for training and testing)


#Network parameters
C = 2e-08                   #Capacitances (farads)
R = 1e+06                   #Resistance (Ohms)
tau = C*R                   #Membrane time constant (seconds)
N = 2000                    #Number of neurons
pNI = 0.2                   #Proportion of inhibitory cells
mean_delays = 0.001/dt      #Average conduction delay
mean_GE = 0.02              #Average excitatory conductance
mean_GI = 0.16              #Average inhibitory conductance
tref = 2e-03/dt             #Average refractory period
p = 0.1                     #Connection density
ITonic = 9                  #Tonic input
G = 1



#Input parameters
N_in = 3                    #Number of input units
start_stim = 0.5            #Starting time of the input
t_stim = int(start_stim/dt)
p_in = 0.3                  #Connection probability from input unit to reservoir units
A = 3                       #Input amplitude (1 = 10 pA)
osc_frequencies = [2,5]         #Input frequencies in Hz
osc_periods = np.random.uniform(osc_frequencies[0],osc_frequencies[1],N_in)

#Target function
td = 0.06                   #Readout decay time constant 
tr = 0.006                  #Readout rising time constant
len_stim = T-start_stim
n_step_stim = int(len_stim/dt)
sigma = 30
lp = 6                      #Cutoff frequency
target = np.zeros(nt)
fs = np.int(1/dt)
nb_tests = 5

targets = np.zeros((nb_tests,nt))
for i in range(nb_tests):
    targets[i,t_stim:] = lowpass_filter(np.random.randn(n_step_stim)*sigma,lp,fs) #Low-pass filtered white noise

#training params
alpha = dt*0.1
step = 50
train_start = int(np.round(start_stim/dt))


#gain_noise_list = np.array([0,0.1,0.3,0.5,1,2,3,4,5,7,10,15,20,30])
gain_noise_list = np.array([0,1,7,30])
nb_noise = len(gain_noise_list)
output = np.zeros((nb_epochs,nt,nb_tests,nb_noise))


input_reg = np.zeros((nb_noise,nt,N_in))
for noise_i in range(nb_noise):
    gNoise = gain_noise_list[noise_i]
    for test_i in range(nb_tests):
        #Initialize network
        net = network.net(N,pNI,mean_delays,tref,G=G,p=p, mean_GE = mean_GE, mean_GI = mean_GI, ITonic=ITonic,tau=tau,tr=tr,td=td)
        net.w_res = np.multiply(np.random.normal(0,1,(N,N_in)),np.random.rand(N,N_in)<p_in)     #Input weights (M in methods)
        target = targets[test_i,:]
        phase = np.random.uniform(0,1,N_in)*np.pi*2     #Initial phase configuration of the oscillators
        input_res = np.zeros((N_in,nt)) #Input to the reservoir
        Pinv = np.eye(net.NE)*alpha
        BPhi = np.zeros(net.NE)
        for inp_cell in range(N_in):
            input_res[inp_cell,t_stim:t_stim+n_step_stim] =  A*(np.sin(2*np.pi*osc_periods[inp_cell]*(np.linspace(0,len_stim,n_step_stim)+phase[inp_cell])) + 1)/2   
        
        for ep_i in range(nb_epochs):
            
            #Simulation variables
            print("Running noise {}/{}, test {}/{}, epoch {} of {}.".format(noise_i+1,nb_noise,test_i+1,nb_tests,ep_i+1,nb_epochs))
            r = np.zeros(net.NE)
            hr = np.zeros(net.NE)                       
            gEx = np.zeros(N)                                                   #Conductance of excitatory neurons
            gIn = np.zeros(N)                                                   #Conductance of inhibitory neurons
            F = np.full(N,np.nan)                                               #Last spike times of each inhibitory cells
            V = np.random.normal(net.mean_VRest,abs(0.04*net.mean_VRest),N)     #Set initial voltage

            inp_net = input_res

        
            for t in range(nt):
                #Conductuances decay exponentially to zero
                gEx = np.multiply(gEx,np.exp(-dt/net.TauE))
                gIn = np.multiply(gIn,np.exp(-dt/net.TauI))
            
                #Update conductance of postsyn neurons
                F_E = np.all([[t-F[net.E_idx]==net.delays[net.E_idx]],[F[net.E_idx] != 0]],axis = 0,keepdims=0)
                SpikesERes = net.E_idx[F_E[0,:]]          #If a neuron spikes x time-steps ago, activate post-syn 
                if len(SpikesERes ) > 0:
                    gEx = gEx + np.multiply(net.GE,np.sum(net.W[:,SpikesERes],axis=1))  #Increase the conductance of postsyn neurons
                F_I = np.all([[t-F[net.I_idx]==net.delays[net.I_idx]],[F[net.I_idx] != 0]],axis = 0,keepdims=0)
                SpikesIRes = net.I_idx[F_I[0,:]]        
                if len(SpikesIRes) > 0:
                    gIn = gIn + np.multiply(net.GI,np.sum(net.W[:,SpikesIRes],axis=1))  #Increase the conductance of postsyn neurons
                         
                #Leaky Integrate-and-fire
                E_current = np.multiply(gEx,net.RE-V)
                I_current = np.multiply(gIn,net.RI-V)
                input_act = inp_net[:,t]  + np.random.normal(0,gNoise,N_in) 
                if ep_i == 0 and test_i == 0:
                    input_reg[noise_i,t,:] = input_act
                Inp_current = np.dot(net.w_res,input_act)
                
                dV_res = ((net.VRest-V) + E_current + I_current + Inp_current + net.ITonic)/net.tau
                V = V + (dV_res * (dt))      #Update membrane potential 
        
        
                r = r*np.exp(-dt/net.tr) + hr*dt
                hr = hr*np.exp(-dt/net.td) + np.divide(V[net.E_idx]>=net.Theta[net.E_idx],net.tr*net.td)        
                z = np.dot(BPhi.T,r)
                output[ep_i,t,test_i,noise_i] = z
                err = z-target[t]
                        
                #RLMS
                if t >= train_start and ep_i < nb_epochs-1:    
                    if t%step == 1:
                        cd = np.dot(Pinv,r)
                        BPhi = BPhi-(cd*err)
                        Pinv = Pinv - np.divide(np.outer(cd,cd.T),1 + np.dot(r.T,cd))
                        
                #Update cells
                Refract = t <= (F + net.Refractory)
                V[Refract] = net.VRest[Refract]                    #Hold resting potential of neurons in refractory period            
                spikers = np.where(V > net.Theta)[0]
                F[spikers] = t                                     #Update the last AP fired by the neuron
                V[spikers] = 0                                     #Membrane potential at AP time
        
                
        
results = {'outputs':output,'targets':targets}
       
with open(output_path+'output_panel_e.p', 'wb') as f:                
    pickle.dump(results,f)
    
with open(output_path+'output_panel_e.p', 'rb') as f:                
    results = pickle.load(f)
    
output = results['outputs']
targets = results['targets']
    
corr_cond = np.zeros((nb_noise,nb_tests))
for noise_i in range(nb_noise):
    for test_i in range(nb_tests):
        corr_cond[noise_i,test_i] = stats.pearsonr(output[-1,t_stim:,test_i,noise_i],targets[test_i,t_stim:])[0]

#Plot results
avg_corr = np.mean(corr_cond,axis=1)
sem_corr = stats.sem(corr_cond,axis=1)
plt.figure(figsize=(6,3))
plt.errorbar(gain_noise_list,avg_corr,yerr=sem_corr,color='k')
plt.xlabel('Sigma')
plt.ylabel('Pearson r')
plt.savefig(output_path+'panel_e.png')
   