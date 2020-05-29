from __future__ import division
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 09:36:50 2018

@author: User1
"""
import numpy as np
import sys
sys.path.append("C:/Users/User1/Google Drive/Philippe/Python/Reservoir_computing/code/demo_simulations")
import network
import matplotlib.pyplot as plt
from scipy import stats
import os
from scipy.signal import iirfilter, lfilter
import scipy.ndimage.filters as filters


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

def bandpass_filter(data, cutoff, fs):
    order = 1
    nyq  = fs/2.0
    cutoff = cutoff/nyq
    b, a = iirfilter(order, cutoff, btype='band',
                     analog=False, ftype='butter')
    filtered_data = lfilter(b, a, data)
    return filtered_data    

def downsample(spks,ds_factor,dt,nb_steps,sigma=0):
    N = spks.shape[0]
    nb_steps_ds = int(np.ceil(nb_steps/ds_factor))
    spikes_ds = np.zeros((N,nb_steps_ds))
    for N_i in range(N):
        N_sp = np.array(np.where(spks[N_i,:])[0])
        N_sp = np.floor(N_sp/ds_factor).astype(int)
        for sp_time in N_sp:
            spikes_ds[N_i,sp_time] += 1
    if sigma > 0:
        spikes_ds = filters.gaussian_filter1d(spikes_ds,sigma=sigma) 
    return spikes_ds

#Sim params
output_path = "outputs/fig4/"
ensure_dir(output_path)


#Network parameters
dt = 5e-05
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
A = 3



#mean_GE = 0.005
#mean_GI = mean_GE*8#0.04#3              #0.055 Conductance (0.001 = 1pS)1.5
fs = np.int(1/dt)
ITonic = 9
#td = 0.04#0.02#0.05 #
#tr = 0.004#0.002#0.01 #
G = 1
gNoise = 0
nb_epochs = 15


#Input parameters
osc_range = [5,7]
N_in = 3
start_stim = 0.5
t_stim = int(start_stim/dt)
p_in = 0.5


#Target function
T = 1.5
nt = int(T/dt)
len_stim = T-start_stim
n_step_stim = int(len_stim/dt)
sigma = 30
lp = 6



#training params
alpha = dt*0.1
step = 50
train_start = int(np.round(start_stim/dt))
rs_factor_list = np.concatenate((np.linspace(0.1,1,10),np.array([1.1,1.3,1.5,1.7,1.9,2,2.3,2.5,2.7,3])),axis=0)
nb_rs = len(rs_factor_list)
max_nt = int(nt*rs_factor_list[-1])
output_rs = np.zeros((nb_rs,max_nt))
target = np.zeros(nt)
#Gaussian process
target[t_stim:] = lowpass_filter(np.random.randn(n_step_stim)*sigma,lp,fs)
net = network.net(N,pNI,mean_delays,tref,G=G,p=p, mean_GE = mean_GE, mean_GI = mean_GI, ITonic=ITonic)
   

osc_periods = np.random.uniform(osc_range[0],osc_range[1],N_in)
input_res = np.zeros((N_in,nt))
if N_in > 0:
    N_in_net = int(np.round(p_in*N))
    temp_w_in = np.abs(np.multiply(np.random.normal(0,1,(N,N_in)),np.random.rand(N,N_in)<p_in))
    w_res = np.zeros((N,N_in))
    w_res = temp_w_in

    net.w_res = w_res
    phase = np.random.uniform(0,1,N_in)*np.pi 

    for inp_cell in range(N_in):
        input_res[inp_cell,t_stim:t_stim+n_step_stim] =  A*(np.sin(2*np.pi*osc_periods[inp_cell]*(np.linspace(0,len_stim,n_step_stim))+phase[inp_cell]) + 1)/2   

            


N = net.N
output = np.zeros((nb_epochs,nt))
Pinv = np.eye(net.NE)*alpha
BPhi = np.zeros(net.NE)
data = {}
delta_per_epoch = int(np.floor(nt/step))
   
total_iEx = np.zeros(nt)
total_iIn = np.zeros(nt)
total_iInp = np.zeros(nt)

for ep_i in range(nb_epochs):
    #Simulation variables
    print('Epoch {}/{}.'.format(ep_i+1,nb_epochs))
    r = np.zeros(net.NE)
    hr = np.zeros(net.NE)                       
    gEx = np.zeros(N)                                            #Conductance of excitatory neurons
    gIn = np.zeros(N)                                            #Conductance of excitatory neurons
    F = np.full(N,np.nan)                                               #Last spike times of each inhibitory cells
    V = np.random.normal(net.mean_VRest,abs(0.04*net.mean_VRest),N)   #Set initial voltage Exc 
 
    sparse_mat = np.zeros((N,nt))
    t_train_i = 0
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
        Inp_current = np.dot(net.w_res,input_res[:,t])
        noise = np.random.normal(0,gNoise,N)
        dV_res = ((net.VRest-V) + E_current + I_current + Inp_current + noise + net.ITonic)                                     #Compute raw voltage change
        V = V + (dV_res * (dt/net.tau))                                        #Update membrane potential based on tau
        

        r = r*np.exp(-dt/net.tr) + hr*dt
        hr = hr*np.exp(-dt/net.td) + np.divide(V[net.E_idx]>=net.Theta[net.E_idx],net.tr*net.td)        
        z = np.dot(BPhi.T,r)
        output[ep_i,t] = z
        err = z-target[t]
                
        #RLMS
        if t >= train_start and (ep_i+1) < nb_epochs:
            if t%step == 1:
                cd = np.dot(Pinv,r)
                BPhi = BPhi-(cd*err)
                Pinv = Pinv - np.divide(np.outer(cd,cd.T),1 + np.dot(r.T,cd))

                
        #Update cells
        Refract = t <= (F + net.Refractory)
        V[Refract] = net.VRest[Refract]                                           #Hold resting potential of neurons in refractory period            
        spikers = np.where(V > net.Theta)[0]
        F[spikers] = t                                                              #Update the last AP fired by the neuron
        V[spikers] = 0                                                             #Membrane potential at AP time
        sparse_mat[spikers,t] = 1 
        total_iEx[t] = np.sum(E_current)
        total_iIn[t] = np.sum(I_current)
        total_iInp[t] = np.sum(Inp_current)
    
print('Running rescaled inputs.')
for rs_i in range(nb_rs):
    rs_factor = rs_factor_list[rs_i]
    osc_periods2 = np.array(osc_periods)/rs_factor
    n_step_stim_rs = int(n_step_stim*rs_factor)
    len_stim_rs = len_stim*rs_factor
    nt_rs = t_stim + n_step_stim_rs
    #Generate rescaled input
    input_res2 = np.zeros((N_in,nt_rs))
    
    for inp_cell in range(N_in):
        input_res2[inp_cell,t_stim:t_stim+n_step_stim_rs] =  A*(np.sin(2*np.pi*osc_periods2[inp_cell]*(np.linspace(0,len_stim_rs,n_step_stim_rs))+phase[inp_cell]) + 1)/2   
        
    
    r = np.zeros(net.NE)
    hr = np.zeros(net.NE)                       
    gEx = np.zeros(N)                                            #Conductance of excitatory neurons
    gIn = np.zeros(N)                                            #Conductance of excitatory neurons
    F = np.full(N,np.nan)                                               #Last spike times of each inhibitory cells
    V = np.random.normal(net.mean_VRest,abs(0.04*net.mean_VRest),N)   #Set initial voltage Exc 
 
    t_train_i = 0
    for t in range(nt_rs):
        #Conductuances decay exponentially to zero
        gEx = np.multiply(gEx,np.exp(-dt/net.TauE))
        gIn = np.multiply(gIn,np.exp(-dt/net.TauI))
    
        #Update conductance 
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
        Inp_current = np.dot(net.w_res,input_res2[:,t])
        noise = np.random.normal(0,gNoise,N)
        dV_res = ((net.VRest-V) + E_current + I_current + Inp_current + noise + net.ITonic)                                     #Compute raw voltage change
        V = V + (dV_res * (dt/net.tau))                                        #Update membrane potential based on tau
        
    
        r = r*np.exp(-dt/net.tr) + hr*dt
        hr = hr*np.exp(-dt/net.td) + np.divide(V[net.E_idx]>=net.Theta[net.E_idx],net.tr*net.td)        
        z = np.dot(BPhi.T,r)
        output_rs[rs_i,t] = z
                
                
        #Update cells
        Refract = t <= (F + net.Refractory)
        V[Refract] = net.VRest[Refract]                                           #Hold resting potential of neurons in refractory period            
        spikers = np.where(V > net.Theta)[0]
        F[spikers] = t                                                              #Update the last AP fired by the neuron
        V[spikers] = 0                                                             #Membrane potential at AP time

#Panel c
targets_rs = np.zeros((nb_rs,max_nt))
rs_nt_list = np.zeros(nb_rs)

error_rs = np.zeros(nb_rs)
for rs_i in range(nb_rs):
    rs_factor = rs_factor_list[rs_i]
    n_step_stim_rs = int(n_step_stim*rs_factor)
    nt_rs = t_stim + n_step_stim_rs
    rs_nt_list[rs_i] = nt_rs
    res_stim_idx = np.floor(np.linspace(0,len(target)-t_stim-1,n_step_stim_rs)).astype(np.int)
    targets_rs[rs_i,t_stim:nt_rs] = target[res_stim_idx+t_stim]
    error_rs[rs_i] = stats.pearsonr(targets_rs[rs_i,t_stim:nt_rs],output_rs[rs_i,t_stim:nt_rs])[0]

plt.figure()
plt.ylabel('Pearson r')
plt.xlabel('Rescaling factor')
plt.plot(rs_factor_list,error_rs)
plt.savefig(output_path+'panel_c.png')


#Panel b
width = 15
height = 8

plt_colors = ['1100E5','3314B7','451EA0','683372','793D5B','9C522D','BF6700']
plt_colors = [tuple(np.array(tuple(int(x[i:i+2], 16) for i in (0, 2 ,4)))/255) for x in plt_colors]

xaxis = np.linspace(0,T*rs_factor_list[-1],max_nt)
rs_plot = [2,4,7,9,11,13,15]
nb_plot = len(rs_plot)


plt.figure(figsize=(width,height))
for i in range(nb_plot):
    rs_i = rs_plot[i]
    rs_factor = rs_factor_list[rs_i]
    stim_len = nt-t_stim
    res_stim_len = int(stim_len*rs_factor)
    offset = 5*i
    plt.plot(xaxis[:t_stim],offset+target[:t_stim],color='k',linestyle='--',alpha=0.5)
    plt.plot(xaxis[t_stim:res_stim_len+t_stim],offset+targets_rs[rs_i,t_stim:res_stim_len+t_stim],'k',linestyle='--',alpha=0.5)
    plt.plot(xaxis[:res_stim_len+t_stim],offset+output_rs[rs_i,:t_stim+res_stim_len],linewidth=2.5,color=plt_colors[i],label=np.round(rs_factor,1))
    plt.xlabel('Time (s)', fontsize=20)
    plt.ylabel('Rescaling factor', fontsize=25)
    plt.tick_params(labelsize=17)
plt.legend()
plt.savefig(output_path+'panel_b.png')
