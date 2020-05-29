# -*- coding: utf-8 -*-
"""
Created on Wed May 23 09:36:50 2018

@author: User1
"""
import numpy as np
import network
import matplotlib.pyplot as plt
from datetime import datetime
import os
import pickle
from scipy.signal import iirfilter, lfilter
from matplotlib.collections import LineCollection

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

def adjust_sampling(coords,nt):
    current_nt = coords.shape[1]
    indices = np.round(np.linspace(0,current_nt-1,nt)).astype(np.int16)
    return coords[:,indices]


#Sim params
startTime = datetime.now()
output_path = "outputs/fig3/"
ensure_dir(output_path)

T = 0.8                     #Time in seconds
dt = 5e-05                  #Integration time step
nt = int(T/dt)
nb_epochs = 40              #Number of epochs (divide by two for training and testing)


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
N_in = 4                    #Number of input units
start_stim = 0.2            #Starting time of the input
t_stim = int(np.round(start_stim/dt))
p_in = 0.3                  #Connection probability from input unit to reservoir units
A = 5                       #Input amplitude (1 = 10 pA)
osc_frequencies = [1,3]         #Input frequencies in Hz


#Target function
len_stim = T-start_stim
n_step_stim = int(len_stim/dt)
sigma = 30



#Target function
path_out = "data/"
with open(path_out+'forms.p','rb') as f:
    targets_raw = pickle.load(f)
N_out = 2
N_tar = 2

target1 = targets_raw['circle']
target1_s = adjust_sampling(target1,n_step_stim)
target2 = targets_raw['star']
target2_s = adjust_sampling(target2,n_step_stim)
targets = np.zeros((N_out,nt,N_tar))
targets[:,t_stim:,0] = target1_s
targets[:,t_stim:,1] = target2_s


nb_epochs_train = 20
nb_epochs_test = 15
nb_epochs = nb_epochs_train + nb_epochs_test

   
for i in range(N_tar):
    targets[:,t_stim:,i] =targets[:,t_stim:,i]/np.max(targets[:,t_stim:,i],axis=1)[:,None]

#training params
alpha = dt*0.1
step = 50

net = network.net(N,pNI,mean_delays,tref,G=G,p=p, mean_GE = mean_GE, mean_GI = mean_GI, ITonic=ITonic)
osc_periods = np.random.uniform(osc_frequencies[0],osc_frequencies[1],N_in)


input_res_list = np.zeros((N_in,nt,N_tar+1))
if N_in > 0:
    N_in_net = int(np.round(p_in*N))
    input_res = np.zeros((N_in,nt)) #Input to the reservoir
    net.w_res = np.multiply(np.random.normal(0,1,(N,N_in)),np.random.rand(N,N_in)<p_in)     #Input weights (M in methods)
    
    for tar in range(N_tar+1):
        phase = np.random.uniform(0,1,N_in)*np.pi*2
        for inp_cell in range(N_in):
            input_res_list[inp_cell,t_stim:t_stim+n_step_stim,tar] =  A*(np.sin(2*np.pi*osc_periods[inp_cell]*(np.linspace(phase[inp_cell],len_stim+phase[inp_cell],n_step_stim))) + 1)/2   


                

Pinv = np.eye(net.NE)*alpha
BPhi = np.zeros((net.NE,N_out))
nb_tests_per_tar = int(nb_epochs_test/3)
output = np.zeros((N_out,nt,N_tar+1,nb_tests_per_tar))



for ep_i in range(nb_epochs):
   
    print('Running epoch {}/{}.'.format(ep_i+1,nb_epochs))
    if ep_i < nb_epochs_train:
        if ep_i %2 == 0:
            idx = 0
        else:
            idx = 1
        target = targets[:,:,idx]
    else:
        if ep_i %3 == 0:
            idx = 0
        elif ep_i % 3 == 1:
            idx = 1
        elif ep_i % 3 == 2:
            idx = 2
    
    
    input_res = input_res_list[:,:,idx]
    r = np.zeros(net.NE)
    hr = np.zeros(net.NE)                       
    gEx = np.zeros(N)                                            #Conductance of excitatory neurons
    gIn = np.zeros(N)                                            #Conductance of excitatory neurons
    F = np.full(N,np.nan)                                               #Last spike times of each inhibitory cells
    V = np.random.normal(net.mean_VRest,abs(0.04*net.mean_VRest),N)   #Set initial voltage Exc 
  
    sparse_mat = np.zeros((N,nt))
    t_train_i = 0
    for t in range(nt):
       
        gEx = np.multiply(gEx,np.exp(-dt/net.TauE))
        gIn = np.multiply(gIn,np.exp(-dt/net.TauI))
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
        dV_res = ((net.VRest-V) + E_current + I_current + Inp_current + net.ITonic)                                     #Compute raw voltage change
        V = V + (dV_res * (dt/net.tau))                                        #Update membrane potential based on tau
        

        r = r*np.exp(-dt/net.tr) + hr*dt
        
        hr = hr*np.exp(-dt/net.td) + np.divide(V[net.E_idx]>=net.Theta[net.E_idx],net.tr*net.td)        
        z = np.dot(BPhi.T,r)
        if ep_i >= nb_epochs_train:
            i = ep_i - nb_epochs_train + 1
            ep_idx = int(np.ceil(i/(N_tar+1)))-1
            output[:,t,idx,ep_idx] = z
        err = z-target[:,t]
                
        #RLMS
        if t >= t_stim and ep_i < nb_epochs_train:
            if t%step == 1:
                cd = np.dot(Pinv,r)
                Pinv = Pinv - np.divide(np.outer(cd,cd.T),1 + np.dot(r.T,cd))
                for ro in range(2):
                    BPhi[:,ro] = BPhi[:,ro]-(cd*err[ro])

        #Update cells
        Refract = t <= (F + net.Refractory)
        V[Refract] = net.VRest[Refract]                                           #Hold resting potential of neurons in refractory period            
        spikers = np.where(V > net.Theta)[0]
        F[spikers] = t                                                              #Update the last AP fired by the neuron
        V[spikers] = 0                                                             #Membrane potential at AP time
        
    




grad = np.linspace(0,len_stim*1000,n_step_stim)
norm = plt.Normalize(grad.min(), grad.max())

output_avg = np.mean(output,axis=3)
fig, axs = plt.subplots(1, 3,figsize=(12,4))
n_plots = N_tar+1

for i in range(n_plots):
    points = np.array([output_avg[0,t_stim:,-(n_plots-i)],output_avg[1,t_stim:,-(n_plots-i)]]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap='winter', norm=norm)
    # Set the values used for colormapping
    lc.set_array(grad)
    lc.set_linewidth(2)
    for j in range(nb_tests_per_tar):
        axs[i].plot(output[0,t_stim:,-(n_plots-i),j],output[1,t_stim:,-(n_plots-i),j],'k',alpha=0.4,linewidth=0.5)


    if i < 2:
        axs[i].plot(targets[0,t_stim:,i],targets[1,t_stim:,i],'k--')
    line = axs[i].add_collection(lc)

plt.savefig('../output/figure3/panel_c.png')
    