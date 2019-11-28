# -*- coding: utf-8 -*-
import numpy as np
import network
import os
import matplotlib.pyplot as plt
from scipy.signal import iirfilter, lfilter, hilbert
import scipy.ndimage.filters as filters
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

np.random.seed(10)

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

def downsample(spks,ds_factor,dt,nb_steps):
    N = spks.shape[0]
    nb_steps_ds = int(np.ceil(nb_steps/ds_factor))
    spikes_ds = np.zeros((N,nb_steps_ds))
    for N_i in range(N):
        N_sp = np.array(np.where(spks[N_i,:])[0])
        N_sp = np.floor(N_sp/ds_factor).astype(int)
        for sp_time in N_sp:
            spikes_ds[N_i,sp_time] += 1
    return spikes_ds

def conv_raster(spikes,sigma):
    spikes = filters.gaussian_filter1d(spikes,sigma=sigma,mode='constant') 
    return spikes

def zscore_raster(raster):
    n_rows = raster.shape[0]
    raster_z = np.zeros(raster.shape)
    for i in range(n_rows):
        row = raster[i,:]
        if np.mean(row) != 0:
            raster_z[i,:] =  (row-np.mean(row))/np.std(row)
    return raster_z

#Sim params
output_path = "outputs/fig6/"
ensure_dir(output_path)
nb_epochs = 1   #Only one lap with training
T = 5
dt = 5e-05
nt = int(T/dt)

#Network parameters (see figure1.py for more detailed comments)
C = 1e-08                   #Capacitances (farads)
R = 1e+06                   #Resistance (Ohms)
tau = C*R                   #Membrane time constant (seconds)
N = 1000
pNI = 0.2
mean_delays = 0.001/dt
mean_GE = 0.01
mean_GI = 0.16
fs = np.int(1/dt)
tref = 2e-03/dt
p = 0.1
ITonic = 9
td = 0.06
tr = 0.006
G = 1

#Input parameters
N_in = 20
osc_frequencies = np.linspace(7.5,8.5,N_in)     #Number of input oscillators (CA3)
start_stim = 0.5
t_stim = int(start_stim/dt)
p_in = 1
len_stim = T-start_stim
n_step_stim = int(len_stim/dt)

#Place cells parameters
N_place = 10
place_cells = np.arange(N_place)                            #The indices of the place cells
place_timing = np.linspace(1,3,N_place) + start_stim        #Time where each place cell is depolarized
t_timing = (place_timing/dt).astype(int)
sine_len = 0.6                                              #Length of the sine wave current associated iwth place fields
sine_nt = int(sine_len/dt)
sine_freq = 10                                              #Frequency of the sine depolarization
sine_I = 6                                                 #Amplitude of the input (1 = 10pA)
sine = (np.sin(2*np.pi*sine_freq*np.linspace(0,sine_len,sine_nt))+1)/2*sine_I

depo = np.zeros((N,nt))                                     #Matrix storing the environemental input to all place cells
for i in range(N_place):
    idx = np.arange(sine_nt).astype(int)+t_timing[i]
    depo[i,idx] = sine

#Initialize network
net = network.net(N,pNI,mean_delays,tref,G=G,p=p, mean_GE = mean_GE, mean_GI = mean_GI, ITonic=ITonic,tau=tau)
net.VRest[:N_place] = -60
net.W[net.E_idx,net.E_idx] = 0

#Input params
sigma_inp = 0.1
input_res = np.zeros((N_in,nt))
net.w_res = np.abs(np.multiply(np.random.normal(0,sigma_inp,(N,N_in)),np.random.rand(N,N_in)<p_in))
phase = np.random.uniform(0,1,N_in)*np.pi*2 
for inp_cell in range(N_in):
    input_res[inp_cell,t_stim:t_stim+n_step_stim] =  np.sin(2*np.pi*osc_frequencies[inp_cell]*np.linspace(0,len_stim,n_step_stim)+phase[inp_cell])



#Training parameters
gmax_in = sigma_inp*5      #Upper bound for input connections
alpha = 0.25                #Learning rate
burst_nb = 3
burst_width = int(0.05/dt)
spikes = np.zeros((N,nt,3))             #Store spikes for pre-training,training and testing

for ep_i in range(nb_epochs+2):
    #Simulation variables
    print("Running epoch {} of {}.".format(ep_i+1,nb_epochs+2))                     
    gEx = np.zeros(N)                                            #Conductance of excitatory neurons
    gIn = np.zeros(N)                                            #Conductance of excitatory neurons
    F = np.full(N,np.nan)                                               #Last spike times of each inhibitory cells
    V = np.random.normal(net.mean_VRest,abs(0.04*net.mean_VRest),N)   #Set initial voltage Exc 
    
    spk_hist = np.zeros((N_place,burst_width))                  #Store recent spikes
    burst_t = np.zeros(N_place)                                 #Store last burst

    for t in range(nt):
        if t%10000==0:
            print('{}/{} seconds'.format(np.round(t*dt,2),T))
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
        if ep_i < (nb_epochs+1) and ep_i > 0:   
            place_sine = depo[:,t]
        else:
            place_sine = np.zeros(N)
        dV_res = ((net.VRest-V) + E_current + I_current + Inp_current + place_sine + net.ITonic)/net.tau                                 
        V = V + (dV_res * (dt))                                        #Update membrane potential
        
        #Update cells
        Refract = t <= (F + net.Refractory)
        V[Refract] = net.VRest[Refract]                                           #Hold resting potential of neurons in refractory period            
        spikers = np.where(V > net.Theta)[0]
        F[spikers] = t                                                              #Update the last AP fired by the neuron
        V[spikers] = 0                                                             #Membrane potential at AP time
        
        spikes_t = V > net.Theta

        #Place cells activity
        spikersT = np.where(spikes_t[:N_place])[0]
        spk_hist = np.concatenate([spk_hist[:,1:],spikes_t[:N_place].reshape(N_place,1)],axis=1)
        nb_spks = np.sum(spk_hist,axis=1)               #Current number of spikes in the burst interval
        burst_cells = np.where(nb_spks>=burst_nb)[0]    #Index of bursting cells at time t
        last_burst = burst_t[burst_cells]               #Last burst timing
        non_ref = (t-last_burst)>=burst_width           #Only keep cells that haven't bursted in a duration = burst_width
        burst_cells = burst_cells[non_ref]              #Bursting cells for weights update
        
        if ep_i < (nb_epochs+1) and ep_i > 0:           #Only train for nb_epochs (first is pretraining and last is testing)    
            if len(burst_cells)>0:
                for cell in burst_cells:
                    net.w_res[cell,:] = net.w_res[cell,:] + alpha*input_res[:,t] 
                    burst_t[cell] = t
                    
            #Make sure the input weight stays within bounds
            net.w_res[net.w_res<0] = 0
            net.w_res[net.w_res>gmax_in] = gmax_in

        spikes[spikers,t,ep_i] = 1

#REPLAY
#Forward        
rs = 0.15 #Rescaling factor 
T_rs = start_stim + len_stim*rs
len_stim_rs = len_stim*rs
nt_rs = int(T_rs/dt) 
n_step_stim_rs = nt_rs-t_stim
osc_frequencies_rs = np.array(osc_frequencies)/rs
input_res_rs = np.zeros((N_in,nt_rs))
for inp_cell in range(N_in):
    input_res_rs[inp_cell,t_stim:] =  np.sin(2*np.pi*osc_frequencies_rs[inp_cell]*np.linspace(0,len_stim_rs,n_step_stim_rs)+phase[inp_cell])  


spikes_replay = np.zeros((N,nt_rs,2))                 #axis2: 0= forward replay, 1= reverse replay
#Simulation variables              
gEx = np.zeros(N)                                            #Conductance of excitatory neurons
gIn = np.zeros(N)                                            #Conductance of excitatory neurons
F = np.full(N,np.nan)                                               #Last spike times of each inhibitory cells
V = np.random.normal(net.mean_VRest,abs(0.04*net.mean_VRest),N)   #Set initial voltage Exc 


print('Generating compressed forward replay')
for t in range(nt_rs):
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
    Inp_current = np.dot(net.w_res,input_res_rs[:,t])

    dV_res = ((net.VRest-V) + E_current + I_current + Inp_current  + net.ITonic+0)/net.tau                                 
    V = V + (dV_res * (dt))                                        #Update membrane potential 

    #Update cells
    Refract = t <= (F + net.Refractory)
    V[Refract] = net.VRest[Refract]                                           #Hold resting potential of neurons in refractory period            
    spikers = np.where(V > net.Theta)[0]
    F[spikers] = t                                                              #Update the last AP fired by the neuron
    V[spikers] = 0                                                             #Membrane potential at AP time
    spikes_replay[spikers,t,0] = 1
    
    
#REVERSE
input_res_rs_rev = np.zeros((N_in,nt_rs))
input_res_rs_rev[:,t_stim:] = np.fliplr(input_res_rs[:,t_stim:])


#Simulation variables              
gEx = np.zeros(N)                                            #Conductance of excitatory neurons
gIn = np.zeros(N)                                            #Conductance of excitatory neurons
F = np.full(N,np.nan)                                               #Last spike times of each inhibitory cells
V = np.random.normal(net.mean_VRest,abs(0.04*net.mean_VRest),N)   #Set initial voltage Exc 


print('Generating compressed reverse replay')
for t in range(nt_rs):
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
    Inp_current = np.dot(net.w_res,input_res_rs_rev[:,t])

    dV_res = ((net.VRest-V) + E_current + I_current + Inp_current  + net.ITonic+0)/net.tau                                 
    V = V + (dV_res * (dt))                                        #Update membrane potential 

    #Update cells
    Refract = t <= (F + net.Refractory)
    V[Refract] = net.VRest[Refract]                                           #Hold resting potential of neurons in refractory period            
    spikers = np.where(V > net.Theta)[0]
    F[spikers] = t                                                              #Update the last AP fired by the neuron
    V[spikers] = 0                                                             #Membrane potential at AP time
    spikes_replay[spikers,t,1] = 1

#Plot params
width = 15
height = 8
xaxis = np.linspace(0,T,nt)

# =============================================================================
# plot training activity
# =============================================================================
#Show target cells
bin_size = 0.001                #Downsampling
ds_factor = bin_size*(1/dt)     #bin_size in ms
nb_steps_ds = int(np.ceil(nt/ds_factor))   
sigma_s = 0.10                  #Spikes convultion in s
sigma = sigma_s/bin_size
n_xt = 6
xticks_lab = np.round(np.linspace(0,5,n_xt),1).astype(int)
xticks = np.linspace(0,5*1e03,n_xt)
learning_phases = ['Pre-learning','Learning','Testing']
plt.figure(figsize=(15,8))
for ep_i in range(3):
    plt.subplot2grid((6,3),(0,ep_i),rowspan=4)
    plt.title(learning_phases[ep_i],fontsize=15)
    if ep_i == 0:
        plt.ylabel('Neuron #', fontsize=15)
    ds_spikes = downsample(spikes[:,:,ep_i],ds_factor,dt,nt)        #Downsample raster
    conv_spikes = conv_raster(ds_spikes,sigma)                      #Convolve raster
    zs_spikes = zscore_raster(conv_spikes) 
    plt.pcolor(zs_spikes[:N_place,:],cmap='jet',vmin=-1.5,vmax=2.5)
    plt.tick_params(labelsize=12)
    plt.xticks(xticks,xticks_lab)

    plt.subplot2grid((6,3),(4,ep_i),rowspan=2)
    [plt.plot(np.where(spikes[x,:,ep_i])[0]*dt,np.ones(len(np.where(spikes[x,:,ep_i])[0]))*(x),'k.',markersize=1) for x in range(N_place)]
    plt.tick_params(labelsize=12)
    plt.xlabel('Time (s)', fontsize=20)
    if ep_i == 0:
        plt.ylabel('Neuron #', fontsize=15)
    plt.ylim([0,N_place])
plt.savefig(output_path+'pannel_g.png')
    
# =============================================================================
# plot replay
# =============================================================================
#Show target cells
xticks_lab = np.round(np.linspace(0,T_rs,n_xt),1)
xticks = np.linspace(0,T_rs,n_xt)
replay_type =  ['Forward','Reversed']
plt.figure(figsize=(10,8))
for ep_i in range(2):
    plt.subplot2grid((6,2),(0,ep_i),rowspan=4)
    plt.title(replay_type[ep_i],fontsize=15)
    if ep_i == 0:
        plt.ylabel('Neuron #', fontsize=15)
                   
    ds_spikes = downsample(spikes_replay[:,:,ep_i],ds_factor,dt,nt_rs)
    conv_spikes = conv_raster(ds_spikes,sigma) 
    zs_spikes = zscore_raster(conv_spikes)                   
    plt.pcolor(zs_spikes[:N_place,:],cmap='jet',vmin=-1.5,vmax=2.5)
    plt.tick_params(labelsize=12)
    plt.xticks(xticks,xticks_lab)

    plt.subplot2grid((6,2),(4,ep_i),rowspan=2)
    [plt.plot(np.where(spikes_replay[x,:,ep_i])[0]*dt,np.ones(len(np.where(spikes_replay[x,:,ep_i])[0]))*(x),'k.',markersize=2) for x in range(N_place)]
    plt.tick_params(labelsize=17)
    plt.yticks([0,5,10])
    plt.xlabel('Time (s)', fontsize=20)
    plt.ylabel('Neuron #', fontsize=20)
    plt.xlim([0,T_rs])
    plt.xticks(xticks,xticks_lab)
plt.savefig(output_path+'pannel_h.png')
