# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.signal import iirfilter, lfilter
import network
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

np.random.seed(1)

def lowpass_filter(data, cutoff, fs):
    order = 3
    nyq  = fs/2.0
    cutoff = cutoff/nyq
    b, a = iirfilter(order, cutoff, btype='lowpass',
                     analog=False, ftype='butter')
    filtered_data = lfilter(b, a, data)
    return filtered_data    

def launch_net(net,Inets,nt,train,w_in,inh_pulse,w_fb,input_Inets,w_inj,target,BPhi,Pinv,nb_epochs,
                     tr,td,step):
    N = net.N
    output = np.zeros((nb_epochs,nt))

    nb_Inets = len(Inets)
    N_Inets = Inets[0].N
    #Flatten input net params
    GE_Inets = np.ndarray.flatten(np.array([Inet.GE for Inet in Inets]))
    GI_Inets = np.ndarray.flatten(np.array([Inet.GI for Inet in Inets]))
    E_idx_Inets = np.ndarray.flatten(np.array([Inet.E_idx+(x*N_Inets) for x,Inet in enumerate(Inets)]))
    I_idx_Inets = np.ndarray.flatten(np.array([Inet.I_idx+(x*N_Inets) for x,Inet in enumerate(Inets)]))
    delays_Inets = np.ndarray.flatten(np.array([Inet.delays for Inet in Inets]))
    VRest_Inets = np.ndarray.flatten(np.array([Inet.VRest for Inet in Inets]))
    RE_Inets = np.ndarray.flatten(np.array([Inet.RE for Inet in Inets]))
    RI_Inets = np.ndarray.flatten(np.array([Inet.RI for Inet in Inets]))
    ITonic_Inets = np.ndarray.flatten(np.array([Inet.ITonic for Inet in Inets]))
    tau_Inets = np.ndarray.flatten(np.array([Inet.tau for Inet in Inets]))
    Refractory_Inets = np.ndarray.flatten(np.array([Inet.Refractory for Inet in Inets]))
    Theta_Inets = np.ndarray.flatten(np.array([Inet.Theta for Inet in Inets]))
    TauE_Inets = np.ndarray.flatten(np.array([Inet.TauE for Inet in Inets]))
    TauI_Inets = np.ndarray.flatten(np.array([Inet.TauI for Inet in Inets]))
    W_Inets = np.zeros((nb_Inets*N_Inets,nb_Inets*N_Inets))
    for i_Inet in range(nb_Inets):
        W_Inets[(i_Inet*N_Inets):((i_Inet+1)*N_Inets),(i_Inet*N_Inets):((i_Inet+1)*N_Inets)] = Inets[i_Inet].W
    
    for i in range(nb_epochs):
        print('Epoch {}/{}.'.format(i+1,nb_epochs))
        r = np.zeros(net.NE)
        hr = np.zeros(net.NE)         
        spikes = np.zeros((net.N,nt))    #Spike times   
        spikes_Inets = np.zeros((nb_Inets*N_Inets,nt))     #Spike times                             
        gEx = np.zeros(N)                                            #Conductance of excitatory neurons
        gIn = np.zeros(N)                                            #Conductance of excitatory neurons
        gEx_Inets = np.zeros(N_Inets*nb_Inets)                                            #Conductance of excitatory neurons
        gIn_Inets = np.zeros(N_Inets*nb_Inets)                                            #Conductance of excitatory neurons
        F = np.full(N,np.nan)                                               #Last spike times of each inhibitory cells
        F_Inets = np.full(N_Inets*nb_Inets,np.nan)
        V = np.random.normal(net.mean_VRest,abs(0.04*net.mean_VRest),N)   #Set initial voltage Exc
        V_Inets = np.random.normal(net.mean_VRest,abs(0.01*net.mean_VRest),N_Inets*nb_Inets)   #Set initial voltage Exc


        for t in range(nt):
            #Conductuances decay exponentially to zero
            gEx = np.multiply(gEx,np.exp(-dt/net.TauE))
            gIn = np.multiply(gIn,np.exp(-dt/net.TauI))
            
            gEx_Inets = np.multiply(gEx_Inets,np.exp(-dt/TauE_Inets))
            gIn_Inets = np.multiply(gIn_Inets,np.exp(-dt/TauI_Inets))
        
            #Update conductance of postsyn neurons
            F_E = np.all([[t-F[net.E_idx]==net.delays[net.E_idx]],[F[net.E_idx] != 0]],axis = 0,keepdims=0)
            SpikesERes = net.E_idx[F_E[0,:]]          #If a neuron spikes x time-steps ago, activate post-syn 
            if len(SpikesERes ) > 0:
                gEx = gEx + np.multiply(net.GE,np.sum(net.W[:,SpikesERes],axis=1))  #Increase the conductance of postsyn neurons
                gEx_Inets = gEx_Inets + np.multiply(GE_Inets,np.sum(w_fb.T[:,SpikesERes],axis=1))
            F_I = np.all([[t-F[net.I_idx]==net.delays[net.I_idx]],[F[net.I_idx] != 0]],axis = 0,keepdims=0)
            SpikesIRes = net.I_idx[F_I[0,:]]        
            if len(SpikesIRes) > 0:
                gIn = gIn + np.multiply(net.GI,np.sum(net.W[:,SpikesIRes],axis=1))  #Increase the conductance of postsyn neurons
                
            #Update conductances of input (oscillatory) networks
            F_E_Inets = np.all([[t-F_Inets[E_idx_Inets]==delays_Inets[E_idx_Inets]],
                                              [F_Inets[E_idx_Inets] != 0]],axis = 0,keepdims=0)
            SpikesEInets = E_idx_Inets[F_E_Inets[0,:]]
            
            if len(SpikesEInets) > 0:
                gEx_Inets = gEx_Inets + np.multiply(GE_Inets,np.sum(W_Inets[:,SpikesEInets],axis=1))  
                gEx = gEx + np.multiply(net.GE,np.sum(w_in[:,np.where(F_E_Inets)[0]],axis=1))
            
            F_I_Inets = np.all([[t-F_Inets[I_idx_Inets]==delays_Inets[I_idx_Inets]],
                                              [F_Inets[I_idx_Inets] != 0]],axis = 0,keepdims=0)
            SpikesIInets = I_idx_Inets[F_I_Inets[0,:]]
            if len(SpikesIInets) > 0:
                gIn_Inets = gIn_Inets + np.multiply(GI_Inets,np.sum(W_Inets.T[:,SpikesIInets],axis=1))  
                gIn = gIn + np.multiply(net.GI,np.sum(w_in[:,np.where(F_I_Inets)[0]],axis=1))

            #Leaky Integrate-and-fire
            dV_res = ((net.VRest-V) + np.multiply(gEx,net.RE-V) +
                      np.multiply(gIn,net.RI-V) + net.ITonic)             
            V = V + (dV_res * (dt/net.tau))                                        #Update membrane potential

            dV_Inets = ((VRest_Inets-V_Inets) + np.multiply(gEx_Inets,RE_Inets-V_Inets) + 
                      np.multiply(gIn_Inets,RI_Inets-V_Inets) + np.dot(w_inj,input_Inets[:,t]) +
                      np.dot(w_inj,inh_pulse[:,t]) + ITonic_Inets)                                  
            V_Inets = V_Inets + (dV_Inets * (dt/tau_Inets))    
            
            r = r*np.exp(-dt/tr) + hr*dt
            hr = hr*np.exp(-dt/td) + np.divide(V[net.E_idx]>=net.Theta[net.E_idx],tr*td)        
            z = np.dot(BPhi.T,r)
            output[i,t] = z
            err = z-target[t]
                    
            #RLMS
            if t >= train_start and train:
                if t%step == 1:
                    cd = np.dot(Pinv,r)
                    BPhi = BPhi-(cd*err)
                    Pinv = Pinv - np.divide(np.outer(cd,cd.T),1 + np.dot(r.T,cd))
                    
            #Update cells
            Refract = t <= (F + net.Refractory)
            Refract_Inets = t <= (F_Inets + Refractory_Inets)
            V[Refract] = net.VRest[Refract]                                           #Hold resting potential of neurons in refractory period            
            V_Inets[Refract_Inets] = VRest_Inets[Refract_Inets]
            spikers = np.where(V > net.Theta)[0]
            spikers_Inets = np.where(V_Inets > Theta_Inets)[0]
            F[spikers] = t                                                              #Update the last AP fired by the neuron
            F_Inets[spikers_Inets] = t
            V[spikers] = 0                                                             #Membrane potential at AP time
            V_Inets[spikers_Inets] = 0
            spikes[spikers,t] = 1
            spikes_Inets[spikers_Inets,t] = 1

    return BPhi,spikes,spikes_Inets,output
    
#Sim params
output_path = "outputs/fig2/"
dt = 5e-05                    #dt (in ms)
T = 2                      #Integration time step
nt = int(T/dt)
nb_epochs = 20

#Network parameters (see figure1.py for more detailed comments)
mean_GE = 0.02
mean_GI = 0.16             
mean_delays = 0.001/dt
tref = 2e-03/dt
N = 1000
p_NI = 0.2
p = 0.1
G = 1
ITonic = 9

#Input parameters
start_stim = 0.5
t_stim = int(start_stim/dt)
len_stim = 1                    #Length of the step-function to the oscillatory networks
n_step_stim = int(len_stim/dt)

#Target function
td = 0.06
tr = 0.006
N_out = 1
sigma = 30
fs = np.int(1/dt)
target = np.zeros(nt)
target[t_stim:t_stim+n_step_stim] = lowpass_filter(np.random.randn(N_out,n_step_stim)*sigma,6,fs)


#Input network parameters
nb_Inets = 3
Inets_path = output_path + "osc_nets.p"
with open(Inets_path,'rb') as f:
    data = pickle.load(f)
net1 = data['nets'][0]
N_Inets = net1.N

#Augmented network parameters
p_fb = 0.5          #Probability of connection from reservoir neuron to oscillatory network neurons
p_in = 0.5          #Probability of connection from oscillatory network neuron to reservoir neurons
A = 2     #1 = 10 pA (before connection weights)
gain_fb = 1
gain_in = 10
input_Inets = np.zeros((nb_Inets,nt))
input_Inets[:,t_stim:t_stim+n_step_stim] = A
Inets = data['nets'][:nb_Inets]

#Inhibitory transients
gain_inh = -A          #Amplitude of the inhibitory step
inh_pulse1 = np.zeros((nb_Inets,nt))
pulse_width1 = [0.1,0.03,0.07]      #Length of the inhibitory steps for input#1
inh_pulse2 = np.zeros((nb_Inets,nt))
pulse_width2 = [0,0.1,0.03]     #Length of the inhibitory steps for input#2
for i in range(nb_Inets):
    inh_pulse1[i,t_stim:t_stim+int(pulse_width1[i]/dt)] = gain_inh
    inh_pulse2[i,t_stim:t_stim+int(pulse_width2[i]/dt)] = gain_inh
    
#training params
alpha = dt*0.1
step = 50
train_start = int(np.round(start_stim/dt))


#Initialize augmented network
net = network.net(N,p_NI,mean_delays,tref,G=G,p=p, mean_GE = mean_GE, mean_GI = mean_GI, ITonic=ITonic)
NE = net.NE
NE_Inets = int(N_Inets-(N_Inets*p_NI))

scale_in = gain_in/np.sqrt(p_in*N_Inets*N)
scale_fb = gain_fb/np.sqrt(p_fb*N*N_Inets)
w_fb = scale_fb*np.abs(np.multiply(np.random.normal(0,1,(N,N_Inets*nb_Inets)),
                                   np.random.rand(N,N_Inets*nb_Inets)<p_fb))    #Feedback (res->osc) connections
w_in = scale_in*np.abs(np.multiply(np.random.normal(0,1,(N,N_Inets*nb_Inets)),
                                   np.random.rand(N,N_Inets*nb_Inets)<p_in))    #Input (osc->res) connections

w_inj = np.zeros((nb_Inets*N_Inets,nb_Inets))           #Condense all Inets (input networks) weights in the same matrix
for i_Inet in range(nb_Inets):
    w_inj[(i_Inet*N_Inets):((i_Inet+1)*N_Inets),i_Inet] = Inets[i_Inet].w_res.ravel()

#Training
BPhi = np.zeros(NE)
Pinv = np.eye(NE)*alpha
train = True         
BPhi_trained,spikes_train,spikes_Inets_train,output_train = launch_net(net,Inets,nt,train,w_in,inh_pulse1,
              w_fb,input_Inets,w_inj,target,BPhi,Pinv,nb_epochs,tr,td,step)

#Testing
train = False       
BPhi,spikes_inp1,spikes_Inets_inp1,output_inp1 = launch_net(net,Inets,nt,train,w_in,inh_pulse1,
              w_fb,input_Inets,w_inj,target,BPhi_trained,Pinv,1,tr,td,step)     #Test with input 1
BPhi,spikes_inp2,spikes_Inets_inp2,output_inp2 = launch_net(net,Inets,nt,train,w_in,inh_pulse2,
              w_fb,input_Inets,w_inj,target,BPhi_trained,Pinv,1,tr,td,step)     #Test with input 2

#PLot settings
width = 15
height = 8
xaxis = dt*np.arange(nt)

#pannel e, oscillatory networks raster
total_N = nb_Inets*N_Inets
plt.figure(figsize=(width,height))
plt.title('Oscillatory networks raster')
[plt.plot(np.where(spikes_Inets_inp1[x,:])[0]*dt,np.ones(len(np.where(spikes_Inets_inp1[x,:])[0]))*(total_N-x),linestyle='none',marker='.',color=np.array([3, 112, 6])/255,markersize=1,alpha=1,label='Input#1') for x in range(total_N)]
[plt.plot(np.where(spikes_Inets_inp2[x,:])[0]*dt,np.ones(len(np.where(spikes_Inets_inp2[x,:])[0]))*(total_N-x),linestyle='none',marker='.',color=np.array([27, 214, 32])/255,markersize=1,alpha=1,label='Input#2') for x in range(total_N)]
plt.savefig(output_path+'pannel_e-top.png')   

#pannel e, reservoir raster
plt.figure(figsize=(width,height))
plt.title('Testing.')
[plt.plot(np.where(spikes_inp1[x,:])[0]*dt,np.ones(len(np.where(spikes_inp1[x,:])[0]))*(N-x),'r.',markersize=1,alpha=1) for x in range(net.NE)]
[plt.plot(np.where(spikes_inp2[x,:])[0]*dt,np.ones(len(np.where(spikes_inp2[x,:])[0]))*(N-x),'b.',markersize=1,alpha=1) for x in range(net.NE,net.NE+net.NI)]
plt.savefig(output_path+'pannel_e-middle.png')   


#pannel f, network output
plt.figure(figsize=(15,4))
plt.plot(xaxis,target,'k--',linewidth=2)
plt.plot(xaxis,output_inp2[0,:],color=np.array([27, 214, 32])/255,linewidth=3,alpha=0.9,label='Input#2')   
plt.plot(xaxis,output_inp1[0,:],color=np.array([3, 112, 6])/255,linewidth=3,alpha=0.9,label='Input#1')    
plt.legend()
plt.savefig(output_path+'pannel_f.png')    


