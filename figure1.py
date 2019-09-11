# -*- coding: utf-8 -*-
import numpy as np
import network
import matplotlib.pyplot as plt
from datetime import datetime
import os
from scipy.signal import iirfilter, lfilter
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
output_path = "outputs/fig1/"
ensure_dir(output_path)
T = 1.5                     #Time in seconds
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
N_in = 2                    #Number of input units
start_stim = 0.5            #Starting time of the input
t_stim = int(start_stim/dt)
p_in = 0.3                  #Connection probability from input unit to reservoir units
A = 3                       #Input amplitude (1 = 10 pA)
osc_frequencies = [4,5]         #Input frequencies in Hz

#Target function
td = 0.06                   #Readout decay time constant 
tr = 0.006                  #Readout rising time constant
len_stim = T-start_stim
n_step_stim = int(len_stim/dt)
sigma = 30
lp = 6                      #Cutoff frequency
target = np.zeros(nt)
fs = np.int(1/dt)
target[t_stim:] = lowpass_filter(np.random.randn(n_step_stim)*sigma,lp,fs) #Low-pass filtered white noise

#training params
alpha = dt*0.1
step = 50
train_start = int(np.round(start_stim/dt))

#Initialize network
net = network.net(N,pNI,mean_delays,tref,G=G,p=p, mean_GE = mean_GE, mean_GI = mean_GI, ITonic=ITonic,tau=tau,tr=tr,td=td)
input_res = np.zeros((N_in,nt)) #Input to the reservoir
net.w_res = np.multiply(np.random.normal(0,1,(N,N_in)),np.random.rand(N,N_in)<p_in)     #Input weights (M in methods)
phase = np.random.uniform(0,1,N_in)*np.pi*2     #Initial phase configuration of the oscillators

for inp_cell in range(N_in):
    input_res[inp_cell,t_stim:t_stim+n_step_stim] =  A*(np.sin(2*np.pi*osc_frequencies[inp_cell]*(np.linspace(0,len_stim,n_step_stim)+phase[inp_cell])) + 1)/2   

#Recording and training variables
output = np.zeros((nb_epochs,nt))
Pinv = np.eye(net.NE)*alpha
BPhi = np.zeros(net.NE)
total_iEx = np.zeros(nt)
total_iIn = np.zeros(nt)
total_iInp = np.zeros(nt)
sparse_mat = np.zeros((N,nt))

for ep_i in range(nb_epochs):
    #Simulation variables
    print("Running epoch {} of {}.".format(ep_i+1,nb_epochs))
    r = np.zeros(net.NE)
    hr = np.zeros(net.NE)                       
    gEx = np.zeros(N)                                                   #Conductance of excitatory neurons
    gIn = np.zeros(N)                                                   #Conductance of inhibitory neurons
    F = np.full(N,np.nan)                                               #Last spike times of each inhibitory cells
    V = np.random.normal(net.mean_VRest,abs(0.04*net.mean_VRest),N)     #Set initial voltage


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

        dV_res = ((net.VRest-V) + E_current + I_current + Inp_current + net.ITonic)/net.tau
        V = V + (dV_res * (dt))      #Update membrane potential 


        r = r*np.exp(-dt/net.tr) + hr*dt
        hr = hr*np.exp(-dt/net.td) + np.divide(V[net.E_idx]>=net.Theta[net.E_idx],net.tr*net.td)        
        z = np.dot(BPhi.T,r)
        output[ep_i,t] = z
        err = z-target[t]
                
        #RLMS
        if t >= train_start and ep_i%2 == 1:    #Train every 2nd epoch (test on the other)
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

        
        if ep_i == nb_epochs-1:       #Only record activity on the last epoch to save memory
            sparse_mat[spikers,t] = 1 
            total_iEx[t] = np.sum(E_current)
            total_iIn[t] = np.sum(I_current)
            total_iInp[t] = np.sum(np.abs(Inp_current))
    


#Fig params
width = 15
height = 8
xaxis = np.linspace(0,T,nt)

#Pannel c, first two plots from top
balance = np.mean((total_iEx[:t_stim]+total_iIn[:t_stim])/N)
plt.figure(figsize=(width,height))
plt.title('Fig 1, pannel c (first two)')
plt.subplot(2,1,1)
plt.plot(xaxis,10*total_iIn[:]/N,'b',label='Inhibition')
plt.plot(xaxis,10*total_iEx[:]/N,'r',label='Excitation')
plt.plot(xaxis,10*(total_iEx[:]+total_iIn[:])/N,'k',label='Balance')
plt.tick_params(labelsize=17)
plt.ylabel('Network currents (pA)', fontsize=20)
plt.legend()
plt.twinx()
plt.plot(xaxis,10*total_iInp[:]/(N*p),'g',label='External drive (input units only)')
plt.tick_params(labelsize=17)
plt.ylabel('External currents (pA)', fontsize=20)
plt.legend()
plt.subplot(2,1,2)
[plt.plot(np.where(sparse_mat[x,:])[0]*dt,np.ones(len(np.where(sparse_mat[x,:])[0]))*(N-x),'r.',markersize=1,alpha=0.7) for x in range(net.NE)]
[plt.plot(np.where(sparse_mat[x,:])[0]*dt,np.ones(len(np.where(sparse_mat[x,:])[0]))*(N-x),'b.',markersize=1,alpha=0.7) for x in range(net.NE,net.NE+net.NI)]
plt.tick_params(labelsize=17)
plt.xlabel('Time (s)', fontsize=20)
plt.ylabel('Neuron #', fontsize=20)
plt.savefig(output_path+'pannel_c-top.png')
           
           
#Pannel c, bottom, instantaneous firing rate
inst_fr = np.zeros(sparse_mat.shape)
for i in range(N):
    events = np.where(sparse_mat[i,:])[0]
    s = 0
    for event_i in range(len(events)):
        event = events[event_i]
        inst_fr[i,s:event] = 1/((event-s)*dt)
        s = event
plt.figure(figsize=(width,6))
plt.plot(xaxis,np.nanmean(inst_fr,axis=0),'k')
plt.xlabel('Time (s)', fontsize=20)
plt.ylabel('Average firing rate\n (spks/s)', fontsize=20)
plt.tick_params(labelsize=17)
plt.title('Instantaneous firing rate')
plt.savefig(output_path+'pannel_c-bottom.png')
           
#Pannel d, readout output
plt_colors = [[255,171,171],
              [255,120,120],
        [255,0,0]]
plt_colors = [tuple(np.array(x)/255) for x in plt_colors]
plt.figure(figsize=(width,6))
plt_numbers = [2,6,38]

labels = ['Epoch #2','Epoch #5', 'Epoch #20']
plt.plot(xaxis,target,'k',label='Target')
[plt.plot(xaxis,output[plt_numbers[x],:],color=plt_colors[x],label=labels[x]) for x in range(len(plt_numbers))]
plt.xlabel('Time (s)', fontsize=20)
plt.ylabel('Output', fontsize=20)
plt.tick_params(labelsize=17)
plt.legend()
plt.savefig(output_path+'pannel_d.png')


#Pannel e, output/target correlation
nb_tests = int(np.floor(nb_epochs/2))
test_error = np.zeros(nb_tests)
for i in range(1,nb_tests):
    test_error[i] = np.corrcoef(target[t_stim:],output[(i*2),t_stim:])[0,1]
plt.figure(figsize=(12,8))   
plt.plot(range(1,nb_tests+1),test_error)
plt.xlabel('Epoch #', fontsize=30)
plt.ylabel('Pearson r', fontsize=30)
plt.tick_params(labelsize=27)
plt.ylim([-0.3,1])
plt.savefig(output_path+'pannel_e.png')


