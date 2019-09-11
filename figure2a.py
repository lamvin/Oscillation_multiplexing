# -*- coding: utf-8 -*-

import numpy as np
import network
import matplotlib.pyplot as plt
import os
import pickle
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

np.random.seed(10)

def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)
        
#Sim params
output_path = "outputs/fig2/"
ensure_dir(output_path)
T = 2.5                     #Time in seconds
dt = 5e-05                  #Integration time step
nt = int(T/dt)
nb_epochs = 10               #Number of epochs for each oscillatory network
nb_networks = 3          #Number of stable networks generated (we need three for the figure2b.py script)

#Network parameters (see figure1.py for more detailed comments)
N = 500
pNI = 0.2
dt = 5e-05
mean_delays = 0.001/dt
mean_GE = 0.03
mean_GI = 0.14
mean_TauFall_I_list = [0.07,0.1,0.13]
A = 2                #Amplitude of the step function: 1 = 10 pA (before connection weights)
tref = 2e-03/dt
p = 1
ITonic = 9
G = 1
GaussSD = 0.01      #Reduce heterogeneity in network compared to reservoirs

#Input parameters
start_stim = 1      #In seconds
end_stim = 2
t_stim = int(start_stim/dt)
e_stim = int(end_stim/dt)
p_in = 1            #Porbability of connection from input unit (step function) to network

data = {}   #Output dictionary used by figure2b.py
data['nets'] = []
cutoff = 0.95       #Cut-off correlation between the activity on each trial to consider a network as stable
for net_i in range(nb_networks):
    input_network = np.zeros(nt)
    N_in = 1
    unstable = True
    mean_TauFall_I = mean_TauFall_I_list[net_i]
    while unstable:
        total_iEx = np.zeros((nb_epochs,nt))
        total_iIn = np.zeros((nb_epochs,nt))
        total_iInp = np.zeros((nb_epochs,nt))
        spikes = np.zeros((N,nt,nb_epochs))
        #Exc cells
        net = network.net(N,pNI,mean_delays,tref,G=G,p=p, mean_GE = mean_GE, mean_GI = mean_GI, ITonic=ITonic, mean_TauFall_I=mean_TauFall_I,GaussSD=GaussSD)
        N_in_net = p_in*net.NE
        temp_w_in = np.abs(np.multiply(np.random.normal(0,1,(N_in_net,N_in)),np.random.rand(N_in_net,N_in)<p_in))
        w_res = np.zeros((N,N_in))
        w_res[net.E_idx,:] = temp_w_in          #Only the excitatory units receive the step input
        net.w_res = w_res
        input_network[t_stim:e_stim] =  A            

        for ep_i in range(nb_epochs):
            #Simulation variables
            gEx = np.zeros(N)                                                  #Conductance of excitatory neurons
            gIn = np.zeros(N)                                                  #Conductance of excitatory neurons
            F = np.full(N,np.nan)                                              #Last spike times of each inhibitory cells
            V = np.random.normal(net.mean_VRest,abs(0.02*net.mean_VRest),N)    #Set initial voltage Exc 

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
                Inp_current = (net.w_res*input_network[t]).reshape((N,))
                dV_res = ((net.VRest-V) + E_current + I_current + Inp_current + net.ITonic)                              
                V = V + (dV_res * (dt/net.tau))                                        #Update membrane potential 

                        
                #Update cells
                Refract = t <= (F + net.Refractory)
                V[Refract] = net.VRest[Refract]                                        #Hold resting potential of neurons in refractory period            
                spikers = np.where(V > net.Theta)[0]
                F[spikers] = t                                                         #Update the last AP fired by the neuron
                V[spikers] = 0 
                
                total_iEx[ep_i,t] = np.sum(E_current)
                total_iIn[ep_i,t] = np.sum(I_current)
                total_iInp[ep_i,t] = np.sum(Inp_current)
                spikes[spikers,t,ep_i] = 1

        sample_trace = total_iEx[0,t_stim:e_stim]
        corr = np.mean([np.corrcoef(total_iEx[x,t_stim:e_stim],sample_trace) for x in range(1,nb_epochs)])
        print('Network {}: r={}'.format(net_i+1,np.round(corr,2)))
        if corr < cutoff :
            pass
        else:
            unstable = False
            data['nets'].append(net)

#Save networks for augmented model (figure2b.py)
with open(output_path+'osc_nets.p','wb') as f:
    pickle.dump(data,f)
        
    
#Fig params
width = 15
height = 8
xaxis = np.linspace(0,T,nt)

#pannel a, top
plt_colors = ['451EA0','683372']
#plt_colors = [tuple(np.array(x)/255) for x in plt_colors]
plt_colors = [tuple(np.array(tuple(int(x[i:i+2], 16) for i in (0, 2 ,4)))/255) for x in plt_colors]
skip_step = int(0/dt)
plt.figure(figsize=(10,6))
for ep_i in range(nb_epochs):
    if ep_i % 2 == 1:
        [plt.plot(np.where(spikes[x,:,ep_i])[0]*dt,np.ones(len(np.where(spikes[x,skip_step:,ep_i])[0]))*(N-x)+(ep_i*N),linestyle='none',marker='.',color=plt_colors[0],markersize=1,alpha=1) for x in range(N)]
    else:
        [plt.plot(np.where(spikes[x,:,ep_i])[0]*dt,np.ones(len(np.where(spikes[x,skip_step:,ep_i])[0]))*(N-x)+(ep_i*N),linestyle='none',marker='.',color=plt_colors[1],markersize=1,alpha=1) for x in range(N)]
plt.xlim([0,2.5])
plt.xlabel('Time (s)',fontsize=20)
plt.tick_params(labelsize=17)
plt.yticks([], [])
plt.savefig(output_path+'pannel_a-top.png')

#pannel a, bottom (last network only)
fig = plt.figure(figsize=(width,height))
plt.plot(xaxis,np.mean(total_iEx[:,:]/N,axis=0),color='r',label='Exc') 
y = np.mean(total_iEx[:,:]/N,axis=0)
err = np.std(total_iEx[:,:]/N,axis=0)
plt.fill_between(xaxis,y-err, y+err,color='orange')
plt.plot(xaxis,np.mean(total_iIn[:,:]/N,axis=0),color='b',label='Inh')
plt.ylabel('Average network currents', fontsize=20)
plt.xlabel('Time (s)', fontsize=20)
plt.tick_params(labelsize=17)
plt.twinx()
plt.plot(xaxis,input_network,color='g',label='Inp')
plt.ylabel('Step current', fontsize=20)
plt.tick_params(labelsize=17)
plt.savefig(output_path+'pannel_a-bottom.png')

