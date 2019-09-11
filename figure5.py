# -*- coding: utf-8 -*-
import numpy as np
import network
import audio.audio_tools as at
import matplotlib.pyplot as plt
import os
from scipy.signal import iirfilter, lfilter
from scipy.io import wavfile
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

def resample(sig,rs):
    N = sig.shape[0]
    nb_steps = sig.shape[1]
    nb_steps_ds = int(np.ceil(nb_steps/rs))
    bin_size = int(np.round(nb_steps/nb_steps_ds))
    nb_bins = int(np.ceil(nb_steps/bin_size))
    sig_ds = np.zeros((N,nb_bins))
    for i in range(nb_bins):
        sig_ds[:,i] = np.mean(sig[:,(i*bin_size):((i+1)*bin_size)],axis=1)
    return sig_ds



#Sim params
output_path = "outputs/fig5/"
ensure_dir(output_path)
nb_epochs = 20              #Number of epochs (divide by two for training and testing)

#Network parameters (see figure1.py for more detailed comments)
N = 1000
pNI = 0.2
dt = 5e-05
mean_delays = 0.001/dt
mean_GE = 0.02
mean_GI = 0.16             
fs = np.int(1/dt)
tref = 2e-03/dt
p = 0.1
ITonic = 9
G=1

#Input parameters
osc_frequencies = np.array([3,4,5])     #Frequency of the input
A = 3
N_in = 3
start_stim = 0.5
t_stim = int(np.round(start_stim/dt))
train_start = t_stim
p_in = 0.3

#Target parameters
td = 0.06
tr = 0.006
fft_size = 2048  # window size for the FFT
step_size = fft_size // 16  # distance to slide along the window (in time)
spec_thresh = 4  # threshold for spectrograms (lower filters out more noise)
lowcut = 500  # Hz # Low cut for our butter bandpass filter
highcut = 15000  # Hz # High cut for our butter bandpass filter
# For mels
N_out = 64  # number of mel frequency channels
shorten_factor = 1  # how much should we compress the x-axis (time)
start_freq = 300  # Hz # What frequency to start sampling our melS from
end_freq = 8000  # Hz # What frequency to stop sampling our melS from

wav_file = output_path + "../../audio/reservoir.wav"

# Filter wav
rate, data = wavfile.read(wav_file)
data = data[:,0]
data = at.butter_bandpass_filter(data, lowcut, highcut, rate, order=1)

#Additional sim params (setting T based on length audio clip)
buffer = 0.1
len_buffer = int(buffer/dt)
T_output = np.shape(data)[0] / float(rate)
len_output = int(T_output/dt)
T = start_stim + T_output + buffer
nt = int(T/dt)
len_stim = T-start_stim-buffer
n_step_stim = int(len_stim/dt)
cut_start = 0.1
cut_t = int(cut_start/dt)
T_out = T-cut_start
nt_out = int(np.round(T_out/dt))


#Convert to spectrogram
wav_spectrogram = at.pretty_spectrogram(data.astype("float64"),fft_size=fft_size,step_size=step_size,log=True,thresh=spec_thresh)
#Create filter for compression
mel_filter, mel_inversion_filter = at.create_mel_filter(
    fft_size=fft_size,
    n_freq_components=N_out,
    start_freq=start_freq,
    end_freq=end_freq,
)
mel_spec = at.make_mel(wav_spectrogram, mel_filter, shorten_factor=shorten_factor)
len_mel = mel_spec.shape[1]
#Resample spectrogram to match the number of time-steps of the simulation
rs_audio = len_output/len_mel
target = np.full((N_out,nt),np.min(mel_spec))
target[:,t_stim:-len_buffer] = mel_spec[:,np.round(np.linspace(0,len_mel-1,len_output)).astype(np.int)]

#Initialize network
net = network.net(N,pNI,mean_delays,tref,G=G,p=p, mean_GE = mean_GE, mean_GI = mean_GI, ITonic=ITonic)
net.tr = tr
net.td = td
input_res = np.zeros((N_in,nt))
N_in_net = int(np.round(p_in*N))
net.w_res = np.abs(np.multiply(np.random.normal(0,1,(N,N_in)),np.random.rand(N,N_in)<p_in))

phase = np.random.uniform(0,1,N_in)*np.pi 
for inp_cell in range(N_in):
    input_res[inp_cell,t_stim:t_stim+n_step_stim] =  A*(np.sin(2*np.pi*osc_frequencies[inp_cell]*(np.linspace(0,len_stim,n_step_stim))+phase[inp_cell]) + 1)/2   

#Training params
alpha = dt*0.1
step = 50
train_start = int(np.round(start_stim/dt))
Pinv = np.eye(net.NE)*alpha
BPhi = np.zeros((net.NE,N_out))

#Recording variables
output = np.full((N_out,nt),np.min(mel_spec))

for ep_i in range(nb_epochs):
    #Simulation variables
    print("Running epoch {} of {}.".format(ep_i+1,nb_epochs))
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
        dV_res = ((net.VRest-V) + E_current + I_current + Inp_current + net.ITonic)/net.tau                                  #Compute raw voltage change
        V = V + (dV_res * (dt))                                        #Update membrane potential 

        r = r*np.exp(-dt/net.tr) + hr*dt
        hr = hr*np.exp(-dt/net.td) + np.divide(V[net.E_idx]>=net.Theta[net.E_idx],net.tr*net.td)        
        z = np.dot(BPhi.T,r)
        output[:,t] = z
        err = z-target[:,t]
                
        Refract = t <= (F + net.Refractory)
        V[Refract] = net.VRest[Refract]                                           #Hold resting potential of neurons in refractory period            
        spikers = np.where(V > net.Theta)[0]
        F[spikers] = t                                                              #Update the last AP fired by the neuron
        V[spikers] = 0                                                             #Membrane potential at AP time
        
        #RLMS
        if t >= train_start and (ep_i+1) < nb_epochs:
            if t%step == 1:
                cd = np.dot(Pinv,r)
                Pinv = Pinv - np.divide(np.outer(cd,cd.T),1 + np.dot(r.T,cd))
                for ro in range(N_out):
                    BPhi[:,ro] = BPhi[:,ro]-(cd*err[ro])
                    
                    
                    
                    
    
#Rescaling parameters    
rs_factor_list = [0.25,0.5,0.75,1,1.25,1.5,2,3]     #Rescaling factors tested
nb_rs = len(rs_factor_list)
nt_post_stim = nt-t_stim-len_buffer
max_nt = int(nt_post_stim*rs_factor_list[-1]) + t_stim + len_buffer     #Longest rescaling factor (to initialize output array)
output_rs = np.full((N_out,max_nt,nb_rs),np.min(mel_spec)) 
#Run with rescaled inputs
for rs_i in range(nb_rs):
    rs_factor = rs_factor_list[rs_i]
    print("Running network with {}X input periods ({}/{}).".format(rs_factor,rs_i+1,nb_rs))
    osc_frequencies_rescaled = np.array(osc_frequencies)/rs_factor
    n_step_stim_rs = int(nt_post_stim*rs_factor)
    len_stim_rs = len_stim*rs_factor
    nt_rs = t_stim + n_step_stim_rs 
    #Generate rescaled input
    input_res_rescaled = np.zeros((N_in,max_nt))
    for inp_cell in range(N_in):
        input_res_rescaled[inp_cell,t_stim:t_stim+n_step_stim_rs] =  A*(np.sin(2*np.pi*osc_frequencies_rescaled[inp_cell]*(np.linspace(0,len_stim_rs,n_step_stim_rs))+phase[inp_cell]) + 1)/2   
    r = np.zeros(net.NE)
    hr = np.zeros(net.NE)                       
    gEx = np.zeros(N)                                            #Conductance of excitatory neurons
    gIn = np.zeros(N)                                            #Conductance of excitatory neurons
    F = np.full(N,np.nan)                                               #Last spike times of each inhibitory cells
    V = np.random.normal(net.mean_VRest,abs(0.04*net.mean_VRest),N)   #Set initial voltage Exc 

    for t in range(max_nt):
        #Conductuances decay exponentially to zero
        gEx = np.multiply(gEx,np.exp(-dt/net.TauE))
        gIn = np.multiply(gIn,np.exp(-dt/net.TauI))
    
        #Update conductance of postsyn neurons
        F_E = np.all([[t-F[net.E_idx]==net.delays[net.E_idx]],[F[net.E_idx] != 0]],axis = 0,keepdims=0)
       
        SpikesERes = net.E_idx[F_E[0,:]]                                        #If a neuron spikes x time-steps ago, activate post-syn 
        if len(SpikesERes ) > 0:
            gEx = gEx + np.multiply(net.GE,np.sum(net.W[:,SpikesERes],axis=1))  #Increase the conductance of postsyn neurons
        F_I = np.all([[t-F[net.I_idx]==net.delays[net.I_idx]],[F[net.I_idx] != 0]],axis = 0,keepdims=0)
        SpikesIRes = net.I_idx[F_I[0,:]]        
        if len(SpikesIRes) > 0:
            gIn = gIn + np.multiply(net.GI,np.sum(net.W[:,SpikesIRes],axis=1))  #Increase the conductance of postsyn neurons
                 
        #Leaky Integrate-and-fire
        E_current = np.multiply(gEx,net.RE-V)
        I_current = np.multiply(gIn,net.RI-V)
        Inp_current = np.dot(net.w_res,input_res_rescaled[:,t])
        dV_res = ((net.VRest-V) + E_current + I_current + Inp_current + net.ITonic)                                   
        V = V + (dV_res * (dt/net.tau))                                        #Update membrane potential

        #Output
        r = r*np.exp(-dt/net.tr) + hr*dt
        hr = hr*np.exp(-dt/net.td) + np.divide(V[net.E_idx]>=net.Theta[net.E_idx],net.tr*net.td)        
        z = np.dot(BPhi.T,r)
        output_rs[:,t,rs_i] = z
                
        #Update cells
        Refract = t <= (F + net.Refractory)
        V[Refract] = net.VRest[Refract]                                           #Hold resting potential of neurons in refractory period            
        spikers = np.where(V > net.Theta)[0]
        F[spikers] = t                                                              #Update the last AP fired by the neuron
        V[spikers] = 0                                                             #Membrane potential at AP time

#Spectrogram inversion to waveform and create output .wav file
gain_volume = 3
spectrograms = []
waves = []
adjust = -1
for rs_i in range(nb_rs):
    rs_factor = rs_factor_list[rs_i]
    n_step_stim_rs = int(nt_post_stim*rs_factor)
    nt_rs = t_stim + n_step_stim_rs - cut_t
    res_output = resample(output_rs[:,cut_t:,rs_i],rs_audio)
    res_output = res_output[:,:adjust]
    spectrograms.append(res_output)

    ratio_len = (len_output/(max_nt-cut_t))
    mel_inverted_spectrogram_out = at.mel_to_spectrogram(
        res_output+gain_volume,
        mel_inversion_filter,
        spec_thresh=spec_thresh,
        shorten_factor=1,
    )
    inverted_mel_audio_out  = at.invert_pretty_spectrogram(
        np.transpose(mel_inverted_spectrogram_out ),
        fft_size=fft_size,
        step_size=step_size,
        log=True,
        n_iter=10,
    )
    waves.append(inverted_mel_audio_out)
    wavfile.write("{}/net_output_X{}.wav".format(output_path,rs_factor_list[rs_i]),rate,inverted_mel_audio_out)


#Plot individual spectrograms in real time (panel b)
rs_i=3 #Index in the rescaling list, by defaut:[0.25,0.5,0.75,1,1.25,1.5,2,3]
len_res = spectrograms[rs_i].shape[1]
n_yt = 4
n_xt = 6
xticks_lab = np.round(np.linspace(0,max_nt*dt,n_xt),1)
yticks_lab = np.linspace(start_freq,end_freq,n_yt)*1e-03
yticks_lab = np.round(yticks_lab,1)
xticks = np.linspace(0,len_res,n_xt)
yticks = np.linspace(0,N_out,n_yt)

plt.figure(figsize=(12,8))
plt.pcolor(spectrograms[rs_i],vmin=-4)
plt.title('Rescaling factor: {}'.format(rs_factor_list[rs_i]),fontsize=30)
plt.xticks(xticks,xticks_lab)
plt.yticks(yticks,yticks_lab)
plt.ylabel('Frequency (kHz)',fontsize=25)
plt.xlabel('Time (s)',fontsize=25)
plt.tick_params(labelsize=20)
plt.colorbar()
plt.savefig(output_path+'spectrogram_X{}.png'.format(rs_factor_list[rs_i]))


#Panel d
error = []
res_target = resample(target[:,cut_t:adjust ],rs_audio)
len_res = spectrograms[rs_i].shape[1]
len_res_tar = res_target.shape[1]
ratio_len = (len_res/(max_nt-cut_t))
ratio_len_tar = (len_res_tar/(nt-cut_t))
#compute the error for each rescaling factor
for rs_i in range(nb_rs):
    rs_factor = rs_factor_list[rs_i]
    n_step_stim_rs = int(nt_post_stim*rs_factor*ratio_len)
    output_s = spectrograms[rs_i]
    comp_output = output_s[:,int((t_stim-cut_t)*ratio_len)+np.linspace(0,n_step_stim_rs,int(nt_post_stim*ratio_len)).astype(np.int)]
    comp_target = res_target[:,int((t_stim-cut_t)*ratio_len_tar)+np.arange(0,int(nt_post_stim*ratio_len_tar)).astype(np.int)]
    corr = [np.corrcoef(comp_output[x,:],comp_target[x,:])[0,1] for x in range(N_out)]      #Get the average pearson r for each readout
    error.append(np.mean(corr))
    
plt.figure(figsize=(10,8))   
plt.plot(rs_factor_list,error)
plt.tick_params(labelsize=17)
plt.ylabel('Pearson r', fontsize=20)
plt.xlabel('Rescaling factor', fontsize=20)
plt.title('Output/target correlation')
plt.savefig(output_path+'pannel_d.png')

#Panel c
plt.figure(figsize=(13,5))
len_res = spectrograms[rs_i].shape[1]
len_res_tar = res_target.shape[1]
ratio_len = (len_res/(max_nt-cut_t))#((max_nt-cut_t)/len_output) 
ratio_len_tar = (len_res_tar/(nt-cut_t))#((max_nt-cut_t)/len_output) 
n_yt = 4
n_xt = 4
xticks_lab = np.round(np.linspace(0,T_output,n_xt),1)
yticks_lab = np.round(np.linspace(start_freq,end_freq,n_yt)*1e-03,1)
xticks = np.linspace(0,int(nt_post_stim*ratio_len),n_xt)
yticks = np.linspace(0,N_out,n_yt)

for rs_i in range(8):
    rs_factor = rs_factor_list[rs_i]
    n_step_stim_rs = int(nt_post_stim*rs_factor*ratio_len)
    output_s = spectrograms[rs_i]
    comp_output = output_s[:,int((t_stim-cut_t)*ratio_len)+np.linspace(0,n_step_stim_rs,int(nt_post_stim*ratio_len)).astype(np.int)]
    comp_target = res_target[:,int((t_stim-cut_t)*ratio_len_tar)+np.arange(0,int(nt_post_stim*ratio_len_tar)).astype(np.int)]
    plt.subplot(2,4,rs_i+1)
    plt.pcolor(comp_output)
    plt.xticks(xticks,xticks_lab)
    plt.yticks(yticks,yticks_lab)
    #plt.ylabel('Frequency (kHz)',fontsize=25)
    #plt.xlabel('Time (s)',fontsize=25)
    plt.tick_params(labelsize=10)
    plt.title('{} X'.format(rs_factor),fontsize=10)
plt.savefig(output_path+'pannel_c.png')