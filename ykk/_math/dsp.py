import torch
import torchaudio
import torchaudio.transforms
import librosa
from torchaudio import functional as F
def dB(x):
    """Calcula o valor em decibels
    """
    return 10*torch.log10(abs(x))

def schroeder(x):
    """Aplica a integral de Schroeder em uma
    h(t). Assume que o dado ainda não é quadrático.
    """
    x2 = x**2
    return torch.sum(x2)-torch.cumsum(x2)

def ensure_mono(x):
    """Recebe um sinal e garante que ele sempre tenha
    o formato (1,N)
    """
    if x.dim() > 1:
        x = torch.mean(x,axis=0)
    x = torch.atleast_2d(x)
    return x

def resample(signal, fs, fs_new):
    if fs==fs_new:
        return signal    
    signal = torchaudio.transforms.Resample(fs,fs_new,)(signal)
    return signal

def pad_until(signal,fs,total_time):
    """Recebe um sinal com formato
    """
    N = signal.shape[1]
    N_after = int(fs*total_time)
    
    if N >= N_after:
        return signal

    output_shape = list(signal.shape)
    output_shape[1] = N_after
    out = torch.zeros(output_shape)
    out[:,:N] = signal
    return out

def normalize(signal,):
    return signal/abs(signal).max()

def pitch_shift(signal,fs,n_st):
    signal = F.pitch_shift(signal,fs,n_st)
    return signal
    # signal = librosa.effects.pitch_shift(signal.numpy(),
    #                                       sr=fs,n_steps=n_st)
    # return torch.as_tensor(signal)

def troll(signal,fs,time):
    """applies circshift of `fs*time` to a signal. 
    This function considers the last axis as time axis
    """
    signal = torch.roll(signal,int(time*fs),-1)
    return signal

def timestretch(signal,stretch_pct):
    signal = librosa.effects.time_stretch(signal.numpy(),
                                          rate=stretch_pct
                                          )
    return torch.as_tensor(signal)

def center_crop(signal,fs,total_time):
    """Recebe um sinal com formato
    """
    N_in = signal.shape[1]
    N_out = int(fs*total_time)
    
    if N_out >= N_in: #No need to crop
        # print('no need at all')
        return signal

    N_to_crop = abs(N_in-N_out)
    Nleft = int(N_to_crop/2)
    Nright = N_to_crop-Nleft

    return signal[...,Nleft:-Nright]
    # output_shape = list(signal.shape)
    # output_shape[1] = N_out
    # out = torch.zeros(output_shape)
    # out[:,Nleft:Nright] = signal
    # return out

def remove_offset(x):
    return x - x.mean()
