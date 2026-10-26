import torch
import torchaudio
import torchaudio.transforms

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
    o formato (N,1)
    """
    if x.dim() > 1:
        x = x[:,0]
    else:
        x = x.unsqueeze(0)

def resample(signal, fs, fs_new):
    if fs==fs_new:
        return signal    
    signal = torchaudio.transforms.Resample(fs,fs_new,)(signal)
    return signal

def zero_pad(signal,fs,total_time):
    pass
