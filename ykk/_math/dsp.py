import torch
import torchaudio
import torchaudio.transforms
import librosa
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
        x = x[0,:]
    
    x = torch.atleast_2d(x)
    return x

def resample(signal, fs, fs_new):
    if fs==fs_new:
        return signal    
    signal = torchaudio.transforms.Resample(fs,fs_new,)(signal)
    return signal

def zero_pad(signal,fs,total_time):
    """Recebe um sinal com formato
    """
    N = signal.shape[1]
    N_after = int(fs*total_time)
    output_shape = list(signal.shape)
    output_shape[1] = N_after
    out = torch.zeros(output_shape)
    out[:,:N] = signal
    return out

def normalize(signal,):
    return signal/abs(signal).max()

def pitch_shift(signal,fs,n_st):
    signal = librosa.effects.pitch_shift(signal.numpy(),
                                          sr=fs,n_steps=n_st)
    return torch.as_tensor(signal)
