#%%
"""For dsp testing purposes
"""
import ykk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ykk.neuralnet import urban8k
import seaborn as sns
ykk.utils.configure()
from tqdm import tqdm
from ykk import dsp
from torchaudio import transforms
from torchaudio import functional as tF
#%%
dataset = urban8k(augmentation=True)

#%%
winsize = 2048

# hop = winsize-
mel_kw = {"n_fft": 2048, "hop_length": 256, "n_mels": 23}
dataset.set_mfcc_params(n_mfcc=20,
                        mel_kw=mel_kw)

#%%
signal,_ = dataset[16]
mfcc_plot = signal.squeeze(0).numpy()

plt.figure(figsize=(10, 4))
plt.imshow(mfcc_plot, aspect='auto', 
           origin='lower', cmap='inferno',vmin=00,interpolation='none')
plt.colorbar(label='Amplitude')
plt.title('MFCC')
plt.xlabel('Quadros (tempo)')
plt.ylabel('Coeficiente cepstral')
plt.tight_layout()
plt.show()

#%%
signal.shape
#%%
signal,fs,_ = dataset.get_wavfile(16)
signal = dsp.ensure_mono(signal)
signal = dsp.resample(signal,fs,44100)
signal = dsp.pad_until(signal,fs,4.0)
signal = dsp.normalize(signal)

ykk.specgram(signal.numpy()[0],fs=44100,
             signal_kw=dict(noverlap=40,nperseg=256))
#%%
# signal2 = tF.pitch_shift(signal,fs,2.5)

# ykk.specgram(signal.numpy()[0],fs=44100,
#              signal_kw=dict(noverlap=40,nperseg=2048))
# plt.show()
# ykk.specgram(signal2.numpy()[0],fs=44100,
#              signal_kw=dict(noverlap=40,nperseg=2048))


#%%

MFCC = transforms.MFCC(44100,n_mfcc=13,                       # número de coeficientes a reter
    melkwargs={"n_fft": 2048, "hop_length": 128, "n_mels": 23},
)

signal_spk = MFCC(signal)
#%%
mfcc_plot = signal_spk.squeeze(0).numpy()

plt.figure(figsize=(10, 4))
plt.imshow(mfcc_plot, aspect='auto', origin='lower', cmap='inferno',vmin=00)
plt.colorbar(label='Amplitude')
plt.title('MFCC')
plt.xlabel('Quadros (tempo)')
plt.ylabel('Coeficiente cepstral')
plt.tight_layout()
plt.show()
#%%
metadata = dataset.metadata

#%%
dataset.find_fold_indexes([1,2,3])
