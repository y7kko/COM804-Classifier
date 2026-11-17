#%%
import ykk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ykk.neuralnet import urban8k
import seaborn as sns
ykk.utils.configure()
from tqdm import tqdm
#%%
dataset = urban8k()

#%%

bar = tqdm(range(len(dataset)))
fs_list = np.zeros(len(dataset))
n_channels = np.zeros(len(dataset))
n_fft = np.zeros(len(dataset))
for i in range(len(dataset)):
    signal,fs_list[i],_ = dataset.get_wavfile(i,bypass_standadization=True)
    n_channels[i] = signal.shape[0]
    n_fft[i] = signal.shape[1]
        
    bar.update(1)
bar.close()
#%%
sc = .75
plt.figure(figsize=(7*sc,3*sc))
valores, contagens = np.unique(fs_list, return_counts=True)

plt.bar(np.arange(len(valores)),(contagens/len(fs_list))*100)
plt.xticks(np.arange(len(valores)),[f'{val:.0f}' for val in valores],rotation=45)
plt.grid()
plt.ylabel('Quant. relativa (%)')
plt.xlabel('$f_\\mathrm{s}$ - Taxa de amostragem (Hz)')
plt.tight_layout()
plt.savefig('plots/fs.pdf',transparent=True)
#%%
valores, contagens = np.unique(n_channels.astype(int), return_counts=True)

sc = .75
plt.figure(figsize=(7*sc,3*sc))
plt.bar(np.arange(len(valores)),(contagens/len(n_channels))*100)
plt.xticks(np.arange(len(valores)),valores,rotation=45)
plt.grid()
plt.ylabel('Quant. relativa (%)')
plt.xlabel('Número de canais de áudio (-)')
plt.tight_layout()
plt.savefig('plots/n_chnl.pdf',transparent=True)
#%%
# valores, contagens = np.unique(n_fft/fs_list, return_counts=True)
sc = .75
plt.figure(figsize=(7*sc,3*sc))
df = pd.DataFrame(dict(duration=n_fft/fs_list))
sns.histplot(df,x='duration',stat='percent',bins=12)

plt.grid()
plt.ylabel('Quant. relativa (%)')
plt.xlabel('Duração do sinal (s)')
plt.tight_layout()
plt.savefig('plots/duration_sec.pdf',transparent=True)

#%%
sc = .75
plt.figure(figsize=(7*sc,3*sc))
metadata = dataset.metadata
sns.histplot(metadata,y='class',stat='percent')
plt.grid()
plt.ylabel('Tipo de evento')
# plt.gca().tick_params('y',rotation=45)
plt.xlabel('Quant. relativa (%)')
plt.tight_layout()
plt.savefig('plots/event_class.pdf',transparent=True)

#%%

#%%

dataset = urban8k()
dataset._binarize_ids()
# %%
sc = .75
plt.figure(figsize=(7*sc,3*sc))
metadata = dataset.metadata
metadata.loc[metadata['classID'] == 0, 'class'] = 'Não Mecânico'
metadata.loc[metadata['classID'] == 1, 'class'] = 'Mecânico'
sns.histplot(metadata,y='class',stat='percent')
# plt.yticks([0,1],['Não Mecânico', 'Mecânico'])
plt.grid()
plt.ylabel('Tipo de evento')
# plt.gca().tick_params('y',rotation=45)
plt.xlabel('Quant. relativa (%)')
plt.tight_layout()
plt.savefig('plots/event_class_binary.pdf',transparent=True)
#%%
len(metadata)
