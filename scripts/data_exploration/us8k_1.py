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
    signal,fs_list[i],_ = dataset.get_wavfile(i)
    n_channels[i] = signal.shape[0]
    n_fft[i] = signal.shape[1]
        
    bar.update(1)
bar.close()
#%%
valores, contagens = np.unique(fs_list, return_counts=True)

plt.bar(np.arange(len(valores)),(contagens/len(contagens))*100)
plt.xticks(np.arange(len(valores)),valores,rotation=45)
plt.grid()
plt.ylabel('Quant. relativa (%)')
plt.xlabel('$f_\\mathrm{s}$ - Taxa de amostragem (Hz)')
plt.savefig('plots/fs.pdf',transparent=True)
#%%
valores, contagens = np.unique(n_channels.astype(int), return_counts=True)

plt.bar(np.arange(len(valores)),(contagens/len(contagens))*100)
plt.xticks(np.arange(len(valores)),valores,rotation=45)
plt.grid()
plt.ylabel('Quant. relativa (%)')
plt.xlabel('Número de canais de áudio (-)')
plt.savefig('plots/n_chnl.pdf',transparent=True)
#%%
# valores, contagens = np.unique(n_fft/fs_list, return_counts=True)

df = pd.DataFrame(dict(duration=n_fft/fs_list))
sns.histplot(df,x='duration',stat='percent',bins=7)

plt.grid()
plt.ylabel('Quant. relativa (%)')
plt.xlabel('Duração do sinal (s)')
plt.savefig('plots/duration_sec.pdf',transparent=True)

#%%
metadata = dataset.metadata
sns.histplot(metadata,y='class',stat='percent')
plt.grid()
plt.ylabel('Tipo de evento')
plt.xlabel('Quant. relativa (%)')
plt.savefig('plots/event_class.pdf',transparent=True)

#%%
