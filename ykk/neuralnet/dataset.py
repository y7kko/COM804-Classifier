import pandas as pd
import torchaudio
from ..math import dsp

class urban8k:
    def __init__(self,root_path='data/urbansound8k'):
        self.data_path = root_path.replace('\\','/').removesuffix('/')
        self.metadata = None
        try:
            self.metadata = pd.read_csv(f'{self.data_path}/metadata/Urbansound8k.csv')
            self.metadata['file_path'] = 'fold' + self.metadata['fold'].astype(str) + '/' + self.metadata['slice_file_name'].astype(str)
        except:
            print('metadata not found! To download use `download_dataset()`')


    def id_to_class(self,id:int) -> str:
        classes = ['air_conditioner','car_horn','children_playing','dog_bark','drilling','engine_idling','gun_shot','jackhammer','siren','street_music']
        return classes[id]


    def _load_file(self,path):
        """Load audio file in specified path
        Returns:
            ndarray,float: Audio frames and sampling frequency
        """
        return torchaudio.load(path,backend="soundfile")

    def __getitem__(self, index):
        signal,fs,label = self.get_wavfile(index)
        signal = dsp.ensure_mono(signal)
        signal = dsp.resample(signal,fs=fs,fs_new=44100)
        signal = dsp.zero_pad(signal,fs,4)
        features = signal #Depois aplicar MFCC
        return features,label

    def get_wavfile(self,index):
        data = self.metadata.iloc[index]
        label = data['classID']
        path = f'{self.data_path}/audio/{data['file_path']}'
        signal, fs = self._load_file(path)
        return signal, fs, label

    def find_fold_indexes(self,fold):
        """Fold must be an integer between [1,10]
        """
        return self.meta[self.meta['fold']==fold].index

    def download_dataset(self,path=None):
        import soundata
        if path is None:
            path = self.data_path
        dataset = soundata.initialize('urbansound8k',
                                    data_home = path)
        dataset.download()  # download the dataset
        dataset.validate()  # validate that all the expected files are there

    def __len__(self):
        return len(self.metadata)
    