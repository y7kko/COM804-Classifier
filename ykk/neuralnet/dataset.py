import pandas as pd
import torchaudio
from torchaudio import transforms, functional as F
from .._math import dsp
from torch.utils.data import Dataset
import numpy as np
import numpy.random as random
import torch
_preloaded = []


class urban8k(Dataset):
    def __init__(self,root_path='data/urbansound8k',
                 augmentation=False, fs=44100, filter_folds=None,random_state=None,binary=False):
        self.data_path = root_path.replace('\\','/').removesuffix('/')
        self.metadata = None
        self.augmentation = augmentation
        self.fs = fs

        self.shift_range = 2.0
        self.troll_range = 4.0
        self.strech_pct = 10

        self.random_state =random_state
        self.rng = np.random.default_rng(seed=random_state)


        try:
            self.metadata = self._load_metadata(data_path=self.data_path)
            self.dataset_is_ok = True
        except:
            self.dataset_is_ok = False
            print('metadata not found! To download use `download_dataset()`')

        mel_kw = {"n_fft": 2048, "hop_length": 256, "n_mels": 23}
        self.set_mfcc_params(n_mfcc=20,mel_kw=mel_kw)

        if binary: #Separa em Mechanical/ Non Mechanical
            self._binarize_ids()

        if not (filter_folds is None):
            # print('tentndo')
            indexes = self.find_fold_indexes(filter_folds)
            self.metadata = self.metadata.loc[indexes].reset_index(drop=True)

        self._cached_features = False


    def set_mfcc_params(self,n_mfcc=20,mel_kw=None):
        self.mfcc_transform = transforms.MFCC(self.fs,n_mfcc=n_mfcc,
                                            melkwargs=mel_kw)

    def __getitem__(self, index):
        if self.cached_features:
            return self.cache_features[index], self.cached_labels[index]

        # load and standardize data
        signal, fs_, label  = self.get_wavfile(index)
        features = self._extract_features(signal,fs_)
        return features,label

    def cache_features(self):
        all_features = []
        all_labels = []
        
        for i in range(len(self)):
            features, label = self[i]  # ou self.__getitem__(i)
            all_features.append(features)
            all_labels.append(label)
        
        # stack -> (N, dim1, dim2)
        self.cached_features = torch.stack(all_features, dim=0)
        self.cached_labels = torch.stack([torch.as_tensor(l) for l in all_labels], dim=0)
        self._cached_features = True
        return self.cached_features, self.cached_labels


    def _extract_features(self,signal,fs_):
        signal              = dsp.remove_offset(signal)
        signal              = dsp.ensure_mono(signal)
        signal              = dsp.resample(signal,fs=fs_,fs_new=self.fs)
        signal              = dsp.normalize(signal)

        if self.augmentation:
            stretch_pct = 1 + self._sym_rand() * (self.strech_pct/100)
            s_st = self._sym_rand() * self.shift_range
            # Muda a altura do sinal
            signal = dsp.pitch_shift(signal,self.fs,n_st=s_st)
            # Muda o velocidade do sinal
            signal = dsp.timestretch(signal,stretch_pct)

        # Mantem apenas o 4s centrais do audio (Sinais muito longos)
        signal = dsp.center_crop(signal, self.fs, 4.0)
        signal = dsp.pad_until(signal, self.fs, 4.0)

        #circshift deve ser apos zero pad, senão
        # o final vira inutil
        if self.augmentation:
            t_roll = self._sym_rand() * self.troll_range
            signal = dsp.troll(signal, self.fs, time=int(t_roll))

        # Applies MFCC transform
        features = self.mfcc_transform(signal)
        return features

    def get_wavfile(self,index):
        """Returns signal correspondent to the given index

        Args:
            index (int): index of signal

        Returns:
            signal(Tensor) ,fs(float),label(int)
        """
        data = self.metadata.iloc[index]
        label = data['classID']
        path = f'{self.data_path}/audio/{data['file_path']}'
        signal, fs = self._load_file(path)
        return signal, fs, label


    def find_fold_indexes(self,folds):
        """Fold must be an integer or a list/tuple 
        with values raging from [1-10]
        """
        if isinstance(folds, (list, tuple)):
            mask = self.metadata['fold'].isin(folds)
        else:
            mask = self.metadata['fold'] == folds
        return self.metadata.index[mask]


    def download_dataset(self,path=None,force_overwrite=False):
        """Downloads dataset, if specified path is None,
        the dataset will be download to `self.path`
        """
        if self.dataset_is_ok and not force_overwrite:
            return
        
        import soundata
        if path is None:
            path = self.data_path
        dataset = soundata.initialize('urbansound8k',
                                    data_home = path)
        dataset.download(force_overwrite=force_overwrite)  # download the dataset
        dataset.validate()  # validate that all the expected files are there
        self.dataset_is_ok = True
        print('dataset downloaded')


    def __len__(self):
        return len(self.metadata)

    @staticmethod
    def _load_metadata(data_path):
        df = pd.read_csv(f'{data_path}/metadata/UrbanSound8K.csv')
        df['file_path'] = ('fold' 
                                + df['fold'].astype(str) 
                                + '/' 
                                + df['slice_file_name'].astype(str)
                                )
        return df
    

    def _load_file(self,path):
        """Load audio file in specified path
        Returns:
            ndarray,float: Audio frames and sampling frequency
        """
        return torchaudio.load(path)


    def decode_id(self,id:int) -> str:
        """Converts ClassID to class title
        """
        classes = ['air_conditioner','car_horn','children_playing',
                   'dog_bark','drilling','engine_idling','gun_shot',
                   'jackhammer','siren','street_music']
        return classes[id]


    def _sym_rand(self,size=None):
        """Returns random values in the half open interval [-1,1[
        """
        return (self.rng.random(size)-.5)*2

    def _binarize_ids(self,):
        class_translate = np.array([1,1,0,0,1,1,1,1,1,0])
        self.metadata['classID'] = class_translate[self.metadata['classID'].to_numpy()]
