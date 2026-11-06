import pandas as pd
import torchaudio
from torchaudio import transforms, functional as F
from .._math import dsp
from torch.utils.data import Dataset
import numpy as np
import numpy.random as random
import torch
from tqdm.auto import tqdm
from audiomentations import Compose,PitchShift,TimeStretch

class urban8k(Dataset):
    def __init__(self,root_path='data/urbansound8k',
                 augmentation=False, fs=44100, filter_folds=None,random_state=None,binary=False):
        self.data_path = root_path.replace('\\','/').removesuffix('/')
        self.metadata = None
        self.augmentation_is_enabled = augmentation
        self.fs = fs

        self.troll_range = 4.0

        self.random_state =random_state
        self.rng = np.random.default_rng(seed=random_state)


        try:
            self.metadata = self._load_metadata(data_path=self.data_path)
            self.dataset_is_ok = True
        except:
            self.dataset_is_ok = False
            print('metadata not found! To download use `download_dataset()`')

        mel_kw = {"n_fft": 2048, "hop_length": 256, "n_mels": 23}
        self.set_transform(transforms.MFCC(self.fs,n_mfcc=20,melkwargs=mel_kw))

        if binary: #Separa em Mechanical/ Non Mechanical
            self._binarize_ids()

        if not (filter_folds is None):
            # print('tentndo')
            indexes = self.find_fold_indexes(filter_folds)
            self.metadata = self.metadata.loc[indexes].reset_index(drop=True)

        self.augment = Compose([
            PitchShift(-2,+2,p=.5),
            TimeStretch(.8,1.2,True,p=.5)
        ])

        self._signals_are_cached = False
        self._features_are_cached = False

    def set_transform(self,transform_object):
        self._transform = transform_object

    def __getitem__(self, index):
        if self._features_are_cached:
            return self._get_cached_features(index)
            # return self.cached_features[index],self.cached_labels[index]

        # load and standardize data
        signal, fs_, label  = self.get_wavfile(index)
        features = self._extract_features(signal,fs_)
        return features,label


    def _extract_features(self,signal,fs_):
        if self.augmentation_is_enabled:
            signal = self.augment(signal.numpy(),self.fs)
            signal = torch.from_numpy(np.ascontiguousarray(signal))
            
            t_roll = self._sym_rand() * self.troll_range
            signal = dsp.troll(signal, self.fs, time=int(t_roll))

        # Applies MFCC transform
        features = self._transform(signal)
        return features

    def get_wavfile(self,index):
        """Returns signal correspondent to the given index

        Args:
            index (int): index of signal

        Returns:
            signal(Tensor) ,fs(float),label(int)
        """
        if self._signals_are_cached:
            return self.cached_signals[index],self.fs,self.cached_labels[index]
        
        data = self.metadata.iloc[index]
        label = data['classID']
        path = f"{self.data_path}/audio/{data['file_path']}"
        signal, fs = self._load_file(path)
        signal              = dsp.remove_offset(signal)
        signal              = dsp.ensure_mono(signal)
        signal              = dsp.resample(signal,fs=fs,fs_new=self.fs)
        signal              = dsp.normalize(signal)
        signal = dsp.center_crop(signal, self.fs, 3.0)
        signal = dsp.pad_until(signal, self.fs, 3.0)
        return signal,self.fs, label


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


    # def cache_features(self,):
    #     if self.augmentation_is_enabled:
    #         print('Warning: augmentation is enabled')

    #     all_features, all_labels =[], []
    #     for i in tqdm(range(len(self)),desc='Caching features'):
    #         features, label = self[i]  # ou self.__getitem__(i)
    #         all_features.append(features)
    #         all_labels.append(label)
        
    #     # stack -> (N, dim1, dim2)
    #     self.cached_features = torch.stack(all_features, dim=0)
    #     self.cached_labels = torch.stack([torch.as_tensor(l) for l in all_labels], dim=0)
    #     self._features_are_cached = True
    #     return self.cached_features, self.cached_labels


    def cache_signals(self,waitbar=True):
        s0, _, _ = self.get_wavfile(0)
        N = len(self)
        self.cached_signals = torch.empty((N, *s0.shape), dtype=s0.dtype)
        self.cached_labels  = torch.empty(N, dtype=torch.long)
        for i in tqdm(range(N), desc='Caching signals',disable=(not waitbar)):
            sig, _, lab = self.get_wavfile(i)
            self.cached_signals[i] = sig      # cópia in-place, sem pico
            self.cached_labels[i]  = lab

        self._signals_are_cached = True
        # return self.cached_signals, self.cached_labels


    def cache_features(self,augmentation_sets=None,waitbar=True):
        s0, _ = self[0]
        N = len(self)

        self._feature_cache_size = 1
        if self.augmentation_is_enabled and not (augmentation_sets is None):
            self._feature_cache_size = augmentation_sets
        
        self.cached_features = []
        self.cached_labels  = torch.empty(N, dtype=torch.long)
        for _set in range(self._feature_cache_size):
            cached_feature_set = torch.empty((N, *s0.shape), dtype=s0.dtype)
            for i in tqdm(range(len(self)),desc=f'Caching features ({_set}/{augmentation_sets-1})',disable=(not waitbar)):
                features, label = self[i]
                cached_feature_set[i] = features
                self.cached_labels[i] = label #sim, podia sair do loop...
            self.cached_features.append(cached_feature_set) # (SET)(N,s01,s02)
        self._features_are_cached = True


    def _get_cached_features(self,index):
        label = self.cached_labels[index]
        if self.augmentation_is_enabled:
            _set = self.rng.integers(0,self._feature_cache_size)
            feature = self.cached_features[_set][index]
        else:
            feature = self.cached_features[0][index]
        return feature,label
