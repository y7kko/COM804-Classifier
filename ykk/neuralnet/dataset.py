import pandas as pd
import torchaudio
from torchaudio import transforms
from .._math import dsp

class urban8k:
    def __init__(self,root_path='data/urbansound8k',augmentation=False,fs=44100):
        self.data_path = root_path.replace('\\','/').removesuffix('/')
        self.metadata = None
        self.augmentation = augmentation
        self.fs = fs
        try:
            self.metadata = pd.read_csv(f'{self.data_path}/metadata/Urbansound8k.csv')
            self.metadata['file_path'] = ('fold' 
                                          + self.metadata['fold'].astype(str) 
                                          + '/' 
                                          + self.metadata['slice_file_name'].astype(str)
                                          )
        except:
            print('metadata not found! To download use `download_dataset()`')

        self.feat_extract = transforms.MFCC(fs,n_mfcc=13,
                                            melkwargs={
                                                "n_fft": 2048, 
                                                "hop_length": 160,
                                                "n_mels": 23
                                                })


    def id_to_class(self,id:int) -> str:
        """Converts ClassID to class title
        """
        classes = ['air_conditioner','car_horn','children_playing',
                   'dog_bark','drilling','engine_idling','gun_shot',
                   'jackhammer','siren','street_music']
        return classes[id]


    def _load_file(self,path):
        """Load audio file in specified path
        Returns:
            ndarray,float: Audio frames and sampling frequency
        """
        return torchaudio.load(path)

    def __getitem__(self, index):
        signal,fs_,label = self.get_wavfile(index)
        signal          = dsp.ensure_mono(signal)
        signal          = dsp.resample(signal,fs=fs_,fs_new=self.fs)
        signal          = dsp.zero_pad(signal,fs_,4.0)
        if self.augmentation:
            pass

        features = self.feat_extract(signal)
        return features,label


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


    def find_fold_indexes(self,fold):
        """Fold must be an integer between [1,10]
        """
        return self.meta[self.meta['fold']==fold].index


    def download_dataset(self,path=None):
        """Downloads dataset, if specified path is None,
        the dataset will be download to `self.path`
        """
        import soundata
        if path is None:
            path = self.data_path
        dataset = soundata.initialize('urbansound8k',
                                    data_home = path)
        dataset.download()  # download the dataset
        dataset.validate()  # validate that all the expected files are there
        print('dataset downloaded')

    def __len__(self):
        return len(self.metadata)
