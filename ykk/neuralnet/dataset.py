import pandas as pd
import torchaudio
from torchaudio import transforms
from .._math import dsp
from torch.utils.data import Dataset


class urban8k(Dataset):
    def __init__(self,root_path='data/urbansound8k',augmentation=False,fs=44100,folds=None):
        self.data_path = root_path.replace('\\','/').removesuffix('/')
        self.metadata = None
        self.augmentation = augmentation
        self.fs = fs

        try:
            self.metadata = self._load_metadata(data_path=self.data_path)
        except:
            print('metadata not found! To download use `download_dataset()`')

        self.feat_extract = transforms.MFCC(fs,n_mfcc=13,
                                            melkwargs={
                                                "n_fft": 2048, 
                                                "hop_length": 160,
                                                "n_mels": 23
                                                })

        if not (folds is None):
            indexes = self.find_fold_indexes(folds)
            self.metadata = self.metadata.loc[indexes].reset_index(drop=True)


    def id_to_class(self,id:int) -> str:
        """Converts ClassID to class title
        """
        classes = ['air_conditioner','car_horn','children_playing',
                   'dog_bark','drilling','engine_idling','gun_shot',
                   'jackhammer','siren','street_music']
        return classes[id]


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


    def find_fold_indexes(self,folds):
        """Fold must be an integer or a list/tuple 
        with values raging from [1-10]
        """
        if isinstance(folds, (list, tuple)):
            mask = self.metadata['fold'].isin(folds)
        else:
            mask = self.metadata['fold'] == folds
        return self.metadata.index[mask]


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

    @staticmethod
    def _load_metadata(data_path):
        metadata = pd.read_csv(f'{data_path}/metadata/Urbansound8k.csv')
        metadata['file_path'] = ('fold' 
                                + metadata['fold'].astype(str) 
                                + '/' 
                                + metadata['slice_file_name'].astype(str)
                                )
    @staticmethod
    def _load_file(path):
        """Load audio file in specified path
        Returns:
            ndarray,float: Audio frames and sampling frequency
        """
        return torchaudio.load(path)
