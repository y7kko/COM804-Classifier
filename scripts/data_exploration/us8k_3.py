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
# dataset._binarize_ids()
