import os
from .utils import set_project_dir

try:
    if ('YKK_NOCHDIR' in os.environ):
        pass
    else:
        shell = get_ipython().__class__.__name__
        ROOT = set_project_dir()
        print(f'chdir changed to {ROOT}')
except:
    print("chdir wasn't changed, if something does't work you may want to call `ykk.set_project_dir()`")
from . import neuralnet
from ._math import dsp
from .utils import specgram
