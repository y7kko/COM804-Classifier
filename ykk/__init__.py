from .utils import set_project_dir

try:
    shell = get_ipython().__class__.__name__
    ROOT = set_project_dir()
    print(f'chdir changed to {ROOT}')
except:
    print("chdir wasn't changed, if something does't work you may want to call `ykk.set_project_dir()`")
    pass
from . import neuralnet
