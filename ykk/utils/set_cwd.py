from pathlib import Path
import os
def set_project_dir():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    os.chdir(PROJECT_ROOT)
    return PROJECT_ROOT
