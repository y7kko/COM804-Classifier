import os
import json

def configure(config_path='configs.json'):
    with open(config_path) as f:
        configs = json.load(f)

    if 'ffmpeg_dll' in configs:
        print('ffmpeg added to path')
        os.add_dll_directory(configs['ffmpeg_dll'])
