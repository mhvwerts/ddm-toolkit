# -*- coding: utf-8 -*-
#
from .utils import isnotebook


if isnotebook():
    print('Running inside a Notebook (Jupyter or otherwise)...')
    try:
        from tqdm.notebook import tqdm
    except ModuleNotFoundError:
        print('Warning: module tqdm not found; using local legacy version.')
        from .tqdm_legacy.notebook import tqdm
else:
    try:
        from tqdm import tqdm
    except ModuleNotFoundError:
        print('Warning; module tqdm not found; using local legacy version.')
        from .tqdm_legacy import tqdm

