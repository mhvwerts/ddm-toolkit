# -*- coding: utf-8 -*-
#
from .utils import isnotebook


if isnotebook():
    print('Running inside a Notebook (Jupyter or otherwise)...')
    from tqdm.notebook import tqdm

else:
    from tqdm import tqdm


