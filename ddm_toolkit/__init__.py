# -*- coding: utf-8 -*-
#

from .ddm import ImageStructureEngine
from .ddm import ImageStructureEngine2
from .ddm import ImageStructureEngine3
from .ddm import ImageStructureEngineSelector
from .ddm import ImageStructureFunction
from .ddm import ImageStructureFunctionRadAvg
from .parameters import sim_params
from .parameters import sim_params_empty
from .analysis import ISFanalysis_simple_brownian
from .functions import isnotebook

if isnotebook():
    print('running inside a Notebook (Jupyter or otherwise)...')
    try:
        from tqdm.notebook import tqdm
    except ModuleNotFoundError:
        print('system tqdm not found; using legacy version.')
        from .tqdm_legacy.notebook import tqdm
else:
    try:
        from tqdm import tqdm
    except ModuleNotFoundError:
        print('system tqdm not found; using legacy version.')
        from .tqdm_legacy import tqdm

#TODO: add
# from tqdm.notebook import tqdm

