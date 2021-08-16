# -*- coding: utf-8 -*-
#

from .ddm import ImageStructureEngine
from .ddm import ImageStructureFunction
from .ddm import ImageStructureFunctionRadAvg
from .parameters import DDMParams
from .parameters import DDMParams_from_configfile_or_defaultpars
from .analysis import ISFanalysis_simple_brownian
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

