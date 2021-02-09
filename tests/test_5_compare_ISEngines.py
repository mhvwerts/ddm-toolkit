#!/usr/bin/env python3
# coding: utf-8
#
# Test 6. Compare results of different ImageStructureEngines
#
# THIS SCRIPT SHOULD BE RUN FROM THE PROJECT ROOT DIRECTORY
#  (this is the parent directory to the directory in which this script is)
#  In order to run using 'Spyder', choose 'Run'->'Configuration per file', and
#  set the working directory to the project's root directory. Personally, I
#  also set 'Execute in a dedicated console' 
#
#
#TODO add test for any new (highly optimized) engine
# an optimized engine should give same result as the (slow) reference engine
#
#




import numpy as np
import matplotlib.pyplot as plt

from ddm_toolkit.tqdm import tqdm

from ddm_toolkit import ImageStructureEngine
from ddm_toolkit import ImageStructureEngine2
from ddm_toolkit import ImageStructureEngine3
from ddm_toolkit import ImageStructureFunction


print('')
print('')
print('5. Test of different ImageStructureEngine types')


# For now, use the video file from the 'simulX' sequence.
# Later we could/should use the file provided by 'datafile.py'
videof = 'datafiles/simul1_result_video.npz'


# CALCULATE VIDEO (IMAGE) STRUCTURE FUNCTION
# video was saved using: np.savez_compressed(videof, img=ims)
ims = np.load(videof)['img']

ISE_Nbuf = 50
ISE_Npx = ims.shape[1]
Nt = ims.shape[0]

#push onto DDM engines
#TODO: this could use some multiprocessing!
ISE1 = ImageStructureEngine(ISE_Npx, ISE_Nbuf)
ISE2 = ImageStructureEngine2(ISE_Npx, ISE_Nbuf)
ISE3 = ImageStructureEngine3(ISE_Npx, ISE_Nbuf)

for it in tqdm(range(Nt)):
    ISE1.push(ims[it])
    ISE2.push(ims[it])
    ISE3.push(ims[it])

ISF1 = ImageStructureFunction.fromImageStructureEngine(ISE1)
ISF2 = ImageStructureFunction.fromImageStructureEngine(ISE2)
ISF3 = ImageStructureFunction.fromImageStructureEngine(ISE3)

good2 = np.allclose(ISF1.ISF, ISF2.ISF)
good3 = np.allclose(ISF1.ISF, ISF3.ISF)
devsq2 = np.sum((ISF2.ISF - ISF1.ISF)**2)
devsq3 = np.sum((ISF3.ISF - ISF1.ISF)**2)
sq = np.sum(ISF1.ISF**2)

print('Engine type#2 gives same result as Reference Engine: ',good2)
print('   total of deviation squared:', devsq2, 'on', sq)
print()
print('Engine type#3 gives same result as Reference Engine: ',good3)
print('   total of deviation squared:', devsq3, 'on', sq)

def test_ImageStructureEngine2_OK():
    assert good2

def test_ImageStructureEngine3_OK():
    assert good3


