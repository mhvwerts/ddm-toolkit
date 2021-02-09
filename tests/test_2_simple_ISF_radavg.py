#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from time import time
import numpy as np

import matplotlib.pyplot as plt


# somewhat clumsy try...except imports
# to enable these test scripts to be run independently from pytest
# for example using spyder
try:
    from ddm_toolkit import ImageStructureEngine, ImageStructureFunction
except:
    import sys
    sys.path.append('./..')
    from ddm_toolkit import ImageStructureEngine, ImageStructureFunction


try:
    from datafile import get_datafilename
except:
    from tests.datafile import get_datafilename

fname = get_datafilename()
im=np.load(fname)
img=im['img']
im.close()
Ni=img.shape[0]

print('')
print('2. Simple test run (default parameters) plus radial averaging')

ISeng_Nbuf = 100
ISeng_Npx = 256
ISeng = ImageStructureEngine(ISeng_Npx, ISeng_Nbuf)

t0 = time()

Ni = 150 # SHORTER RUN

t0 = time()
for it in range(Ni):
    ISeng.push(img[it])
    print('\r\tframe #{0:d}'.format(it), end='')
t1 = time()
print('')
print('Accumulated {0:d} ISF frames in {1:6.3g} s'.format(Ni, t1-t0))

# direct imagestructurefunction creation
ISF_a = ImageStructureFunction.fromImageStructureEngine(ISeng)

taufi = 1
plt.figure("test_2 window 1")
plt.clf()
plt.imshow(ISF_a.ISF[taufi])

plt.figure("test_2 window 2")
plt.clf()
for taufi in range(len(ISF_a.tauf)):
    plt.plot(ISF_a.u,ISF_a.radavg(taufi))

print('Wait for graph windows to close...')
plt.pause(2.0)



def test_ISFcount():
    assert ISeng.ISFcount == (Ni - ISeng_Nbuf)
    
def test_tauf():
    assert len(ISF_a.tauf) == (ISeng_Nbuf + 1)
    
