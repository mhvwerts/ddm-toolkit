#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from time import time
import numpy as np

import matplotlib.pyplot as plt

# somewhat clumsy try...except imports
# to enable these test scripts to be run independently from pytest
# for example using spyder
try:
    from ddm_toolkit import ImageStructureEngine
except:
    import sys
    sys.path.append('./..')
    from ddm_toolkit import ImageStructureEngine

try:
    from datafile import get_datafilename
except:
    from tests.datafile import get_datafilename

fname = get_datafilename()
im=np.load(fname)
img=im['img']
im.close()
Ni=img.shape[0]

#    Instantiation
#    -------------
#    ise = ImageStructureEngine(
#        Npx, Nbuf, pick = Npick, avg = Navg, drop = Ndrop, 
#        apodization = 'No', fillbuf = True
#        )

print('1. Testing pick, average, drop logic')
# this is done by varying the values supplied to pick, avg and drop
#  and checking the behaviour of the internal counter variables
# no need to do the FFTs, so dummyrun is set for fast run

ISF_Nbuf = 80
ISF_Npx = 256

ISF = ImageStructureEngine(ISF_Npx, ISF_Nbuf,
                           pick = 3, avg = 1, drop = 0,
                           dummyrun=True)

t0 = time()

#        self.bufN = 0
#        self.ISFcount = 0 # ISF accumulation counter
#        self.totalframes = 0 # total frames processed
#        self.idrop = 0 # number of dropped frames
#        self.ipick = 0 # pick counter
#        self.iavg = 0 # averager counter

isum = np.zeros(Ni)
iavg = np.zeros(Ni)
ipick = np.zeros(Ni)
idrop = np.zeros(Ni)
ISFcount = np.zeros(Ni)
bufN = np.zeros(Ni) 
for it in range(Ni):
    ISF.push(img[it])
    isum[it] = ISF.isum
    iavg[it] = ISF.iavg
    ipick[it] = ISF.ipick
    idrop[it] = ISF.idrop
    ISFcount[it] = ISF.ISFcount
    bufN[it] = ISF.bufN
    print('\r\tframe #{0:d}'.format(it), end='')
t1 = time()
print('')
print('Accumulated {0:d} ISF frames in {1:6.3g} s'.format(Ni, t1-t0))

plt.figure("test_1 window 1")
plt.clf()
plt.subplot(511)

itmax = Ni
ymax = ISF.Nbuf * 1.1
#itmax=30
#ymax=5
plt.plot(isum)
plt.xlim(0,itmax)
plt.subplot(512)
plt.plot(iavg)
plt.xlim(0,itmax)
plt.subplot(513)
plt.plot(ipick)
plt.xlim(0,itmax)
plt.subplot(514)
plt.plot(bufN)
plt.xlim(0,itmax)
plt.ylim(0,ymax)
plt.subplot(515)
plt.plot(ISFcount)
plt.xlim(0,itmax)

plt.pause(2.0)


# is this necessary? => not sure if it is used elsewhere
#outfp = 'datafiles/test_1_ISF_dummy.npz'
#ISF.save(outfp)

print('TODO: test_1: add many more test (periodicity etc.) - look at graph')

def test_isum1():
    assert np.allclose(isum, 1.0)
    
def test_iavg_fluctuations():
    assert np.isclose(np.min(iavg), 1.0)
    assert np.isclose(np.max(iavg), 3.0)
    
def test_ipick_fluctuations():
    assert np.isclose(np.min(ipick), 0.0)
    assert np.isclose(np.max(ipick), 2.0)   



