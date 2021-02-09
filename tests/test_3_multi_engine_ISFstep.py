#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from time import time
import numpy as np
from ddm_toolkit import ImageStructureEngine, ImageStructureFunction
import matplotlib.pyplot as plt

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
print('3. Test run: multi-engine run')

teststep = 10
ISeng_Nbuf = 100
ISeng_Npx = 256

ISeng = ImageStructureEngine(ISeng_Npx, ISeng_Nbuf, ISFstep=teststep)
ISengB = ImageStructureEngine(ISeng_Npx, ISeng_Nbuf, ISFstep=1)

t0 = time()

Ni = 200 # SHORTER RUN

t0 = time()
for it in range(Ni):
    ISeng.push(img[it])
    ISengB.push(img[it])
    print('\r\tframe #{0:d}'.format(it), end='')
t1 = time()
print('')
print('Accumulated {0:d} ISF frames in {1:6.3g} s'.format(Ni, t1-t0))

# direct imagestructurefunction creation
ISF_a = ImageStructureFunction.fromImageStructureEngine(ISeng)
ISF_b = ImageStructureFunction.fromImageStructureEngine(ISengB)

taufi = 3
plt.figure("test_3 window 1")
plt.clf()
plt.imshow(ISF_a.ISF[taufi])
plt.title('ISF at tau = {0:d} frames'.format(ISF_a.tauf[taufi]))

taufi = 30
plt.figure("test_3 window 10")
plt.clf()
plt.imshow(ISF_b.ISF[taufi])
plt.title('ISF(b) at tau = {0:d} frames'.format(ISF_b.tauf[taufi]))

plt.figure("test_3 window 2")
plt.clf()
for taufi in range(len(ISF_a.tauf)):
    plt.plot(ISF_a.u,ISF_a.radavg(taufi),
             label='tau={0:d}'.format(ISF_a.tauf[taufi]))
plt.legend()
plt.xlabel('u / pix-1')
plt.ylabel('ISF')

ss = np.sum(ISF_a.ISF, axis = 1)
sss = np.sum(ss, axis = 1)
ssB = np.sum(ISF_b.ISF, axis = 1)
sssB = np.sum(ssB, axis = 1)
plt.figure("test_3 window 3")
plt.clf()
plt.plot(ISF_b.tauf, sssB,'-', label='ISFstep = 1')
plt.plot(ISF_a.tauf, sss,'o', label='ISFstep = 10')
plt.xlabel('tau / frames')
plt.ylabel('integrated ISF')
plt.legend()

plt.figure("test_3 window 5")
plt.clf()
# quick and dirty test for equivalence
diff = sss - sssB[::teststep]
plt.plot(ISF_a.tauf, diff)
print('EVERYTHING OK: ', np.allclose(diff, 0.0))

print('Wait for graph windows to close...')
plt.pause(4.0)

def test_ISFequivalent():
    assert np.allclose(diff, 0.0)

