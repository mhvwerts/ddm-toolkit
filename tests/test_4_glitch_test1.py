#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
First version of 'glitch detection' that detects glitches in
videos of Brownian motion, i.e. skipped frames?

It relies on calculating a stack of ISF at constant lag and then
analyzing that stack, e.g. using SVD (but at present it just calculates
total power)

It requires to set a maximum number of processed ISFs

For the video it can be start and end frames
"""
import numpy as np
import matplotlib.pyplot as plt

# somewhat clumsy try...except imports
# to enable these test scripts to be run independently from pytest
# for example using spyder
try:
    from ddm_toolkit import glitchdetect_array
except:
    import sys
    sys.path.append('./..')
    from ddm_toolkit import glitchdetect_array

try:
    from datafile import get_datafilename
except:
    from tests.datafile import get_datafilename

print('')
print('4. Glitch detection test')

#===
#process parameters
#----------


infile = get_datafilename()
ISF_Npx = 256

im=np.load(infile)
imgorig=im['img']
im.close()

# create a 'glitch' (dropped frames)
Noff = 1
Ni = 150 # must be smaller than len(imgorig)
glitchpos = 17
Nglitch = 1
img = np.zeros_like(imgorig[Noff+0:Noff+Ni])
img[0:glitchpos] = imgorig[Noff+0:Noff+glitchpos]
img[glitchpos:] = imgorig[Noff+glitchpos+Nglitch:Noff+Ni+Nglitch]

print(len(imgorig))
print(len(img))
tot = glitchdetect_array(img)

plt.figure('test_glitch_test_1')
plt.clf()
plt.plot(tot)
print('Wait for graph windows to close...')
plt.pause(2.0)

#'some statistics should be applied'
# but now just see where the detected glitch is

detectedglitchpos = np.argmax(tot) + 1

def test_glitchposfound():
    assert detectedglitchpos == glitchpos





