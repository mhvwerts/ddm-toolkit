# -*- coding: utf-8 -*-

from ddm_toolkit import isnotebook


isnotebookOK = False
try:
    print('isnotebook(): ', isnotebook())
    isnotebookOK = True
except:
    isnotebookOK = False
   

def test_isnotebook_operational():
    assert isnotebookOK
    