# -*- coding: utf-8 -*-

# somewhat clumsy try...except imports
# to enable these test scripts to be run independently from pytest
# for example using spyder
try:
    from ddm_toolkit import isnotebook
except:
    import sys
    sys.path.append('./..')
    from ddm_toolkit import isnotebook



isnotebookOK = False
try:
    print('isnotebook(): ', isnotebook())
    isnotebookOK = True
except:
    isnotebookOK = False
   

def test_isnotebook_operational():
    assert isnotebookOK
    