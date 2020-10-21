# build C module 
# Usage: ``python build_c.py build_ext --inplace``
from distutils.core import setup, Extension
import numpy
setup(name='_tifffile',
      ext_modules=[Extension('_tifffile', ['_tifffile.c'],
                             include_dirs=[numpy.get_include()])])
