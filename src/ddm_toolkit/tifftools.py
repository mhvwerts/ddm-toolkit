# -*- coding: utf-8 -*-


# tifffile aliases (DDM toolkit depends on Gohlke's tifffile package)

from tifffile import TiffFile
from tifffile import imread as tiff_imread
from tifffile import imsave as tiff_imsave
from tifffile import TiffWriter
from tifffile import __version__ as tifffile_version

# print('tifffile version:', tifffile_version)
