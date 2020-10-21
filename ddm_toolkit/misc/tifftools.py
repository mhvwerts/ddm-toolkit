# -*- coding: utf-8 -*-


# tifffile: use installed tifffile if available, else
# use 'tifffile_fork' legacy tifffile
try:
    from tifffile import TiffFile
    from tifffile import imread as tiff_imread
    from tifffile import __version__ as tifffile_version
except:
    from .tifffile_fork import TiffFile
    from .tifffile_fork import imread as tiff_imread
    from .tifffile_fork import __version__ as tifffile_version
# print('tifffile version:', tifffile_version)

