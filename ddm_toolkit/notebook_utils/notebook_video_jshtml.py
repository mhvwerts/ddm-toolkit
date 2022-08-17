"""ddm_toolkit: A Toolkit for Differential Dynamic Microscopy (DDM)

notebook_utils.py:
    specific utility functions for using the DDM Toolkit in Jupyter Notebooks (or Google Colab etc.)
"""

from time import time

import matplotlib.pyplot as plt
import matplotlib.animation

def notebook_video_jshtml(ims, Nframes = 50, vmin = None, vmax = None):
    t0 = time()

    if (len(ims) < Nframes):
        Nframes = len(ims)

    # initialize figure to be animated
    fig, ax = plt.subplots()
    # set axes scaling
    # ax.axis([0,L,0,1.2])
    l = ax.imshow(ims[0], vmin = vmin, vmax = vmax)

    # the animation receives an index 'ix' for the profile
    # to be plotted
    def animate(ix):
        l.set_data(ims[ix])

    # animate to HTML/JavaScript
    ani = matplotlib.animation.FuncAnimation(fig, animate, frames=Nframes, interval = 100)
    jshtml_animation = ani.to_jshtml()
    t1 = time()
    print("Time needed for generating 50 frame JS animation: {0:.2f}s".format(t1 - t0))

    return jshtml_animation
