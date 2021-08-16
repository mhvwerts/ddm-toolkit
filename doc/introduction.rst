============
Introduction
============

These documents will become the documentation of the DDM Toolkit. At present, refer to the ``README.rst`` of the `DDM Toolkit GitHub repository`_ for basic documentation.

.. _DDM Toolkit GitHub repository: https://github.com/mhvwerts/ddm-toolkit

We aim to use `Sphinx`_ for managing the documentation of DDM Toolkit. To this end we have put in place the present ``doc`` directory, using ``sphinx-quickstart`` and following the `"first steps" instructions`_.

.. _Sphinx: https://www.sphinx-doc.org
.. _"first steps" instructions: https://www.sphinx-doc.org/en/master/usage/quickstart.html

The native format for Sphinx documents is ReStructuredText (RST), so we will use that. Incidentally, NumPy docstring (that we use for documenting the code) are also a flavour of RST.

The ``doc`` directory contains a ``Makefile`` (and a ``make.bat``). To generate the full documentation as HTML, and as PDF (requires ``latexmk``), use the following commands inside ``doc\``:

.. code-block::

   make html
   make latexpdf



To do
=====

- Simplify the ``README.rst`` of the GitHub repository, and move the details to this documentation.
- Organize the documentation.
- `nbsphinx`_ may be handy: it enables the inclusion of Jupyter Notebooks into the documentation 
- This introduction should become the introduction to DDM Toolkit, not the introduction to the process of documenting DDM Toolkit!
- Add the (NumPy-style) docstrings in the Toolkit's code to the automatic Sphinx documentation (I believe that this is typically some 'Appendix' describing the 'API').
- Improve the docstrings of the code.

.. _nbsphinx: https://nbsphinx.readthedocs.io

