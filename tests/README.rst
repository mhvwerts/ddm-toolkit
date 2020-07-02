========================================================
ddm_toolkit: Toolkit for Differential Dynamic Microscopy
========================================================

The `tests` directory contains simple unit testing scripts that were used during development, and should still run after code has been added or changed.

We wish to use `pytest`_.

.. _pytest: https://docs.pytest.org/en/latest/index.html

Currently, unit testing is largely incomplete, but it will probably detect new changes to the code that break existing functionality. For this type of scientific code, *functional* testing (i.e. using 'real' benchmark test cases and verifying the overall result) is much more important than *unit* testing.

-----------
How to test
-----------

You should run ``pytest`` from the project root directory. This runs all tests. You can run ``pytest`` on specific tests by invoking ``pytest tests/test_<name-of-the-test>.py``

Before doing ``pytest``, you may also directly run the ``tests/test_<name-of-the-test>.py`` script in Python using ``python -m tests.test_<name-of-the-test>`` (without the ``.py`` extension) so that you can inspect the console output before applying the ``pytest``.

All tests are best run from the project's root directory (i.e. have this root directory as the current working directory). Running the script from within the ``tests`` directory may not work (problems with importing modules). Some tests, however, can be run with ``tests`` as the working directory so that they can be edited and run using Spyder (these have a work-around for the import problem).


-----
TO DO
-----

- The test scripts are presently really verbose, displaying graphs and generating ample stdout output. There are also a number of redundant parts in the scripts. The test scripts need more actual ``test_something`` functions with ``assert``  instead of just running the code and plotting graphs.
- Create 'silent' testing! (Perhaps as an option to be supplied, or just in a general sense)



