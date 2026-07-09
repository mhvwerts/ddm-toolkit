# ddm_toolkit: Toolkit for Differential Dynamic Microscopy

The `tests` directory contains simple unit testing scripts that were used during development, and should still run after code has been added or changed.

We use [pytest](https://docs.pytest.org/en/latest/index.html).

Currently, unit testing is incomplete, but it will likely detect new changes to the code that break existing functionality. For this type of scientific code, *functional* testing (i.e. using 'real' benchmark test cases and verifying the overall result) is much more important than *unit* testing.

## How to test

You should run `pytest -s` from the project root directory. This runs all tests. You can run `pytest` on specific tests by invoking `pytest tests/test_<name-of-the-test>.py`


## TO DO

- The test scripts are presently really verbose, displaying graphs and     generating ample stdout output. There are also a number of redundant parts in the scripts. The test scripts need more actual `test_something` functions instead of just running the code and plotting graphs.

