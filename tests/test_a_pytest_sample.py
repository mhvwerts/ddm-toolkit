#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def func(x):
    return x + 2


def test_answer():
    assert func(3) == 5
    
# what if assert in main?
#assert func(2) == 4
# well, basically this crashes your script, and pytesting is not proceeding correctly
# the test assert needs to be in a test_function 
