from sys import argv
from configparser import ConfigParser
from .DDMParams import DDMParams

def DDMParams_from_configfile_or_defaultpars(simulation = True):
    """
    Get parameters either from the default parameters (defined in this module)
    or from a config file (if specified as command line argument)

    Parameters
    ----------
    simulation (optional) : boolean
        Use parameter set for simulation-analysis, else real video parameters


    Returns
    -------
    DDMParams object with DDM parameters


    """

    argc = len(argv)
    # print('Argument count :',argc)
    # print('Argument vector :',argv)
    if argc == 1:
        print('Using default (simulation) parameters from ddm_toolkit.parameters.')
        params = DDMParams.defaultSimulationParams()
    elif argc == 2:
        parfn = argv[1]
        print('Using parameters from file:', parfn)

        # Determine if this file is a simulation or a real video configuration
        #
        #TODO:
        # This section, together with the 'fromSimulationConfigFile'
        # and the 'fromRealVideoConfigFile', may be refactored
        CP = ConfigParser(interpolation=None)
        CP.read(parfn)
        simulation = True
        try:
            if CP['general']['real_video']=='True':
                simulation = False
        except KeyError:
            pass
        if simulation:
            params = DDMParams.fromSimulationConfigFile(parfn)
        else:
            params = DDMParams.fromRealVideoConfigFile(parfn)
    else:
        print('argc = ',argc)
        print('argv = ',argv)
        raise Exception('invalid number of arguments')
    return params
