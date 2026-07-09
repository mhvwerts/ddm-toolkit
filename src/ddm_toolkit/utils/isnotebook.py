# Environment checking/initialization functions

def isnotebook():
    """Check if running in a Jupyter Notebook.

    from: https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook

    If someone is interested in detecting whether the notebook is running
    on Google Colab you can check this:
      get_ipython().__class__.__module__ == "google.colab._shell"
    The 'get_ipython().__class__.__module__' may actually provide a more precise
    evaluation of the running environment

    Spyder => 'spyder_kernels.console.kernel'
    Google Colab => 'google.colab._shell'
    iPython console => 'IPython.terminal.interactiveshell'

    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        elif shell == 'Shell':
            return True   # Google Colab notebook returns this
        elif shell == 'SpyderShell':
            return False  # running in Spyder Terminal
        else:
            print("WARNING: unknown ipython shell type '"+shell+"'")
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter
