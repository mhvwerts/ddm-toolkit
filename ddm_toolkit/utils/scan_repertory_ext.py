#%%
import os

#%% Scan repertory to extract only '.txt' config
def scan_repertory_ext(folder_path, ext):

    '''
    '''

    temp_file_name = [f.name for f in os.scandir(folder_path) if f.is_file()] # Generate a list with the name of each file in a given path

    # Retrieve only files with a given extension

    file_list = [] # Initialize the list containing file name

    for i in temp_file_name: # Enumerate each element of temp list
        if i.endswith(ext): # Look for file having '.ext' extension
            file_list.append(i) # Store the filename

    return file_list
