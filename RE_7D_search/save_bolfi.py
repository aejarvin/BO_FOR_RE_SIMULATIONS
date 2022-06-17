# Standard python packages
import pickle
import numpy as np
# Import elfi
import elfi

def save_bolfi(bolfi,fname='bolfi_save.pkl'):
    """ This function saves the bolfi sample dictionary to a pickle file
    Parameters:
    -----------
    bolfi : bolfi instance
    fname : output filename
    """
    # Extract X and Y for the samples.
    X = bolfi.target_model.X
    Y = bolfi.target_model.Y
    counter = 0
    di = {}
    # Build a dictionary of the sample X and Y values
    for i in bolfi.model.parameter_names:
        di[i] = np.array(X[:,counter])
        counter = counter + 1
    di['log_d'] = np.array(Y.T)
    # Write the dictionary to a file
    with open(fname, 'wb') as f:
        pickle.dump(di, f)
    return 
