import pickle
import numpy as np
import elfi

def save_bolfi(bolfi,fname='bolfi_save.pkl'):
    X = bolfi.target_model.X
    Y = bolfi.target_model.Y
    counter = 0
    di = {}
    for i in bolfi.model.parameter_names:
        di[i] = np.array(X[:,counter])
        counter = counter + 1
    di['log_d'] = np.array(Y.T)
    with open(fname, 'wb') as f:
        pickle.dump(di, f)
    return 
    
    
