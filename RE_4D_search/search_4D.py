# Standard python packages.
import numpy as np
import matplotlib.pyplot as plt
import h5py
import pickle
import scipy.stats as stats
import multiprocessing
import sys

# Python packages for Bayesian inference and GPR
import elfi
import GPy
from elfi.methods.bo.gpy_regression import GPyRegression
from elfi.methods.bo.acquisition import RandMaxVar
from elfi.model.extensions import ModelPrior

sys.path.append('../common/')

# Import the function that runs DREAM
from run_CQ_fluid_simulation import run_CQ_fluid_simulation

# Import the summary function
from IP_summary_25ms import IP_summary_25ms

# A function to save the sample dictionary to a pickle file
from save_bolfi import save_bolfi

def sim_fn(Tf, nAr, alpha, beta, batch_size=1, random_state=None):
    """ This is a wrapper function for the simulation model.
    
    Parameters
    ----------
    Tf : Electron temperature (eV)
    nAr : Argon assimilation fraction (%)
    alpha : alpha parameter of the RE seed distribution (Gamma distribution pdf)
    beta : beta parameter of the RE seed distribution (Gamma distribution pdf)
    batch_size : a dummy variable required by BOLFI, but not used
    random_state : a dummy variable required by BOLFI, but not used
    
    Returns
    -------
    Total plasma current as a function of time as an np.array.
    """
    file1 = run_CQ_fluid_simulation(T_f=Tf[0],nAr_frac=nAr[0], 
                                    walltime=5.0, 
                                    alpha=alpha[0],
                                    beta=beta[0], 
                                    Nre = 1e10,
                                    inputfilename = 'bolfi_4D_runs/test1i.h5',
                                    outputfilename = 'bolfi_4D_runs/test1o.h5', 
                                    initoutfile = 'bolfi_4D_runs/'+str(float(Tf))+'init_out.h5',
                                    CQfluidoutfile = 'bolfi_4D_runs/'+str(float(Tf))+'CQ_fluid_out.h5')
    timev = np.array(file1['grid']['t'][:])
    currentv = np.array(file1['eqsys']['I_p'][:])
    currentv = np.squeeze(currentv)
    output = np.array([np.transpose(timev),np.transpose(currentv)])
    return output

def distance1(summary, observed):       
    """Calculate the L1 norm between the simulated and observed plasma currents.
    Parameters
    ----------
    summary : Simulated Ip mapped to common time axis with the measured Ip
    observed : Measured Ip mapped to common time axis qith the simulated Ip
    Returns
    -------
    distance as an np.array 
    """
    y = observed 
    x = summary

    # L1 norm 
    dist = np.sum(np.abs(x - y))
    dist = np.array([dist])
    return dist

def get_model(true_params=None, seed_obs=None, **kwargs):
     """ A convenience function that defines the ELFI model
    
    Returns
    -------
    An ELFI model.
    
    """
    # Initialize an ELFI model.
    m = elfi.ElfiModel()

    # Define the prior distributions for the uncertain parameters
    priors = []
    priors.append(elfi.Prior('uniform', 1.0, 19, model=m, name='T_f'))
    priors.append(elfi.Prior('uniform', 0.001, 100, model=m, name='nAr'))
    priors.append(elfi.Prior('uniform', 0.001, 10, model=m, name='alpha'))
    priors.append(elfi.Prior('uniform', 0.001, 10, model=m, name='beta'))

    # Load observed/measured data
    # N.B. The IP_out.h5 corresponds to experimental data from
    # JET and is not given in this repository. However, it is possible 
    # to create artificial current evolution data to try the algorithm.
    exp_data = h5py.File('../experimental_data/IP_out.h5','r')
    exp_datar = exp_data['IP_data']
    exp_d = np.array(exp_datar)

    # Define the ELFI simulator. Here ELFI links to the wrapper
    # function defined above.
    elfi.Simulator(sim_fn, *priors, observed=exp_d, name='CQf')

    # Define the summary function. IP_summary_25ms is given
    # externally and is simply maps the Ip to a unified
    # timebase between 0 and 25 ms.
    S1 = elfi.Summary(IP_summary_25ms, m['CQf'], name='IPsum')

    # Define the method to measure discrepancy. This is given
    # by the distance1 function defined above.
    elfi.Discrepancy(distance1, m['IPsum'], name='d')
    
    return m

if __name__ == '__main__':
    """ This run script completes the steps to run BOLFI for the 
    search task defined above.
    """
    # Setup the multiprocessing environment. Standard multiprocessing
    # seems to work fine. Also IPyparallel was tested but brings more
    # overhead in terms of setting up. For very large simulations it 
    # might anyway be smart to take them out of the python wrapper.   
    elfi.set_client('multiprocessing')

    # Get the ELFI model.
    m = get_model()

    # Take logarithm of the discrepancy
    log_d = elfi.Operation(np.log, m['d'])

    # If a pre-existing dictionary of samples is to be loaded,
    # give the dictionary filename here, otherwise set the
    # dictionary_name = None.
    dictionary_name = None
    if dictionary_name == None:
        result_dictionary = None
    else:
        with open(dictionary_name, 'rb') as f:
                result_dict = pickle.load(f)
        result_dictionary = result_dict

    # Define the GPR kernel
    kernel = GPy.kern.RatQuad(input_dim=4, ARD=True)
    kernel.lengthscale[[0]].constrain_bounded(1e-3,1.0)
    kernel.lengthscale[[1]].constrain_bounded(1e-3,0.5)
    kernel.lengthscale[[2]].constrain_bounded(1e-3,0.5)
    kernel.lengthscale[[3]].constrain_bounded(1e-3,1.0)
    kernel.power.constrain_bounded(1e-10,0.03)

    # Define the GPR
    bounds_dict = {'T_f':(1.0,20), 'nAr':(0.001,100),
                                                  'alpha':(0.001,10),
                                                  'beta':(0.001,10)}
    tmn = GPyRegression(m.parameter_names, bounds=bounds_dict,
                        kernel=kernel, normalizer=True)

    # Setup BOLFI
    bolfi = elfi.BOLFI(log_d, batch_size=1, initial_evidence=result_dictionary, update_interval=1,
                       bounds=bounds_dict, acq_noise_var=0, target_model=tmn,
                       async_acq=True, batches_per_acquisition=1, max_parallel_batches=10)

    # Change acquisition method to RandMaxVar
    bolfi.acquisition_method = RandMaxVar(model=bolfi.target_model, noise_var=0, prior=ModelPrior(bolfi.model),sampler='metropolis',n_samples=200) 

    # Setup the number of initialisation points. This the number to be sampled randomly
    # before applying the acquisition function.
    bolfi.n_initial_evidence=30

    # Collect the initialization points
    iteration_accepted = False
    while iteration_accepted == False:
        try:
            bolfi.fit(n_evidence=bolfi.n_initial_evidence)
            iteration_accepted =True
        except:
            bolfi.batches.cancel_pending()
            bolfi.batches._next_batch_index = batch_index+1

    # Save the initial evidence to a dictionary
    save_bolfi(bolfi,fname='BOLFI_4D_initial_'+str(bolfi.n_initial_evidence)+'.pkl')

    # Initial fit of the GPR
    inst = tmn.instance
    inst.Gaussian_noise.variance=1e-10
    inst.Gaussian_noise.variance.fix()
    inst.randomize()
    for i in range(4):
        inst.RatQuad.lengthscale[i] = (bolfi.target_model.bounds[i][1] -
                                       bolfi.target_model.bounds[i][0])/bolfi.state['n_evidence']
    inst.optimize()

    # Collect samples in batches of 10
    for i in range(20):
        if i > 0:
            iteration_accepted = False
            print('i: ', i)
            while iteration_accepted == False:
                try:
                    bolfi.fit(n_evidence=10*i)
                    save_bolfi(bolfi,fname='BOLFI_4D_at_'+str(10*i)+'.pkl')
                    inst = tmn.instance
                    inst.Gaussian_noise.variance = 1e-10
                    inst.Gaussian_noise.variance.fix()
                    # For even rounds, use lengthscale.constrain_positive()
                    # For odd rouns, use lengthscale.constrain_bounded()
                    if i%2 == 0:
                        kernel.lengthscale.constrain_positive()
                        inst.randomize()
                        for i in range(4):
                            inst.RatQuad.lengthscale[i] = (bolfi.target_model.bounds[i][1] -
                                                           bolfi.target_model.bounds[i][0])/bolfi.state['n_evidence']
                        inst.optimize()
                    else:
                        # Temperature lengthscale constraint
                        kernel.lengthscale[[0]].constrain_bounded(1e-3,1.0)
                        # Alpha lengthscale constraint
                        kernel.lengthscale[[1]].constrain_bounded(1e-3,0.5)
                        # Beta lengthscale constraint
                        kernel.lengthscale[[2]].constrain_bounded(1e-3,0.5)
                        # Argon assimilation fraction lengthscale constraint
                        kernel.lengthscale[[3]].constrain_bounded(1e-3,1.0)
                        inst.randomize()
                        inst.optimize()
                    iteration_accepted = True
                except:
                    # If something goes wrong, print 'Exception' and reset the
                    # remaining batches. This handles situations where the simulations
                    # fail, causing an exception in BOLFI. 
                    print('Exception!')
                    bolfi.batches.reset()
