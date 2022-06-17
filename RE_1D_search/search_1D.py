# Standard python packages.
import numpy as np
import matplotlib.pyplot as plt
import h5py
import pickle
import scipy.stats as stats
import multiprocessing

# Bayesian inference and Gaussian Process packages
import elfi
import GPy
from elfi.methods.bo.gpy_regression import GPyRegression

# Import the function that runs DREAM
from run_CQ_fluid_simulation_uniform_RE import run_CQ_fluid_simulation_uniform_RE

# Import the summary function
from IP_summary_25ms import IP_summary_25ms

# Plotting and analysis routines
from plot_gp_and_aq import plot_gp_and_aq
from estimate_CI_from_pdf import estimate_CI_from_pdf


def sim_fn(Tf1, batch_size=1, random_state=None):
    """ This is a wrapper function for the simulation model.
    
    Parameters
    ----------
    Tf1 : Electron temperature (eV)
    batch_size : a dummy variable required by BOLFI, but not used
    random_state : a dummy variable required by BOLFI, but not used
    
    Returns
    -------
    Total plasma current as a function of time as an np.array.

    """
    
    Tf1 = float(Tf1)
    # N.B. The T_ne_data_JET95135.mat corresponds to experimental data from
    # JET and is not given in this repository. However, it is possible 
    # to create artificial density and temperature profiles to try the algorithm.
    file1 = run_CQ_fluid_simulation_uniform_RE(T_f=Tf1, nAr_frac=15, 
                                    walltime=5,
                                    filename='T_ne_data_JET95135.mat',
                                    initoutfile = 'bolfi_1D_runs/'+str(float(Tf1))+'init_out.h5',
                                    CQfluidoutfile = 'bolfi_1D_runs/'+str(float(Tf1))+'CQ_fluid_out.h5')
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

def get_model():
    """ A convenience function that defines the ELFI model
    
    Returns
    -------
    An ELFI model.

    """
    # Initialize an ELFI model
    m = elfi.ElfiModel()
    
    # Define the prior distribution for the uncertain parameter
    prior=elfi.Prior('uniform', 1.0, 19, model=m, name='Tf1')

    # Load observed/measured data
    # N.B. The IP_out.h5 corresponds to experimental data from
    # JET and is not given in this repository. However, it is possible 
    # to create artificial current evolution data to try the algorithm.
    exp_data = h5py.File('IP_out.h5','r')
    exp_datar = exp_data['IP_data']
    exp_d = np.array(exp_datar)

    # Define the ELFI simulator. Here ELFI links to the wrapper
    # function defined above.
    elfi.Simulator(sim_fn, prior, observed=exp_d, name='CQf')
    
    # Define the summary function. IP_summary_25ms is given
    # externally and it simply maps the Ip to a unified
    # timebase between 0 and 25 ms. 
    S1 = elfi.Summary(IP_summary_25ms, m['CQf'], name='IPsum')

    # Define the method to measure discrepancy. This is given
    # by the distance1 function defined above
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

    # Get the ELFI model
    m = get_model()

    # Take a logarithm of the discrepancy
    log_d = elfi.Operation(np.log, m['d'])

    # Define the GPR kernel parameters
    kernel = GPy.kern.RatQuad(input_dim=1)
    kernel.lengthscale.constrain_bounded(1e-10,1.0)

    # Define the GPR
    bounds_dict = {'Tf1':(1.0,20)}
    tmn = GPyRegression(m.parameter_names, bounds=bounds_dict,
                        kernel=kernel, normalizer=True)

    # Setup BOLFI 
    bolfi = elfi.BOLFI(log_d, batch_size=1, initial_evidence=None, update_interval=1,
                       bounds=bounds_dict, acq_noise_var=None, exploration_rate=5.0, target_model=tmn,
                       async_acq=False, batches_per_acquisition=1)

    # Set the number of initialization points
    bolfi.n_initial_evidence=3

    # Run the initialization simulations
    print('Starting the initial fit')
    bolfi.fit(n_evidence=3)

    # Fit the GPR
    inst = tmn.instance
    inst.Gaussian_noise.variance.constrain_bounded(1e-100,1e-4)
    inst.optimize()

    # This section diagnoses the progress and creates plots
    xtest = np.linspace(1,20,10000)
    results = []
    X = bolfi.target_model.X
    Y = bolfi.target_model.Y
    mean, var = bolfi.target_model.predict(xtest)
    acq = bolfi.acquisition_method.evaluate(xtest,t=bolfi._get_acquisition_index(bolfi.batches.next_index))
    xacq = bolfi.acquisition_method.acquire(1,t=bolfi._get_acquisition_index(bolfi.batches.next_index))
    yacq = bolfi.acquisition_method.evaluate(xacq,t=bolfi._get_acquisition_index(bolfi.batches.next_index))
    plot_gp_and_aq(X,Y,xtest,mean,var,acq,xacq,yacq,fname='1D_RQ_example_aut_'+str(3)+'.svg')
    out = bolfi.extract_result()
    post = bolfi.extract_posterior()
    pdf1 = post.pdf(xtest)
    xmin, xmax = estimate_CI_from_pdf(xtest, pdf1/sum(pdf1), 0.05)
    results.append([3, float(out.x_min['Tf1']),xmin, xmax])

    # Repeat to collect total of 12 samples
    for i in range(4,13):
        # Collect sample and refit GPR
        bolfi.fit(n_evidence=i)
        inst = tmn.instance
        inst.Gaussian_noise.variance.constrain_bounded(1e-100,1e-4)
        inst.optimize()

        # Collect the optimization progress and create plots
        X = bolfi.target_model.X
        Y = bolfi.target_model.Y
        mean, var = bolfi.target_model.predict(xtest)
        acq = bolfi.acquisition_method.evaluate(xtest,t=bolfi._get_acquisition_index(bolfi.batches.next_index)+1)
        xacq = bolfi.acquisition_method.acquire(1,t=bolfi._get_acquisition_index(bolfi.batches.next_index)+1)
        yacq = bolfi.acquisition_method.evaluate(xacq,t=bolfi._get_acquisition_index(bolfi.batches.next_index)+1)
        plot_gp_and_aq(X,Y,xtest,mean,var,acq,xacq,yacq,fname='1D_RQ_example_aut_'+str(i)+'.svg')
        out = bolfi.extract_result()
        post = bolfi.extract_posterior()
        pdf1 = post.pdf(xtest)
        xmin, xmax = estimate_CI_from_pdf(xtest, pdf1/sum(pdf1), 0.05)
        results.append([i, float(out.x_min['Tf1']),xmin, xmax])

    # Create the convergence plots
    res1 = np.array(results)
    plt.plot(res1[:,0],res1[:,1],'ko-')
    plt.plot(res1[:,0],res1[:,2],'bo--')
    plt.plot(res1[:,0],res1[:,3],'bo--')
    plt.ylim(0,20)
    plt.xlim(2,13)
    plt.gca().set_aspect(11.0/20.0)
    plt.savefig('1D_RQ_example_aut_convergence.svg')
                                               

    
    
