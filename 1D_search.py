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
    """ This is the wrapper function for the simulation model.
    
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
    try:
        file1 = run_CQ_fluid_simulation_uniform_RE(T_f=Tf1, nAr_frac=15, 
                                    walltime=5,
                                    initoutfile = 'bolfi_1D_runs/'+str(float(Tf1))+'init_out.h5',
                                    CQfluidoutfile = 'bolfi_1D_runs/'+str(float(Tf1))+'CQ_fluid_out.h5')
        timev = np.array(file1['grid']['t'][:])
        currentv = np.array(file1['eqsys']['I_p'][:])
    except:
        timev = np.array([0,10])
        currentv = np.array([20e6,20e6])
    currentv = np.squeeze(currentv)
    output = np.array([np.transpose(timev),np.transpose(currentv)])
    return output

def distance1(*summaries, observed):       
    """Calculate an L1-based distance between the simulated and observed summaries.

    Follows the simplified single-distance approach in:
    Gutmann M U, Corander J (2016). Bayesian Optimization for Likelihood-Free Inference
    of Simulator-Based Statistical Models. JMLR 17(125):1−47, 2016.

    Parameters
    ----------
    *summaries : k np.arrays of shape (m, n)
    observed : list of k np.arrays of shape (1, n)

    Returns
    -------
    np.array of shape (m,)

    """
    summaries = np.stack(summaries)
    observed = np.stack(observed)
   
    y = observed 
    x = summaries

    # L1 norm 
    dist = np.sum(np.abs(x - y))
    dist = np.array([dist])
    return dist

def get_model():
    """ A convenience function that defines the ELFI model
    
    """
    m = elfi.ElfiModel()
    priors = []
    sumstats = []
    priors.append(elfi.Prior('uniform', 1.0, 19, model=m, name='Tf1'))

    exp_data = h5py.File('../experimental_data/IP_out.h5','r')
    exp_datar = exp_data['IP_data']
    exp_d = np.array(exp_datar)

    elfi.Simulator(sim_fn, *priors, observed=exp_d, name='CQf')

    S1 = elfi.Summary(IP_summary_25ms, m['CQf'], name='IPsum')
    elfi.Discrepancy(distance1, m['IPsum'], name='d')
    
    return m

if __name__ == '__main__':
    elfi.set_client('multiprocessing')
    m = get_model()
    log_d = elfi.Operation(np.log, m['d'])
    kernel = GPy.kern.RatQuad(input_dim=1, ARD=True)
    kernel.lengthscale.constrain_bounded(1e-10,1.0)
    bounds_dict = {'Tf1':(1.0,20)}
    tmn = GPyRegression(m.parameter_names, bounds=bounds_dict,
                        kernel=kernel, normalizer=True)
    bolfi = elfi.BOLFI(log_d, batch_size=1, initial_evidence=None, update_interval=1,
                       bounds=bounds_dict, acq_noise_var=None, exploration_rate=5.0, target_model=tmn,
                       async_acq=False, batches_per_acquisition=1)
    bolfi.n_initial_evidence=3
    print('Starting the initial fit')
    bolfi.fit(n_evidence=3)
    inst = tmn.instance
    inst.Gaussian_noise.variance.constrain_bounded(1e-100,1e-4)
    inst.optimize()
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
    for i in range(4,13):
        bolfi.fit(n_evidence=i)
        inst = tmn.instance
        inst.Gaussian_noise.variance.constrain_bounded(1e-100,1e-4)
        inst.optimize()
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

    res1 = np.array(results)
    plt.plot(res1[:,0],res1[:,1],'ko-')
    plt.plot(res1[:,0],res1[:,2],'bo--')
    plt.plot(res1[:,0],res1[:,3],'bo--')
    plt.ylim(0,20)
    plt.xlim(2,13)
    plt.gca().set_aspect(11.0/20.0)
    plt.savefig('1D_RQ_example_aut_convergence.svg')
                                               

    
    
