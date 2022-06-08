# BO_FOR_RE_SIMULATIONS

This repository contains the python files that have been created for exploring Bayesian Optimization and Inference for validation of runaway electron simulations with the simulation code DREAM (https://ft.nephy.chalmers.se/dream/index.html). The algorithms are based on the Engine for Likelihood-Free Inference, ELFI, (https://elfi.readthedocs.io/en/latest/index.html), using the method Bayesian Optimization for Likelihood-Free Inference, BOLFI (https://elfi.readthedocs.io/en/latest/usage/BOLFI.html). The probabilistic surrogate modelling used in the Bayesian optimization is conducted with Gaussian Process (GP) Regression and the GPy GP framework is used for that (https://gpy.readthedocs.io/en/deploy/). 

The proof-of-principle cases are given in separate folders containing all the files, except the experimental data files, necessary to run the cases, as long as all the software depencies, listed below, are appropriately installed on the system. Some degree of repetition is present as same or similar functions are given in the separate folders, but this was considered to provide a clean way to provide example cases that do not depend on each other. 

The work is related to a publication 'Bayesian approach for validation of runaway electron simulations' submitted to JOURNAL_NAME on DATE and also available at ARXIV-LINK HERE.

# Dependencies
DREAM: https://github.com/chalmersplasmatheory/DREAM

ELFI: https://github.com/elfi-dev/elfi

GPy: https://github.com/SheffieldML/GPy 
