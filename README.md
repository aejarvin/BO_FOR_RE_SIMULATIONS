# BO_FOR_RE_SIMULATIONS

This repository contains the python files that have been created for exploring Bayesian Optimization and Inference for validation of runaway electron simulations with the simulation code DREAM (https://ft.nephy.chalmers.se/dream/index.html). The algorithms are based on the Engine for Likelihood-Free Inference, ELFI, (https://elfi.readthedocs.io/en/latest/index.html), using the method Bayesian Optimization for Likelihood-Free Inference, BOLFI (https://elfi.readthedocs.io/en/latest/usage/BOLFI.html). The probabilistic surrogate modelling used in the Bayesian optimization is conducted with Gaussian Process (GP) Regression and the GPy GP framework is used for that (https://gpy.readthedocs.io/en/deploy/). 

The proof-of-principle cases are given in separate folders. The functions that are called by more than one of the cases are located in the 'common' folder, and the search routines apply 'sys.path.append('../common/')'.  The experimental data files are not given here. 

The work is related to a publication 'Bayesian approach for validation of runaway electron simulations' submitted to Journal of Plasma Physics on 1st of August 2022 and also available at ARXIV-LINK HERE.

# Dependencies
DREAM: https://github.com/chalmersplasmatheory/DREAM

ELFI: https://github.com/elfi-dev/elfi

GPy: https://github.com/SheffieldML/GPy 
