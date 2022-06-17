# Standard python packages.
import numpy as np
import sys
import scipy.io as sio
import datetime,time
import timeit
import matplotlib.pyplot as plt
import h5py
import os
import scipy.stats as stats
from sklearn.ensemble import RandomForestRegressor
#import GPy

# Add path to the DREAM python wrapper
sys.path.append('../../DREAM/py')

# Import the DREAM python wrapper
import DREAM 
from DREAM.DREAMSettings import DREAMSettings
import DREAM.Settings.Equations.DistributionFunction as DistFunc
import DREAM.Settings.Equations.IonSpecies as Ions
import DREAM.Settings.Equations.RunawayElectrons as Runaways
import DREAM.Settings.Solver as Solver
import DREAM.Settings.CollisionHandler as Collisions
import DREAM.Settings.Equations.ElectricField as Efield
import DREAM.Settings.Equations.ColdElectronTemperature as T_cold
import DREAM.Settings.Equations.HotElectronDistribution as FHot
from DREAM.Settings.Equations.ElectricField import ElectricField
from DREAM.Settings.Equations.ColdElectronTemperature import ColdElectronTemperature
import DREAM.Settings.TimeStepper as TimeStepper
import DREAM.Settings.XiGrid as XiGrid
from DREAM.runiface import *


def run_CQ_fluid_simulation(T_f=10, t_T_f=None, nAr_frac=15, walltime=5, 
                            alpha=1.0, beta=1.0, 
                            R=2.643, B0=3.0, V_plasma=100,  a=63, wallRadius=73,
                            Ip_init=1.421e6, Ip=6.7202e5, E_init=20.8104, 
                            nAr0=7.9862e18, E_wall=0.0, 
                            filename='../experimental_data/T_ne_data_JET95135.mat', 
                            nr=30, Nre=1e5, pMax_re=160, Np_re=160, Nxi_re=120, 
                            tMax_init=10e-3, nt_init=100, tMax=44.6e-3, nt=2000,
                            nbr_saveSteps = int(1000), re_grid = False,
                            inputfilename = '../sim_data_automatic/test1i.h5',
                            outputfilename = '../sim_data_automatic/test1o.h5', 
                            initoutfile = '../sim_data_automatic/init_out.h5',
                            CQfluidoutfile = '../sim_data_automatic/CQ_fluid_out.h5'):

    '''
    This function runs DREAM using the runiface and returns the output. 
    List of inputs:
        R:              Magnetic axis (m)
        B0:             Toroidal B-field on magnetic axis (T)
        V_plasma:       Plasma volume (m^3) 
        a:              Plasma minor radius (cm)
        wallRadius:     Machine wall minor radius (cm)
        Ip_init:        Max Ip from experimental data (A)
        Ip:             RE plateu plasma current (A)
        E_init:         Initial electric field (V/m)
        T_f:            Temperature after thermal quench (eV)
        nAr0:           Injected argon density (NAr/V_plasma)
        nAr_frac:       Amount of assimilated argon (%)
        E_wall:         Assumed boundary electric field (V/m)
        walltime:       Characteristic time for the wall (ms)
        filename:       Experimental data.
        nr:             Number of radial grid points
        Nre:            Scaling factor for number of runaways
        pMax_re:        Maximum momentum in units of m_e*c
        Np_re:          Number of momentum grid points       
        Nxi_re:         Number of pitch grid points
        tMax_init:      Length of the init. simulation (s) 
        nt_init:        Number of time steps in the init. simulation
        tMax:           Maximum time of the CQ simulation (s)
        nt:             Maximum number of time steps in the CQ simulation
        alpha:          Parameter for spatial RE-distribution
        beta:           Parameter for spatial RE-distribution
        nbr_saveSteps:  Number of time steps that are saved
        re_grid:        Flag for usage of RE grid
        inputfilename:  Name of the CQ simulation input file
        outputfilename: Name of the CQ simulation output file
    '''
    data = sio.loadmat(filename)
    dNe = np.squeeze(data['dNe'])
    ne_profile = dNe[:,2]*1e19         # (m^-3) From experimental JET data
    r = np.linspace(0,a/100,nr)     # (m) radial coordinates   
    r_exp = np.squeeze(dNe[:,0])- np.squeeze(dNe[0,0])
    #n_re = np.array([(pow(beta, alpha)*pow((rs/max(r)),
    #                (alpha-1))*np.exp(-beta*(rs/max(r)))) for rs in r])
    n_re = stats.gamma.pdf(r, a=alpha, scale=1.0/beta)
    # In the case that the distribution approaches infinity at r -> 0,
    # use linear extrapolation towards r -> 0.
    if np.isinf(n_re[0])==True:
        n_re[0] = n_re[1] + r[1]*(n_re[1] - n_re[2])/(r[2] - r[1])
        #n_re[0] = stats.gamma.pdf(0.0001, a=alpha, scale=1.0/beta)
    n_re = n_re/sum(n_re)
    n_rei = n_re                       # Initial profile
    nAr = (nAr_frac*1e-2)*nAr0          # Amount of assimilated argon
    inverse_walltime = 1/(walltime*1e-3)    # (1/s)
    n_re = Nre*n_re                         # Final radial denstiy distribution of runaways

    current_ok = False
    plateau_current_ok = False
    
    
    # Create DREAMSettings object
    ds=DREAMSettings()

    # Collision type
    ds.collisions.collfreq_type = Collisions.COLLFREQ_TYPE_PARTIALLY_SCREENED
    
    # Electric field
    ds.eqsys.E_field.setPrescribedData(E_init,radius=r)

    # Set Eceff mode
    ds.eqsys.n_re.setEceff(Runaways.COLLQTY_ECEFF_MODE_FULL)

    # Set temperature
    if t_T_f == None:
        ds.eqsys.T_cold.setPrescribedData(T_f, radius=r)
        Tion = T_f
    else:
        ds.eqsys.T_cold.setPrescribedData(T_f[0,0], radius=r, times=t_T_f)
        Tion = T_f[0,0]
    # Set ions
    ds.eqsys.n_i.addIon(name='D2', Z=1, T=Tion, iontype=Ions.IONS_DYNAMIC_FULLY_IONIZED, n=ne_profile, r=r_exp)
    ds.eqsys.n_i.addIon(name='Ar', Z=18, iontype=Ions.IONS_PRESCRIBED_NEUTRAL, n=nAr) 

    # Hot-tail grid settings
    ds.hottailgrid.setEnabled(False)

    # Runaway grid settings
    ds.runawaygrid.setEnabled(False)

    # Set up radial grid
    ds.radialgrid.setB0(B0) 
    ds.radialgrid.setNr(nr)
    ds.radialgrid.setMinorRadius(a/100) 
    ds.radialgrid.setWallRadius(wallRadius/100)        
    
    # Set solver type
    ds.solver.setType(Solver.NONLINEAR)                       # Semi-implicit time stepping
    #ds.solver.setVerbose(True)                               # Print info from Newton iterations

    ds.solver.setLinearSolver(3)

    # Include otherquantities to save to output
    ds.other.include('fluid')

    # Set time stepper
    ds.timestep.setTmax(tMax_init)
    ds.timestep.setNt(nt_init)

    # Print progress
    ds.output.setTiming(stdout=True, file=True)

    # Set file name of output file
    # ds.output.setFilename(init_output_str)
    # This section of the code runs the initialization simulation.
    # The current_ok flag is used to iterate the initial electric field
    # to get the correct current.
    
    while current_ok != True:
        if os.path.exists(initoutfile):
            os.remove(initoutfile)
        try: 
            f = runiface(ds, outfile=initoutfile, quiet = True)
            current = f['eqsys']['I_p'][-1][0]
        except:
            print('Exception already in the initialization')
        f.close()
        if abs(current - Ip_init) > 1e4:
            E_init = E_init*(Ip_init/current)
            ds.eqsys.E_field.setPrescribedData(E_init,radius=r)
        else:
            current_ok = True
    currents = []
    Nres = []
    counter = 0
    Flag_above = False
    Flag_below = False
    mult = 1000
    Ip_prev = 0
    Nre_below = 0
    Nre_above = 0
    Ip_below = 0
    Ip_above = 0
    print('Case ' + CQfluidoutfile + ' at temperature ' + str(T_f) + ' and walltime ' + str(walltime) + 
          ' starting')
    while plateau_current_ok != True:    
        timestepok = False
        if counter%10 == 0:
            print('Case ' + CQfluidoutfile + ' at temperature ' + str(T_f) + ' counter ' + str(counter))
        if Nre < 1.0e-3:
            break
        if nt > 1e5:
            break
        while timestepok == False:
            ds1 = DREAMSettings(ds)
            ds1.fromOutput(initoutfile, ignore=['n_re', 'T_cold'])
            if t_T_f == None:
                ds1.eqsys.T_cold.setPrescribedData(T_f, radius=r)
            else:
                ds1.eqsys.T_cold.setPrescribedData(T_f, radius=r, times=t_T_f)
            ds1.eqsys.E_field.setType(Efield.TYPE_SELFCONSISTENT)
            ds1.eqsys.E_field.setBoundaryCondition(bctype = Efield.BC_TYPE_SELFCONSISTENT, 
                                                       inverse_wall_time = inverse_walltime,
                                                       R0=R)
            ds1.eqsys.n_i.getIon('Ar').initialize_dynamic_neutral(
                    interpr=ds.eqsys.n_i.getIon('Ar').r, n=nAr)
            ds1.timestep.setTmax(tMax)
            ds1.timestep.setNt(nt)
            ds1.timestep.setNumberOfSaveSteps(nbr_saveSteps)
            ds1.eqsys.n_re.setInitialProfile(density=n_re,radius=r)
            ds1.eqsys.n_re.setAvalanche(Runaways.AVALANCHE_MODE_FLUID)
            ds1.eqsys.n_re.setDreicer(Runaways.DREICER_RATE_DISABLED)
            ds1.hottailgrid.setEnabled(False)
            ds1.runawaygrid.setEnabled(False)
            if os.path.exists(CQfluidoutfile):
                os.remove(CQfluidoutfile)
            try:
                f = runiface(ds1, outfile = CQfluidoutfile, quiet = True)
                current = f['eqsys']['I_p'][-1][0]
            except:
                print('Reducing time step ', 'nt: ', nt, 'walltime: ', walltime)
                current = 1.0
            try:
                timev = f['eqsys']['grid']['t'][:]
            except: 
                timev = [0.0]
            if current == 1.0:
                nt = nt*1.2
                ds1.timestep.setNt(nt)
                Nre = Nre*0.1
                mult = mult*0.1
                if mult < 2:
                    mult = 2
                n_re = Nre*n_rei
                if nt > 1e5:
                    break
            else:
                timestepok = True
            f.close()
                        
            step_size = 1.0
            if abs(current - Ip) > 1e3 and timestepok == True:
                #print('Current Ip: ', current, ' Target Ip: ', Ip, ' Nre: ', Nre, ' T_f: ', T_f, ' nAr_frac: ', nAr_frac)
                if Flag_below == False or Flag_above == False:
                    if (current > Ip):
                        Nre_above = Nre
                        Ip_above = current
                        if Flag_below == False:
                            Nre = Nre/mult
                        else:
                            Nre = 0.5*(Nre_above + Nre_below)
                        Flag_above = True
                    else:
                        Nre_below = Nre
                        Ip_below = current
                        if Flag_above == False:
                            Nre = Nre*mult
                        else:
                            Nre = 0.5*(Nre_above + Nre_below)
                        Flag_below = True
                else:
                    if (current > Ip):
                        Nre_above = Nre
                        Ip_above = current
                    else:
                        Nre_below = Nre
                        Ip_below = current
                    Nre = 0.5*(Nre_above + Nre_below)
                    #print('Nre: ', Nre)
                n_re = Nre*n_rei
                current_ok = False
                counter = counter + 1
                if Nre < 1.0:
                    break
            else:
                if timestepok == True:
                    plateau_current_ok = True
            
    if re_grid:
        ds1.solver.tolerance.set('j_re', abstol=1)
        ds1.runawaygrid.setEnabled(True)
        ds1.runawaygrid.setNxi(Nxi_re)
        ds1.runawaygrid.setNp(Np_re)
        ds1.runawaygrid.setPmax(pMax_re)
        ds1.runawaygrid.setBiuniformGrid(thetasep =0.4,nthetasep_frac=0.5)
        f = np.zeros((1,1,1))                                 #nr x np x nXi
        ds1.eqsys.f_re.setInitialValue(f,r=[0],p =[0],xi=[0])
    
        # Collision mode
        ds1.collisions.collfreq_mode = Collisions.COLLFREQ_MODE_ULTRA_RELATIVISTIC
    
        # Flux limiter
        ds1.eqsys.f_re.setAdvectionInterpolationMethod(ad_int=DistFunc.AD_INTERP_TCDF,
            ad_jac=DistFunc.AD_INTERP_JACOBIAN_UPWIND)
        ds1.eqsys.f_re.setSynchrotronMode(DistFunc.SYNCHROTRON_MODE_INCLUDE)
    else:
        ds1.runawaygrid.setEnabled(False)
    if t_T_f == None:
        Tf_out = T_f
    else:
        Tf_out = T_f[0,0]
    print(' T_f: ', Tf_out , ' nAr_frac: ', nAr_frac, 'Alpha: ', alpha, 'Beta: ', beta, 'Walltime: ', walltime)
    print('Case ' + CQfluidoutfile + ' at temperature ' + str(T_f) + 
          ' finished successfully')
    #ds.output.setFilename(outputfilename)
    #ds.save(inputfilename)
    f = runiface(ds1, outfile = CQfluidoutfile, quiet = True)
    f.close()
    file1 = h5py.File(CQfluidoutfile,'r')
    #timev = np.array(file1['grid']['t'][:])
    #timev.reshape(len(timev),1)
    #currentv = np.array(file1['eqsys']['I_p'][:])
    #currentv.reshape(len(currentv),1)
    #output = np.c_[timev,currentv]
    return file1
