# Standard Python packages
import matplotlib.pyplot as plt
import numpy as np
# Import GPy
import GPy

def plot_gp_and_aq(X,Y,xtest,mean,var,acq,next_x, next_y,fname=None):
    # Plot the confidence interval of the GPR
    plt.fill_between(xtest,mean.flatten()-1.96*np.sqrt(var.flatten()),mean.flatten()+1.96*np.sqrt(var.flatten()),alpha=0.75)
    # Plot the mean of the GPR
    plt.plot(xtest,mean,'b-')
    # Plot the collected samples
    plt.plot(X,Y,'ko')
    # Plot the acquisition function
    plt.plot(xtest,acq,'r-')
    # Plot the optimum of the acquisition function
    plt.plot(next_x, next_y,'rs')
    # Set limits, aspect ratio, and axis labels
    plt.xlim(0,20)
    plt.ylim(17,21)
    plt.gca().set_aspect(20.0/4.0)
    plt.xlabel('Temperature (eV)')
    plt.ylabel('Logarithmic distance')
    # Show or save the figure
    if fname == None:
        plt.show()
    else:
        plt.savefig(fname)
        plt.close()
