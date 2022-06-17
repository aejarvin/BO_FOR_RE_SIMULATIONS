# Standard python packages
import numpy as np

# Define IP summary function
def IP_summary_25ms(data):
    ''' This function simply maps the plasma current to a standardized 
    timebase between 0 and 25 ms.
    
    Input:
        data : np.array of 2 columns: time, Ip
    Return:
        IPout : np.array of 1 column: Ip mapped to a standardized timebase
    '''
    timev = data[0,:]
    IPv = data[1,:]
    timebase = 1e-3*25.0*np.array(range(0,2000))/2000.0
    IPout = []
    for i in timebase:
        index1 = np.where(abs(timev - i )==min(abs(timev - i)))
        index1 = index1[0][0]
        IPout.append(abs(IPv[index1]))
    IPout = np.array(IPout)
    return IPout
