# Standard python packages
import numpy as np

# Define IP summary function
def IP_summary_25ms(data):
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
