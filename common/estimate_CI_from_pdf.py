# Standard Python packages
import numpy as np

def estimate_CI_from_pdf(xvals,pdf,CI):
    # This function simply integrates the pdf and 
    # returns the xmin and xmax that correspond to
    # locations for which the integrated pdf is 50%
    # CI below xmin and above xmax. CI is defined
    # as CI = 1 - confidence_interval. For
    # 95% confidence interval CI is 0.05. Then
    # xmin corresponds to 2.5% of the pdf being below xmin, and
    # xmax as 2.5% of the pdf being above xmax.
    sumval = 0
    counter = 0
    while sumval < CI/2.0:
        sumval = sumval + pdf[counter]
        counter = counter + 1
    xmin = xvals[counter]
    while sumval < 1 - CI/2.0:
        sumval = sumval + pdf[counter]
        counter = counter + 1
    xmax = xvals[counter]
    return xmin, xmax

