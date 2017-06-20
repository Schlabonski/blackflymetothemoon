import cython
import numpy as np
cimport numpy as np
from scipy.optimize import curve_fit

#DTYPE = np.int
#DTYPEf = np.float64
#ctypedef np.int_t DTYPE_y
#ctypedef np.float64_t DTYPEf_t

cdef np.float64_t A = 1/np.sqrt(2*np.pi)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def gaussian_estimate(np.ndarray x, np.ndarray y):
    cdef np.ndarray y_normalized, y_smooth, cdf, y_grounded
    cdef np.float64_t mu_0, sigma_0, C0, A0, delta_x, popt, pcov
    cdef int N = 20 # size of moving average window
    cdef int sigma_0_ind, mu_0_ind
    cdef list p0
    
    # I In the first part we want to get stable estimates for the fit
    # parameters. For this we smooth the data, calculate the cdf and extract the values for
    # mean and standard deviation
    
        
    # extract and substract offset
    C0 = min(y)
    y_grounded = y - C0
        
    # 1. apply a moving average to data
    y_smooth = np.convolve(y_grounded, np.ones((N,))/float(N), mode='same')

    # 2. calculate cdf and extract scaling factor
    delta_x = x[1] - x[0] # TODO: Fix value when working with pixels
    cdf = np.cumsum(y_smooth)*delta_x
    A0 = np.max(cdf)
    cdf /= A0
    
    # 3. extract stdev and mean
    sigma_0_ind = np.argmax(cdf > 0.1586)
    mu_0_ind = np.argmax(cdf > 0.5)
    mu_0_ind = min(len(x)-1, max(0, mu_0_ind))
    sigma_0_ind = min(len(x)-1, max(0, sigma_0_ind))
    mu_0 = x[mu_0_ind]
    sigma_0 = mu_0 - x[sigma_0_ind]
    
    
    # II. In the second part we fit a gaussian to the data
    
    # TODO: Maybe its not really necessary to perform a fit. A good estimate might
    # be enough for the purpose of beam profiling
    y_fit = gaussianC(x, mu_0, sigma_0, C0, A0)
    
    return y_fit, mu_0, sigma_0, C0, A0

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
#cdef gaussianC(np.ndarray x, np.float64_t mu, np.float64_t sigma, np.float64_t C, np.float64_t A0):
cdef gaussianC(np.ndarray x, np.float64_t mu, np.float64_t sigma, np.float64_t C, np.float64_t A0):
    return A0 * A /sigma * np.exp(-(x-mu)**2/(2*sigma**2)) + C

def gaussianC_py(x, mu, sigma, C, A0):
    return A0 * A /sigma * np.exp(-(x-mu)**2/(2*sigma**2)) + C

def gaussian_fit(x, y):
    estimate = True
    p0 = gaussian_estimate(x, y)[1:]
    try:
        popt, pcov = curve_fit(gaussianC_py, x, y, p0)
        estimate = False
    except RuntimeError:
        print("No fitted result, only estimate possible.")
        popt = p0
    y_fit = gaussianC(x, popt[0], popt[1], popt[2], popt[3])
    return y_fit, estimate, popt
