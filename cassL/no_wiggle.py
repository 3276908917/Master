from scipy.interpolate import UnivariateSpline
import scipy.fft as fft
import generate_emu_data as ged
import numpy as np

def crunch_spectra(lhc, big_k_arr, small_k_arr, samples, priors):
    if len(lhc) != len(samples):
        raise ValueError("The parameter configs and spectra do not appear " \
                         "to come from the same data sets.")
    
    crunched_wiggle = np.empty((len(samples), len(small_k_arr)))
    crunched_nowiggle = np.empty((len(samples), len(small_k_arr)))
    
    for i in range(len(samples)):
        print(i)
        lhc_row = lhc[i]
        omega_m = get_omegam(lhc_row, priors)
        
        pk_huge = samples[i]
        pk_wiggle, pk_nw = get_nowiggle_pk(small_k_arr, big_k_arr, pk_huge,
                                            omega_m)
            
        crunched_wiggle[i] = pk_wiggle
        crunched_nowiggle[i] = pk_nw
    return np.array(crunched_wiggle), np.array(crunched_nowiggle)
    

def get_omegam(lhc_row, priors):
    denormalized_row = ged.denormalize_row(lhc_row, priors)
    return denormalized_row[0] + denormalized_row[1]


def get_nowiggle_pk(kfinal, karr, Parr, omega_m, range_imin=np.array([80,150]),
                    range_imax=np.array([200,300]), threshold=0.04, offset=-25):
    
    logParr = np.log(karr * Parr)
    xiarr = fft.dst(logParr, type=2)
    xi_even = xiarr[::2]
    xi_odd = xiarr[1::2]
    xi_even_spline = UnivariateSpline(np.arange(2**15),xiarr[::2],k=3,s=0)
    xi_odd_spline = UnivariateSpline(np.arange(2**15),xiarr[1::2],k=3,s=0)
    
    range_imin = (range_imin*(0.1376591/omega_m)**(1./4)).astype(int)
    range_imax = (range_imax*(0.1376591/omega_m)**(1./4)).astype(int)
    
    imin_start = np.argmin(xi_even_spline.derivative(n=2)(np.arange(range_imin[0],range_imin[1]))) + range_imin[0] + offset
    imax_start = np.where(xi_even_spline.derivative(n=2)(np.arange(range_imax[0],range_imax[1])) < threshold)[0][0] + range_imax[0]
    
    def remove_bump(imin, imax):
        r = np.delete(np.arange(2**15), np.arange(imin,imax))
        xi_even_nobump = np.delete(xi_even, np.arange(imin,imax))
        xi_odd_nobump = np.delete(xi_odd, np.arange(imin,imax))
        xi_even_nobump_spline = UnivariateSpline(r, (r+1)**2*xi_even_nobump, k=3, s=0)
        xi_odd_nobump_spline = UnivariateSpline(r, (r+1)**2*xi_odd_nobump, k=3, s=0)

        xi_nobump = np.zeros(2**16)
        xi_nobump[::2] = xi_even_nobump_spline(np.arange(2**15))/np.arange(1,2**15+1)**2
        xi_nobump[1::2] = xi_odd_nobump_spline(np.arange(2**15))/np.arange(1,2**15+1)**2

        logkpk_nowiggle = fft.idst(xi_nobump, type=2)
        return UnivariateSpline(karr, np.exp(logkpk_nowiggle)/karr, k=3, s=0)
    
    P = UnivariateSpline(karr, Parr, k=3, s=0)
    Pnw = remove_bump(imin_start, imax_start)
        
    return P(kfinal), Pnw(kfinal)