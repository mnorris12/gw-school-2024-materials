
###
"""
By Kyle Gersbach for the VIPER PTA Summer School 2024

This file contains a few helpful extended functions for the OS. Feel free to use
these in your projects!

"""
###

import numpy as np
from pair_covariance import _linear_solve



def binned_pair_correlations(xi, rho, sig, bins=10):
    """Create binned separation vs correlations with even pairs per bin.

    This function creates a binned version of the xi, rho, and sig values to better
    vizualize the correlations as a function of pulsar separation. This function uses
    even number of pulsar pairs per bin. Note that this function only works with continuous 
    ORFs in pulsar separation space.

    Args:
        xi (numpy.ndarray): A vector of pulsar pair separations
        rho (numpy.ndarray): A vector of pulsar pair correlated amplitude
        sig (numpy.ndarray): A vector of uncertainties in rho
        bins (int): Number of bins to use. Defaults to 10.

    Returns:
        xiavg (numpy.ndarray): The average pulsar separation in each bin
        rhoavg (numpy.ndarray): The weighted average pulsar pair correlated amplitudes
        sigavg (numpy.ndarray): The uncertainty in the weighted average pair amplitudes
    """
    temp = np.arange(0,len(xi),len(xi)/bins,dtype=np.int16)
    ranges = np.zeros(bins+1)
    ranges[0:bins]=temp
    ranges[bins]=len(xi)
    
    xiavg = np.zeros(bins)
    rhoavg = np.zeros(bins)
    sigavg = np.zeros(bins)
    
    #Need to sort by pulsar separation
    sortMask = np.argsort(xi)
    
    for i in range(bins):
        #Mask and select range of values to average
        subXi = xi[sortMask]
        subXi = subXi[int(ranges[i]):int(ranges[i+1])]
        subRho = rho[sortMask]
        subRho = subRho[int(ranges[i]):int(ranges[i+1])]
        subSig = sig[sortMask]
        subSig = subSig[int(ranges[i]):int(ranges[i+1])]
        
        subSigSquare = np.square(subSig)
        
        xiavg[i] = np.average(subXi)
        rhoavg[i] = np.sum(subRho/subSigSquare)/np.sum(1/subSigSquare)
        sigavg[i] = 1/np.sqrt(np.sum(1/subSigSquare))
    
    return xiavg,rhoavg,sigavg


def binned_pair_covariant_correlations(xi, rho, Sigma, bins=10, orf='hd'):
    """Create binned separation vs correlations with pulsar pair covariances.

    This function creates a binned version of the xi, rho, and sig values to better
    vizualize the correlations as a function of pulsar separation while including
    some of the effects of pulsar pair covariances. Note that this does not account 
    for correlations between the different bins. This function uses even number 
    of pulsar pairs per bin. Note that this function only works with continuous 
    ORFs in pulsar separation space.
    Also note that orf can be replaced with a custom function which must accept 
    pulsar positions (cartesian) as its only 2 arguments.
    Predefined orf names are:
        'hd' - Hellings and downs
        'dipole' - Dipole
        'monopole' - Monopole
        'gw_dipole' - Gravitational wave dipole
        'gw_monopole' - Gravitational wave monopole
        'st' - Scalar transverse

    Args:
        xi (numpy.ndarray): A vector of pulsar pair separations
        rho (numpy.ndarray): A vector of pulsar pair correlated amplitude
        Sigma (numpy.ndarray): The pulsar pair covariance matrix
        orf (str, function): The name of a predefined ORF function or custom function 
        bins (int): Number of bins to use. Defaults to 10.

    Returns:
        xiavg (numpy.ndarray): The average pulsar separation in each bin
        rhoavg (numpy.ndarray): The weighted average pulsar pair correlated amplitudes
        sigavg (numpy.ndarray): The uncertainty in the weighted average pair amplitudes
    """
    temp = np.arange(0,len(xi),len(xi)/bins,dtype=np.int16)

    ranges = np.zeros(bins+1)
    ranges[0:bins]=temp
    ranges[bins]=len(xi)
    
    xiavg = np.zeros(bins)
    rhoavg = np.zeros(bins)
    sigavg = np.zeros(bins)
    
    #Need to sort by pulsar separation
    sortMask = np.argsort(xi)
    
    for i in range(bins):
        #Mask and select range of values to average
        l,h = int(ranges[i]), int(ranges[i+1])
        subXi = (xi[sortMask])[l:h]
        subRho = (rho[sortMask])[l:h]
        subSig = (Sigma[sortMask,:][:,sortMask])[l:h,l:h]
        subORF = orf_xi(subXi,orf)[:,None]

        cinv = np.linalg.pinv(subSig)

        r,s2,_ = _linear_solve(subORF, subRho, cinv)

        xiavg[i] = np.average(subXi)
        bin_orf = orf_xi(xiavg[i],orf)
        rhoavg[i] = bin_orf * r
        sigavg[i] = np.abs(bin_orf)*np.sqrt(s2)
    
    return xiavg, rhoavg, sigavg


def orf_xi(xi, orf='hd'):
    """A function to turn pulsar separations into correlations using a set ORF

    Given a pulsar separation or separations, compute the correlation factor
    for that separation and given overlap reduction function. Note that orf can be 
    replaced with a custom function which must accept pulsar positions (cartesian) 
    as its only 2 arguments.
    Predefined orf names are:
        'hd' - Hellings and downs
        'dipole' - Dipole
        'monopole' - Monopole
        'gw_dipole' - Gravitational wave dipole
        'gw_monopole' - Gravitational wave monopole
        'st' - Scalar transverse

    Args:
        xi (numpy.ndarray or float): A vector or float of pulsar pair separation(s)
        orf (str, function): The name of a predefined ORF function or custom function 

    Raises:
        ValueError: If given a string of an unrecognized ORF

    Returns:
        _type_: correlation(s) for the pair separation(s)
    """
    if type(orf) == str:
        from enterprise_extensions import model_orfs
        if orf.lower() == 'hd':
            orf_func = model_orfs.hd_orf
        elif orf.lower() == 'dipole':
            orf_func = model_orfs.dipole_orf
        elif orf.lower() == 'monopole':
            orf_func = model_orfs.monopole_orf
        elif orf.lower() == 'gw_dipole':
            orf_func = model_orfs.gw_dipole_orf
        elif orf.lower() == 'gw_monopole':
            orf_func = model_orfs.gw_monopole_orf
        elif orf.lower() == 'st':
            orf_func = model_orfs.st_orf
        else:
            raise ValueError(f"Undefined ORF name '{orf}'")
        orf_func
    else:
        orf_func = orf
    
    orf_lamb = lambda x: orf_func([1,0,0], [np.cos(x),np.sin(x),0])

    if np.array(xi).size>1:
        return np.array([orf_lamb(x) for x in xi])
    else:
        return orf_lamb(xi)    


def uncertainty_sample(A2, A2s, n_samples=1000):
    """A function to calculate the amplitude square distribution from the NMOS.

    A function to implement uncertainty sampling to account for all underlying 
    uncertainty in the optimal statistic A^2 and noise parameters. This function 
    uses Gaussians at each point.

    Args:
        A2 (np.ndarray): The array of amplitude estimators from the NMOS
        A2s (np.ndarray): The array of uncertainties in the amplitude estimators from the NMOS.
        n_samples (int, optional): The number of points to represent each Gaussian. Defaults to 100.

    Returns:
        tot_A2: The total distribution on A^2
    """
    tot_A2 = np.random.normal(A2,A2s,(n_samples,len(A2)))
    tot_A2 = np.reshape(tot_A2,[-1])
    return tot_A2

