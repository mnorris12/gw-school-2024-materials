
###
"""
By Kyle Gersbach for the VIPER PTA Summer School 2024

This file contains my implementation of the pair covariant Optimal Statistic. This function
is a preview for a more comprehensive OS package currently in development. Feel free to
use this function as long as you credit me! If you plan to modify these functions, reach 
out to me. I may have already done what you want!

"""
###

import numpy as np
import scipy.linalg as sl
from tqdm import tqdm

from enterprise.signals.utils import powerlaw

from itertools import combinations_with_replacement, combinations


def pair_covariant_OS(OS_obj, params, gwb_name='gw', use_tqdm=False):
    """The optimal statistic including pair covariance

    A version of the Optimal Statistic that accounts for pair covariance in cases
    where the gravitational wave background is within the intermediate or strong
    signal regimes. Developed based on Allen, Romano 2023 and implemented using 
    techniques found in Gersbach et al 2024.

    Args:
        OS_obj (OptimalStatistic): An Optimal Statistic object
        params (dict): A dictionary of PTA model parameter values
        gwb_name (str): The name of the GWB in the PTA model
        use_tqdm (bool): Whether to include a progress bar for the covariance generation

    Returns:
        xi (array): An array of pair separation angles [N_pairs]
        rho (array): An array of pair correlated powers [N_pairs]
        Sigma (array): A matrix representing the correlations between pairs [N_pairs x N_pairs]
        a2 (float): The least squares fit for the power of the background at f_year
        a2_sig (float): The uncertainty in a2
    """
    # Need to get X and Z matrix products
    X,Z = _compute_XZ(OS_obj,params,gwb_name)

    npsr = len(X)
    pairs = np.array([(i,j) for i in range(npsr) for j in range(i+1, npsr)])
    a,b = pairs[:,0],pairs[:,1]

    if OS_obj.gamma_common is None and OS_obj.gwb_name+'_gamma' in params.keys():
        phiIJ = powerlaw(OS_obj.freqs, log10_A=0, gamma=params[gwb_name+'_gamma'])
    else:
        phiIJ = powerlaw(OS_obj.freqs, log10_A=0, gamma=OS_obj.gamma_common)

    # Same thing as X[a].T @ phi @ X[b] where n represents pairs
    tops = np.einsum('ni,i,ni->n',X[a],phiIJ,X[b])
    # Same thing as trace(Z[a] @ phi @ Z[b] @ phi) where n represents pairs
    bots = np.einsum('nij,j,nji,i->n',Z[a],phiIJ,Z[b],phiIJ)
        
    xi = np.arccos([np.dot(OS_obj.psrlocs[i],OS_obj.psrlocs[j]) for (i,j) in pairs])
    rho = tops/bots
    sig = 1/np.sqrt(bots)

    ORF = np.array([OS_obj.orf(OS_obj.psrlocs[i],OS_obj.psrlocs[j]) for (i,j) in pairs])

    # Get the ORF as a matrix for use later
    orf_matrix = np.zeros((npsr,npsr))
    orf_matrix[a,b] = ORF
    orf_matrix[b,a] = ORF
    
    # We need to get an estimate on A2. Use CURN estimate
    a2 = 10**(2*params[gwb_name+'_log10_A']) 

    # We need to get the covariance matrix, fact_Sigma needs factors of A**2
    fact_Sigma = _get_factored_covariance(Z, phiIJ, orf_matrix,sig, use_tqdm, 300)
    Sigma = 1*fact_Sigma[0] + a2*fact_Sigma[1] + a2**2*fact_Sigma[2]
    
    W = np.linalg.pinv(Sigma)
    fit_a2, fit_sig2, _ = _linear_solve(ORF,rho,W) #Unused value is SNR
    
    return xi, rho, Sigma, fit_a2, np.sqrt(fit_sig2)


def pair_covariant_NMOS(OS_obj, chain, param_names, Niter=100, gwb_name='gw'):
    """The pair covariant noise marginalized optimal statistic

    A function applying the pair covariance paradigm to the noise marginalized
    optimal statistic. Keep in mind that this function may take substantial time
    depending on the number of pulsar pairs in your dataset.

    Args:
        OS_obj (OptimalStatistic): An Optimal Statistic object
        chain (np.ndarray): The CURN chain to take samples from
        param_names (list): A list of the corresponding parameter names for the CURN chain
        Niter (int): The number of NMOS iterations to run
        gwb_name (str): The name of the GWB in the PTA model

    Returns:
        a2 (np.ndarray): An array of the NMOS A^2 values.
        a2_sig (np.ndarray): An array of 1-sigma uncertainties in a2
    """
    a2 = np.zeros(Niter)
    a2_sig = np.zeros(Niter)

    rand_idx = np.random.randint(0,len(chain),size=Niter)
    for i in tqdm(range(Niter)):
        idx = rand_idx[i]
        params = {p:v for p,v in zip(param_names,chain[idx])}

        _,_,_,a2[i],a2_sig[i] = pair_covariant_OS(OS_obj, params, gwb_name, False)

    return a2, a2_sig




def _get_factored_covariance(Z,phiIJ,orf_mat,pair_sig, use_tqdm, max_chunk):
    """This function will return the GW amplitude factored covariance matrix.

    This function uses numpy array indexing shenanigans and numpy vectorized
    operations to compute a factored version of the pulsar pair covariance matrix.
    The format of the returned covariance matrix is a 3 x N_pairs x N_pairs matrix
    where the first term should be multiplied by 1, the second by A^2_gw, and
    the 3rd by A^4_gw before summing. 
    NOTE: The memory usage of this function scales with (N_pulsar^4 * N_freq^2)

    Args:
        Z (array): An N_pulasr array of 2N_frequencies x 2N_frequencies Z matrices from the OS
        phiIJ (array): A 2N_frequencies array of the spectral shape of the GWB
        orf_mat (array): A N_pulsar x N_pulsar matrix of the ORF for each pair of pulsars
        pair_sig (array): A N_pair array of OS uncertainties on individual pair correlated amplitudes
        use_tqdm (bool): A boolean of wether to use a progress bar
        max_chunk (int): The maximum number of simultaneous matrix multiplications. 100-1000 seem to work best
       
    Returns:
        array: A 3 x N_pairs x N_pairs matrix of the final amplitude factored covariance matrix
    """
    npsr = len(Z)
    nfreq = len(Z[0])

    pairs_idx = np.array(list( combinations(range(npsr),2) ),dtype=int)

    # Get pairs of pairs, both the indices of the pairs, and the pulsar indices
    PoP_idx = np.array(list( combinations_with_replacement(range(len(pairs_idx)),2) ),dtype=int)
    
    PoP = np.zeros((len(PoP_idx),4),dtype=int)
    PoP[:,(0,1)] = pairs_idx[PoP_idx[:,0]] 
    PoP[:,(2,3)] = pairs_idx[PoP_idx[:,1]]

    # It is also helpful to create some basic filters. From (ab,cd)
    psr_match = (PoP[:,(0,1)] == PoP[:,(2,3)]) # checks (a==c,b==d)
    psr_inv_match = (PoP[:,(0,1)] == PoP[:,(3,2)]) # checks (a==d,b==c)

    # It will be faster to pre-compute some quantities
    Zphi = Z@np.diag(phiIJ)
    ZphiZphi = np.zeros((npsr,npsr,nfreq,nfreq))
    a,b = pairs_idx[:,0],pairs_idx[:,1]
    ZphiZphi[a,b] = Zphi[a] @ Zphi[b]
    ZphiZphi[b,a] = Zphi[b] @ Zphi[a]


    def case1(a,b,c,d): #(ab,cd)
        a0 = np.zeros_like(a)
        a2 = np.zeros_like(a)
        a4 = orf_mat[a,c] * orf_mat[d,b] * np.einsum('ijk,ikj->i', ZphiZphi[b,a], ZphiZphi[c,d]) + \
             orf_mat[a,d] * orf_mat[c,b] * np.einsum('ijk,ikj->i', ZphiZphi[b,a], ZphiZphi[d,c])
        return [a0,a2,a4]
    
    def case2(a,b,c): #(ab,ac)
        a0 = np.zeros_like(a)
        a2 = orf_mat[b,c] * np.einsum('ijk,ikj->i', ZphiZphi[b,a], Zphi[c])
        a4 = orf_mat[a,c] * orf_mat[a,b] * np.einsum('ijk,ikj->i', ZphiZphi[b,a], ZphiZphi[c,a])
        return [a0,a2,a4]

    def case3(a,b): #(ab,ab)
        a0 = np.trace(ZphiZphi[b,a],axis1=1,axis2=2)
        a2 = np.zeros_like(a)
        a4 = orf_mat[a,b]**2 * np.einsum('ijk,ikj->i', ZphiZphi[b,a], ZphiZphi[b,a])
        return [a0,a2,a4]

    
    # --------------------------------------------------------------------------
    # Now lets calculate them!
    C_m = np.zeros((3,len(pairs_idx),len(pairs_idx)),dtype=np.float64)

    if use_tqdm:
        from tqdm import tqdm
        ntot = len(PoP)
        progress = tqdm(total=ntot,desc='Pairs of pairs',ncols=80)


    try: # This is used to close the progress bar if an exception occurs

        # Case1: no matching pulsars--------------------------------------------
        mask = (~psr_match[:,0] & ~psr_match[:,1]) & \
               (~psr_inv_match[:,0] & ~psr_inv_match[:,1])
        
        p_idx1,p_idx2 = PoP_idx[mask].T
        a,b,c,d = PoP[mask].T
    
        for i in range( int(len(a)/max_chunk)+1 ):
            l,h = i*max_chunk,(i+1)*max_chunk
            temp = case1(a[l:h],b[l:h],c[l:h],d[l:h])
            C_m[:,p_idx1[l:h],p_idx2[l:h]] = temp
            C_m[:,p_idx2[l:h],p_idx1[l:h]] = temp
            if use_tqdm: progress.update(len(a[l:h]))
    

        # Case2: 1 matching pulsar----------------------------------------------
        mask = (psr_match[:,0] & ~psr_match[:,1]) # Check for (ab,ac)
        p_idx1,p_idx2 = PoP_idx[mask].T
        a,b,_,c = PoP[mask].T

        for i in range( int(len(a)/max_chunk)+1 ):
            l,h = i*max_chunk,(i+1)*max_chunk
            temp = case2(a[l:h],b[l:h],c[l:h])
            C_m[:,p_idx1[l:h],p_idx2[l:h]] = temp
            C_m[:,p_idx2[l:h],p_idx1[l:h]] = temp
            if use_tqdm: progress.update(len(a[l:h]))


        mask = (~psr_inv_match[:,0] & psr_inv_match[:,1]) # Check for (ab,bc)
        p_idx1,p_idx2 = PoP_idx[mask].T
        b,a,_,c = PoP[mask].T # Index swap a with b

        for i in range( int(len(a)/max_chunk)+1 ):
            l,h = i*max_chunk,(i+1)*max_chunk
            temp = case2(a[l:h],b[l:h],c[l:h])
            C_m[:,p_idx1[l:h],p_idx2[l:h]] = temp
            C_m[:,p_idx2[l:h],p_idx1[l:h]] = temp
            if use_tqdm: progress.update(len(a[l:h]))


        mask = (~psr_match[:,0] & psr_match[:,1]) # Check for (ab,cb)
        p_idx1,p_idx2 = PoP_idx[mask].T
        b,a,c,_ = PoP[mask].T # Index swap a with b

        for i in range( int(len(a)/max_chunk)+1 ):
            l,h = i*max_chunk,(i+1)*max_chunk
            temp = case2(a[l:h],b[l:h],c[l:h])
            C_m[:,p_idx1[l:h],p_idx2[l:h]] = temp
            C_m[:,p_idx2[l:h],p_idx1[l:h]] = temp
            if use_tqdm: progress.update(len(a[l:h]))

        # Case3: 2 matching pulsars---------------------------------------------
        mask = psr_match[:,0] & psr_match[:,1] # Check for (ab,ab)
        
        p_idx1,p_idx2 = PoP_idx[mask].T
        a,b,_,_ = PoP[mask].T 

        for i in range( int(len(a)/max_chunk)+1 ):
            l,h = i*max_chunk,(i+1)*max_chunk
            temp = case3(a[l:h],b[l:h])
            C_m[:,p_idx1[l:h],p_idx2[l:h]] = temp
            C_m[:,p_idx2[l:h],p_idx1[l:h]] = temp
            if use_tqdm: progress.update(len(a[l:h])) 

    except Exception as e:
        if use_tqdm: progress.close()
        print('Exception occured during pair covariance creation!')
        raise e
    if use_tqdm: progress.close()

    # Include the final sigmas
    C_m[:] *= np.outer(pair_sig**2,pair_sig**2)

    return C_m


def _compute_XZ(OS_obj, params, gwb_name='gw'):
        """A function to quickly calculate the OS' matrix quantities

        This function calculates the X and Z matrix quantities from the appendix A
        of Pol, Taylor, Romano, 2022: (https://arxiv.org/abs/2206.09936). X and Z
        can be represented as X = F^T @ P^{-1} @ r and Z = F^T @ P^{-1} @ F.

        Args:
            OS_obj (OptimalStatistic): An enterprise extensions optimal statistic object
            params (dict): A dictionary containing the parameter name:value pairs for the PTA

        Returns:
            (np.array, np.array): A tuple of X and Z. X is an array of vectors for each pulsar 
                (N_pulsar x 2N_frequency). Z is an array of matrices for each pulsar 
                (N_pulsar x 2N_frequency x 2N_frequency)
        """

        X, Z = [], []
        
        for psr_signal in OS_obj.pta:
            # Need residuals r, GWB Fourier design F, and pulsar design matrix T = [M F]
            r = psr_signal._residuals
            F = psr_signal[gwb_name].get_basis(params)
            T = psr_signal.get_basis(params)

            # Used in creating P^{-1}
            # Need N, use .solve() for inversions
            N = psr_signal.get_ndiag(params)

            # sigma = B^{-1} + T^T @ N^{-1} @ T
            sigma = sl.cho_factor( np.diag(psr_signal.get_phiinv(params)) + psr_signal.get_TNT(params) )

            FNr = N.solve(r,F) # F^T @ N^{-1} @ r
            TNr = N.solve(r,T) # T^T @ N^{-1} @ r
            FNT = N.solve(T,F) # F^T @ N^{-1} @ T
            FNF = N.solve(F,F) # F^T @ N^{-1} @ F
        
            # X = F^T @ P^{-1} @ r =
            # F^T @ N^{-1} @ r - F^T @ N^{-1} @ T @ sigma^{-1} @ T^T @ N^{-1} @ r
            X.append( FNr - FNT @ sl.cho_solve(sigma, TNr) )

            # Z = F^T @ P^{-1} @ F =
            # F^T @ N^{-1} @ F - F^T @ N^{-1} @ T @ sigma^{-1} @ T^T @ N^{-1} @ F
            Z.append( FNF - FNT @ sl.cho_solve(sigma, FNT.T) )

        return np.array(X), np.array(Z)


def _linear_solve(x,r,cinv):
    """A basic chi-square solver

    Minimizing the equation (r - x*theta)^T c^(-1) (r - x*theta)
    If you have n data points and want to fit for m parameters 
    then x should have shape (n x m) [n rows with m columns]

    If x is given as a (n) vector, it will be converted to an (n x 1) matrix for you.
    
    Args:
        x (array): Design matrix [(n x m) matrix]
        r (array): Data values [(n) vector]
        cinv (array): uncertainties on each parameter [(n x n) matrix]
    
    Returns:
        theta (array): Solution vector theta [ (m) vector or scalar if 1d]
        cov (array): Covariance matrix [ (m x m) matrix or scalar if 1d]
        snr (float): Total snr [scalar]
    """

    xin = np.array(x)
    if len(xin.shape)==1:
        xin=xin[:,None]

    rin = np.array(r)
    if len(rin.shape)==1:
        rin = r[:,None]

    fisher = xin.T @ cinv @ xin
    cov = np.linalg.pinv(fisher)
    term2 = xin.T @ cinv @ rin
    theta = cov @ term2
    snr = np.sqrt(theta.T @ fisher @ theta).item()

    theta = np.squeeze(theta)
    cov = np.squeeze(cov)

    return theta, cov, snr

