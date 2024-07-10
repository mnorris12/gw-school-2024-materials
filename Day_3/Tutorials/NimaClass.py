import numpy as np
import json, os, shutil
from functools import cached_property
from tqdm import tqdm
import scipy.special as spec
from joblib import Parallel, delayed

class Nimapta(object):
    '''
    Simplets PTA Object to be used with the injections that Nima does!!!

    :param: res: a pulsar's residulas (MJD)
    :param: toas: a pulsar's toas (MJD)
    :param wn_sigma: Sigma value of the injected white noise

    :param: self.Mmat: Design Matrix
    :param: fit : quadratic fit
    :param: gw_res: in case you know it!
    :param: RN_res: in case you know it!
    '''
    def __init__(self, res, toas, wn_sigma, psr_pos = None, Mmat = None, fit = False, gw_res = None, RN_res = None):

        self.toas = toas * 86400
        self.sigma = wn_sigma
        self.psr_pos = psr_pos
        self.res = res
        self.fit = fit
        self.DMat = Mmat
        self.gw_res = gw_res
        self.RN_res = RN_res

    def Mmat(self):
        if not np.any(self.DMat):
            DesignM = np.ones((len(self.toas), 3))
            DesignM[:,1] = self.toas
            DesignM[:,2] = DesignM[:,1]**2
            return DesignM
        else:
            return self.DMat

    def WN_inv(self):
        if type(self.sigma) == float or type(self.sigma) == np.float64:
            return np.diag([1./self.sigma**2 for t in self.toas])
        else:
            return np.diag(1./self.sigma**2)

    def Rmat(self):
        '''
        The Rmatrix used to do the quadratic fitting
        '''
        I = np.identity(len(self.toas))
        DesignM = self.Mmat()
        return I - np.einsum('kn,nm,pm->kp', DesignM, np.linalg.inv(np.einsum('km,kn->nm',DesignM,DesignM)),DesignM)

    def post_fit_res(self):
        if self.fit:
            return self.res @ self.Rmat()
        else:
            return self.res

class BayesPower(object):

    def __init__(self, Nimapta, crn_bins, num_samples, inj_amp, gamma,red_amp = None, red_gamma = None,
                low = 1e-18, high = 1e-8,
                Baseline = False):
        """
        Model class contains calls to functions required to
        apply the 'Bayes Power' technique to a simple simulated
        PTA Data set. In developement!!!!!:

        :param Nimapta: Nima's simple pta object. See above!!!
        :param crn_bins: Number of frequency bins for CRN process
        :param inj_amp: Injected Amplitude for GWB
        :param gamma: Spectral Index of the GWB
        :param num_samples: Number of Draws Used in Both Gibbs Sampling and Random Sampling from a Distribution
        .
        .
        .
        """
        yr_to_sec = 86400 * 365.25
        self.ref_f = 1/(yr_to_sec)

        self.pta = Nimapta
        self.psr_pos = Nimapta.psr_pos
        self.Mmat = Nimapta.Mmat()
        self.Nmat_inv = Nimapta.WN_inv()

        self.rhohigh = high
        self.rholow = low

        self.crn_bins = crn_bins
        self.inj_amp = inj_amp
        self.gamma = gamma
        self.red_amp = red_amp
        self.red_gamma = red_gamma
        self.toas = Nimapta.toas
        self.int_Tspan = max(self.toas) - min(self.toas)
        self.res = Nimapta.post_fit_res()
        self.RN_res = Nimapta.RN_res
        self.gw_res = Nimapta.gw_res

        if Baseline:
            self.Tspan = Baseline * yr_to_sec
        else:
            self.Tspan = max(self.toas) - min(self.toas)
        self.freqs = np.arange(1/self.Tspan, (crn_bins + .01)/self.Tspan, 1/self.Tspan)
        self.int_freqs = np.arange(1/self.Tspan, 30.01/self.Tspan, 1/self.Tspan)

        self.num_samples = num_samples
        self.rand_idx = np.random.randint(0, self.num_samples, 1500)
        #self.randgen = np.random.default_rng()

        self._b = None
        self._rho = None
        self._rhoCommon = None
        self.INWTF = None

    @cached_property
    def Fmat(self):
        '''
        The 'F-matrix' used to do a discrete Fourier transform

        Author: Nima Laal
        '''
        freqs=self.freqs
        nmodes = len(freqs)
        toas = self.toas
        N = len(toas)
        F = np.zeros((N, 2 * nmodes))
        F[:, 0::2] = np.sin(2 * np.pi * toas[:, None] * freqs[None, :])
        F[:, 1::2] = np.cos(2 * np.pi * toas[:, None] * freqs[None, :])
        return F, F.T

    def Freq_to_Time_2D(self, Mat):
        '''
        Discrete Fourier transform from frequency domain to time domain
        for a 2D n_freq by n_freq matrix.

            :param Mat: 2D n_freq by n_freq matrix

        Author: Nima Laal
        '''

        sin_M_sin = self.Fmat[0][:, 0::2] @ Mat @ self.Fmat[1][0::2, :]
        cos_M_cos = self.Fmat[0][:, 1::2] @ Mat @ self.Fmat[1][1::2, :]
        return sin_M_sin + cos_M_cos

    def Time_to_Freq_2D(self, Mat, diag = False):
        '''
        Discrete Fourier transform from time domain to frequency domain
        for a 2D n_toa by n_toa matrix.

            :param Mat: 2D n_toa by n_toa matrix

        Author: Nima Laal
        '''

        sin_M_sin = self.Fmat[1][0::2, :] @ Mat @ self.Fmat[0][:, 0::2]
        cos_M_cos = self.Fmat[1][1::2, :] @ Mat @ self.Fmat[0][:, 1::2]

        if diag:
            return np.repeat(np.diagonal(sin_M_sin + cos_M_cos), 2)
        else:
            return sin_M_sin + cos_M_cos

    @cached_property
    def Nmat(self):
        '''
        The White noise covaraince matrix

        Author: Nima Laal
        '''
        return np.identity(len(self.toas)) * self.pta.sigma**2

    @cached_property
    def Dmat(self):
        '''
        The Inverse of 'D-matrix' which is like the N inverse matrix,
        but it takes into account marginalization over timing model parameters

        Author: Nima Laal
        '''

        N_inv = self.Nmat_inv
        M = self.Mmat
        MNM_inv = np.linalg.inv(M.T@N_inv@M)
        D_inv = N_inv - N_inv @ M @ MNM_inv @ M.T @ N_inv
        return D_inv

    @cached_property
    def TNT(self):
        F = self.Fmat
        #return F[1] @ self.Nmat_inv @ F[0]
        return F[1] @ self.Dmat @ F[0]

    @cached_property
    def Gmat(self):
        '''
        The G-Matrix used for fitting timing residulas

        Author: Joeseph Romano
        '''

        U, S, V = np.linalg.svd(self.Mmat, full_matrices=True)
        # extract G from U
        N_TOA = len(U)
        N_par = len(V)
        m = N_TOA-N_par
        G = U[:, N_par:]
        return G, G.T

    @cached_property
    def Rmat(self):
        '''
        The Rmatrix used to do the quadratic fitting

        Author: Nima Laal
        '''
        I = np.identity(len(self.toas))
        DesignM = self.Mmat
        return I - np.einsum('kn,nm,pm->kp', DesignM, np.linalg.inv(np.einsum('km,kn->nm',DesignM,DesignM)),DesignM)

    @cached_property
    def TFunc(self):
        '''
        Transmission fucntion

        Author: Nima Laal
        '''

        return np.diagonal(self.Time_to_Freq_2D(Mat = self.Gmat[0] @ self.Gmat[1]))/len(self.toas)

    @cached_property
    def HF(self):
        '''
        The bias factor when using 'mu' instead of 'bchain'.
        It must be used in the denominator of FDOS.

        Author: Nima Laal
        '''
        D_inv = self.Dmat
        F = self.Fmat
        TNT = F[1] @ D_inv @ F[0]
        HF = []
        for idx in range(self.num_samples):
            phiinv = self.phiinv_mat(self._rho[:, idx])
            HF.append(np.linalg.inv(TNT + np.diag(phiinv)) @ F[1] @ D_inv @ F[0])
        return np.array(HF)[self.rand_idx,:,:].T
        #return np.array(HF)[:,:,:].T

    @cached_property
    def max_b(self):
        '''
        Maximum likelihood estimate of the Fourier coefficients 'b'.

        Author: Nima Laal
        '''

        D_inv = self.Dmat
        F = self.Fmat
        TNT = F[1] @ D_inv @ F[0]
        b_hat = np.linalg.inv(TNT) @ F[1] @ D_inv @ self.res
        return b_hat

    def RedPower(self, idx, truth):
        '''
        Total red noise power estimated from a free-spectal run

            :param idx: The index of the chain to marginalize over
            :param truth: if set to 'True' will use the injected values

        Author: Nima Laal
        '''

        if truth:
            PW_i = 10**(2 * self.red_truth)
            PW_c = 10**(2 * self.truth())
            #PW_i = 0
            #PW_c = 10**(2 * np.mean(self._fit_rho, axis = 1))
        else:
            '''
            PW_ii = 10**(2 * self._rho[:, idx])
            PW_cc = 10**(2 * self.truth())
            if PW_ii[0] > PW_cc[0]:
                PW_i = 0
                PW_c = PW_ii
            else:
                PW_i = 0
                PW_c = PW_cc
            '''
            PW_i = 0
            PW_c = 10**(2 * self._rho[:, idx])
        return self.Tspan * (PW_c + PW_i)

    @cached_property
    def WNPower(self):
        '''
        Power from inverse white noise covariance matrix.

        Author: Nima Laal
        '''

        return 1/np.diagonal(self.Time_to_Freq_2D(self.Nmat_inv)) * (2 * self.Tspan)

    @cached_property
    def r_I(self):
        '''
        Fitted residuals.

        Author: Nima Laal
        '''
        #return BP.Gmat[1] @ BP.Fmat[0] @ BP._b
        #return  self.Gmat[1] @ np.mean(self.Fmat[0] @ self._b, axis = 1)
        #return BP.Gmat[1] @ BP.Fmat[0] @ BP.max_b
        return self.Gmat[1] @ self.res

    @cached_property
    def P_inv_I(self, idx = None, truth = True, method = 'power'):
        '''
        P_invese of time domain OS

        Author: Nima Laal
        '''
        return np.linalg.inv(self.Gmat[1] @ (self.RNCov(idx, truth, method) + self.Nmat) @ self.Gmat[0])

    def FanN(self, idx, truth, method):
        '''
        Noise inverted trasmission function/FDOS weights estimation.

            :param idx: The index of the chain to marginalize over
            :param truth: if set to 'True' will use the injected values
            :param method: options are 'exact', 'app', and 'white'

        Author: Nima Laal
        '''
        if method == 'fitted':

            Mat = self.Gmat[0] @ np.linalg.inv(self.Gmat[1] @ (self.RNCov(idx = idx, truth = truth))\
            @ self.Gmat[0]) @ self.Gmat[1]
            calc = np.diagonal(self.Time_to_Freq_2D(Mat = Mat))/(2 * self.Tspan)

        elif method == 'unfitted':
            calc = 1/self.RedPower(idx, truth)

        else:
             raise ValueError("options are 'fitted', and 'unfitted'")

        return np.repeat(calc, 2)

    def calc_INWTF(self, truth, method, NMarg):
        '''
        Function to cache the output of 'FanN'.

            :param truth: if set to 'True' will use the injected values
            :param method: options are 'exact', 'app', and 'white'

        Author: Nima Laal
        '''

        if not np.any(self.INWTF):

            if not truth:
                    self.INWTF = np.array(Parallel(n_jobs=16)(delayed(self.FanN)(i, False, method) for i in self.rand_idx)).T
            else:
                if NMarg:
                    self.INWTF = np.repeat(self.FanN(None, True, method), len(self.rand_idx)).reshape(2 * self.crn_bins, len(self.rand_idx))
                else:
                    self.INWTF = self.FanN(None, True, method)

    def RNCov(self, idx, truth, method, int_red = True):
        '''
        Red noise covariance matrix estimation

            :param idx: The index of the chain to marginalize over
            :param truth: if set to 'True' will use the injected values
            :param method: if set to 'power', the psd is used for estimation.
            Otherwise, trapezoid rule is used to do integration

        Author: Nima Laal
        Author: Jeff Hazboun
        '''
        if method == 'exact':
            nmax = 20
            gamma = self.gamma
            norm = self.inj_amp**2 /(12 * np.pi**2 * self.ref_f**(3-gamma))
            calc = np.zeros((nmax, len(self.toas), len(self.toas)))
            t1, t2 = np.meshgrid(self.toas, self.toas)
            tau = 2 * np.pi * np.abs(t1-t2)
            del t1; del t2
            p1 = tau**(gamma - 1) * spec.gamma(1-gamma) * np.cos(np.pi/2 * (1-gamma))
            idxs = np.arange(0, nmax + 2, 2, dtype = int)
            for ct, n in enumerate(idxs):
                    calc[ct, :, :] = tau**n * np.cos(np.pi/2*n) * (self.freqs[0])**(n-gamma+1)/(np.math.factorial(n) * (n-gamma+1))
            p2 = np.sum(calc, axis = 0)

            del calc

            cov0 = 2 * norm * (p1 - p2)

            if int_red:
                gamma = self.red_gamma
                norm = self.red_amp**2 /(12 * np.pi**2 * self.ref_f**(3-gamma))
                calc = np.zeros((nmax, len(self.toas), len(self.toas)))
                t1, t2 = np.meshgrid(self.toas, self.toas)
                tau = 2 * np.pi * np.abs(t1-t2)
                del t1; del t2
                p1 = tau**(gamma - 1) * spec.gamma(1-gamma) * np.cos(np.pi/2 * (1-gamma))
                idxs = np.arange(0, nmax + 2, 2, dtype = int)
                for ct, n in enumerate(idxs):
                        calc[ct, :, :] = tau**n * np.cos(np.pi/2*n) * (self.freqs[0])**(n-gamma+1)/(np.math.factorial(n) * (n-gamma+1))
                p2 = np.sum(calc, axis = 0)

                del calc

                cov1 = 2 * norm * (p1 - p2)
            else:
                cov1 = 0

            return cov0 + cov1

        elif method == 'power':
            PW = self.RedPower(idx, truth)
            return self.Freq_to_Time_2D(np.diag(PW/(2 * self.Tspan)))

        elif method == 'hasasia':
            PW = self.RedPower(idx, truth)
            t1, t2 = np.meshgrid(self.toas, self.toas, indexing='ij')
            tm = np.abs(t1-t2)
            integrand = PW*np.cos(2*np.pi*self.freqs*tm[:,:,np.newaxis])
            return np.trapz(integrand, axis=2, x=self.freqs)

        else:
             raise ValueError("options are 'exact', 'power', and 'hasasia'")

    def truth(self, hat = False, repeat = True):
        '''
        The true values of GWB power in each frequency bin (based on the given injection parameters)

            :param: hat: if set to 'true', the shape of spectrum is returned.

        Author: Nima Laal
        '''
        if not hat:
            return 0.5 * np.log10(self.inj_amp**2/(12 * np.pi**2 * self.freqs**3 * self.Tspan) * (self.freqs/self.ref_f)**(3-self.gamma))
        else:
            calc = 1/(12 * np.pi**2 * self.freqs**3) * (self.freqs/self.ref_f)**(3-self.gamma)
            if repeat:
                return np.repeat(calc, 2)
            else:
                return calc

    @cached_property
    def red_truth(self):
        '''
        The true values of intrinsic red power in each frequency bin (based on the given injection parameters)

        Author: Nima Laal
        '''
        return 0.5 * np.log10(self.red_amp**2/(12 * np.pi**2 * self.freqs**3 * self.Tspan) * (self.freqs/self.ref_f)**(3-self.red_gamma))

    @cached_property
    def total_truth(self):
        p1 = self.red_amp**2/(12 * np.pi**2 * self.freqs**3 * self.Tspan) * (self.freqs/self.ref_f)**(3-self.red_gamma)
        p2 = self.inj_amp**2/(12 * np.pi**2 * self.freqs**3 * self.Tspan) * (self.freqs/self.ref_f)**(3-self.gamma)
        return 0.5 * np.log10(p1 + p2)

    def guess_x0(self, informed = True):
        '''
        Initial value of rho for gibbs sampling. choose between informative and un-informative.

        Author: Nima Laal
        '''
        means = 0.5 * np.log10(self.inj_amp**2/(12 * np.pi**2 * self.freqs**3 * self.Tspan) * (self.freqs/self.ref_f)**(3-self.gamma))
        if informed:
            return np.array([np.random.normal(loc = mean, scale = .02) for mean in means])
        else:
            np.random.uniform(-5.5, -4, size = self.crn_bins)

    def phiinv_mat(self, rho, norm = False):
        '''
        The inverse of phi matrix.
        '''
        if not norm:
            return np.repeat(1/(1 * (10**(2 * rho))), 2)
        else:
            return np.repeat(1/(self.Tspan * (10**(2 * rho))), 2)

    def phi_mat(self, rho, norm = False):
        '''
        The phi matrix
        '''
        if not norm:
            return np.repeat(10**(2 * rho), 2)
        else:
            return np.repeat(self.Tspan * 10**(2 * rho), 2)

    def ptainvgamma(self, tau, low = None, high = None):
        '''
        Inverse gamma distribution which has an upper and a lower bound for its domain.
        Little bit hard to exaplain the form here!!! I will show the derivation in a seprate document.

        Author: Stepehan Taylor
        '''
        if not low:
            low = self.rholow
        if not high:
            high = self.rhohigh
        #low = 10**(2 * (self.truth - 1.5))
        #high = 10**(2 * (self.truth + 1.5))
        #Norm = 1/(np.exp(-tau/high) - np.exp(-tau/low))
        #x = np.random.uniform(0, 1, size = tau.shape)
        #return -tau/np.log(x/Norm + np.exp(-tau/low))
        eta = np.random.uniform(0, 1-np.exp((tau/high) - (tau/low)))
        return tau / ((tau/high) - np.log(1-eta))

    def b_given_rho(self, rho, res):
        '''
        Calculate Fourier coefficients form rho values.

        Author: Nima Laal
        '''
        phiinv = self.phiinv_mat(rho)

        D_inv = self.Dmat
        F = self.Fmat
        TNT = F[1] @ D_inv @ F[0]
        #TNT = self.TNT
        #TNT = self.Time_to_Freq_2D(D_inv)
        Sigma = TNT + np.diag(phiinv)
        var = np.linalg.inv(Sigma)
        mean = np.array(var @ F[1] @ D_inv @ res)
        try:
            b = np.random.default_rng().multivariate_normal(mean = mean, cov = var, check_valid = 'raise', method = 'svd')
        except:
            b = np.random.default_rng().multivariate_normal(mean = mean, cov = var, check_valid = 'raise', method = 'cholesky')
        return b

    def rho_given_b(self, b):
        '''
        Calculate rho from Fourier coefficients.

        Author: Nima Laal
        '''
        tau = (b[0::2]**2 + b[1::2]**2) / 2
        rhonew = self.ptainvgamma(tau = tau)
        return 0.5 * np.log10(rhonew)

    @cached_property
    def bmean(self):
        meanb  = np.zeros((2*self.crn_bins, self.num_samples))
        for idx in range(self.num_samples):
            var = np.linalg.inv(self.TNT + np.diag(self.phiinv_mat(self._rho[:, idx])))
            meanb[:, idx] = var @ self.Fmat[1] @ self.Dmat @ self.res
        return meanb

    @cached_property
    def bmean_unb(self):
        meanb  = np.zeros((2*self.crn_bins, self.num_samples))
        for idx in range(self.num_samples):
            var = np.linalg.inv(self.TNT + np.diag(self.phiinv_mat(self._rho[:, idx])))
            meanb[:, idx] = (var @ self.Fmat[1] @ self.Dmat @ self.res)/np.diagonal(self.HF[:, :, idx])
        return meanb

    def gibbs_sampler(self, num_samples = None, progress = False):
        '''
        Gibbs Sampling

        Author: Nima Laal
        '''
        if not num_samples:
            num_samples = self.num_samples

        b = np.zeros((2*self.crn_bins, num_samples+1))
        rho = np.zeros((self.crn_bins, num_samples+1))
        rho[:,0] = self.guess_x0(informed = True)

        if progress:
            for ii in tqdm(range(num_samples), colour="GREEN"):
                b[:,ii] = self.b_given_rho(rho = rho[:,ii], res = self.res)
                rho[:,ii+1] = self.rho_given_b(b[:,ii]) #self.guess_x0(informed = True)
        else:
            for ii in range(num_samples):
                b[:,ii] = self.b_given_rho(rho = rho[:,ii], res = self.res)
                rho[:,ii+1] = self.rho_given_b(b[:,ii])

        b[:,ii+1] = self.b_given_rho(rho = rho[:,ii+1], res = self.res)
        return b[:, 1:], rho[:, 1:]
