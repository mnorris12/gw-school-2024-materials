import numpy as np
import scipy.interpolate as interp
import glob, os, json, pickle, math, copy, shutil
from itertools import product
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.coordinates import ICRS
import libstempo as T2, libstempo.toasim as LT
import libstempo.plot as LP
from enterprise.pulsar import Pulsar
import scipy.special as ss
from altorfs import HD_ORF, ST_ORF
from functools import cached_property

np.seterr(all='ignore')
plt.style.use('dark_background')

day = 24*3600
year = 365.25 * day
euler_const = 0.5772156649
kpc_to_meter = 3.086e19
light_speed = 299792458
f_yr = 1/year #reference frequency in seconds

def do_chol(chi_reshaped):
    try:
        H = np.linalg.cholesky(chi_reshaped)
        return H
    except:
        return False


def UniformPulsarDist(Numpulsars, seed = None, name = True, skyplot = True):
    '''
    This function creates a population of random pulsars distributed uniformly in the sky. The coordinates are reported in the Geocentric-true-ecliptic frame.

    :param: Npulsars: Number of pulsars to make
    :param: seed: sets the seed for the random number generator
    :param: name: if set to true, each generated pulsar will have its own unique name based on the usual pulsar naming convension.
    :param: skyplot: if set to true, a sky plot of the pulsar population will be displayed.
    :output: lam, beta, pname: Geocentric-true-ecliptic longitude, latitude, and pulsar names (if name is set to true)

    Author: Nima Laal
    '''
    if seed:
        np.random.seed(seed)

    dec = np.arcsin(np.random.uniform(-1,1, size = Numpulsars))
    ra = np.random.uniform(0,2*np.pi, size = Numpulsars)
    sc = SkyCoord(ra = ra , dec = dec, unit = 'rad', frame='icrs')
    lam = sc.geocentrictrueecliptic.lon.deg
    beta = sc.geocentrictrueecliptic.lat.deg

    if skyplot:
        plt.figure(dpi = 170, figsize=(8, 4))
        plt.subplot(projection="aitoff")

        c = SkyCoord(ra*180/np.pi, dec*180/np.pi, unit = 'deg', frame='icrs')
        ra_rad = c.ra.wrap_at(180 * u.deg).radian
        dec_rad = c.dec.radian
        plt.scatter(ra_rad  , dec_rad,marker=(5, 2),color = 'r',label = 'Pulsars\nM = {}'.format(Numpulsars))

        plt.xticks(ticks=np.radians([-150, -120, -90, -60, -30, 0, \
                                     30, 60, 90, 120, 150]),
                   labels=['10h', '8h', '6h', '4h', '2h', '0h', \
                           '22h', '20h', '18h', '16h', '14h'])

        plt.xlabel('Right Ascension in hours')
        plt.ylabel('Declination in deg.')
        plt.grid(True)
        plt.legend(loc = 'upper right')
        plt.show()

    if name:
        pname = []
        for ii in range(Numpulsars):
            coo = ICRS(ra[ii]*u.rad, dec[ii]*u.rad)
            to_replace = ['h','m','s']
            ra_coo = coo.ra.to_string(u.hourangle)
            dec_coo = coo.dec.to_string(u.hourangle)
            for tr in to_replace:
                ra_coo = ra_coo.replace(tr, ':')
                dec_coo = dec_coo.replace(tr, ':')

            dec_str = dec_coo.split(':')
            ra_str = ra_coo.split(':')
            if len(ra_str[0]) == 1:
                ra_str[0] = '0' + ra_str[0]

            if float(dec_str[0]) < 0 or dec_str[0] == '-0':
                pname.append("J" + ra_str[0] + ra_str[1] + dec_str[0] + dec_str[1])
            else:
                pname.append("J" + ra_str[0] + ra_str[1] + "+" +  dec_str[0] + dec_str[1])


        return lam, beta, pname, np.vstack((sc.cartesian.x.value, sc.cartesian.y.value, sc.cartesian.z.value))

    return lam, beta, np.vstack((sc.cartesian.x.value, sc.cartesian.y.value, sc.cartesian.z.value))


def make_parfile(donor_loc, save_loc , lam, beta, pname = None):
    '''
    Generates fake par files form donor parfiles. Only pulsar name and coordnoates will be changed from the donor par file.

    :param: donor_loc: directory of the donor parfiles
    :param: save_loc: directory to save the fake par files
    :param: lam: ecliptic longitude coordinate
    :param: beta: ecliptic latitude coordinate
    :param: pname: name of the pulsars. If set to none, J1, J2, ..., Jn will be the names of the pulsars.

    Author: Nima Laal
    '''

    tempParfiles = sorted(glob.glob(donor_loc + '/*.par'))
    if not pname:
        pname = ['J' + str(x) for x in np.arange(1, len(tempParfiles) + 1, dtype = int)]

    for kk, par in enumerate(tempParfiles):
        with open(par, 'r') as pfile:
            li = pfile.read().splitlines()

            line0 = li[0].split(' ')[:-1]
            line0.append(pname[kk])
            line0 = ' '.join(line0)

            line1 = li[1].split(' ')
            for ii, el in enumerate(line1):
                try:
                    float(el)
                    idx = ii
                    break
                except ValueError:
                    continue
            line1[idx] = str(lam[kk])
            line1 = ' '.join(line1)

            line2 = li[2].split(' ')
            for ii, el in enumerate(line2):
                try:
                    float(el)
                    idx = ii
                    break
                except ValueError:
                    continue
            line2[idx] = str(beta[kk])
            line2 = ' '.join(line2)

            li[0] = line0
            li[1] = line1
            li[2] = line2

            np.savetxt(save_loc  + '/{}.par'.format(pname[kk]), li, fmt="%s")

def make_timfile(toas, parfile_path, save_loc,toaerr = 0.5):
    '''
    Generates fake tim files form given parfiles.

    :param: toas: a list or an array of toas. The dimension must be (len(parfiles), N)
    :param: toaerr: a constant toa error in units of microseconds. You can change this per pulsar while doing injection later.
    :param: parfile_dir: location of the parfile(s)
    :param: save_loc: directory to save the fake tim file(s)

    Author: Nima Laal
    '''
    p = LT.fakepulsar(parfile_path , obstimes = toas , toaerr = toaerr)
    p.savetim(save_loc + '/' + p.name + '.tim')

def make_noise_dict(psrname, efac, equad, ecorr, log10_A, gamma, pdist):
    '''
    makes a noise dictionary out of given white noise and intrinsic red noise values.

    :param: psrnmae: name of the pulsar
    :param: efac: efac value of the pulsar
    :param: equad: log10(equad) value of the pulsar
    :param: ecorr: log10(ecorr) value of the pulsar
    :param: log10_A: log10(A) value of the pulsar's intrinsic red noise
    :param: gamma: spectral index value of the pulsar's intrinsic red noise

    Author: Nima Laal
    '''
    noise_dict = {}
    ef = "_efac"
    eq = "_log10_equad"
    ec = "_log10_ecorr"
    A = "_red_noise_log10_A"
    g = "_red_noise_gamma"
    d = "_crn_p_dist"
    noise_dict.update({psrname + ef: efac})
    noise_dict.update({psrname + eq: equad})
    noise_dict.update({psrname + ec: ecorr})
    noise_dict.update({psrname + A: log10_A})
    noise_dict.update({psrname + g: gamma})
    noise_dict.update({psrname + d: pdist})

    return noise_dict


class PTASIM(object):

    def __init__(self, toas, psrlist, Amp, alpha, psr_locs, MG = 'TT', RNamp = np.array([None]), RNalpha = np.array([None]), 
                 pdist = np.array([None]), kappa = 0, Nrea = 1,
                 seed = 156457, cad = 14, howml = 10, wn_sig = 1e-6):

            self.MG = MG
            self.Amp = Amp
            self.RNamp = RNamp
            self.RNalpha = RNalpha
            self.alpha = alpha
            self.pdist = pdist
            self.kappa = kappa
            self.seed = seed
            self.cad = cad
            self.Nrea = Nrea
            self.wn_sig = wn_sig
            self.howml = howml
            self.psrlist = psrlist
            self.Npulsars = len(self.psrlist)
            self.psr_locs = psr_locs
            self.auto_indx = np.arange(0, self.Npulsars*self.Npulsars + 1, self.Npulsars + 1, int)
            self.toas = toas
            t = np.hstack(self.toas)
            # gw start and end times for entire data set
            start = t.min() * day - 1 * day
            stop = t.max() * day + 1 * day
            self.dur = stop - start
            self.npts = int(self.dur/(day*self.cad))
            # make a vector of evenly sampled data points
            self.ut = np.linspace(start, stop, self.npts)
            # time resolution in seconds
            self.dt = self.dur/self.npts
            # Define frequencies spanning from DC to Nyquist.
            # This is a vector spanning these frequencies in increments of 1/(dur*howml).
            self.freqs = np.arange(0, 1/(2*self.dt), 1/(self.dur*self.howml))
            self.f_norm = self.freqs * kpc_to_meter/light_speed
            self.freqs[0] = self.freqs[1] # avoid divide by 0 warning
            self.Nf = len(self.freqs)

    @cached_property
    def w(self):
        if self.seed:
            np.random.seed(self.seed)

        return (np.random.normal(loc = 0, scale = 1, size = (self.Nrea, self.Npulsars, len(self.freqs))) +\
            1j*np.random.normal(loc = 0, scale = 1, size = (self.Nrea, self.Npulsars, len(self.freqs))))

    @cached_property
    def C(self):
        calc = np.zeros((len(self.Amp), self.Nf))
        for ii, (amp, alp) in enumerate(zip(self.Amp, self.alpha)):
            calc[ii,:] = ((1 + self.kappa**2) / (1 + self.kappa**2 * (self.freqs /  f_yr)**(-2/3))) *\
            (self.dur * self.howml  * amp**2/(48 * np.pi**2 * f_yr**(2*alp)) * self.freqs**(2*alp - 3))
        return calc

    @cached_property
    def C_RN(self):
        calc = np.zeros((self.Npulsars, len(self.freqs)))
        for ii, (amp, alp) in enumerate(zip(self.RNamp, self.RNalpha)):
            calc[ii,:] = (self.dur * self.howml  * amp**2/(48 * np.pi**2 * f_yr**(2*alp)) * self.freqs**(2*alp - 3))
        return calc

    @cached_property
    def ang_info(self):
        names = []
        prod = product(self.psrlist,self.psrlist)
        for i in list(prod):
            names.append(list(i))
        N = int(self.Npulsars * self.Npulsars)
        d = []
        for _ in range (N):
            p_a = names[_][0]
            p_b = names[_][1]
            n_a = self.psr_locs[self.psrlist.index(p_a)]
            n_b = self.psr_locs[self.psrlist.index(p_b)]

            # dist = [self.pdist[self.psrlist.index(p_a)],self.pdist[self.psrlist.index(p_b)]]
            # L_a = max(dist)
            # L_b = min(dist)
            d.append([1.,1.,np.arccos(np.dot(n_b,n_a))])
            #d.append([p_a,p_b,np.arccos(np.dot(n_b,n_a))])
        return np.array(d, dtype = object)

    @cached_property
    def RN(self):
        '''
        Generates timing residuals in freqeuncy domain from a TT-type SGWB.

        :param: output of "inj_params" function

        Author: Nima Laal
        '''
        H = np.identity(self.Npulsars)
        Res_f = np.einsum('IJ,rJk,Jk->rIk', H, self.w, np.sqrt(self.C_RN))
        Res_f[:, :,0] = 0
        Res_f[:, :,-1] = 0
        return Res_f

    @cached_property
    def TT(self):
        '''
        Generates timing residuals in freqeuncy domain from a TT-type SGWB.

        Author: Nima Laal
        '''

        ang = self.ang_info[:,2].astype(dtype = float)
        chi = HD_ORF(ang)
        chi[self.auto_indx] = 1
        chi = chi.reshape((self.Npulsars, self.Npulsars))
        H = np.linalg.cholesky(chi)
        Res_f = np.einsum('IJ,rJk,k->rIk', H, self.w, np.sqrt(self.C[0]))
        Res_f[:, :,0] = 0
        Res_f[:, :,-1] = 0
        return Res_f

    @cached_property
    def ST(self):
        '''
        Generates timing residuals in freqeuncy domain from a TT-type SGWB.

        Author: Nima Laal
        '''

        ang = self.ang_info[:,2].astype(dtype = float)
        chi = ST_ORF(ang)
        chi[self.auto_indx] = 1
        chi = chi.reshape((self.Npulsars, self.Npulsars))
        H = np.linalg.cholesky(chi)
        Res_f = np.einsum('IJ,rJk,k->rIk', H, self.w, np.sqrt(self.C[1]))
        Res_f[:, :,0] = 0
        Res_f[:, :,-1] = 0
        return Res_f

    @cached_property
    def TTST(self):
        '''
        Generates timing residuals in freqeuncy domain from a TT + ST-type SGWB.

        Author: Nima Laal
        '''

        ang = self.ang_info[:,2].astype(dtype = float)
        C = self.C

        chi = np.zeros((self.Nf, len(ang)))
        for k in range(self.Nf):
            chi[k, :] = 1/(C[0][k] + C[1][k]) * (C[0][k] * HD_ORF(ang) + C[1][k] * ST_ORF(ang))
            chi[k, self.auto_indx] = 1

        chi = chi.reshape((self.Nf, self.Npulsars, self.Npulsars))
        H = np.linalg.cholesky(chi)
        Res_f = np.einsum('kIJ,rJk,k->rIk', H, self.w, np.sqrt(C[0] + C[1]))
        Res_f[:, :,0] = 0
        Res_f[:, :,-1] = 0
        return Res_f

    @cached_property
    def VL(self):
        '''
        Generates timing residuals in freqeuncy domain from a VL-type SGWB.

        Author: Nima Laal
        '''

        ang = self.ang_info[:,2].astype(dtype = float)
        chi = np.zeros((self.Nf, len(ang)))
        for ii in range(self.Nf):
            chi[ii][:] = VL_ORF(self.ang_info, self.f_norm[ii])
            chi[ii][self.auto_indx] = autovl(self.f_norm[ii], self.pdist)

        H = False; tol = 0
        while(True):
            H = do_chol(chi.reshape((self.Nf, self.Npulsars, self.Npulsars)))
            if np.any(H):
                break
            tol += .01
            for ii in range(self.Nf):
                chi[ii][self.auto_indx] = (1 + tol) * autovl(self.f_norm[ii], self.pdist)
            print('*** Chi matrix is not positive definite. Increasing the auto-terms by {} percent.'.format(round(tol * 100)))

        Res_f = np.einsum('kij,jk->ik', H, self.w*np.sqrt(self.C[2]))
        Res_f[:, :, 0] = 0
        Res_f[:, :,-1] = 0
        return Res_f

    @cached_property
    def TTVL(self):
        '''
        Generates timing residuals in freqeuncy domain from a TT + VL-type SGWB.

        Author: Nima Laal
        '''

        ang = self.ang_info[:,2].astype(dtype = float)
        chi = np.zeros((self.Nf, len(ang)))
        C = self.C

        for k in range(self.Nf):
            chi[k, :] = 1/(C[0][k] + C[1][k]) * (C[0][k] * HD_ORF(ang) + C[1][k] * VL_ORF(self.ang_info, self.f_norm[k]))
            chi[k, self.auto_indx] = 1/2 + autovl(self.f_norm[k], self.pdist)
            
        H = False; tol = 0
        while(True):
            H = do_chol(chi.reshape((self.Nf, self.Npulsars, self.Npulsars)))
            if np.any(H):
                break
            tol += .01
            for ii in range(self.Nf):
                chi[ii][self.auto_indx] = (1 + tol) * autovl(self.f_norm[ii], self.pdist)
            print('*** Chi matrix is not positive definite. Increasing the auto-terms by {} percent.'.format(round(tol * 100)))

        chi = chi.reshape((self.Nf, self.Npulsars, self.Npulsars))
        H = np.linalg.cholesky(chi)
        Res_f = np.einsum('kIJ,rJk,k->rIk', H, self.w, np.sqrt(C[0] + C[1]))
        Res_f[:, :,0] = 0
        Res_f[:, :,-1] = 0
        return Res_f

    def ind_res(self, Res_f):
        '''
        Completes the injection process by turning the residulas in freqeuncy domain to the time domain

        :param: Res_f: residuals in frequency domain

        '''

        Res_f2 = np.zeros((self.Nrea, self.Npulsars, 2*self.Nf-2), complex)
        Res_t = np.zeros((self.Nrea, self.Npulsars, 2*self.Nf-2))
        Res_f2[:, :,0:self.Nf] = Res_f[:, :,0:self.Nf]
        Res_f2[:, :, self.Nf:(2*self.Nf-2)] = np.conj(Res_f[:, :,(self.Nf-2):0:-1])
        Res_t = np.real(np.fft.ifft(Res_f2)/self.dt)
        # shorten data and interpolate onto TOAs
        Res = np.zeros((self.Nrea, self.Npulsars, self.npts))

        res_gw = []
        for rr in range(self.Nrea):
            dummy = []
            for ll in range(self.Npulsars):
                Res[rr, ll,:] = Res_t[rr, ll, 10:(self.npts+10)]
                f = interp.interp1d(self.ut, Res[rr, ll,:], kind='linear')
                dummy.append(f(self.toas[ll]*day))
            res_gw.append(dummy)

        return np.array(res_gw, dtype = object)


    @cached_property
    def create_altpol_res(self):
        '''
        User-friendly way to use all other functions to create residuals in the time domain.

        Author: Nima Laal
        '''

        if self.MG == 'TT':
            res = self.ind_res(self.TT)
        elif self.MG == 'ST':
            res = self.ind_res(self.ST)
        elif self.MG == 'VL':
            res = self.ind_res(self.VL)
        elif self.MG == 'TTST':
            res = self.ind_res(self.TTST)
        elif self.MG == 'TTVL':
            res = self.ind_res(self.TTVL)
        else:
            raise ValueError("option not implemented yet")

        if self.RNamp.any():
            res = res + self.ind_res(self.RN)
        return res

    def add_white_noise(self, wn_sig, toas):
        return np.random.normal(loc = 0, scale = wn_sig, size = len(toas))

    def total_res(self):

        total_res = []
        gw_res = self.create_altpol_res
        for pidx in range(self.Npulsars):
            dummy_res = []
            R = self.Rmat(self.toas[pidx])
            for rr in range(self.Nrea):
                dummy_res.append(R @ gw_res[rr][pidx] + self.add_white_noise(self.wn_sig, self.toas[pidx]))
            total_res.append(dummy_res)

        #return np.transpose(total_res, (1,0))
        return total_res
    
    def Mmat(self, toas):
        '''
        Simple three-column design matrix for quadratic timing model.
        '''

        DesignM = np.ones((len(toas), 3))
        DesignM[:,1] = toas
        DesignM[:,2] = DesignM[:,1]**2
        return DesignM

    def Rmat(self, toas):
        '''
        The Rmatrix used to do the quadratic fitting
        '''
        I = np.identity(len(toas))
        DesignM = self.Mmat(toas)
        return I - np.einsum('kn,nm,pm->kp', DesignM, np.linalg.inv(np.einsum('km,kn->nm',DesignM,DesignM)),DesignM)
