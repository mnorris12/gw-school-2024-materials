"""
An example script to run the PTA CURN MCMC analysis on simulated data
"""

import os, pickle
import numpy as np
from enterprise.signals import gp_priors, parameter, gp_signals, signal_base, white_signals
from enterprise_extensions.model_orfs import hd_orf
from enterprise_extensions.model_utils import get_tspan
from enterprise_extensions.sampler import setup_sampler
from enterprise_extensions.sampler import (get_parameter_groups,
                                           get_psr_groups)
import la_forge.core as co

# open simulated PTA data with pickle
data_loc = '../Tutorials/data/sim_ng_psrs.pkl'

with open(data_loc, 'rb') as f:
    psrs = pickle.load(f)

Tspan = get_tspan(psrs)  # compute timespan of data

# setting up our model
# timing model
tm = gp_signals.MarginalizingTimingModel()

# white noise model
efac = parameter.Constant(1.0)
wn = white_signals.MeasurementNoise(efac=efac)

# intrinsic red noise
log10_A_rn = parameter.Uniform(-18, -12)
gamma_rn = parameter.Uniform(0, 7)
rn_psd = gp_priors.powerlaw(log10_A=log10_A_rn, gamma=gamma_rn)
rn = gp_signals.FourierBasisGP(rn_psd, Tspan=Tspan, components=30)

# gwb
log10_A_gw = parameter.Uniform(-18, -12)('log10_A_gw')
#gamma_gw = parameter.Uniform(0, 7)('gamma_gw')
gamma_gw = parameter.Constant(13/3)('gamma_gw')
gw_psd = gp_priors.powerlaw(log10_A=log10_A_gw, gamma=gamma_gw)
hd = gp_signals.FourierBasisCommonGP(gw_psd, hd_orf(), Tspan=Tspan,
                                     components=10, name='gw_hd')  # HD

# full model
model_curn = tm + wn + rn + hd

# set up PTA
pta_curn = signal_base.PTA([model_curn(p) for p in psrs])

# we add a list of lists of parameters that we want to group together
groups = get_parameter_groups(pta_curn)
groups.extend(get_psr_groups(pta_curn))

outdir = './results/30fPLirn_10fPLhd_gamma4p33_20yrsim/'

# do you have any empirical distributions? Load them and include them!

better_sampler = setup_sampler(pta_curn, outdir=outdir, groups=groups,
                               empirical_distr=None)

x0 = np.hstack([p.sample() for p in pta_curn.params])
better_sampler.sample(x0, 5e6)

# save the chain as a la_forge core object
c0 = co.Core(outdir)
c0.save(outdir + '30fPLirn_10fPLcrn_gamma4p33_20yrsim.core')
