"""
An example script to run the PTA CURN MCMC analysis on simulated data
"""

import os, pickle
import numpy as np
from enterprise.signals import gp_priors, parameter, gp_signals, signal_base, white_signals
from enterprise_extensions.models import model_general
from enterprise_extensions.model_utils import get_tspan
from enterprise_extensions.sampler import setup_sampler
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
gamma_gw = parameter.Uniform(0, 7)('gamma_gw')
gw_psd = gp_priors.powerlaw(log10_A=log10_A_gw, gamma=gamma_gw)
curn = gp_signals.FourierBasisGP(gw_psd, Tspan=Tspan,
                                 components=10, name='gw')  # CURN

# full model
model_curn = tm + wn + rn + curn

# set up PTA
pta_curn = signal_base.PTA([model_curn(p) for p in psrs])

# we add a list of lists of parameters that we want to group together
groups = [[ii for ii in range(len(pta_curn.params))]]
[groups.append([2*ii, 2*ii+1]) for ii in range(len(psrs))]
[groups.append([-2, -1]) for ii in range(10)]

outdir = './results/my_first_pta_ptmcmc/'

better_sampler = setup_sampler(pta_curn, outdir=outdir, groups=groups)

x0 = np.hstack([p.sample() for p in pta_curn.params])
better_sampler.sample(x0, 5e6)

# save the chain as a la_forge core object
c0 = co.Core(outdir)
c0.save(outdir + '30fPLirn_10fPLcrn_20yrsim.core')
