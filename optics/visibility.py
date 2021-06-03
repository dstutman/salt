# Standard lib
import plotly.graph_objects as go
import plotly.subplots as splt
from os import name
from pathlib import Path
from math import pi, radians
from enum import Enum, auto

# External lib
import scipy.constants as cns
import pandas as pd
import numpy as np
import numpy.linalg as la

# Configuration
ds_path = Path("optics/Datasets/PS_2021.05.20_08.13.20.csv")
prefilter = True
pt_auth = 50  # Degrees
show_figs = True
wavelength = 20E-6
baseline = 100
required_snr = 5
instrumental_throughput = 0.10
max_exposure = 60*60*30
phase_variance = 2*pi * 1.5E-9/10E-6 # TODO: This is path length error maximum, set to 3sigma for variance
mirror_radius = 1
n_mirrors = 4

# Configuration checks
if pt_auth > 90:
    print(f'Clamping pointing authority (was ${pt_auth}')
    pt_auth = 90

# Basic formulas
'''
Calculates the time average null depth.
    st_ang_dia: The apparent angular diameter of the star (rad)
    lam: Wavelength (m)
    baseline: Baseline length (m)
    phs_var: Phase variance (rad**2)
    frac_inten_var: Fractional intensity variations (-)
'''


def null_depth(st_ang_dia, lam=wavelength, baseline=baseline, phs_var=phase_variance, frac_inten_var=0):
    st_leak = (pi**2)/4*(baseline*st_ang_dia/lam)**2
    return 1/4*(phs_var + st_leak + frac_inten_var)


'''
Calculates the black-body emissions in watts.
    bd_eq_temp: Body equilibrium temperature (K)
'''


def blackbody_flux(bd_eq_tem):
    return cns.sigma*bd_eq_tem**4


'''
Calculates the shot noise in x/sqrt(s).
    pl_flux: Planetary flux (x/s)
    st_flux: Stellar flux (x/s)
    bg_flux: Background flux (x/s)
'''


def shot_noise(pl_flux, st_flux, bg_flux=0):
    return np.sqrt(pl_flux + bg_flux + st_flux)


'''
Calculates the photon energy in J/ph.
    lam: The wavelength (m)
'''


def photon_energy(lam):
    return cns.h*cns.c/lam


'''
Calculates the RMS variation of the null depth (-).
    phs_var: Phase variance (rad**2)
    frac_inten_var: Fractional intensity variations (-)
'''


def rms_null_variation(phs_var, frac_inten_var=0):
    return np.sqrt((phs_var**2 + frac_inten_var**2)/8)


# Load the dataset
ds = pd.read_csv(ds_path, comment='#', usecols=[
                 'soltype', 'pl_controv_flag', 'ra', 'dec', 'sy_dist', 'pl_name', 'pl_rade', 'pl_eqt', 'st_teff', 'st_rad'])
if prefilter:
    ds = ds.dropna()
    ds = ds[ds['soltype'] == 'Published Confirmed']
    ds = ds[ds['pl_controv_flag'] == False]

# Convert dataset units
ds[['ra', 'dec']] *= radians(1)
ds['sy_dist'] *= cns.parsec
ds['st_rad'] *= 696342E3  # Solar radii to m
ds['pl_rade'] *= 6371E3  # Earth radii to m

# Set up the visibility column


class Visibility(Enum):
    VISIBLE = auto()
    OUT_OF_FOR = auto()
    OUT_OF_RANGE = auto()
    SNR_TOO_LOW = auto()


ds['visibility'] = Visibility.VISIBLE

# Determine geometric planet visibility
'''
Defines the rotation from the ICRS frame
to the target pointing frame
'''


def R(ra, dec):
    def Rz(theta):
        return np.array([[np.cos(theta), np.sin(theta), 0],
                         [-np.sin(theta), np.cos(theta), 0],
                         [0, 0, 1]])

    def Ry(theta):
        return np.array([[np.cos(theta), 0, -np.sin(theta)],
                         [0, 1, 0],
                         [np.sin(theta), 0, np.cos(theta)]])

    return Ry(-dec) @ Rz(ra)


ra0 = radians(269.9949)  # Moon celestial north right ascention
dec0 = radians(66.5392)  # Moon celestial north declination
mn_spole = -la.inv(R(ra0, dec0)) @ np.array([1, 0, 0]).T

ds['pt_vect'] = ds[['ra', 'dec']].apply(
    lambda ra_dec: la.inv(R(*ra_dec)) @ np.array([1, 0, 0]).T, axis=1)
ds['sep_ang'] = ds['pt_vect'].apply(lambda v: np.arccos(np.dot(v, mn_spole)))
ds.loc[ds['sep_ang'] > radians(pt_auth), 'visibility'] = Visibility.OUT_OF_FOR

# Fluxes at source
pl_sfca = 4*pi*ds['pl_rade']**2
ds['pl_watt_flux'] = blackbody_flux(ds['pl_eqt'])
ds['pl_watt'] = ds['pl_watt_flux']*pl_sfca
ds['pl_phps'] = ds['pl_watt']/photon_energy(wavelength)

st_sfca = 4*pi*ds['st_rad']**2
ds['st_watt_flux'] = blackbody_flux(ds['st_teff'])
ds['st_watt'] = ds['st_watt_flux']*st_sfca
ds['st_phps'] = ds['st_watt']/photon_energy(wavelength)

# Measured fluxes
# TODO: Aperture effective area
sy_spha = (4*pi*ds['sy_dist']**2)
ap_area = n_mirrors*pi*mirror_radius**2
ds['pl_meas_phps'] = ds['pl_phps']*(1/2)*instrumental_throughput/sy_spha*ap_area

ds['st_angdia'] = 2*ds['st_rad']/ds['sy_dist']
ds['st_meas_phps'] = ds['st_phps']*null_depth(ds['st_angdia'])/sy_spha*ap_area

# SNR
ds['ins_noise'] = ds['st_meas_phps']*rms_null_variation(phase_variance)
ds['sh_noise'] = shot_noise(ds['pl_meas_phps'], ds['st_meas_phps'])
ds['t_int'] = (required_snr*np.sqrt(ds['sh_noise']**2 + ds['ins_noise']**2)/ds['pl_meas_phps']*300)**2
ds.loc[(ds['visibility'] == Visibility.VISIBLE) & (
    ds['t_int'] > max_exposure), 'visibility'] = Visibility.SNR_TOO_LOW

# Plotting
fig = splt.make_subplots(rows=2, cols=2, specs=[[{'type': 'scene'}, {'type': 'xy'}],
                                                [{'type': 'xy'}, {'type': 'xy'}]])
fig.update_layout(scene_aspectmode='data')

# ICRS origin
fig.add_scatter3d(x=[0], y=[0], z=[0], name='ICRS Origin',
                  mode='markers', row=1, col=1)

# Earth North Pole
fig.add_scatter3d(x=[0], y=[0], z=[1],
                  name='Earth North Pole', mode='markers', row=1, col=1)

# Moon south pole
fig.add_scatter3d(x=[mn_spole[0]], y=[mn_spole[1]], z=[
                  mn_spole[2]], name='Moon South Pole', mode='markers', row=1, col=1)

# Objects out of Field of Regard
xyz_oof = np.stack(
    ds.loc[ds['visibility'] == Visibility.OUT_OF_FOR, 'pt_vect']).T
fig.add_scatter3d(x=xyz_oof[0], y=xyz_oof[1], z=xyz_oof[2],
                  name='Out of FOR', mode='markers', opacity=0.1, row=1, col=1)

# Objects with excessive integration times
xyz_int = np.stack(
    ds.loc[ds['visibility'] == Visibility.SNR_TOO_LOW, 'pt_vect']).T
fig.add_scatter3d(x=xyz_int[0], y=xyz_int[1], z=xyz_int[2],
                  name='Integration Too Long', mode='markers', row=1, col=1)

# Visible objects
xyz_vis = np.stack(ds.loc[ds['visibility'] == Visibility.VISIBLE, 'pt_vect']).T
fig.add_scatter3d(x=xyz_vis[0], y=xyz_vis[1], z=xyz_vis[2],
                  name='Visible', mode='markers', row=1, col=1)

# Integration times vs distance
fig.update_yaxes(type='log', row=1, col=2)
dist_int = ds.loc[(ds['visibility'] == Visibility.VISIBLE) | (
    ds['visibility'] == Visibility.SNR_TOO_LOW), ['sy_dist', 't_int']]
fig.add_scatter(x=dist_int['sy_dist'],
                y=dist_int['t_int'], mode='markers', row=1, col=2)

if show_figs:
    fig.show()