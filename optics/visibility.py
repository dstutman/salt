# Standard lib
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
pt_auth = 50 # Degrees
show_figs = False

# Configuration checks
if pt_auth > 90:
    print(f'Clamping pointing authority (was ${pt_auth}')
    pt_auth = 90


# Basic formulas
'''
Calculates the time average null depth.
    st_ang_dia: The apparent angular diameter of the star (rad)
    lam: The wavelength (m)
    baseline: The baseline length (m)
    phs_var: The phase variance (rad**2)
    frac_inten_var: The fractional intensity variations (-)
'''
def null_depth(st_ang_dia, lam=10E-6, baseline=100, phs_var=0, frac_inten_var=0):
    st_leak = (pi**2)/4 * (baseline*st_ang_dia/lam) ** 2 
    return phs_var + st_leak + frac_inten_var

'''
Calculates the black-body emissions in watts.
    bd_eq_temp: Body equilibrium temperature (K)
    bd_rad: Body radius (m)
'''
def bbd_wattage(bd_eq_tem, bd_rad):
    bd_sfcarea = 4*pi * bd_rad**2
    return cns.k * bd_eq_tem**4 * bd_sfcarea

'''
Calculates the spatial dissipation from sphere with
radius r1 to sphere with radius r2.
    r1: First radius (*)
    r2: Second radius (*)
'''
def spatial_dissipation(r1, r2):
    return r1**2 / r2**2

# Load the dataset
ds = pd.read_csv(ds_path, comment='#', usecols=['soltype', 'pl_controv_flag', 'ra', 'dec', 'sy_dist', 'pl_name', 'pl_rade', 'pl_eqt', 'st_teff', 'st_rad'])
if prefilter:
    ds = ds.dropna()
    ds = ds[ds['soltype'] == 'Published Confirmed']
    ds = ds[ds['pl_controv_flag'] == False]

# Convert dataset units
ds[['ra', 'dec']] *= radians(1)
ds['sy_dist'] *= cns.parsec
ds['st_rad'] *= 696342E3 # Solar radii to m
ds['pl_rade'] *= 6371E3 # Earth radii to m

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

ra0 = radians(269.9949) # Moon celestial north right ascention
dec0 = radians(66.5392) # Moon celestial north declination
mn_spole = -la.inv(R(ra0, dec0)) @ np.array([1, 0, 0]).T

ds['pt_vect'] = ds[['ra', 'dec']].apply(lambda ra_dec: la.inv(R(*ra_dec)) @ np.array([1, 0, 0]).T, axis=1)
ds['sep_ang'] = ds['pt_vect'].apply(lambda v: np.arccos(np.dot(v, mn_spole)))
ds.loc[ds['sep_ang'] > radians(pt_auth), 'visibility'] = Visibility.OUT_OF_FOR


# Signal and noise calculations
ds['pl_watt'] = ds[['pl_eqt', 'pl_rade']].apply(lambda eqt_rad: bbd_wattage(*eqt_rad), axis=1)

# Plotting
import plotly.graph_objects as go

fig = go.Figure()
fig.update_layout(scene_aspectmode='data')

# ICRS origin
fig.add_scatter3d(x=[0], y=[0], z=[0], name='ICRS Origin', mode='markers')

# Earth North Pole
fig.add_scatter3d(x=[0], y=[0], z=[1], name='Earth North Pole', mode='markers')

# Moon south pole
fig.add_scatter3d(x=[mn_spole[0]], y=[mn_spole[1]], z=[mn_spole[2]], name='Moon South Pole', mode='markers')

# Objects out of Field of Regard
xyz_oof = np.stack(ds.loc[ds['visibility'] == Visibility.OUT_OF_FOR, 'pt_vect']).T
fig.add_scatter3d(x=xyz_oof[0], y=xyz_oof[1], z=xyz_oof[2], name='Out of FOR', mode='markers', opacity=0.1)

# Visible objects
xyz_vis = np.stack(ds.loc[ds['visibility'] == Visibility.VISIBLE, 'pt_vect']).T
fig.add_scatter3d(x=xyz_vis[0], y=xyz_vis[1], z=xyz_vis[2], name='Visible', mode='markers')

if show_figs:
    fig.show()