from visibility import null_depth
import plotly.subplots as splt
from pathlib import Path
from math import pi, radians
from enum import Enum, auto
import scipy.constants as sconst
import pandas as pd
import numpy as np
import numpy.linalg as la

# Configuration
data_path = Path('Datasets/PS_2021.05.20_08.13.20.csv')

# Constants
sun_radius = 695508E3  # https://solarsystem.nasa.gov/solar-system/sun/by-the-numbers/
earth_radius = 6371E3  # https://solarsystem.nasa.gov/solar-system/sun/by-the-numbers/
lun_north_ra = radians(269.9949)  # Lunar celestial north right-ascention
lun_north_dec = radians(66.5392)  # Lunar celestial north declination

# Types


class Visibility(Enum):
    '''Defines the possible visibility modifiers.'''

    VISIBLE = auto()
    OUT_OF_FOR = auto()
    OUT_OF_RANGE = auto()
    SNR_TOO_LOW = auto()
    INT_TOO_LONG = auto()


# Basic formulas
def circle_area(r):
    return pi * r**2


def sphere_area(r):
    return 4 * pi * r**2


def R(ra, dec):
    '''Create rotation matrix from ICRS frame to target pointing frame.'''
    def Rz(theta):
        return np.array([[np.cos(theta), np.sin(theta), 0],
                         [-np.sin(theta), np.cos(theta), 0],
                         [0, 0, 1]])

    def Ry(theta):
        return np.array([[np.cos(theta), 0, -np.sin(theta)],
                         [0, 1, 0],
                         [np.sin(theta), 0, np.cos(theta)]])
    return Ry(-dec) @ Rz(ra)


def blackbody_flux(bd_eq_tem):
    '''
    Calculate the black-body emissions in watts.

    bd_eq_temp: Body equilibrium temperature (K)
    '''
    return sconst.sigma*bd_eq_tem**4


def photon_energy(lam):
    '''
    Calculate the photon energy in J/ph.

    lam: The wavelength (m)
    '''
    return sconst.h*sconst.c/lam


def rms_null_variation(phs_var, frac_inten_var=0):
    '''
    Calculate the RMS variation of the null depth (-).

    phs_var: Phase variance (rad**2)
    frac_inten_var: Fractional intensity variations (-)
    '''
    return np.sqrt((phs_var**2 + frac_inten_var**2)/8)

# Main functionality


def load_dataset(path, only_uncontroversial=True, only_confirmed=True):
    '''
    Load a NASA Exoplanet Database dataset and performs basic filtering.

    path: Path to dataset
    only_uncontroversial: Filter out controversial planets
    only_confirmed: Filter out unconfirmed planets
    '''
    df = pd.read_csv(path, comment='#', usecols=[
                     'soltype', 'pl_controv_flag', 'ra', 'dec', 'sy_dist',
                     'pl_name', 'pl_rade', 'pl_eqt', 'st_teff', 'st_rad'])

    df['pl_controv_flag'] = df['pl_controv_flag'] == 1.0
    if only_uncontroversial:
        df = df[~df['pl_controv_flag']]

    if only_confirmed:
        df = df[df['soltype'] == 'Published Confirmed']

    df = df.dropna()

    df[['ra', 'dec']] *= sconst.degree
    df['sy_dist'] *= sconst.parsec
    df['st_rad'] *= sun_radius
    df['pl_rade'] *= earth_radius

    df = df.rename(columns={'pl_rade': 'pl_rad', 'ra': 'sy_ra',
                   'dec': 'sy_dec', 'pl_eqt': 'pl_temp', 'st_teff': 'st_temp'})

    return df


def calculate_southern_zenith_angle(df):
    '''
    Calculate the zenith angle between the lunar south pole and all planets in radians.
    This adds a 'zenith_angle' column to df.


    df: The planet dataset
    '''

    df = df.copy()

    lun_south = -1 * la.inv(R(lun_north_ra, lun_north_dec)
                            ) @ np.array([1, 0, 0]).T

    df['sy_ptvect'] = df[['sy_ra', 'sy_dec']].apply(
        lambda ra_dec: la.inv(R(*ra_dec)) @ np.array([1, 0, 0]).T, axis=1)
    df['sy_sepang'] = df['sy_ptvect'].apply(
        lambda v: np.arccos(np.dot(v, lun_south)))

    return df


def calculate_emissions(df, wavelength=10E-6):
    '''
    Calculate the emission powers of stars and planets in photons per second.

    df: The planet dataset
    '''

    df = df.copy()

    pl_wattage = blackbody_flux(df['pl_temp']) * sphere_area(df['pl_rad'])
    st_wattage = blackbody_flux(df['st_temp']) * sphere_area(df['st_rad'])
    df['pl_phps'] = pl_wattage / photon_energy(wavelength)
    df['st_phps'] = st_wattage / photon_energy(wavelength)

    return df


def calculate_local_fluxes(df):
    '''
    Calculate the local fluxes of stars and planets in photons per second per square meter.

    df: The planet dataset
    '''

    df = df.copy()

    df['pl_phpspm2_loc'] = df['pl_phps'] / sphere_area(df['sy_dist'])
    df['st_phpspm2_loc'] = df['st_phps'] / sphere_area(df['sy_dist'])

    return df


def calculate_nulling(df, wavelength, baseline, phase_variance,
                      fractional_intensity_variance):
    '''
    Calculate the time average null depth for each system.

    df: The planet dataset
    wavelength: Wavelength (m)
    baseline: Nulling baseline (m)
    phase_variance: Phase variance due to OPD (rad**2)
    fractional_intensity_variance: Fractional intensity variance due to OPA (-)
    '''

    df = df.copy()

    st_angular_diameter = 2 * df['st_rad'] / df['sy_dist']
    st_leakage = pi**2 / 4 * (baseline * st_angular_diameter / wavelength)**2
    df['st_nulldepth'] = 1 / 4 * \
        (phase_variance + st_leakage + fractional_intensity_variance)

    return df


def calculate_detections(df, mirror_radius, instrument_throughput,
                         quantum_efficiency, rotationally_modulated=True):
    '''
    Calculate the detections of the optical subsystem in electrons per second.

    df: The planet dataset
    mirror_radius: The mirror radius (m)
    instrument_throughput: The instrument throughput from aperture to detector (-)
    quantum_efficiency: The detector quantum efficiency (e/ph)
    rotationally_modulated: Is the planet signal rotationally modulated for de-correlation?
    '''

    df = df.copy()

    # Calculate incident photons per second on detector
    pl_phps_incident = df['pl_phpspm2_loc'] * \
        circle_area(mirror_radius) * instrument_throughput
    st_phps_incident = df['st_phpspm2_loc'] * \
        circle_area(mirror_radius) * df['st_nulldepth'] * instrument_throughput

    if rotationally_modulated:
        pl_phps_incident *= 0.5

    df['pl_eps'] = pl_phps_incident * quantum_efficiency
    df['st_eps'] = st_phps_incident * quantum_efficiency

    return df


def calculate_shot_noise_snr(df, integration_time, resolution, phase_variance):
    '''
    Calculate the shot noise after a given integration time.
    TODO: Does not account for thermal background.

    df: The planet dataset
    integration_time: The allowable integration time (s)
    resolution: The spectral resolution (-)
    phase_variance: Phase variance due to OPD (rad**2)
    '''

    df = df.copy()

    instrumental_noise = df['st_eps'] * rms_null_variation(phase_variance)
    shot_noise = np.sqrt(resolution * (df['pl_eps'] + df['st_eps']))
    df['shot_snr_for_time'] = np.sqrt(
        integration_time) * df['pl_eps'] / np.sqrt(shot_noise**2 + instrumental_noise**2)

    return df


def calculate_shot_noise_time(df, target_snr, resolution, phase_variance):
    '''
    Calculate the exposure time for a given SNR
    TODO: Does not account for thermal background.

    df: The planet dataset
    target_snr: The target SNR (-)
    resolution: The spectral resolution (-)
    phase_variance: Phase variance due to OPD (rad**2)
    '''

    df = df.copy()

    instrumental_noise = df['st_eps'] * rms_null_variation(phase_variance)
    shot_noise = np.sqrt(resolution * (df['pl_eps'] + df['st_eps']))
    df['shot_time_for_snr'] = (target_snr / df['pl_eps'] *
                               np.sqrt(shot_noise**2 + instrumental_noise**2))**2

    return df


def determine_visibility(df, pointing_range, max_integration_time,
                         minimum_instantaneous_snr):
    '''
    Assign a visibility tag to each planet.

    df: The planet dataset
    pointing_range: The off zenith pointing angle (rad)
    max_integration_time: The maximum allowable integration time (s)
    minimum_instantaneous_snr: The minimum planet-star contrast (-)
    '''
    df = df.copy()

    # Create an indexing series that is all True
    # TODO: Find a more robust way to do this
    visible_selector = df['pl_name'] != ''

    out_of_for_selector = df['sy_sepang'] > pointing_range
    visible_selector &= ~out_of_for_selector

    low_snr_selector = ((df['pl_eps'] / df['st_eps']) < minimum_instantaneous_snr) & visible_selector
    visible_selector &= ~low_snr_selector

    excessive_integration_selector = (df['shot_time_for_snr'] > max_integration_time) & visible_selector
    visible_selector &= ~excessive_integration_selector

    df.loc[out_of_for_selector, 'visibility'] = Visibility.OUT_OF_FOR
    df.loc[low_snr_selector, 'visibility'] = Visibility.SNR_TOO_LOW
    df.loc[excessive_integration_selector,
           'visibility'] = Visibility.INT_TOO_LONG
    df.loc[visible_selector, 'visibility'] = Visibility.VISIBLE

    return df


if __name__ == '__main__':
    df = load_dataset(data_path)
    df = calculate_southern_zenith_angle(df)
    df = calculate_emissions(df)
    df = calculate_local_fluxes(df)
    df = calculate_nulling(df, 10E-6, 100, 0, 0)
    df = calculate_detections(df, 1, 1, 1)
    df = calculate_shot_noise_time(df, 5, 300, 1.5E-9/10E-6)
    df = determine_visibility(df, radians(40), 60*60, 5)
    print(df['visibility'].value_counts())

# TODO: No planets are not visible due to excessive integration time. This does not match the previous model.
# TODO: Port the unit test harness and achieve full coverage
# TODO: Add all necessary plots and possible Monte-Carlos