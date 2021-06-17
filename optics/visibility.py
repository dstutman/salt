import plotly.subplots as sbplt
import plotly.graph_objects as gpho
from pathlib import Path
from math import pi, radians, exp
from enum import Enum, auto
import scipy.constants as sconst
import pandas as pd
import numpy as np
import numpy.linalg as la
from scipy.constants import constants

# Configuration
data_path = Path('Datasets/PS_2021.06.08_01.44.53.csv')

# Constants
sun_radius = 695508E3  # https://solarsystem.nasa.gov/solar-system/sun/by-the-numbers/
earth_radius = 6371E3  # https://solarsystem.nasa.gov/solar-system/sun/by-the-numbers/
lun_north_ra = radians(269.9949)  # Lunar celestial north right-ascention
lun_north_dec = radians(66.5392)  # Lunar celestial north declination
num_mirrors = 4

# Types


class Visibility(Enum):
    '''Defines the possible visibility modifiers.'''
    HIDDEN = auto()
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


def south_pole():
    '''
    Calculate the lunar celestial south pole.
    '''
    return -1 * la.inv(R(lun_north_ra, lun_north_dec)
                       ) @ np.array([1, 0, 0]).T


def blackbody_flux(bd_eq_tem):
    '''
    Calculate the black-body emissions in watts.

    bd_eq_temp: Body equilibrium temperature (K)
    '''
    return sconst.sigma*bd_eq_tem**4


def photon_energy(wavelength):
    '''
    Calculate the photon energy in J/ph.

    lam: The wavelength (m)
    '''
    return sconst.h*sconst.c/wavelength


def rms_null_variation(phs_var, frac_inten_var=0):
    '''
    Calculate the RMS variation of the null depth (-).

    phs_var: Phase variance (rad**2)
    frac_inten_var: Fractional intensity variations (-)
    '''
    return np.sqrt((phs_var**2 + frac_inten_var**2)/8)


def plancks_law(wavelength, T):
    '''
    Calculate the spectral intensity at a given wavelength for a given temperature (W/m**2/m).

    wavelength: Wavelength (m)
    T: Black-body temperature (K)
    '''
    return 2 * constants.h * constants.c**2 / wavelength**5 * 1 / (np.exp(constants.h * constants.c / constants.k / wavelength / T) - 1)  # http://rossby.msrc.sunysb.edu/~marat/MAR542/ATM542-Chapter2.pdf


# def lowest_intensity_wavelength(T, lower=6E-6, upper=20E-6):
#    '''
#    Calculate the lowest spectral intensity wavelength in a band (m).
#
#    This is guaraunteed to give the global minimum within the band
#    because Wein's law is monotonically non-increasing when viewed
#    from the peak to each spectral bound.
#    T: Black-body temperature (K)
#    lower: The lower bound of the band (m)
#    upper: The upper bound of the band (m)
#    '''
#    low_intensities = plancks_law(lower, T)
#    up_intensities =  plancks_law(upper, T)
#    selector = low_intensities < up_intensities
#    ret = np.zeros(len(selector))
#    ret[selector] = lower
#    ret[~selector] = upper
#    #if plancks_law(lower, T) < plancks_law(upper, T):
#    #    return lower
#    #else:
#    #    return upper


def weins_law(T):
    '''
    Calculate the wavelength emission peak (m).

    T: Black-body temperature (K)
    '''
    return 2897 / T * 1E-6  # http://rossby.msrc.sunysb.edu/~marat/MAR542/ATM542-Chapter2.pdf


# Main functionality


def load_dataset(path, only_uncontroversial=True, only_confirmed=True, only_unary=True, only_solitary=True):
    '''
    Load a NASA Exoplanet Database dataset and performs basic filtering.

    path: Path to dataset
    only_uncontroversial: Filter out controversial planets
    only_confirmed: Filter out unconfirmed planets
    '''
    df = pd.read_csv(path, comment='#', usecols=[
                     'soltype', 'pl_controv_flag', 'sy_snum', 'sy_pnum', 'ra', 'dec', 'sy_dist',
                     'pl_name', 'pl_rade', 'pl_eqt', 'st_rad', 'st_teff'])
    df = df.dropna()

    df['pl_controv_flag'] = df['pl_controv_flag'] == 1.0
    df[['ra', 'dec']] *= sconst.degree
    df['sy_dist'] *= sconst.parsec
    df['st_rad'] *= sun_radius
    df['pl_rade'] *= earth_radius

    if only_uncontroversial:
        df = df[~df['pl_controv_flag']]

    if only_confirmed:
        df = df[df['soltype'] == 'Published Confirmed']

    if only_unary:
        df = df[df['sy_snum'] == 1]

    if only_solitary:
        df = df[df['sy_pnum'] == 1]

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

    lun_south = south_pole()

    df['sy_ptvect'] = df[['sy_ra', 'sy_dec']].apply(
        lambda ra_dec: la.inv(R(*ra_dec)) @ np.array([1, 0, 0]).T, axis=1)
    df['sy_sepang'] = df['sy_ptvect'].apply(
        lambda v: np.arccos(np.dot(v, lun_south)))

    return df


def calculate_peak_wavelength(df):
    """
    Calculate the highest intensity wavelength in a band (m).

    df: The planet dataset
    """
    df = df.copy()

    df['peak_wavelength'] = weins_law(df['pl_temp'])

    return df


def calculate_lowest_intensity_wavelength(df, lowest=6E-6, highest=20E-6, resolution=300):
    """
    Calculate the lowest intensity wavelength in a band (m).

    df: The planet dataset
    lowest: The lowest wavelength in the band (m)
    highest: The highest wavelength in the band (m)
    resolution: The spectral resolution (-)
    """
    df = df.copy()

    lower_intensities = plancks_law(lowest, df['pl_temp'])
    upper_intensities = plancks_law(highest, df['pl_temp'])
    selector = lower_intensities < upper_intensities

    df.loc[selector, 'worst_wavelength'] = lowest
    df.loc[~selector, 'worst_wavelength'] = highest
    df['spectral_oneband_width'] = (highest - lowest)/resolution
    return df


def calculate_emissions(df):
    '''
    Calculate the emission powers of stars and planets in photons per second.
    Uses the worst case intensity band.

    df: The planet dataset
    '''

    df = df.copy()

    pl_wattage = sphere_area(df['pl_rad']) * \
        plancks_law(df['worst_wavelength'], df['pl_temp']) * \
        df['spectral_oneband_width']
    st_wattage = sphere_area(df['st_rad']) * \
        plancks_law(df['worst_wavelength'], df['st_temp']) * \
        df['spectral_oneband_width']

    # pl_wattage = blackbody_flux(df['pl_temp']) * sphere_area(df['pl_rad'])
    # st_wattage = blackbody_flux(df['st_temp']) * sphere_area(df['st_rad'])
    df['pl_phps'] = pl_wattage / photon_energy(df['worst_wavelength'])
    df['st_phps'] = st_wattage / photon_energy(df['worst_wavelength'])

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


def calculate_nulling(df, baseline, phase_variance,
                      fractional_intensity_variance):
    '''
    Calculate the time average null depth for each system.

    df: The planet dataset
    baseline: Nulling baseline (m)
    phase_variance: Phase variance due to OPD (rad**2)
    fractional_intensity_variance: Fractional intensity variance due to OPA (-)
    '''

    df = df.copy()

    st_angular_diameter = 2 * df['st_rad'] / df['sy_dist']
    st_leakage = pi**2 / 4 * \
        (baseline * st_angular_diameter / df['worst_wavelength'])**2
    df['st_nulldepth'] = 1 / 4 * \
        (phase_variance + st_leakage + fractional_intensity_variance)

    return df


def force_nulling(df, nulling):
    '''
    Force the nulling to the specified value.

    df: The planet dataset
    nulling: The desired nulling (-)
    '''
    df = df.copy()

    df['st_nulldepth'] = nulling

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
        circle_area(mirror_radius) * instrument_throughput * num_mirrors
    st_phps_incident = df['st_phpspm2_loc'] * \
        circle_area(mirror_radius) * \
        df['st_nulldepth'] * instrument_throughput * num_mirrors

    if rotationally_modulated:
        pl_phps_incident *= 0.5

    df['pl_eps'] = pl_phps_incident * quantum_efficiency
    df['st_eps'] = st_phps_incident * quantum_efficiency

    return df


def calculate_shot_noise_snr(df, integration_time, phase_variance):
    '''
    Calculate the shot noise after a given integration time.
    TODO: Does not account for thermal background.

    df: The planet dataset
    integration_time: The allowable integration time (s)
    phase_variance: Phase variance due to OPD (rad**2)
    '''

    df = df.copy()

    instrumental_noise = df['st_eps'] * rms_null_variation(phase_variance)
    shot_noise = np.sqrt(df['pl_eps'] + df['st_eps'])
    df['shot_snr_for_time'] = np.sqrt(
        integration_time) * df['pl_eps'] / np.sqrt(shot_noise**2 + instrumental_noise**2)

    return df


def calculate_shot_noise_time(df, target_snr, phase_variance):
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
    shot_noise = np.sqrt(df['pl_eps'] + df['st_eps'])
    df['shot_time_for_snr'] = (target_snr / df['pl_eps'] *
                               np.sqrt(shot_noise**2 + instrumental_noise**2))**2

    return df


def determine_visibility(df, pointing_range, plotting_range, max_integration_time,
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

    out_of_plot_selector = df['sy_sepang'] > plotting_range
    visible_selector &= ~out_of_for_selector

    low_snr_selector = ((df['pl_eps'] / df['st_eps']) <
                        minimum_instantaneous_snr) & visible_selector
    visible_selector &= ~low_snr_selector

    excessive_integration_selector = (
        df['shot_time_for_snr'] > max_integration_time) & visible_selector
    visible_selector &= ~excessive_integration_selector

    df.loc[out_of_for_selector, 'visibility'] = Visibility.OUT_OF_FOR
    df.loc[low_snr_selector, 'visibility'] = Visibility.SNR_TOO_LOW
    df.loc[out_of_plot_selector, 'visibility'] = Visibility.HIDDEN
    df.loc[excessive_integration_selector,
           'visibility'] = Visibility.INT_TOO_LONG
    df.loc[visible_selector, 'visibility'] = Visibility.VISIBLE

    return df


def plot_visibility(df):  # pragma: no cover
    fig = gpho.Figure()
    fig.update_layout(template='simple_white', title=dict(text='Celestial Visibility', xanchor='center', x=0.5), scene=dict(aspectmode='data', xaxis=dict(
        showticklabels=False), yaxis=dict(showticklabels=False), zaxis=dict(showticklabels=False)), scene_camera=dict(eye=dict(x=-0.25, y=-0.25, z=2)))

    # ICRS origin
    fig.add_scatter3d(x=[0], y=[0], z=[0], name='ICRS Origin', mode='markers')

    # Earth North Pole
    fig.add_scatter3d(x=[0], y=[0], z=[1],
                      name='Earth North Pole', mode='markers')

    # Moon south pole
    lun_south = south_pole()
    fig.add_scatter3d(x=[lun_south[0]], y=[lun_south[1]], z=[
                      lun_south[2]], name='Moon South Pole', mode='markers')

    # Objects out of Field of Regard
    xyz_oof = np.stack(
        df.loc[df['visibility'] == Visibility.OUT_OF_FOR, 'sy_ptvect']).T
    fig.add_scatter3d(x=xyz_oof[0], y=xyz_oof[1], z=xyz_oof[2],
                      name='Out of FOR', mode='markers', opacity=1)

    # SNR Too Low
    snr_low = np.stack(
        df.loc[df['visibility'] == Visibility.SNR_TOO_LOW, 'sy_ptvect']).T
    fig.add_scatter3d(x=snr_low[0], y=snr_low[1], z=snr_low[2],
                      name='SNR Too Low', mode='markers', opacity=1)

    # Integration too long
    xyz_lng = np.stack(
        df.loc[df['visibility'] == Visibility.INT_TOO_LONG, 'sy_ptvect']).T
    fig.add_scatter3d(x=xyz_lng[0], y=xyz_lng[1], z=xyz_lng[2],
                      name='Integration Too Long', mode='markers', opacity=1)

    # Visible objects
    xyz_vis = np.stack(
        df.loc[df['visibility'] == Visibility.VISIBLE, 'sy_ptvect']).T
    fig.add_scatter3d(x=xyz_vis[0], y=xyz_vis[1],
                      z=xyz_vis[2], name='Visible', mode='markers')

    return fig


def plot_visibility_flat(df):  # pragma: no cover
    fig = gpho.Figure()
    fig.update_layout(template='simple_white', title=dict(text='Celestial Visibility', xanchor='center',
                      x=0.5), xaxis_title='Right ascention (rad)', yaxis_title='Declination (rad)')

    # ICRS origin
    #fig.add_scatter(x=[0], y=[0], z=[0], name='ICRS Origin', mode='markers')

    # Earth North Pole
    # fig.add_scatter3d(x=[0], y=[0], z=[1],
    #                  name='Earth North Pole', mode='markers')

    # Moon south pole
    #fig.add_scatter(x=[lun_north_ra-pi], y=[lun_north_dec-pi],
    #                name='Moon South Pole', mode='markers')

    # Objects out of Field of Regard
    xyz_oof = df.loc[(df['visibility'] == Visibility.OUT_OF_FOR) | (
        df['visibility'] == Visibility.HIDDEN), ['sy_ra', 'sy_dec']]
    fig.add_scatter(x=xyz_oof['sy_ra'], y=xyz_oof['sy_dec'],
                    name='Out of FOR', mode='markers', opacity=1)

    # Low SNR objects
    # xyz_low = np.stack(
    #    df.loc[df['visibility'] == Visibility.SNR_TOO_LOW, 'sy_ptvect']).T
    # fig.add_scatter3d(x=xyz_low[0], y=xyz_low[1], z=xyz_low[2],
    #                  name='SNR Too Low', mode='markers', opacity=0.1)

    # Integration too long
    xyz_lng = df.loc[df['visibility'] ==
                     Visibility.INT_TOO_LONG, ['sy_ra', 'sy_dec']]
    fig.add_scatter(x=xyz_lng['sy_ra'], y=xyz_lng['sy_dec'],
                    name='Integration Too Long', mode='markers', opacity=1)

    # SNR too low
    snr_low = df.loc[df['visibility'] ==
                     Visibility.SNR_TOO_LOW, ['sy_ra', 'sy_dec']]
    fig.add_scatter(x=snr_low['sy_ra'], y=snr_low['sy_dec'],
                    name='SNR Too Low', mode='markers', opacity=1)

    # Visible objects
    xyz_vis = df.loc[df['visibility'] ==
                     Visibility.VISIBLE, ['sy_ra', 'sy_dec']]
    fig.add_scatter(x=xyz_vis['sy_ra'], y=xyz_vis['sy_dec'],
                    name='Visible', mode='markers')

    return fig


def plot_integration_visibility(df):  # pragma: no cover
    int_too_long = df.loc[(df['visibility'] == Visibility.VISIBLE) | (
        df['visibility'] == Visibility.INT_TOO_LONG), ['sy_dist', 'shot_time_for_snr']]

    fig = gpho.Figure()
    fig.update_layout(template='simple_white', title=dict(text='Integration Time vs Distance',
                      xanchor='center', x=0.5), xaxis_title='Distance (pc)', yaxis_title='Integration Time (s)')

    fig.update_yaxes(type='log')
    fig.add_scatter(x=int_too_long['sy_dist'],
                    y=int_too_long['shot_time_for_snr'], mode='markers')
    return fig


def plot_detections_diameter(df, from_dia=0.25, to_dia=6, num_samples=100):  # pragma: no cover
    dias = np.linspace(from_dia, to_dia, num_samples)
    dets = np.zeros(num_samples)

    for idx, dia in enumerate(dias):
        df = df.copy()

        df = calculate_detections(df, dia/2, 0.5, 0.2)
        df = calculate_shot_noise_time(df, 10, 2*pi*1.5E-9/10E-6)
        df = determine_visibility(df, radians(60), radians(90), 60*60*10, 10)

        dets[idx] = sum(df['visibility'] == Visibility.VISIBLE)

    fig = gpho.Figure()
    fig.update_layout(template='simple_white', title=dict(text='Visible Planets vs Mirror Diameter',
                      xanchor='center', x=0.5), xaxis_title='Mirror Diameter (m)', yaxis_title='Visible Planets (-)')
    fig.add_scatter(x=dias, y=dets, mode='markers')
    return fig


def plot_detections_area(df, from_dia=0.25, to_dia=6, num_samples=100):  # pragma: no cover
    dias = np.linspace(from_dia, to_dia, num_samples)
    dets = np.zeros(num_samples)

    for idx, dia in enumerate(dias):
        df = df.copy()

        df = calculate_detections(df, dia/2, 0.5, 0.2)
        df = calculate_shot_noise_time(df, 10, 2*pi*1.5E-9/10E-6)
        df = determine_visibility(df, radians(60), radians(90), 60*60*10, 10)

        dets[idx] = sum(df['visibility'] == Visibility.VISIBLE)

    fig = gpho.Figure()
    fig.update_layout(template='simple_white', title=dict(text='Visible Planets vs Mirror Area',
                      xanchor='center', x=0.5), xaxis_title='Mirror Area (m2)', yaxis_title='Visible Planets (-)')
    fig.add_scatter(x=pi*(dias/2)**2, y=dets, mode='markers')
    return fig


def plot_peak_wavelengths(df):  # pragma: no cover
    fig = gpho.Figure()
    fig.update_layout(template='simple_white', title=dict(text='Spectral Peak Occurrence',
                      xanchor='center', x=0.5), xaxis_title="Wavelength (mum)", yaxis_title="Planets (-)")
    fig.add_histogram(x=df['peak_wavelength']*1E6)
    return fig


def plot_integration_histogram(df):
    fig = gpho.Figure()
    fig.update_layout(template='simple_white', title=dict(text='Integration Time Occurrence (T < 100 hours)',
                      xanchor='center', x=0.5), xaxis_title="Time (hours)", yaxis_title="Planets (-)")
    fig.add_histogram(x=df[df['shot_time_for_snr'] < 60*60*100]['shot_time_for_snr']/60/60)
    return fig


if __name__ == '__main__':  # pragma: no cover
    df = load_dataset(data_path)
    df = calculate_southern_zenith_angle(df)
    df = calculate_peak_wavelength(df)
    df = calculate_lowest_intensity_wavelength(df)
    df = calculate_emissions(df)
    df = calculate_local_fluxes(df)
    #df = calculate_nulling(df, 10E-6, 1000, 0, 0)
    df = force_nulling(df, 1E-5)
    df = calculate_detections(df, 2/2, 0.5, 0.2)
    df = calculate_shot_noise_time(df, 10, 2*pi*1.5E-9/10E-6)
    df = determine_visibility(df, radians(60), radians(90), 60*60*10, 10)
    print(df['visibility'].value_counts())
    vis = plot_visibility(df)
    vis.show()
    vis.write_html('out/celestial_visibility.html')
    vis.write_image('out/celestial_visibility.png')
    flat_vis = plot_visibility_flat(df)
    flat_vis.show()
    flat_vis.write_html('out/celestial_visibility_flat.html')
    flat_vis.write_image('out/celestial_visibility_flat.png')
    int_vis = plot_integration_visibility(df)
    int_vis.show()
    int_vis.write_html('out/integration_vs_distance.html')
    int_vis.write_image('out/integration_vs_distance.png')
    det_dia = plot_detections_diameter(df)
    det_dia.show()
    det_dia.write_html('out/visibility_vs_diameter.html')
    det_dia.write_image('out/visibility_vs_diameter.png')
    det_area = plot_detections_area(df)
    det_area.show()
    det_area.write_html('out/visibility_vs_area.html')
    det_area.write_image('out/visibility_vs_area.png')
    peaks = plot_peak_wavelengths(df)
    peaks.show()
    peaks.write_html('out/spectral_peaks.html')
    peaks.write_image('out/spectral_peaks.png')
    int_hist = plot_integration_histogram(df)
    int_hist.show()
    int_hist.write_html('out/integration_time_histogram.html')
    int_hist.write_image('out/integration_time_histogram.png')
# TODO: Port the unit test harness and achieve full coverage
# TODO: Add all necessary plots and possible Monte-Carlos
# TODO: Add plot details like titles and axis names
