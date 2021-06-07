"""
Tests for the visibility model.

Unless otherwise stated, the planet used for comparisons
with exoplanet_data_snr_updated.xslx is: CoRoTID 223977153 b
"""

import visibility
from pytest import approx
from math import sqrt, pi


#def test_null_depth():
#    from scipy.constants import arcsec
#    assert visibility.null_depth(0.0000127378756436371000000000*arcsec,
#                                 10E-6, 100, 0, 0) == approx(2.3524E-07, 1E-3)


#def test_blackbody_flux():
#    from scipy.constants import Stefan_Boltzmann
#
#    # Sanity test
#    assert visibility.blackbody_flux(1) == Stefan_Boltzmann
#
#    # Verified with https://www.spectralcalc.com
#    assert visibility.blackbody_flux(255) == approx(239.764, 1E-3)


#def test_shot_noise():
#    # Verified with exoplanet_data_snr_updated.xslx
#    assert visibility.shot_noise(2.30E+40, 3.59E+42) == approx(1.90101E+21, 1E-3)  # noqa: E501


#def test_photon_energy():
#    # Verified with https://www.sensorsone.com
#    assert visibility.photon_energy(10E-6) == approx(1.98644586e-20, 1E-3)

#def test_rms_null_variation():
#    # Verified by hand
#    assert visibility.rms_null_variation(0.000942, 0.01) == approx(0.00355, 1E-3)

def test_end_to_end():
    df = visibility.load_dataset(visibility.data_path, True, True, True, True)
    df = visibility.calculate_emissions(df)
    df = visibility.calculate_local_fluxes(df)
    df = visibility.calculate_nulling(df, 10E-6, 100, 0, 0)
    df = visibility.calculate_detections(df, 1/sqrt(pi), 0.5, 1, False)
    df = visibility.calculate_shot_noise_snr(df, 1, 1, 0)
    pl = df[df['pl_name'] == 'CoRoTID 223977153 b'].iloc[0]
    assert(pl['st_nulldepth'] == approx(2.35246239517305E-07))
    assert(pl['st_phpspm2_loc'] == approx(3452459, rel=5E-3))
    assert(pl['pl_eps'] == approx(5.79E0, abs=0.05))
    assert(pl['st_eps'] == approx(4.06E-1, rel=5E-3))
    assert(pl['shot_snr_for_time'] == approx(2.29, abs=0.05))