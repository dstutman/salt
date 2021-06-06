"""
Tests for the visibility model.

Unless otherwise stated, the planet used for comparisons
with exoplanet_data_snr_updated.xslx is: CoRoTID 223977153 b
"""

from pytest import approx
import visibility


#def test_null_depth():
#    from scipy.constants import arcsec
#    assert visibility.null_depth(0.0000127378756436371000000000*arcsec,
#                                 10E-6, 100, 0, 0) == approx(2.3524E-07, 1E-3)


#def test_blackbody_flux():
    from scipy.constants import Stefan_Boltzmann

    # Sanity test
    assert visibility.blackbody_flux(1) == Stefan_Boltzmann

    # Verified with https://www.spectralcalc.com
    assert visibility.blackbody_flux(255) == approx(239.764, 1E-3)


#def test_shot_noise():
#    # Verified with exoplanet_data_snr_updated.xslx
#    assert visibility.shot_noise(2.30E+40, 3.59E+42) == approx(1.90101E+21, 1E-3)  # noqa: E501


#def test_photon_energy():
#    # Verified with https://www.sensorsone.com
#    assert visibility.photon_energy(10E-6) == approx(1.98644586e-20, 1E-3)

#def test_rms_null_variation():
#    # Verified by hand
#    assert visibility.rms_null_variation(0.000942, 0.01) == approx(0.00355, 1E-3)