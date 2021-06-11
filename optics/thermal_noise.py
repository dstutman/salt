import plotly.graph_objects as gpho
import numpy as np
from visibility import plancks_law, photon_energy

wavelengths = np.linspace(6E-6, 20E-6)
intensities_ten = (lambda w: plancks_law(w, 10))(wavelengths)/photon_energy(wavelengths)
intensities_forty = (lambda w: plancks_law(w, 40))(wavelengths)/photon_energy(wavelengths)

fig = gpho.Figure()
fig.add_scatter(x=wavelengths, y=intensities_ten, name='Emission Intensities (10 K)')
fig.add_scatter(x=wavelengths, y=intensities_forty, name='Emission Intensities (40 K)')
fig.update_yaxes(type='log')
fig.show()