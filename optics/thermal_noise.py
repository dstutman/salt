import numpy as np
from visibility import plancks_law, photon_energy
import plotly.graph_objects as gpho

wavelengths = np.linspace(6E-6, 20E-6)
emissions_ten = plancks_law(wavelengths, 10)/photon_energy(wavelengths) * 1E6
emissions_forty = plancks_law(wavelengths, 40)/photon_energy(wavelengths) * 1E6

fig = gpho.Figure()
fig.update_layout(template='simple_white', xaxis_title=r'Wavelength [\mu m]', yaxis_title=r'Intensity [$ph/(m^2 \cdot \mu m \cdot sr)]')
fig.update_yaxes(type='log')
fig.add_scatter(x=wavelengths*1E6, y=emissions_ten, name='Emissions at 10K')
fig.add_scatter(x=wavelengths*1E6, y=emissions_forty, name='Emissions at 40K')
fig.show()
fig.write_html('out/thermal_noise.html')
fig.write_image('out/thermal_noise.png')
