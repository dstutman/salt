import numpy as np
from visibility import plancks_law, photon_energy
import plotly.graph_objects as gpho

wavelengths = np.linspace(6E-6, 20E-6)
emissions_ten = plancks_law(wavelengths, 10)/photon_energy(wavelengths)
emissions_forty = plancks_law(wavelengths, 40)/photon_energy(wavelengths)

fig = gpho.Figure()
fig.update_layout(template='seaborn', xaxis_title=r'$\text{Wavelength}\: [\mu m]$', yaxis_title=r'$\text{Intensity}\: [ph/(m^2 \cdot \mu m \cdot sr)]$')
fig.update_yaxes(type='log')
fig.add_scatter(x=wavelengths*1E6, y=emissions_ten, name='Emissions at 10K')
fig.add_scatter(x=wavelengths*1E6, y=emissions_forty, name='Emissions at 40K')
fig.show()
fig.write_html('out/thermal_noise.html')
fig.write_image('out/thermal_noise.png')
