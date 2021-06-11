import plotly.graph_objects as gpho
import scipy.constants as constants
import numpy as np

# Parameters
wavelength=10E-6
bu = 1000
bv = 200
omegau = 2*np.pi/(wavelength/bu)
omegav = 2*np.pi/(wavelength/bv)


# Generate celestial sphere coordinates
x = np.linspace(2*-2*np.pi/omegau, 2*2*np.pi/omegau, 500)
y = np.linspace(2*-2*np.pi/omegav, 2*2*np.pi/omegav, 500)

# Responses vary sinusoidally along each axis
ru = np.cos(omegau*x)
rv = np.cos(omegav*y)

# The response is outer of individual responses
r = np.outer(ru, rv)

# Remap 1-(-1) to 1-0 linearly
r = (1-r)*0.5

# Scale constants
to_mas = 1000*360*60*60/2/np.pi

x *= to_mas
y *= to_mas
# Plotting
fig = gpho.Figure()

# Celestial sphere is 2d manifold
fig.update_layout(template='simple_white', title=dict(text='Array Transmission Map', xanchor='center', x=0.5), xaxis_title='x (mas)', yaxis_title='y (mas)')
fig.add_heatmap(x=x, y=y, z=r, colorbar=dict(title='Normalized Transmittance'))
fig.show()
fig.write_html('out/transmission_map.html')
fig.write_image('out/transmission_map.png')
