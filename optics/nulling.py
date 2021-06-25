import plotly.graph_objects as gpho
import scipy.constants as constants
import numpy as np

# Parameters
wavelength=10E-6
bu = 200
bv = 200
omegau = 2*np.pi/(wavelength/bu)
omegav = 2*np.pi/(wavelength/bv)

rad = 1.47958052E-7 / constants.arcsec * 1000 * 1/6

theta = np.linspace(0, 2*np.pi) 
xt = np.cos(theta) * rad
yt = np.sin(theta) * rad

# Generate celestial sphere coordinates
x = np.linspace(-2*np.pi/omegau, 2*np.pi/omegau, 500)
y = np.linspace(-2*np.pi/omegav, 2*np.pi/omegav, 500)

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
fig.update_layout(template='seaborn', xaxis_title=r'$\text{x}\: [mas]$', yaxis_title=r'$\text{y}\: [mas]$')
fig.add_heatmap(x=x, y=y, z=r, colorbar=dict(title='Normalized Transmittance'))
fig.add_scatter(x=xt, y=yt)
fig.add_shape(type='circle', x0=-1, x1=1, y0=-1, y1=1, fillcolor='orange', opacity=1)
fig.show()
fig.write_html('out/transmission_map.html')
fig.write_image('out/transmission_map.png')
