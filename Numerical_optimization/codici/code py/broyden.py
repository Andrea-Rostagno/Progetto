import numpy as np
import matplotlib.pyplot as plt
import os

# Run the following code only if latex is available on the system
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'font.serif': 'Times',
    'font.size': 14,
    'text.usetex': True,
    'pgf.rcfonts': False,
})


def plot_surf(f, x, y, name):
    x, y = np.meshgrid(x, y)
    z = f(x, y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, cmap='viridis')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$f(x_1, x_2)$')
    ax.zaxis.labelpad = 9
    ax.zaxis.set_tick_params(pad=5)
    ax.set_box_aspect(None, zoom=.8)
    fig.tight_layout()
    
    plt.show()

def generalized_broyden(x, y):
    return 1/2 * (((3-2*x)*x + 1 - y)**2 + ((3-2*y)*y + 1 - x)**2)

# Generate data for the plot
x = np.linspace(-5, 6, 1000)
y = np.linspace(-5, 6, 1000)

plot_surf(generalized_broyden, x, y, "generalized_broyden")