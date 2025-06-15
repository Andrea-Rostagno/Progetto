import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def extended_rosenbrock(x, y):
    return 100 * (y - x**2)**2 + (1 - x)**2

# Domain for plotting
x = np.linspace(-2, 2, 400)
y = np.linspace(-1, 3, 400)
X, Y = np.meshgrid(x, y)
Z = extended_rosenbrock(X, Y)

# Plotting the 3D surface
fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
ax.set_title('Extended Rosenbrock Function (n = 2)')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$f(x)$')
plt.tight_layout()
plt.savefig("ext_rosen_3d.png", dpi=300)
plt.show()
