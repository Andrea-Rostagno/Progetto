import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Banded Trigonometric Function (n = 2 slice)
def banded_trig(x, y):
    return 1 * ((1 - np.cos(x)) + np.sin(0) - np.sin(y))

# Domain for x and y
x = np.linspace(-2*np.pi, 2*np.pi, 400)
y = np.linspace(-2*np.pi, 2*np.pi, 400)
X, Y = np.meshgrid(x, y)

# Compute function values
Z = banded_trig(X, Y)

# Plotting
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

ax.set_title('Banded Trigonometric Function (n = 2 slice)')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$f(x)$')

plt.tight_layout()
plt.show()
