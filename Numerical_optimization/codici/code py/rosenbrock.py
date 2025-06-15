import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def rosenbrock(x, y):
    return 100 * (y - x**2)**2 + (1 - x)**2

x = np.linspace(-2, 2, 400)
y = np.linspace(-1, 3, 400)
X, Y = np.meshgrid(x, y)
Z = rosenbrock(X, Y)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.9, edgecolor='none')
ax.set_xlabel('x₁')
ax.set_ylabel('x₂')
ax.set_zlabel('f(x₁, x₂)')
ax.set_title('Rosenbrock Function')

# Optional: plot trajectory
# trajectory = np.array([...])
# ax.plot(trajectory[:,0], trajectory[:,1], rosenbrock(trajectory[:,0], trajectory[:,1]), 'r-', label='path')

plt.tight_layout()
plt.savefig('rosen3D_A.png')  # or rosen3D_B.png
plt.show()
