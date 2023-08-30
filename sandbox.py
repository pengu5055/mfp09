import numpy as np
import matplotlib.pyplot as plt

# Generate some random data points
np.random.seed(0)
x = np.random.randn(1000)
y = np.random.randn(1000)

# Create a hexbin plot
plt.hexbin(x, y, gridsize=30, cmap='inferno')

# Add a colorbar
plt.colorbar()

# Set labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Hexbin Heatmap')

# Show the plot
plt.show()
