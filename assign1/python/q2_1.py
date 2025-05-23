import numpy as np
import matplotlib.pyplot as plt

# Define the points in image space.
points = [(10, 10), (20, 20), (30, 30)]

# Create an array of theta values from 0 to pi.
theta = np.linspace(0, 2*np.pi, 180)  

plt.figure(figsize=(8, 6))

# Plot the sinusoid for each point.
for (x, y) in points:
    # Compute rho for each theta.
    rho = x * np.cos(theta) + y * np.sin(theta)
    plt.plot(theta, rho, label=f'Point ({x},{y})')

# For collinear points (on y = x), the sinusoids intersect at (rho, theta) = (0, 3pi/4).
theta_intersect = 3 * np.pi / 4
rho_intersect = 0

#add the transfrom of m and c in teh class probelm
m = -np.cos(theta_intersect) / np.sin(theta_intersect)
c = rho_intersect / np.sin(theta_intersect)

# Plot the intersection point.
plt.plot(theta_intersect, rho_intersect, 'ko', markersize=8, label='Intersection')

# Add labels, title, legend, and grid.
plt.xlabel('Theta (radians)')
plt.ylabel('Rho')
plt.title('Hough Transform Sinusoids for Points (10,10), (20,20), (30,30)')
plt.legend()
plt.grid(True)


# Create a new figure and plot the line y = mx + c
plt.figure(figsize=(8, 6))
x_vals = np.linspace(0, 40, 100)  # Generate x values
y_vals = m * x_vals + c  # Calculate corresponding y values

plt.plot(x_vals, y_vals, label=f'Line: y = {m:.2f}x + {c:.2f}')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Line Equation: y = mx + c')
plt.grid(True)
plt.legend()

# Plot the original points on the line plot
for x, y in points:
    plt.plot(x, y, 'bo', markersize=8, label=f'Point ({x}, {y})')

# Remove duplicate labels
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())


plt.legend()
plt.show()


print(f"Equation of the line: y = {m:.2f}x + {c:.2f}")