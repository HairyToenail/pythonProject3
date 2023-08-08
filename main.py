import sys
import torch
import numpy as np

np.set_printoptions(threshold=sys.maxsize)
# np.set_printoptions(suppress=True)
import pandas as pd
import pywavefront
import pyglet
import matplotlib.pyplot as plt
import pyvista as pv
from pyvista import examples
import math

def fire(a, spread_probability):
    a[48, 30] = -1

    # Loop through each cell in the a.
    count = 0
    for row in range(len(a[:, 0])):
        for column in range(len(a[0, :])):
            # If the cell is not on fire, check if it is adjacent to a burning cell.
            if a[row, column] > 0:
                for neighbor_row in range(-1, 2):
                    for neighbor_column in range(-1, 2):
                        count += 1
                    if 0 <= row + neighbor_row < len(a[:, 0]) and 0 <= column + neighbor_column < len(a[0, :]):
                        if a[row + neighbor_row][column + neighbor_column] == 1:
                            # If it is adjacent to a burning cell, set it on fire with probability `spread_probability`.
                            if np.random.random() < spread_probability:
                                a[row, column] = -1
def distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


def ant_colony_optimization(points, n_ants, n_iterations, alpha, beta, evaporation_rate, Q):
    n_points = len(points)
    pheromone = np.ones((n_points, n_points))
    best_path = None
    best_path_length = np.inf

    for iteration in range(n_iterations):
        paths = []
        path_lengths = []

        for ant in range(n_ants):
            visited = [False] * n_points
            current_point = np.random.randint(n_points)
            visited[current_point] = True
            path = [current_point]
            path_length = 0

            while False in visited:
                unvisited = np.where(np.logical_not(visited))[0]
                probabilities = np.zeros(len(unvisited))

                for i, unvisited_point in enumerate(unvisited):
                    probabilities[i] = pheromone[current_point, unvisited_point] ** alpha / distance(
                        points[current_point], points[unvisited_point]) ** beta

                probabilities /= np.sum(probabilities)

                next_point = np.random.choice(unvisited, p=probabilities)
                path.append(next_point)
                path_length += distance(points[current_point], points[next_point])
                visited[next_point] = True
                current_point = next_point

            paths.append(path)
            path_lengths.append(path_length)

            if path_length < best_path_length:
                best_path = path
                best_path_length = path_length

        pheromone *= evaporation_rate

        for path, path_length in zip(paths, path_lengths):
            for i in range(n_points - 1):
                pheromone[path[i], path[i + 1]] += Q / path_length
            pheromone[path[-1], path[0]] += Q / path_length

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='o')

    for i in range(n_points - 1):
        ax.plot([points[best_path[i], 0], points[best_path[i + 1], 0]],
                [points[best_path[i], 1], points[best_path[i + 1], 1]],
                [points[best_path[i], 2], points[best_path[i + 1], 2]],
                c='g', linestyle='-', linewidth=2, marker='o')

    ax.plot([points[best_path[0], 0], points[best_path[-1], 0]],
            [points[best_path[0], 1], points[best_path[-1], 1]],
            [points[best_path[0], 2], points[best_path[-1], 2]],
            c='g', linestyle='-', linewidth=2, marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()



mesh = pv.read("scan_2.obj")
# mesh = mesh.rotate_x(180, transform_all_input_vectors=True, inplace=True)

texture = pv.read_texture("scan_2.jpg")


def generate_points(subset=1):  # bigger subset more points ohhhh okok
    """A helper to make a 3D NumPy array of points (n_points by 3)"""
    dataset = pv.PolyData("scan_2.obj")  # this is where it is btw ohhh ok thanks
    ids = np.random.randint(low=0, high=dataset.n_points - 1, size=int(dataset.n_points * subset))
    return dataset.points[ids]


points = generate_points()

print(len(points))
# Print first 5 rows to prove its a numpy array (n_points by 3)
# Columns are (X Y Z)
point_cloud = pv.PolyData(points)  # this is where we create a point cloud
data = points[:, 1]
point_cloud["elevation"] = data #adds color
point_cloud = point_cloud.rotate_x(90, transform_all_input_vectors=True, inplace=True)

#point_cloud.plot(render_points_as_spheres=True)
print(len(points))
#mesh.plot(texture=texture)  # adds color? - no

# converts the coordinates of cloud ohhhhh okok
new = pv.convert_array(pv.convert_array(point_cloud.points))
x = new[:, 0]
x_dif = max(x) - min(x)
y = new[:, 1]
y_dif = max(y) - min(y)
z = new[:, 2]
new[:, 0] = new[:, 0] - min(x)
new[:, 1] = new[:, 1] - min(y)
new[:, 2] = new[:, 2] - min(z)
a = np.zeros((int(max(new[:, 1])*10)+1, int(max(new[:, 0])*10)+1), dtype=float)
b=np.zeros((int(max(new[:, 1])*10)+1, int(max(new[:, 0])*10)+1), dtype=float)

# print(new)

test = []
print(max(new[:, 2]))
print(a.shape)

#puts in values for z based on (x,y) coordinates
for i in range(len(new[:, 0])):
    h = new[i, 2]
    x_coord = int(new[i, 1]*10)
    y_coord = int(new[i, 0]*10)
    if (h > a[x_coord, y_coord] and h<2.35): #adding to floor map
        a[x_coord, y_coord] = h
    if (h > b[x_coord, y_coord] and h>2.35): #adding to ceiling map
        b[x_coord, y_coord] = h

print(len(test))


# removes 0s from columns and rows ohhhhhokok
a = a[:, ~np.all(a == 0, axis=0)]
a = a[~np.all(a == 0, axis=1), :]
print(b)
print(new)

#np.savetxt("data5.csv", a, delimiter=',')
#np.savetxt("data6.csv", b, delimiter=',')

circle1 = pv.Circle(1)
circle1 = circle1.translate((2.5, 15, 3))
circle2 = pv.Circle(1)
circle2 = circle2.translate((8, 15, 3))

# graphs the point cloud and if u right click it gives u the coordinates
def callback(point):
    """Create a cube and a label at the click point."""
    pl.add_mesh(point_cloud)
    pl.add_point_labels(point, [f"{point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f}"])

pl = pv.Plotter()
pl.add_mesh(point_cloud)
pl.add_mesh(circle1)
pl.add_mesh(circle2)
pl.enable_surface_point_picking(callback=callback, show_point=False)
pl.show()
# -------------------------------------------------------------------
#Fire simulation
count = 0
while not((a!=-1).all()):
    fire(a, 0.5)
    if(count%5000==0):
        np.savetxt(f"fire{count}.csv", a, delimiter=',')
    count+=1

# ant_colony_optimization(a, 500, 50032, 21, 11, 0.5, 3)
