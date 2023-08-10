import sys
from asyncio import Queue
from multiprocessing import RLock
import random
import torch
import numpy as np
import queue
np.set_printoptions(threshold=sys.maxsize)
# np.set_printoptions(suppress=True)
import pandas as pd
import matplotlib.pyplot as plt
import pyvista as pv


class adjMat(): #not sure if this actually works or if we need it
    def __intit__(self, vertex, matrix):
        self.vertex=vertex
        self.matrix=matrix
    def addEdge(self, source, destination):
        self.matrix[source, destination] = 1
        self.matrix[destination, source] =1

class SynchronousQueue(object):

    def __init__(self):
        self.q = Queue(1)
        self.put_lock = RLock()

    def get(self):
        value = self.q.get(block=True)
        self.q.task_done()
        return value

    def put(self, item):
        with self.put_lock:
            self.q.put(item, block=True)
            self.q.join()

def fire(a, spread_probability, startY, startX):
    a[startY, startX] = -1  # starting point
    # Loop through each cell in the a.
    for row in range(len(a[:, 0])):
        for column in range(len(a[0, :])):
            # If the cell is not on fire, check if it is adjacent to a burning cell.
            if a[row, column] > 0:
                for neighbor_row in range(-1, 2):
                    for neighbor_column in range(-1, 2):
                        if 0 <= row + neighbor_row < len(a[:, 0]) and 0 <= column + neighbor_column < len(a[0, :]):
                            if a[row + neighbor_row, column + neighbor_column] <  0 and a[row, column]>0:
                            # If it is adjacent to a burning cell, set it on fire with probability `spread_probability`.
                                if np.random.random() < spread_probability:
                                    a[row, column] = a[row + neighbor_row, column + neighbor_column]-1
    return a

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
fg = a
cY = random.randint(0, len(a[:, 0]))
cX = random.randint(0, len(a[0, :]))
for i in range(100):
    fire(fg, 0.1, cY, cX)
    if(i%10==0):
        np.savetxt(f"fire{i}.csv", fg, delimiter=',')
print(True)
