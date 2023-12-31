from main import *
import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)

realArray = a.copy()
d = a.copy()
finalPath = []
#set the exit and enter
for i in range(len(a)):
    for o in range(len(a[i])):
        if realArray[i][o] > .5:
            realArray[i][o] = 1
        else:
            realArray[i][o] = 0  # floor

startY = 49
startX = 31

endY = 31
endX = 31

# realArray[startY][startX] = 3 #start
# realArray[endY][endX] = 4 #exit


d[startY][startX] = 400
d[startY][startX] = 400
#realArray is 0 and 1
# print(realArray)


# class QItem:
#     def __init__(self, row, col, dist):
#         self.row = row
#
#         self.col = col
#         self.dist = dist
#
#     def __repr__(self):
#         return f"QItem({self.row}, {self.col}, {self.dist})"
#
# def minDistance(grid):
#     source = QItem(0, 0, 0, list())
#
#     # Finding the source to start from
#     for row in range(len(grid)):
#         for col in range(len(grid[row])):
#             if grid[row][col] == 3:
#                 source.row = row
#                 source.col = col
#                 d[row][col]=400
#                 break
#     # To maintain location visit status
#     visited = [[False for _ in range(len(grid[0]))]
#                for _ in range(len(grid))]
#
#     # applying BFS on matrix cells starting from source
#     queue = []
#     queue.append(source)
#     visited[source.row][source.col] = True
#     i=0
#     while len(queue) != 0:
#
#         source = queue.pop(0)
#         print(source)
#         # moving up
#         if isValid(source.row- 1, source.col, grid, visited):
#             visited[source.row - 1][source.col] = True
#
#             if (grid[source.row-1][source.col] == 4):
#                 return source.dist-1, source.g, len(source.g)
#             d[source.row-1][source.col]=500
#         # moving down
#         if isValid(source.row + 1, source.col, grid, visited):
#             queue.append(QItem(source.row + 1, source.col, source.dist + 1))
#             visited[source.row + 1][source.col] = True
#             if (grid[source.row+1][source.col] == 4):
#                 return source.dist-1, source.g, len(source.g)
#             d[source.row + 1][source.col] = 500
#
#
#         # moving left
#         if isValid(source.row, source.col - 1, grid, visited):
#             queue.append(QItem(source.row, source.col - 1, source.dist + 1))
#             visited[source.row][source.col - 1] = True
#             if (grid[source.row][source.col+1] == 4):
#                 return source.dist-1, source.g, len(source.g)
#             d[source.row][source.col-1] = 500
#
#         # moving right
#         if isValid(source.row, source.col + 1, grid, visited):
#             queue.append(QItem(source.row, source.col + 1, source.dist + 1))
#             visited[source.row][source.col + 1] = True
#             if (grid[source.row][source.col-1] == 4):
#                 return source.dist-1, source.g, len(source.g)
#             d[source.row][source.col+1] = 500
#
#
#     return -1
# np.savetxt("tttt.csv", d, delimiter=',')

# checking where move is valid or not
def isValid(x, y, grid, visited):
    if ((x >= 0 and y >= 0) and
            (x < len(grid) and y < len(grid[0])) and
            (grid[x][y] == 0) and visited[x][y]== False):
        finalPath.append([x,y])
        return True
    else:
        return False


# Driver code
tt = realArray.copy()
def path(start, end, arr):
    path = []
    open = []
    closed = []
    open.append(start)
    parent = start
    while len(open)!=0:
        open.sort()
        current = open.pop(0)
        closed.insert(0,current)
        if(current==end):
            while(current!=start):
                if(len(closed)==0):
                    return path
                path.insert(0,current)
                current = parent
                parent = closed.pop(0)

        neighbors = [(current[0]+1, current[1]),
                     (current[0]-1, current[1]),
                     (current[0]+1, current[1]+1),
                     (current[0]-1, current[1]-1),
                     (current[0], current[1]-1),
                     (current[0], current[1]+1),
                     (current[0]+1, current[1]-1),
                     (current[0]-1, current[1]+1)]
        for neighbor in neighbors:

            if(arr[neighbor]!=0 or neighbor in closed):
                continue
            cost = arr[current] + cost_estimate(current, neighbor)
            if(cost<arr[neighbor] or not (neighbor in open)):
                arr[neighbor] = cost
                parent = current
                if(not(neighbor in open)):
                    open.insert(0, neighbor)

    return -1
tt[startY][startX] = 400
tt[startY][startX] = 400
def cost_estimate(current, next):
    dX = abs(current[0]-next[0])
    dY = abs(current[1]-next[1])
    if(dX>dY):
        return 14 * dY + 10 * (dX - dY)
    return 14 * dX + 10 * (dY - dX)
if __name__ == '__main__':
    grid = realArray

    for vals in path((startX, startY), (endX, endY), tt):
        tt[vals[0]][vals[1]]=500
    np.savetxt("path.csv", tt, delimiter=',')

