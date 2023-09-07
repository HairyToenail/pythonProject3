import queue
from main import *
import numpy as np
realArray = a.copy()

#realArray = exit.realArray

startY = 49
startX = 31

endY = 112
endX = 71

realArray[startY][startX] = 3 #start
realArray[endY][endX] = 4 #exit

def printMaze(maze, moves):
    # for x, pos in enumerate(maze[startY]): #y coord
    #     if pos == 3: #set start
    #         start = x

    i = startX
    j = startY
    pos = set()

    for move in moves:
        if move == 'L':
            i -= 1
        if move == 'R':
            i += 1
        if move == 'U':
            j -= 1
        if move == 'D':
            j += 1
        realArray[j][i] = 69
    #
    # for j, row in enumerate(maze):
    #     for i, col in enumerate(row):
    #         if (j, i) in pos:
    #            realArray[j][i] = 69


def valid(maze, moves):
    # for x, pos in enumerate(maze[0]): #y coord
    #     if pos == 3: #set start
    #         start = x

    i = startX
    j = startY

    for move in moves:
        if move == 'L':
            i -= 1
        if move == 'R':
            i += 1
        if move == 'U':
            j -= 1
        if move == 'D':
            j += 1

        if not (0 <= i < len(maze[0]) and 0 <= j < len(maze)):
            return False
        elif (maze[j][i] == 1):
            return False

    return True


def findEnd(maze, moves):
    # for x, pos in enumerate(maze[0]): #y coordinate
    #     if pos == 3:
    #         start = x

    i = startX #x coord
    j = startY #y coordinate

    for move in moves:
        if move == "L":
            i -= 1

        elif move == "R":
            i += 1

        elif move == "U":
            j -= 1

        elif move == "D":
            j += 1

    if maze[j][i] == 4:  # exit
        print("Found: " + moves)
        printMaze(maze, moves)
        return True


# MAIN ALGORITHM
nums = queue.Queue()
nums.put("")
add = ""
maze = realArray

while not findEnd(maze, add):
    add = nums.get()
    #print(add)
    for j in ["L", "R", "U", "D"]:
        put = add + j
        if valid(maze, put):
            nums.put(put)

print(realArray)

np.savetxt("data8.csv", realArray, delimiter=',')

