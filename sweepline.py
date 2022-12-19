# Skyline Problem: Vertical Line Sweep Algo
import matplotlib.pyplot as plt
import pylab as pl
import numpy as np
import heapq as hq

def getSkyline(buildings):
    '''
    Purpose:
        Compute the contour line of a list of rectangle buildings that are all grounded on y=0, and
        return the skyline of the buildings.
    Parameters:
        buildings : a list of lists where buildings[i] = [lefti, righti, heighti] where
                         - lefti is the x coordinate of the left edge of the ith building.
                         - righti is the x coordinate of the right edge of the ith building.
                         - heighti is the height of the ith building.
    Return:
        a list of "key points" sorted by their x-coordinate in the form [[x1,y1],[x2,y2],...].
        Each key point is the left endpoint of some horizontal segment in the skyline except the
        last point in the list, which always has a y-coordinate 0 and is used to mark the skyline's
        termination where the rightmost building ends.
    '''

    tallq = []
    res = []
    prev = -1

    for b in buildings:
        # check previous stopping points
        while tallq and tallq[0][1] <= b[0]:
            stoppingPt = tallq[0][1]
            height = tallq[0][0]
            hq.heappop(tallq)
            while tallq and (tallq[0][1] <= stoppingPt):
                hq.heappop(tallq)
            if tallq and tallq[0][0] != height:
                res.append([stoppingPt, -tallq[0][0]])
            elif not tallq:
                res.append([stoppingPt, 0])
                
        # consider adding the current pt (collinear case included)
        if not (tallq and (tallq[0][0] == -b[2] or (tallq[0][0] < -b[2] and tallq[0][1] > b[0]))):
            if res and res[-1][0] == b[0]:
                res[-1][1] = max(b[2], res[-1][1])
                if len(res) >= 2 and res[-2][1] == res[-1][1]:
                    res = res[:-1]
            else:
                res.append([b[0], b[2]])

        # maintain invarients: tallest building live beyond this point
        while tallq and tallq[0][1] == b[0]:
            hq.heappop(tallq)

        hq.heappush(tallq, [-b[2], b[1]])
        prev = b[0]

    # check the rest stopping points
    while tallq:
        stoppingPt = tallq[0][1]
        height = tallq[0][0]
        hq.heappop(tallq)
        while tallq and tallq[0][1] <= stoppingPt:
            hq.heappop(tallq)
        if tallq and tallq[0][0] != height:
            res.append([stoppingPt, -tallq[0][0]])
        elif not tallq:
            res.append([stoppingPt, 0])
        prev = stoppingPt

    # remove duplicates
    distinctRes = []
    prev = [-1, -1]
    for i in range(len(res)):
        if res[i][1] != prev[1] and res[i][0] != prev[0]:
            distinctRes.append(res[i])
        if res[i][0] == prev[0]:
            distinctRes[-1][1] = max(distinctRes[-1][1], res[i][1])
        prev = distinctRes[-1]

    return res


def drawRectangles(buildings, xRange, yRange, ax):
    '''
    Purpose:
        Plot the outline of the given rectangle buildings
    Parameters:
        buildlings : a list of lists - same as in getSkyline function
        xRange : the max x-coord shown on canvas
        yRange : the max y-coord shown on canvas
        ax : the subplot axes
    Return:
        None
    '''
    N = len(buildings)
    xs = []
    ys = []
    ws = []
    hs = []
    for b in buildings:
        xs.append(b[0])
        ys.append(0)
        ws.append(b[1] - b[0])
        hs.append(b[2])
    vs = np.random.randn(N)
    normal = pl.Normalize(vs.min(), vs.max())
    colors = pl.cm.jet(normal(vs))

    for x,y,w,h,c in zip(xs,ys,ws,hs,colors):
        rect = pl.Rectangle((x,y),w,h,color=c,fill=None)  #remove the fill=None to show colors filled in rectangles
        ax.add_patch(rect)

    ax.set_xlim(0,xRange)  #320
    ax.set_ylim(0,yRange)  #60


def drawContour(res, xRange, yRange, ax):
    '''
    Purpose:
        Plot the computed skyline of the buildings
    Parameters:
        res : the skyline that the getSkyline function would return (as a list of sorted points)
        xRange : the width of the canvas
        yRange : the height of the canvas
        ax : the subplot axes
    Return:
        None
    '''
    plt.xlim(0, xRange)
    plt.ylim(0, yRange)
    color = '#000000'
    
    prevx = 0
    prevy = 0

    for c in after:
        plt.plot(c[0], c[1], marker="o", markersize=7, markeredgecolor="red", markerfacecolor="red")
        
        # draw a horizontal  line
        xs = [prevx, c[0]]
        ys = [prevy, prevy]
        ax.plot(xs, ys, c=color)

        # draw a vertical line
        x_values = [c[0], c[0]]
        y_values = [prevy, c[1]]
        ax.plot(x_values, y_values, c=color)
        prevx, prevy = c[0], c[1]


import random
def randomBuildings(N, width, height, prev):
    '''
    Purpose:
        Randomly generate a list of N buildings with the constraints
    Parameters:
        N : the number of buildings
        width : max-width of each rectangle
        prev : smallest starting point of 1st rectangle
    Return:
        A list of buildings that's ready to be fed into the getSkyline function
    '''
    xs,ys,ws = [],[],[]

    for i in range(prev, N * width, width):
        xs.append(random.randint(prev,i))
        ys.append(random.randint(i, i+height))
        ws.append(random.randint(5, height))
        prev = i

    return [list(ele) for ele in list(zip(xs, ys, ws))]

# Randomly generate rectangles
N = 15        # number of rectangles
width = 20    # max-width of each rectangle
height = 50   # max-height of each rectangle
prev = 10     # smallest starting point of 1st rectangle
xRange = 300  # canvas x range
yRange = 60   # canvas y range
buildings = randomBuildings(N, width, height, prev)

# Create a canvas
plt.rcParams["figure.figsize"] = [7.50, 7.50]
plt.rcParams["figure.autolayout"] = True
fig, (ax1, ax2) = plt.subplots(2)

# Plot the randomly generated rectangles
drawRectangles(buildings, xRange, yRange, ax1)

# Compute the skyline
after = getSkyline(buildings)

# Plot the skyline
drawContour(after, xRange, yRange, ax2)
plt.show()
