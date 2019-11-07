"""
dbscan.py: implementation of dbscan

* implementation based on https://github.com/chrisjmccormick/dbscan
"""

import numpy as np
import random

def get_distance(p0, p1):
    return np.sum((np.array(p0) - np.array(p1))**2).astype(float)

def region_query(points, point, epsilon):
    neighbors = []

    for point in range(0, len(points)):
        if get_distance(point, points[point]) < epsilon:
            neighbors.append(point)

    return neighbors

def grow_cluster(points, labels, point, neighbor_points, crnt_cluster, epsilon, min_points):
    labels[point] = crnt_cluster

    i = 0
    while i < len(neighbor_points):
        point = neighbor_points[i]

        if labels[point] == -1:
            labels[point] = crnt_cluster
        elif labels[point] == 0:
            labels[point] = crnt_cluster

            point_neighbors = region_query(points, point, epsilon)

            if len(point_neighbors) >= min_points:
                neightbor_points = neighbor_points + point_neighbors

        i += 1

def dbscan(points, epsilon, min_points):
    labels = [0]*len(points)
    crnt_cluster = 0

    for point in range(0, len(points)):
        if not (labels[point] == 0):
            continue

        neighbor_points = region_query(points, point, epsilon)

        if len(neighbor_points) < min_points:
            labels[point] = -1
        else:
            crnt_cluster += 1
            grow_cluster(points, labels, point, neighbor_points, 
                         crnt_cluster, epsilon, min_points)

    return labels
