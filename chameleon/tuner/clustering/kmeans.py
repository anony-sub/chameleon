"""
kmeans.py: implementation of kmeans

* implementation based on https://gist.github.com/pmsosa/5454ade527adbee105dd51066ef30c5f
"""

import numpy as np
import random

# p0 and p1 are tuples
def distance(p0, p1):
    return np.sum((np.array(p0) - np.array(p1))**2).astype(float)

def kmeans(points, k, max_iter=100):
    # initialize centroids and clusters
    centroids = [points[i] for i in np.random.randint(len(points), size=k)]
    
    cluster = [0] * len(points)
    prev_cluster = [-1] * len(points)

    # start
    i = 0
    force_recalculation = False
    while (cluster != prev_cluster) or (i > max_iter) or (force_recalculation):
        if i > max_iter * 2:
            break

        prev_cluster = list(cluster)
        force_recalculation = False
        i += 1
        
        # update cluster
        for p in range(0, len(points)):
            min_dist = float("inf")

            # check for min distance to assign cluster
            for c in range(0, len(centroids)):
                dist = distance(points[p], centroids[c])

                if dist < min_dist:
                    min_dist = dist
                    cluster[p] = c

        # update centroids
        for c in range(0, len(centroids)):
            new_centroid = [0] * len(points[0])
            members = 0

            # add
            for p in range(0, len(points)):
                if cluster[p] == c:
                    for j in range(0, len(points[0])):
                        new_centroid[j] += points[p][j]
                    members += 1

            # divide
            for j in range(0, len(points[0])):
                if members != 0:
                    new_centroid[j] = int(new_centroid[j] / float(members))

                # force recalculation
                else:
                    new_centroid = random.choice(points)
                    force_recalculation = True

            centroids[c] = tuple(new_centroid)

    loss = 0
    for p in range(0, len(points)):
        loss += distance(points[p], centroids[cluster[p]])

    return centroids, cluster, loss
