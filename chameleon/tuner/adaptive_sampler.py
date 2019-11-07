# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=no-else-return,invalid-name,consider-using-enumerate,abstract-method
"""Sampler that uses Adaptive Sampling"""
import numpy as np

from .sampler import Sampler

from ..env import GLOBAL_SCOPE

class AdaptiveSampler(Sampler):
    """Sampler that uses Adaptive Sampling
    
    Parameters
    ----------
    valid_dims: int
        The number of valid dimensions used to compare samples
    max_sample_cnt: int
        The maximum number of samples to be returned
    sample_cnt_step: int
        The number of parallel measurements
    """

    def __init__(self, max_sample_cnt, sample_cnt_step=8):
        self.max_sample_cnt = max_sample_cnt
        self.sample_cnt_step = sample_cnt_step

    def sample(self, samples, dims, threshold=2.5, valid_dims=2):
        """Samples using Adaptive Sampling

        Parameters
        ----------
        samples: Array of int
            configs in knob form
        threshold: float
            hyperparameter used for adaptive sampling 
        
        Returns
        -------
        reduced set of configs in knob form

        """
        cardinal_dims = np.argsort(-np.array(dims))[:valid_dims]
        adaptive_range = np.arange(self.sample_cnt_step, self.max_sample_cnt+1, self.sample_cnt_step)
        cardinal_dims_samples = [np.array(sample)[cardinal_dims] for sample in samples]
        
        last_loss = np.inf
        for k in adaptive_range:
            centroids, cluster, loss = clusterize(cardinal_dims_samples, k)
            
            if loss >= last_loss / threshold:
                break
            else:
                last_loss = loss
        reduced_samples = reduce_samples(samples, centroids, cluster, dims, cardinal_dims)

        return reduced_samples


def distance(x, y, l=2):
    """calculates distance between two points"""
    return np.sum((x - y)**l).astype(float)


def clusterize(points, k, max_iter=100):
    """k-means clustering algorithm"""
    centroids = [points[i] for i in np.random.randint(len(points), size=k)]
    new_assignment = [0] * len(points)
    old_assignment = [-1] * len(points)

    i = 0
    split = False
    while i < max_iter or split == True and new_assignment != old_assignment:
        old_assignment = list(new_assignment)
        split = False
        i += 1
        
        for p in range(len(points)):
            distances = [distance(points[p], centroids[c]) for c in range(len(centroids))]
            new_assignment[p] = np.argmin(distances)

        for c in range(len(centroids)):
            members = [points[p] for p in range(len(points)) if new_assignment[p] == c]
            if members:
                centroids[c] = np.mean(members, axis=0).astype(int)
            else:
                centroids[c] = points[np.random.choice(len(points))]
                split = True

    loss = np.sum([distance(points[p], centroids[new_assignment[p]]) for p in range(len(points))])

    return centroids, new_assignment, loss

def synthesize_sample(centroid, sample_dims, dims, cardinal_dims, reduction='sample'):
    """sample synthesis"""
    sample = []
    for d in range(len(dims)):
        if d in cardinal_dims:
            sample.append(centroid[list(cardinal_dims).index(d)])
        else:
            if reduction == 'sample':
                sample.append(np.random.choice(sample_dims[d]))
            elif reduction == 'mode':
                sample.append(max(set(sample_dims[d]), key=sample_dims[d].count))

    return sample

def reduce_samples(samples, centroids, cluster, dims, cardinal_dims):
    """reduce samples to ones that subsumes the input samples"""
    reduced_samples = []
    for c in range(len(centroids)):
        members = [samples[s] for s in range(len(samples)) if cluster[s] == c]
        sample_dims = []
        if not members:
            sample_dims = [list(range(dims[d])) for d in range(len(dims))]
        else:
            sample_dims = [[s[d] for s in samples] for d in range(len(dims))]
        
        unique = False
        while unique is False:
            new_sample = synthesize_sample(centroids[c], sample_dims, dims, cardinal_dims)
            if new_sample not in reduced_samples:
                reduced_samples.append(new_sample)
                unique = True

    return reduced_samples
