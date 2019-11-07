"""
Cost model optimizer based on reinforcement learning
"""

import heapq
import logging
import time

import numpy as np
import tensorflow as tf

from ..util import sample_ints
from .model_based_tuner import ModelOptimizer, knob2point, point2knob
from .reinforcement_learning import ppo_core

logger = logging.getLogger('autotvm')

class ReinforcementLearningOptimizer(ModelOptimizer):
    """parallel reinforcement learning optimization algorithm"""
    def __init__(self, task, n_iter=500, temp=(1, 0), persistent=False, parallel_size=128,
                 early_stop=50, log_interval=50):
        super(ReinforcementLearningOptimizer, self).__init__()

        self.task = task
        self.dims = [len(x) for x in self.task.config_space.space_map.values()]

        self.n_iter = n_iter
        self.temp = temp
        self.persistent = persistent
        self.parallel_size = min(parallel_size, len(self.task.config_space))
        self.early_stop = early_stop or 1e9
        self.log_interval = log_interval
        self.points = None

        self.obs_space = (len(self.dims), )
        self.act_space = [3] * len(self.dims)

        # NOTE this seems sometimes cause hang after few tasks...
        sess = tf.get_default_session()
        if sess != None:
            sess.close()
        tf.reset_default_graph()

        self.policy = ppo_core.PolicyWithValue(self.obs_space, self.act_space, 'policy')
        self.old_policy = ppo_core.PolicyWithValue(self.obs_space, self.act_space, 'old_policy')

        self.agent = ppo_core.PPOAgent(self.policy, self.old_policy,
                                       horizon=-1,
                                       learning_rate=1e-3,
                                       epochs=3,
                                       batch_size=32,
                                       gamma=0.9,
                                       lmbd=0.99,
                                       clip_value=0.2,
                                       value_coeff=1.0,
                                       entropy_coeff=0.3)

    def find_maximums(self, model, num, exclusive):
        temp, n_iter, early_stop, log_interval = \
                self.temp, self.n_iter, self.early_stop, self.log_interval

        start = time.time()

        if self.persistent and self.points is not None:
            points = self.points
        else:
            points = np.array(sample_ints(0, len(self.task.config_space), self.parallel_size))
        
        scores = model.predict(points)

        # build heap and insert initial points
        heap_items = [(float('-inf'), -i) for i in range(num)]
        heapq.heapify(heap_items)
        in_heap = set(exclusive)
        in_heap.update([-i for i in range(num)])

        for s, p in zip(scores, points):
            if s > heap_items[0][0] and p not in in_heap:
                pop = heapq.heapreplace(heap_items, (s, p))
                in_heap.remove(pop[1])
                in_heap.add(p)

        startstart = time.time()
        for i, p in enumerate(points):
            k = 0

            # create buffer which will be used to save trajectory
            new_points = []
            new_scores = []

            point = p
            good = True

            while k < n_iter and good == True:

                # action
                new_point, good = rl_walk(point, self.dims, self.agent)
                new_points.append(int(new_point))
                point = new_point
                
                k += 1
            
            pad = False
            if len(new_points) < 150:
                orig_len = len(new_points)
                pad = True
                new_points = new_points + [0] * (150 - orig_len)

            new_scores = model.predict(new_points)
            
            if pad == True:
                new_points = new_points[0:orig_len]
                new_scores = new_scores[0:orig_len]

            self.agent.observe_all_and_learn(new_scores) 
            
            # put points into heap
            for s, p in zip(new_scores, new_points):
                if s > heap_items[0][0] and p not in in_heap:
                    pop = heapq.heapreplace(heap_items, (s, p))
                    in_heap.remove(pop[1])
                    in_heap.add(p)

        heap_items.sort(key=lambda item: -item[0])

        if self.persistent:
            self.points = points

        return [x[1] for x in heap_items]

def rl_walk(p, dims, agent):
    n_dims = len(dims)
    n_dims_thres = n_dims * 0.75

    good_cnt = 0

    # transform to knob form
    old = point2knob(p, dims)
    new = list(old)

    # normalize input
    observation = [a / b for a, b, in zip(new, dims)]

    # get action
    action, value = agent.action(observation)
    
    # for each action, make movements
    for i, a in enumerate(action):
        if a == 0:
            # min value is 0
            new[i] = max(0, new[i] - 1)
            if new[i] == 0:
                good_cnt += 1
        elif a == 1:
            new[i] = new[i]
            good_cnt += 1
        elif a == 2:
            # if dims[i] is 10, max value is 9
            new[i] = min(dims[i] - 1, new[i] + 1)
            if new[i] == dims[i] - 1:
                good_cnt += 1
        else:
            pass

    if good_cnt < n_dims_thres:
        good = True
    else:
        good = False

    return knob2point(new, dims), good

def random_walk(p, dims):
    """random walk as local transition

    Parameters
    ----------
    p: int
        index of the ConfigEntity
    dims: Array of int
        sizes of each dimension

    Returns
    -------
    new_p: int
        new neighborhood index
    """
    # transform to knob form
    old = point2knob(p, dims)
    new = list(old)

    # mutate
    while new == old:
        from_i = np.random.randint(len(old))
        to_v = np.random.randint(dims[from_i])
        new[from_i] = to_v

    # transform to index form
    return knob2point(new, dims)
