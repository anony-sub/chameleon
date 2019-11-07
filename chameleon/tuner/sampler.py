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
"""Base class for sampler
This type of sampler will build an internal method to prune or reduce number
of configs measured on hardware to speed up tuning without performance loss.
"""
import numpy as np

from ..env import GLOBAL_SCOPE

class Sampler(object):
    """Base class for sampler
    This type of sampler will build an internal method to prune or reduce number
    of configs measured on hardware to speed up tuning without performance loss.

    Parameters
    ----------
    dims: list
        knob form of the dimensions for the configs
    """

    def __init__(self, dims):
        self.dims = dims

    def sample(self, xs):
        """Sample a subset of configs from a larger set of configs
        
        Parameters
        ----------
        xs: Array of int
            The indexes of configs from a larger set

        Returns
        -------
        a reduced set of configs
        """
        raise NotImplementedError()

