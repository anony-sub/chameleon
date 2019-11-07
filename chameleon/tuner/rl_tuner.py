# pylint: disable=abstract-method
"""Tuner that uses reinforcement learning"""

from .model_based_tuner import ModelBasedTuner, ModelOptimizer, CostModel
from .xgboost_cost_model import XGBoostCostModel
from .treernn_cost_model import TreeRNNCostModel
from .sa_model_optimizer import SimulatedAnnealingOptimizer
from .rl_model_optimizer import ReinforcementLearningOptimizer

class RLTuner(ModelBasedTuner):
    # rank --> reg because we need regression of the true value over mere relative rank
    def __init__(self, task, cost_model='xgb', rnn_params=None, plan_size=64,
                 feature_type='itervar', loss_type='reg', num_threads=None,
                 optimizer='rl', sampler=None, diversity_filter_ratio=None, log_interval=50):
        cost_model = XGBoostCostModel(task,
                                      feature_type=feature_type,
                                      loss_type=loss_type,
                                      num_threads=num_threads,
                                      log_interval=log_interval // 2)
        if optimizer == 'rl':
            optimizer = ReinforcementLearningOptimizer(task, log_interval=log_interval)
        else:
            assert isinstance(optimizer, ModelOptimizer), "Optimizer must be " \
                                                          "a supported name string" \
                                                          "or a ModelOptimizer object."
        if sampler == None:
            sampler = None
        elif sampler == 'adaptive':
            sampler = AdaptiveSampler(plan_size)
        else:
            assert isinstance(sampler, Sampler), "Sampler must be None," \
                                                 "a supported name string," \
                                                 "or a Sampler object."

        super(RLTuner, self).__init__(task, cost_model, optimizer,
                                       plan_size, diversity_filter_ratio)

    def tune(self, *args, **kwargs):  # pylint: disable=arguments-differ
        super(RLTuner, self).tune(*args, **kwargs)

        ## manually close pool to avoid multiprocessing issues
        #self.cost_model._close_pool()
        #self.model_optimizer.agent._close_session()
