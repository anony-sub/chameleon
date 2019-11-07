import tensorflow as tf
import numpy as np
import copy
import time

"""
act_space will be represented using list.
5 dimension of 12, 3, 5, 6, 8 will be represented by 
"""

class PolicyWithValue:
    def __init__(self, observation_space, action_space, name, temp=0.1):
        
        self.ob_space = observation_space
        self.act_space = action_space

        with tf.variable_scope(name):
            self.obs = tf.placeholder(dtype=tf.float32, shape=[None] + list(self.ob_space), name='observation')
            
            factor = 1.0

            with tf.variable_scope('policy_net'):
                layer_1 = tf.layers.dense(inputs=self.obs, units=20*factor, activation=tf.tanh)
                layer_2 = tf.layers.dense(inputs=layer_1, units=20*factor, activation=tf.tanh)
                
                self.act_probs = []
                for i in range(len(self.act_space)):
                    layer_3 = tf.layers.dense(inputs=layer_2, units=self.act_space[i], activation=tf.tanh)
                    layer_4 = tf.layers.dense(inputs=layer_3, units=self.act_space[i], activation=tf.tanh)
                    self.act_probs.append(tf.layers.dense(inputs=tf.divide(layer_4, temp), units=self.act_space[i], activation=tf.nn.softmax))

            with tf.variable_scope('value_net'):
                layer_1 = tf.layers.dense(inputs=self.obs, units=20*factor, activation=tf.tanh)
                layer_2 = tf.layers.dense(inputs=layer_1, units=20*factor, activation=tf.tanh)
                layer_3 = tf.layers.dense(inputs=layer_2, units=20*factor, activation=tf.tanh)
                self.v_preds = tf.layers.dense(inputs=layer_3, units=1, activation=None)
            
            # for stochastic
            self.act_stochastic = []
            for i in range(len(self.act_space)):
                act_multinomial = tf.multinomial(tf.log(self.act_probs[i]), num_samples=1)
                self.act_stochastic.append(tf.reshape(act_multinomial, shape=[-1]))
            
            # for deterministic
            self.act_deterministic = []
            for i in range(len(self.act_space)):
                self.act_deterministic.append(tf.argmax(self.act_probs[i], axis=1))

            self.scope = tf.get_variable_scope().name
    
    def _get_action(self, sess, obs, stochastic=False):
        if stochastic:
            return sess.run([self.act_stochastic, self.v_preds], feed_dict={self.obs: obs})
        else:
            return sess.run([self.act_deterministic, self.v_preds], feed_dict={self.obs: obs})
    
    def _get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

class PPOAgent:
    def __init__(self, policy, old_policy, horizon, learning_rate, epochs, 
                 batch_size, gamma, lmbd, clip_value, value_coeff, entropy_coeff):

        print('open session')
        # FIXME no gpu
        self.sess = tf.Session()
        #self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        current_time = int(time.time())
        self.writer = tf.summary.FileWriter('./log/train_'+str(current_time), self.sess.graph)

        self.policy = policy
        self.old_policy = old_policy

        self.horizon = horizon
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.gamma = gamma
        self.lmbd = lmbd

        print('horizon       : {}'.format(self.horizon))
        print('batch_size    : {}'.format(self.batch_size))
        print('epochs        : {}'.format(self.epochs))
        print('learning_rate : {}'.format(self.learning_rate))

        print('gamma         : {}'.format(self.gamma))
        print('lambda        : {}'.format(self.lmbd))

        self.iteration = 0
        self.list_observations = []
        self.list_actions = []
        self.list_v_preds = []
        self.list_rewards = []

        pi_trainable = self.policy._get_trainable_variables()
        old_pi_trainable = self.old_policy._get_trainable_variables()
        
        # assignment operation to update old_policy with policy
        with tf.variable_scope('assign_op'):
            self.assign_ops = []
            for v_old, v in zip(old_pi_trainable, pi_trainable):
                self.assign_ops.append(tf.assign(v_old, v))

        # inputs for train operation
        with tf.variable_scope('train_input'):
            # discrete action space --> dtype=tf.int32
            # multidimensional action space --> shape=[None, len(???)]
            self.actions = tf.placeholder(dtype=tf.int32, shape=[None, len(self.policy.act_space)], name='actions')
            self.rewards = tf.placeholder(dtype=tf.float32, shape=[None], name='rewards')
            self.v_preds_next = tf.placeholder(dtype=tf.float32, shape=[None], name='v_preds_next')
            self.gaes = tf.placeholder(dtype=tf.float32, shape=[None], name='gaes')

        loss = []
        for i in range(len(self.policy.act_space)):
            act_probs = self.policy.act_probs[i]
            act_probs_old = self.old_policy.act_probs[i]
            
            # probability of actions chosen with the policy
            act_probs = act_probs * tf.one_hot(indices=self.actions[:, i], depth=act_probs.shape[1])
            act_probs = tf.reduce_sum(act_probs, axis=1)

            # probabilities of actions which agent took with old policy
            act_probs_old = act_probs_old * tf.one_hot(indices=self.actions[:, i], depth=act_probs_old.shape[1])
            act_probs_old = tf.reduce_sum(act_probs_old, axis=1)

            # clipped surrogate objective (7)
            # TODO adaptive KL penalty coefficient can be added (8)
            with tf.variable_scope('loss/clip'):
                ratios = tf.exp(tf.log(act_probs) - tf.log(act_probs_old))
                clipped_ratios = tf.clip_by_value(ratios, clip_value_min=1-clip_value, clip_value_max=1+clip_value)
                loss_clip = tf.minimum(tf.multiply(self.gaes, ratios), tf.multiply(self.gaes, clipped_ratios))
                loss_clip = tf.reduce_mean(loss_clip)
                tf.summary.scalar('loss_clip', loss_clip)

            # entropy bonus (9)
            with tf.variable_scope('loss/entropy'):
                entropy = -tf.reduce_sum(self.policy.act_probs[i] * 
                                         tf.log(tf.clip_by_value(self.policy.act_probs[i], 1e-10, 1.0)), axis=1)
                entropy = tf.reduce_mean(entropy, axis=0)
                tf.summary.scalar('entropy', entropy)

            with tf.variable_scope('loss'):
                loss.append(loss_clip + entropy_coeff * entropy)
        
        # squared difference between value (9)
        with tf.variable_scope('loss/value'):
            v_preds = self.policy.v_preds
            loss_v = tf.squared_difference(self.rewards + self.gamma * self.v_preds_next, v_preds)
            loss_v = tf.reduce_mean(loss_v)
            tf.summary.scalar('loss_value', loss_v)
        
        # loss (9)
        with tf.variable_scope('loss'):
            # c_1 = value_coeff, c_2 = entropy_coeff
            # loss = \sum ((clipped loss) + c_2 * (entropy bonus)) - c_1 * (value loss)
            loss = tf.add_n(loss) - value_coeff * loss_v
            # loss : up
            # clipped loss : up
            # value loss : down
            # entropy : up
            tf.summary.scalar('loss', loss)
            
        # gradient ascent using adam optimizer
        loss = -loss
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-5)
        self.train_op = optimizer.minimize(loss, var_list=pi_trainable)
        
        self.sess.run(tf.global_variables_initializer())
        
        self.merge_op = tf.summary.merge_all()

    def action(self, obs, stochastic=True):
        obs = np.stack([obs]).astype(dtype=np.float32)
        act, v_pred = self.policy._get_action(sess=self.sess, obs=obs, stochastic=stochastic)
        
        act = [np.asscalar(a) for a in act]
        v_pred = np.asscalar(v_pred)
        #FIXME
        #print(act, v_pred)

        self.list_observations.append(obs)
        self.list_actions.append(act)
        self.list_v_preds.append(v_pred)

        return act, v_pred

    def observe_all_and_learn(self, rewards):
        self.list_rewards = rewards
        
        # make v_preds_next from v_preds
        self.list_v_preds_next = self.list_v_preds[1:] + [0]
        
        #print('gaes')
        # get generalized advantage estimations
        self.list_gaes = self._get_gaes(self.list_rewards, 
                                          self.list_v_preds, 
                                          self.list_v_preds_next)

        # make list_* into numpy array to feed to placeholders
        np_observations = np.reshape(self.list_observations, newshape=[-1] + list(self.policy.ob_space))
        np_actions = np.array(self.list_actions).astype(dtype=np.int32)
        np_rewards = np.array(self.list_rewards).astype(dtype=np.float32)
        np_v_preds_next = np.array(self.list_v_preds_next).astype(dtype=np.float32)
        np_gaes = np.array(self.list_gaes).astype(dtype=np.float32)
        np_gaes = (np_gaes - np_gaes.mean()) / np_gaes.std()

        input_samples = [np_observations, np_actions, np_rewards, np_v_preds_next, np_gaes]
        
        # update old policy with current policy
        self._update_old_policy()
        
        # sample horizon
        if self.horizon != -1:
            horizon_indices = np.random.randint(low=0, high=np_observations.shape[0], size=self.horizon)
            horizon_samples = [np.take(a=input_sample, indices=horizon_indices, axis=0) for input_sample in input_samples]

        #print('learn')
        # learn
        for epoch in range(self.epochs):
            #print(epoch)

            # sample batch
            if self.horizon != -1:
                batch_indices = np.random.randint(low=0, high=self.horizon, size=self.batch_size)
                batch_samples = [np.take(a=input_sample, indices=batch_indices, axis=0) for input_sample in horizon_samples]
            else:
                batch_indices = np.random.randint(low=0, high=np_observations.shape[0], size=self.batch_size)
                batch_samples = [np.take(a=input_sample, indices=batch_indices, axis=0) for input_sample in input_samples]

            self._learn(observations=batch_samples[0], 
                        actions=batch_samples[1], 
                        rewards=batch_samples[2], 
                        v_preds_next=batch_samples[3], 
                        gaes=batch_samples[4])
        
        summary = self._record(observations=batch_samples[0], 
                               actions=batch_samples[1], 
                               rewards=batch_samples[2], 
                               v_preds_next=batch_samples[3], 
                               gaes=batch_samples[4])[0]
        
        self.writer.add_summary(summary, self.iteration)
        
        self.iteration += 1
           
        self.list_observations = []
        self.list_actions = []
        self.list_v_preds = []
        self.list_rewards = []

    def observe_and_learn(self, reward, terminal, score=False):
        self.list_rewards.append(reward)
        
        if terminal == False:
            # if have not reached end of the episode yet, wait
            return
        else:
            # if have reached end of the episode, train

            # make v_preds_next from v_preds
            self.list_v_preds_next = self.list_v_preds[1:] + [0]
            
            # get generalized advantage estimations
            self.list_gaes = self._get_gaes(self.list_rewards, 
                                              self.list_v_preds, 
                                              self.list_v_preds_next)

            # make list_* into numpy array to feed to placeholders
            np_observations = np.reshape(self.list_observations, newshape=[-1] + list(self.policy.ob_space))
            np_actions = np.array(self.list_actions).astype(dtype=np.int32)
            np_rewards = np.array(self.list_rewards).astype(dtype=np.float32)
            np_v_preds_next = np.array(self.list_v_preds_next).astype(dtype=np.float32)
            np_gaes = np.array(self.list_gaes).astype(dtype=np.float32)
            np_gaes = (np_gaes - np_gaes.mean()) / np_gaes.std()

            input_samples = [np_observations, np_actions, np_rewards, np_v_preds_next, np_gaes]
            
            # update old policy with current policy
            self._update_old_policy()
            
            # sample horizon
            if self.horizon != -1:
                horizon_indices = np.random.randint(low=0, high=np_observations.shape[0], size=self.horizon)
                horizon_samples = [np.take(a=input_sample, indices=horizon_indices, axis=0) for input_sample in input_samples]

            # learn
            for epoch in range(self.epochs):
                # sample batch
                if self.horizon != -1:
                    batch_indices = np.random.randint(low=0, high=self.horizon, size=self.batch_size)
                    batch_samples = [np.take(a=input_sample, indices=batch_indices, axis=0) for input_sample in horizon_samples]
                else:
                    batch_indices = np.random.randint(low=0, high=np_observations.shape[0], size=self.batch_size)
                    batch_samples = [np.take(a=input_sample, indices=batch_indices, axis=0) for input_sample in input_samples]

                self._learn(observations=batch_samples[0], 
                            actions=batch_samples[1], 
                            rewards=batch_samples[2], 
                            v_preds_next=batch_samples[3], 
                            gaes=batch_samples[4])
            
            summary = self._record(observations=batch_samples[0], 
                                   actions=batch_samples[1], 
                                   rewards=batch_samples[2], 
                                   v_preds_next=batch_samples[3], 
                                   gaes=batch_samples[4])[0]
        
            self.writer.add_summary(summary, self.iteration)
            self.writer.add_summary(tf.Summary(value=[
                                    tf.Summary.Value(tag='score', 
                                                     simple_value=score)]), 
                                    self.iteration)
            
            self.iteration += 1
               
            self.list_observations = []
            self.list_actions = []
            self.list_v_preds = []
            self.list_rewards = []

    def _learn(self, observations, actions, rewards, v_preds_next, gaes):
        self.sess.run([self.train_op], feed_dict={self.policy.obs: observations,
                                                  self.old_policy.obs: observations,
                                                  self.actions: actions,
                                                  self.rewards: rewards,
                                                  self.v_preds_next: v_preds_next,
                                                  self.gaes: gaes})

    def _record(self, observations, actions, rewards, v_preds_next, gaes):
        return self.sess.run([self.merge_op], feed_dict={self.policy.obs: observations,
                                                         self.old_policy.obs: observations,
                                                         self.actions: actions,
                                                         self.rewards: rewards,
                                                         self.v_preds_next: v_preds_next,
                                                         self.gaes: gaes})

    def _get_gaes(self, rewards, v_preds, v_preds_next):
        deltas = [r + self.gamma * v_next - v for r, v_next, v in zip(rewards, v_preds_next, v_preds)]
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(gaes) - 1)):
            gaes[t] = gaes[t] + self.gamma * self.lmbd * gaes[t+1]

        return gaes

    def _update_old_policy(self):
        self.sess.run(self.assign_ops)

    def _close_session(self):
        print('reset graph')
        tf.reset_default_graph()
        print('close session')
        self.sess.close()
