from ppo import *
import gym

# TODO PARAMETERS
render = False
#render = True
env = gym.make('CartPole-v1')
#env = gym.make('BipedalWalker-v2')

# NOTE input of the original OpenAI Gym Environment
print(len(env.observation_space.shape)) # Box
print(len(env.action_space.shape)) # Discrete

if len(env.action_space.shape) >= 1:
    obs_space = env.observation_space.shape
    act_space = env.action_space.shape
else:
    obs_space = env.observation_space.shape
    act_space = env.action_space
    
policy = PolicyWithValue(obs_space, act_space, 'policy')
old_policy = PolicyWithValue(obs_space, act_space, 'old_policy')

agent = PPOAgent(policy, old_policy, 
                 horizon=-1, 
                 learning_rate=1e-4, 
                 epochs=4, 
                 batch_size=64, 
                 gamma=0.95, 
                 lmbd=1.0, 
                 clip_value=0.2, 
                 value_coeff=1.0, 
                 entropy_coeff=0.01)

# Initialize the agent
for e in range(2000):
    
    # Initialize OpenAI Gym Environment
    observation = env.reset()

    for t in range(500):
        if render:
            env.render()

        # Query the agent for its action decision
        action, value  = agent.action(observation)
        #print(action, value)

        # Execute the decision and retrieve the current performance
        observation, reward, done, info = env.step(action)

        # Modify reward so that negative reward is given when it finishes too early

        # Pass feedback about performance (and termination) to the agent
        agent.observe_and_learn(reward=reward, terminal=done, score=t+1)

        if done:
            print("Episode {} finished after {} timesteps".format(e+1, t+1))
            break
