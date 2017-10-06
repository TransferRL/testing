import gym
# Wiki:
# https://github.com/openai/gym/wiki/CartPole-v0
# Environment page:
# https://gym.openai.com/envs/CartPole-v0

'''
Learning: 

1. get OpenAI env
2. reset
3. render
4. different types of spaces
    - Discrete
    - Box
5. action_space, observation space
6. step, and what is returns


'''



# To get the environment:
env = gym.make('CartPole-v0')

# run 20 episodes
for i_episode in range(20):

    # env.reset() puts yourself at the starting state. The return value is the state.
    observation = env.reset()

    # what do the state variables mean?
    # Num Observation Min Max
    # 0 Cart Position -2.4  2.4
    # 1 Cart Velocity -Inf  Inf
    # 2 Pole Angle  ~ -41.8°  ~ 41.8°
    # 3 Pole Velocity At Tip  -Inf  Inf

    # for each episode, run up to 100 timestamps.
    for t in range(100):

        # render the environment. The window should pop up here.
        env.render()

        # action is of type Discrete, which allows a fixed range of non-negative numbers (in cartpole, 0 or 1)
        # action -> Discrete(2)
        # These are the things we can do:
        # env.action_space.contains       env.action_space.n              env.action_space.to_jsonable
        # env.action_space.from_jsonable  env.action_space.sample
        action = env.action_space.sample() #sampling random actions from the action space

        # box is of type Box, which is like an array of observations
        # box -> Box(4, )
        # Things we can do with observation space:
        # box.contains       box.high           box.sample         box.to_jsonable
        # box.from_jsonable  box.low            box.shape
        box = env.observation_space

        # env.step does an action
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

