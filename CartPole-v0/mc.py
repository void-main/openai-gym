import gym
from gym import wrappers
import numpy as np
import sys
import os
from collections import defaultdict

record_name="/tmp/voidmain-cartpole-experiment-mc-1"
openai_apikey=os.environ["OPENAI_KEY"]

env = gym.make('CartPole-v0')
env = wrappers.Monitor(env, record_name, force=True)

env.reset()

def create_random_policy(nA):
    """
    Creates a random policy function.
    
    Args:
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes an observation as input and returns a vector
        of action probabilities
    """
    A = np.ones(nA, dtype=float) / nA
    def policy_fn(observation):
        return A
    return policy_fn


def create_greedy_policy(Q):
    """
    Creates a greedy policy based on Q values.
    
    Args:
        Q: A dictionary that maps from state -> action values
        
    Returns:
        A function that takes an observation as input and returns a vector
        of action probabilities.
    """
    
    def policy_fn(state_key):
        A = np.zeros_like(Q[state_key], dtype=float)
        best_action = np.argmax(Q[state_key])
        A[best_action] = 1.0
        return A
    return policy_fn

def state_to_key(state):
	return state.tostring()

def mc_control_importance_sampling(env, num_episodes, behavior_policy, discount_factor=1.0):
    """
    Monte Carlo Control Off-Policy Control using Weighted Importance Sampling.
    Finds an optimal greedy policy.
    
    Args:
        env: OpenAI gym environment.
        num_episodes: Nubmer of episodes to sample.
        behavior_policy: The behavior to follow while generating episodes.
            A function that given an observation returns a vector of probabilities for each action.
        discount_factor: Lambda discount factor.
    
    Returns:
        A tuple (Q, policy).
        Q is a dictionary mapping state -> action values.
        policy is a function that takes an observation as an argument and returns
        action probabilities. This is the optimal greedy policy.
    """
    
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    C = defaultdict(lambda: np.zeros(env.action_space.n))
    
    target_policy = create_greedy_policy(Q)
    
    for i in range(1, num_episodes + 1):
        if (i + 1) % 1000 == 0:
            print("Episode {}/{}.\n".format((i + 1), num_episodes), end="")
            sys.stdout.flush()
        
        episode = []
        state = env.reset()
        while True:
            probs = behavior_policy(state)
            action = np.random.choice(np.arange(len(probs)), p=probs)

            # render env
            env.render()
            next_state, reward, done, _ = env.step(action)

            episode.append((state, action, reward))
            if done:
                break
            state = next_state
        
        G = 0.0
        W = 1.0
        for t in range(len(episode))[::-1]:
            state, action, reward = episode[t]
            G = discount_factor * G + reward

            state_key = state_to_key(state)

            C[state_key][action] += W
            Q[state_key][action] += (W / C[state_key][action]) * (G - Q[state_key][action])
            if action !=  np.argmax(target_policy(state_key)):
                break
            W = W * 1./behavior_policy(state_key)[action]
        
    return Q, target_policy


random_policy = create_random_policy(env.action_space.n)
Q, policy = mc_control_importance_sampling(env, num_episodes=5000, behavior_policy=random_policy)

env.close()
gym.upload(record_name, api_key=openai_apikey)
