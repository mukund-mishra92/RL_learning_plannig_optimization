from dqn_agent import DQNAgent
from dqn_environment import DQNPathPlanningEnvironment
from config import get_dqn_config
import numpy as np

def dqn_path_planning_training_system(env=None, agent=None, num_episodes=None):
    config = get_dqn_config()
    state_size = config['state_size']
    action_size = config['action_size']
    num_episodes = num_episodes or config['num_episodes']
    max_steps = config['max_steps']

    env = env or DQNPathPlanningEnvironment(state_size, action_size)
    agent = agent or DQNAgent(state_size, action_size)

    episode_rewards = []
    for ep in range(num_episodes):
        state = env.reset()
        total_reward = 0
        for step in range(max_steps):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            state = next_state
            total_reward += reward
            if done:
                break
        episode_rewards.append(total_reward)
        if (ep+1) % 50 == 0:
            print(f"Episode {ep+1}/{num_episodes} | Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.3f}")
        if (ep+1) % 100 == 0:
            agent.update_target_network()
    return episode_rewards, agent