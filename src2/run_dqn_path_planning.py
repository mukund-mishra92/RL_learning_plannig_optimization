import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from dqn_agent import DQNAgent
from dqn_environment import DQNPathPlanningEnvironment
from visualization import plot_dqn_training_results
from config import get_dqn_config
import numpy as np

def main():
    agent_cfg = get_dqn_config('agent')
    env_cfg = get_dqn_config('environment')
    num_agents = agent_cfg['num_agents']
    env = DQNPathPlanningEnvironment(state_size=agent_cfg['state_size'], action_size=agent_cfg['action_size'], num_agents=num_agents, max_steps=env_cfg['max_steps'])
    agents = [DQNAgent(agent_cfg['state_size'], agent_cfg['action_size']) for _ in range(num_agents)]
    num_episodes = get_dqn_config('training')['num_episodes']
    max_steps = env_cfg['max_steps']
    all_rewards = [[] for _ in range(num_agents)]

    for ep in range(num_episodes):
        states = env.reset()
        total_rewards = [0] * num_agents
        for step in range(max_steps):
            actions = [agents[i].act(states[i]) for i in range(num_agents)]
            next_states, rewards, dones, info = env.step(actions)
            for i in range(num_agents):
                agents[i].remember(states[i], actions[i], rewards[i], next_states[i], dones[i])
                agents[i].replay()
                total_rewards[i] += rewards[i]
            states = next_states
            if all(dones):
                break
        for i in range(num_agents):
            all_rewards[i].append(total_rewards[i])
        if (ep+1) % 50 == 0:
            print(f"Episode {ep+1}/{num_episodes} | Rewards: {[round(r,2) for r in total_rewards]} | Deadlocks: {info['deadlocks']}")
        if (ep+1) % 100 == 0:
            for agent in agents:
                agent.update_target_network()
    # Save models for transfer learning
    for i, agent in enumerate(agents):
        agent.save(f"dqn_agent_{i}_policy.pt")
    # Performance evaluation on random test data
    test_rewards = [[] for _ in range(num_agents)]
    for _ in range(50):
        states = env.reset()
        total_rewards = [0] * num_agents
        for step in range(max_steps):
            actions = [agents[i].act(states[i]) for i in range(num_agents)]
            next_states, rewards, dones, info = env.step(actions)
            for i in range(num_agents):
                total_rewards[i] += rewards[i]
            states = next_states
            if all(dones):
                break
        for i in range(num_agents):
            test_rewards[i].append(total_rewards[i])
    print("Test Rewards (mean per agent):", [round(np.mean(r),2) for r in test_rewards])
    # Plot training results
    for i in range(num_agents):
        plot_dqn_training_results(all_rewards[i])

if __name__ == "__main__":
    main()
