import matplotlib.pyplot as plt

def plot_dqn_training_results(episode_rewards):
    plt.figure(figsize=(10,5))
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('DQN Training Progress')
    plt.grid(True)
    plt.show()
