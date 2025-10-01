from dqn_training_system import dqn_path_planning_training_system
from visualization import plot_dqn_training_results

def execute_complete_dqn_path_planning_pipeline():
    episode_rewards, agent = dqn_path_planning_training_system()
    plot_dqn_training_results(episode_rewards)
    return episode_rewards, agent
