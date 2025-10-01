
# ⚙️ Configuration File for DQN Path Planning System
"""
Configuration settings for the DQN-based Path Planning System
"""

# 📊 Training Configuration
TRAINING_CONFIG = {
    'num_episodes': 2000,
    'save_interval': 500,
    'max_steps_per_episode': 2000,
    'use_transfer_learning': True,
    'policy_directory': './dqn_path_planning_policies/',
    'results_directory': './results/'
}

# 🤖 Agent Configuration
AGENT_CONFIG = {
    'num_agents': 5,
    'state_size': 12,
    'action_size': 7,
    'learning_rate': 1e-3,
    'discount_factor': 0.99,
    'exploration_rate': 1.0,
    'exploration_decay': 0.995,
    'exploration_min': 0.01,
    'batch_size': 64
}

# 🗺️ Environment Configuration
ENVIRONMENT_CONFIG = {
    'max_steps': 1500,
    'warehouse_size': (10, 10),
    'obstacle_probability': 0.1,
    'reward_settings': {
        'step_penalty': -2,
        'collision_penalty': -15,
        'goal_reward': 50,
        'efficiency_bonus_multiplier': 20,
        'exploration_bonus': 3,
        'loop_penalty': -5,
        'critical_node_bonus': 2
    }
}

# 🔌 Database Configuration
DATABASE_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'database': 'neo-sim-noon-minutes',
    'table': 'location_master',
    'use_mock_data_on_failure': True
}

# 📊 Visualization Configuration
VISUALIZATION_CONFIG = {
    'save_plots': True,
    'figure_size': (20, 16),
    'dpi': 300,
    'plot_style': 'default',
    'color_scheme': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#7209B7']
}

def get_dqn_config(section=None):
    if section == 'training':
        return TRAINING_CONFIG
    elif section == 'agent':
        return AGENT_CONFIG
    elif section == 'environment':
        return ENVIRONMENT_CONFIG
    elif section == 'database':
        return DATABASE_CONFIG
    elif section == 'visualization':
        return VISUALIZATION_CONFIG
    else:
        return {
            'training': TRAINING_CONFIG,
            'agent': AGENT_CONFIG,
            'environment': ENVIRONMENT_CONFIG,
            'database': DATABASE_CONFIG,
            'visualization': VISUALIZATION_CONFIG
        }