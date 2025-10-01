# ⚙️ Configuration File for RL Path Planning System
"""
Configuration settings for the RL-based Path Planning System
"""

# 📊 Training Configuration
TRAINING_CONFIG = {
    'num_episodes': 2000,
    'save_interval': 500,
    'max_steps_per_episode': 2000,
    'use_transfer_learning': True,
    'policy_directory': './rl_path_planning_policies/',
    'results_directory': './results/'
}

# 🤖 Agent Configuration
AGENT_CONFIG = {
    'num_agents': 5,
    'state_size': 12,
    'action_size': 7,
    'learning_rate': 0.1,
    'discount_factor': 0.95,
    'exploration_rate': 1.0,
    'exploration_decay': 0.995,
    'exploration_min': 0.01
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

# 🧪 Testing Configuration
TESTING_CONFIG = {
    'num_test_episodes': 50,
    'evaluation_mode': True,  # No exploration during testing
    'test_metrics': ['completion_rate', 'path_efficiency', 'collision_count', 'average_reward']
}

# 🔄 Transfer Learning Configuration
TRANSFER_LEARNING_CONFIG = {
    'enabled': True,
    'fine_tuning_episodes': 100,
    'similarity_threshold': 0.7,
    'knowledge_inheritance_rate': 0.7,
    'base_knowledge_weight': 0.3
}

# 📁 File Paths
PATHS = {
    'models': './trained_models/',
    'policies': './rl_path_planning_policies/',
    'results': './results/',
    'logs': './logs/',
    'visualizations': './visualizations/'
}

# 🚨 Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_logging': True,
    'console_logging': True
}

# 🎯 Performance Thresholds
PERFORMANCE_THRESHOLDS = {
    'min_completion_rate': 0.8,  # 80% completion rate target
    'min_path_efficiency': 0.7,   # 70% path efficiency target
    'max_collision_rate': 0.1,    # Max 10% collision rate
    'convergence_episodes': 100    # Episodes to check for convergence
}

def get_config(config_name):
    """
    Get configuration by name
    
    Args:
        config_name (str): Name of configuration section
        
    Returns:
        dict: Configuration dictionary
    """
    config_map = {
        'training': TRAINING_CONFIG,
        'agent': AGENT_CONFIG,
        'environment': ENVIRONMENT_CONFIG,
        'database': DATABASE_CONFIG,
        'visualization': VISUALIZATION_CONFIG,
        'testing': TESTING_CONFIG,
        'transfer_learning': TRANSFER_LEARNING_CONFIG,
        'paths': PATHS,
        'logging': LOGGING_CONFIG,
        'thresholds': PERFORMANCE_THRESHOLDS
    }
    
    return config_map.get(config_name.lower(), {})

def print_all_configs():
    """Print all configuration settings"""
    print("⚙️ RL PATH PLANNING SYSTEM CONFIGURATION")
    print("="*50)
    
    configs = [
        ('Training', TRAINING_CONFIG),
        ('Agent', AGENT_CONFIG),
        ('Environment', ENVIRONMENT_CONFIG),
        ('Database', DATABASE_CONFIG),
        ('Visualization', VISUALIZATION_CONFIG),
        ('Testing', TESTING_CONFIG),
        ('Transfer Learning', TRANSFER_LEARNING_CONFIG),
        ('Paths', PATHS),
        ('Performance Thresholds', PERFORMANCE_THRESHOLDS)
    ]
    
    for name, config in configs:
        print(f"\n📊 {name} Configuration:")
        for key, value in config.items():
            print(f"   {key}: {value}")

if __name__ == "__main__":
    print_all_configs()