# 🛣️ RL Path Planning System - Modular Package
"""
Advanced RL-Based Path Planning with Deadlock-Aware Multi-Agent System

This package implements a comprehensive Reinforcement Learning Path Planning system 
for warehouse navigation with integrated deadlock detection and avoidance capabilities.

Key Features:
- 🛣️ RL-Based Path Planning: Agents learn optimal routes through reinforcement learning
- 🤖 Multi-Agent Coordination: Multiple robots planning paths simultaneously  
- 🚫 Integrated Deadlock Detection: Real-time conflict prevention during path planning
- 🔄 Transfer Learning: Knowledge transfer for faster path learning in new layouts
- 📊 Dynamic Route Optimization: Continuous improvement of navigation strategies
- 🎯 Goal-Oriented Navigation: RL agents learn to reach destinations efficiently

Author: RL Path Planning System
Date: September 2025
"""

__version__ = "1.0.0"
__author__ = "RL Path Planning Team"

# Import main components for easy access
from .database_connector import connect_to_warehouse_database, load_location_data, create_mock_location_data
from .path_planning_environment import RLPathPlanningEnvironment
from .rl_path_planning_agent import RLPathPlanningAgent
from .training_system import rl_path_planning_training_system
from .visualization import plot_rl_path_planning_results
from .pipeline import execute_complete_rl_path_planning_pipeline, test_rl_path_planning_policies
from .config import get_config, print_all_configs

__all__ = [
    'connect_to_warehouse_database',
    'load_location_data', 
    'create_mock_location_data',
    'RLPathPlanningEnvironment',
    'RLPathPlanningAgent',
    'rl_path_planning_training_system',
    'plot_rl_path_planning_results',
    'execute_complete_rl_path_planning_pipeline',
    'test_rl_path_planning_policies',
    'get_config',
    'print_all_configs'
]

print("🚀 RL Path Planning System package loaded successfully!")