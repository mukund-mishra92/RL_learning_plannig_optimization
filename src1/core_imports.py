# 📚 Core Imports
"""
Central imports module for the RL Path Planning System
"""

# Core scientific computing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Database connectivity
import mysql.connector
import getpass

# Reinforcement Learning
import gymnasium as gym
from gymnasium import spaces

# Standard library
import random
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional, Any
import pickle
import os
import time
from datetime import datetime

# Graph algorithms
import networkx as nx

# Utilities
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

# Create directories
os.makedirs("trained_models", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("rl_path_planning_policies", exist_ok=True)

print("📚 All core imports loaded successfully!")
print("📁 Directory structure initialized!")

# Export commonly used items
__all__ = [
    'np', 'pd', 'plt', 'sns', 'mysql', 'getpass', 'gym', 'spaces',
    'random', 'deque', 'defaultdict', 'Dict', 'List', 'Tuple', 'Optional', 'Any',
    'pickle', 'os', 'time', 'datetime', 'nx', 'warnings'
]