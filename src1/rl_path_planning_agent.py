# 🧠 RL Path Planning Agent with Deadlock Awareness
"""
Advanced RL Agent for Path Planning with Deadlock Detection and Transfer Learning
"""

import numpy as np
import random
import pickle

class RLPathPlanningAgent:
    """
    Advanced RL Agent for Path Planning with Deadlock Detection and Transfer Learning
    
    This agent learns optimal routing strategies while avoiding deadlocks
    """
    
    def __init__(self, agent_id, state_size=12, action_size=7, 
                 learning_rate=0.1, discount_factor=0.95, exploration_rate=1.0):
        self.agent_id = agent_id
        self.state_size = state_size
        self.action_size = action_size
        
        # RL Hyperparameters for path planning
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = 0.995
        self.exploration_min = 0.01
        
        # Q-Table for path planning (state-action values)
        self.q_table = {}
        self.path_planning_memory = {}  # Store successful paths
        
        # Path Planning Performance Tracking
        self.episode_rewards = []
        self.path_efficiency_history = []
        self.route_discovery_count = 0
        self.successful_paths = {}  # Cache of learned optimal paths
        
        # Deadlock Detection Components
        self.deadlock_memory = []  # Store deadlock situations for learning
        self.conflict_resolution_patterns = {}
        
        # Transfer Learning Components
        self.transfer_learning_active = False
        self.base_knowledge = None
        self.fine_tuning_episodes = 0
        
        print(f"🧠 RL Path Planning Agent {agent_id} initialized with enhanced capabilities")
    
    def get_state_key(self, observation):
        """Convert observation to state key for Q-table lookup"""
        # Discretize continuous values for Q-table
        if len(observation) >= 8:
            # Primary path planning features (discretized)
            current_x = int(observation[0])
            current_y = int(observation[1])
            current_z = int(observation[2])
            target_x = int(observation[3])
            target_y = int(observation[4])
            target_z = int(observation[5])
            
            # Distance to target (discretized into bins)
            distance = observation[6]
            distance_bin = min(int(distance // 5), 20)  # Bin distances
            
            # Path efficiency (discretized)
            efficiency = observation[7]
            efficiency_bin = int(efficiency * 10)  # 0-10 scale
            
            # Basic state representation for path planning
            state_key = (current_x, current_y, current_z, 
                        target_x, target_y, target_z, 
                        distance_bin, efficiency_bin)
            
            return state_key
        else:
            # Fallback for incomplete observations
            return tuple(int(obs) for obs in observation[:min(len(observation), 6)])
    
    def choose_action(self, observation, training=True):
        """Choose action using epsilon-greedy with path planning bias"""
        state_key = self.get_state_key(observation)
        
        # Exploration vs Exploitation for path planning
        if training and random.random() < self.exploration_rate:
            # Exploration: Random action
            action = random.randint(0, self.action_size - 1)
        else:
            # Exploitation: Best known action for path planning
            if state_key in self.q_table:
                # Choose action with highest Q-value
                action = max(self.q_table[state_key], key=self.q_table[state_key].get)
            else:
                # Intelligent default for new states in path planning
                action = self.get_path_planning_heuristic_action(observation)
        
        # Apply transfer learning bias if active
        if self.transfer_learning_active and self.base_knowledge:
            action = self.apply_transfer_learning_bias(state_key, action)
        
        return action
    
    def get_path_planning_heuristic_action(self, observation):
        """Get heuristic action for path planning when no Q-value exists"""
        if len(observation) < 6:
            return random.randint(0, self.action_size - 1)
        
        # Calculate direction towards target
        current_pos = observation[:3]
        target_pos = observation[3:6]
        
        # Direction vector
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        dz = target_pos[2] - current_pos[2]
        
        # Choose action based on largest component (greedy path planning)
        abs_dx, abs_dy, abs_dz = abs(dx), abs(dy), abs(dz)
        
        if abs_dx >= abs_dy and abs_dx >= abs_dz:
            return 0 if dx > 0 else 1  # XP or XN
        elif abs_dy >= abs_dz:
            return 2 if dy > 0 else 3  # YP or YN
        else:
            return 4 if dz > 0 else 5  # ZP or ZN
    
    def apply_transfer_learning_bias(self, state_key, action):
        """Apply transfer learning knowledge to improve path planning"""
        if not self.base_knowledge or not isinstance(self.base_knowledge, dict):
            return action
        
        # Check if we have prior knowledge for similar states
        for known_state, q_values in self.base_knowledge.items():
            if self.states_similar(state_key, known_state):
                # Use knowledge from similar state
                best_action = max(q_values, key=q_values.get)
                # Blend with current action (exploration vs exploitation)
                if random.random() < 0.7:  # 70% chance to use transfer knowledge
                    return best_action
        
        return action
    
    def states_similar(self, state1, state2):
        """Check if two states are similar for transfer learning"""
        if len(state1) != len(state2):
            return False
        
        # States are similar if position and target are close
        position_similar = (abs(state1[0] - state2[0]) <= 2 and 
                          abs(state1[1] - state2[1]) <= 2)
        target_similar = (abs(state1[3] - state2[3]) <= 3 and 
                        abs(state1[4] - state2[4]) <= 3)
        
        return position_similar and target_similar
    
    def update_q_value(self, state, action, reward, next_state, done):
        """Update Q-value using Q-learning for path planning"""
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        # Initialize Q-table entries if they don't exist
        if state_key not in self.q_table:
            self.q_table[state_key] = {i: 0.0 for i in range(self.action_size)}
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = {i: 0.0 for i in range(self.action_size)}
        
        # Q-learning update rule for path planning
        current_q = self.q_table[state_key][action]
        
        if done:
            # Terminal state
            max_next_q = 0
        else:
            # Maximum Q-value for next state
            max_next_q = max(self.q_table[next_state_key].values())
        
        # Enhanced Q-learning with path planning focus
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state_key][action] = new_q
        
        # Store successful path patterns
        if reward > 10:  # Significant positive reward
            self.store_successful_path_pattern(state_key, action, reward)
        
        # Store deadlock patterns for avoidance
        if reward < -10:  # Significant negative reward (likely deadlock/collision)
            self.store_deadlock_pattern(state_key, action)
    
    def store_successful_path_pattern(self, state, action, reward):
        """Store successful path planning patterns for reuse"""
        pattern_key = (state[:6], action)  # Position + target + action
        
        if pattern_key not in self.successful_paths:
            self.successful_paths[pattern_key] = {
                'success_count': 1,
                'average_reward': reward,
                'total_reward': reward
            }
        else:
            pattern = self.successful_paths[pattern_key]
            pattern['success_count'] += 1
            pattern['total_reward'] += reward
            pattern['average_reward'] = pattern['total_reward'] / pattern['success_count']
    
    def store_deadlock_pattern(self, state, action):
        """Store deadlock patterns for avoidance learning"""
        deadlock_pattern = {
            'state': state,
            'action': action,
            'timestamp': len(self.deadlock_memory)
        }
        self.deadlock_memory.append(deadlock_pattern)
        
        # Keep memory manageable
        if len(self.deadlock_memory) > 1000:
            self.deadlock_memory = self.deadlock_memory[-800:]  # Keep recent patterns
    
    def detect_potential_deadlock(self, observation, other_agents_obs=None):
        """Detect potential deadlock situations in path planning"""
        if other_agents_obs is None or len(other_agents_obs) == 0:
            return False
        
        current_pos = observation[:3]
        
        # Check for head-to-head conflicts
        for other_obs in other_agents_obs:
            other_pos = other_obs[:3]
            
            # Calculate distance between agents
            distance = np.sqrt(sum((current_pos[i] - other_pos[i])**2 for i in range(3)))
            
            # If agents are very close and moving towards each other
            if distance < 2:
                # Check if they have conflicting targets or paths
                current_target = observation[3:6]
                other_target = other_obs[3:6]
                
                # Simple deadlock heuristic: targets are in opposite directions
                direction_to_target = np.array(current_target) - np.array(current_pos)
                other_direction = np.array(other_target) - np.array(other_pos)
                
                # Check if directions are roughly opposite (dot product < 0)
                dot_product = np.dot(direction_to_target, other_direction)
                if dot_product < -0.5:
                    return True
        
        return False
    
    def get_deadlock_avoidance_action(self, observation, other_agents_obs):
        """Get action that avoids detected deadlock"""
        # Find alternative actions that move away from conflict
        current_pos = observation[:3]
        
        # Calculate centroid of other agents
        if other_agents_obs:
            other_positions = [obs[:3] for obs in other_agents_obs]
            centroid = np.mean(other_positions, axis=0)
            
            # Move away from centroid
            away_direction = np.array(current_pos) - centroid
            
            # Choose action that moves in away direction
            abs_x, abs_y, abs_z = abs(away_direction[0]), abs(away_direction[1]), abs(away_direction[2])
            
            if abs_x >= abs_y and abs_x >= abs_z:
                return 0 if away_direction[0] > 0 else 1  # XP or XN
            elif abs_y >= abs_z:
                return 2 if away_direction[1] > 0 else 3  # YP or YN
            else:
                return 4 if away_direction[2] > 0 else 5  # ZP or ZN
        
        # Fallback: stay in place
        return 6
    
    def update_exploration_rate(self):
        """Update exploration rate for path planning learning"""
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay
    
    def save_path_planning_policy(self, filepath):
        """Save learned path planning policy and knowledge"""
        policy_data = {
            'agent_id': self.agent_id,
            'q_table': self.q_table,
            'successful_paths': self.successful_paths,
            'deadlock_memory': self.deadlock_memory,
            'episode_rewards': self.episode_rewards,
            'path_efficiency_history': self.path_efficiency_history,
            'route_discovery_count': self.route_discovery_count,
            'exploration_rate': self.exploration_rate,
            'learning_parameters': {
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor,
                'exploration_decay': self.exploration_decay
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(policy_data, f)
        
        print(f"📊 Path planning policy saved for agent {self.agent_id}")
        print(f"   🛣️ Q-table size: {len(self.q_table)} states")
        print(f"   ✅ Successful patterns: {len(self.successful_paths)}")
        print(f"   ⚠️ Deadlock patterns: {len(self.deadlock_memory)}")
    
    def load_path_planning_policy(self, filepath, transfer_learning=True):
        """Load path planning policy with optional transfer learning"""
        try:
            with open(filepath, 'rb') as f:
                policy_data = pickle.load(f)
            
            if transfer_learning:
                # Transfer learning mode: use as base knowledge
                self.base_knowledge = policy_data.get('q_table', {})
                self.transfer_learning_active = True
                
                # Inherit successful patterns
                loaded_patterns = policy_data.get('successful_paths', {})
                for pattern, data in loaded_patterns.items():
                    if pattern not in self.successful_paths:
                        self.successful_paths[pattern] = data
                
                # Learn from previous deadlock patterns
                self.deadlock_memory.extend(policy_data.get('deadlock_memory', []))
                
                print(f"🔄 Transfer learning activated for agent {self.agent_id}")
                print(f"   📚 Base knowledge: {len(self.base_knowledge)} states")
                print(f"   ✅ Inherited patterns: {len(loaded_patterns)}")
                
            else:
                # Direct loading: replace current policy
                self.q_table = policy_data.get('q_table', {})
                self.successful_paths = policy_data.get('successful_paths', {})
                self.deadlock_memory = policy_data.get('deadlock_memory', [])
                self.episode_rewards = policy_data.get('episode_rewards', [])
                self.exploration_rate = policy_data.get('exploration_rate', self.exploration_rate)
                
                print(f"📥 Policy loaded directly for agent {self.agent_id}")
                print(f"   🧠 Q-table: {len(self.q_table)} states")
                print(f"   📈 Episode history: {len(self.episode_rewards)} episodes")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load policy: {str(e)}")
            return False
    
    def fine_tune_mode(self, enable=True, episodes=100):
        """Enable/disable fine-tuning mode for transfer learning"""
        if enable:
            self.transfer_learning_active = True
            self.fine_tuning_episodes = episodes
            self.exploration_rate = max(0.2, self.exploration_rate)  # Increase exploration for fine-tuning
            print(f"🎯 Fine-tuning mode enabled for {episodes} episodes")
        else:
            self.transfer_learning_active = False
            self.fine_tuning_episodes = 0
            print(f"✅ Fine-tuning mode disabled")
    
    def get_path_planning_stats(self):
        """Get comprehensive path planning performance statistics"""
        if len(self.episode_rewards) == 0:
            return {}
        
        recent_rewards = self.episode_rewards[-50:] if len(self.episode_rewards) >= 50 else self.episode_rewards
        
        return {
            'total_episodes': len(self.episode_rewards),
            'average_reward': np.mean(self.episode_rewards),
            'recent_average_reward': np.mean(recent_rewards),
            'best_episode_reward': max(self.episode_rewards),
            'exploration_rate': self.exploration_rate,
            'q_table_size': len(self.q_table),
            'successful_patterns_learned': len(self.successful_paths),
            'deadlock_patterns_learned': len(self.deadlock_memory),
            'route_discoveries': self.route_discovery_count,
            'path_efficiency_trend': np.mean(self.path_efficiency_history[-10:]) if self.path_efficiency_history else 0,
            'transfer_learning_active': self.transfer_learning_active
        }

print("🧠 RL Path Planning Agent class defined with deadlock awareness and transfer learning!")