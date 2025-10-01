# 🛣️ RL-Based Path Planning Multi-Agent Environment
"""
Reinforcement Learning-based Path Planning Environment for Multi-Agent Warehouse Navigation
"""

import numpy as np
import pandas as pd
import networkx as nx
import random
from gymnasium import spaces

class RLPathPlanningEnvironment:
    """
    Reinforcement Learning-based Path Planning Environment for Multi-Agent Warehouse Navigation
    
    This environment enables agents to learn optimal paths through RL while avoiding conflicts
    """
    
    def __init__(self, location_df, num_agents=3, max_steps=1000):
        self.location_df = location_df.copy()
        self.num_agents = num_agents
        self.max_steps = max_steps
        
        # Setup warehouse map and path planning components
        self.setup_warehouse_map()
        self.build_path_planning_graph()
        
        # Agent states for path planning
        self.agents = {}
        self.step_count = 0
        
        # Action space: 0=XP, 1=XN, 2=YP, 3=YN, 4=ZP, 5=ZN, 6=STAY
        self.action_space = spaces.Discrete(7)
        
        # Enhanced observation space for path planning
        # [current_x, current_y, current_z, target_x, target_y, target_z, 
        #  distance_to_target, path_efficiency, other_agents...]
        obs_size = 8 + (num_agents - 1) * 4  # Enhanced state for path planning
        self.observation_space = spaces.Box(low=-100, high=1000, 
                                          shape=(obs_size,), dtype=np.float32)
        
        # Path planning metrics
        self.path_efficiency_history = []
        self.route_discovery_stats = {}
        
    def setup_warehouse_map(self):
        """Setup warehouse map with enhanced path planning features"""
        self.locations = {}
        self.location_types = {}
        self.movement_matrix = {}
        self.coordinates = {}  # Enhanced coordinate tracking
        
        for idx, row in self.location_df.iterrows():
            if 'location_id' in row:
                location_id = int(row['location_id'])
            else:
                location_id = int(row.iloc[0])
            
            # Get coordinates with enhanced precision
            x = float(row.get('x', row.iloc[1]))
            y = float(row.get('y', row.iloc[2]))
            z = float(row.get('z', row.iloc[3]) if len(row) > 3 else 0)
            
            # Store coordinate mapping for path planning
            self.coordinates[(x, y, z)] = location_id
            
            location_type = row.get('type', row.iloc[4] if len(row) > 4 else 'PATH')
            
            # Movement possibilities for path planning
            moves = {
                'XP': int(row.get('XP', row.iloc[5] if len(row) > 5 else 0)),
                'XN': int(row.get('XN', row.iloc[6] if len(row) > 6 else 0)),
                'YP': int(row.get('YP', row.iloc[7] if len(row) > 7 else 0)),
                'YN': int(row.get('YN', row.iloc[8] if len(row) > 8 else 0)),
                'ZP': int(row.get('ZP', row.iloc[9] if len(row) > 9 else 0)),
                'ZN': int(row.get('ZN', row.iloc[10] if len(row) > 10 else 0))
            }
            
            self.locations[location_id] = {
                'x': x, 'y': y, 'z': z, 
                'type': location_type,
                'connectivity': sum(moves.values())  # Path planning connectivity score
            }
            self.movement_matrix[location_id] = moves
            self.location_types[location_id] = location_type
        
        # Build adjacency matrix for path planning algorithms
        self.build_adjacency_matrix()
        
        print(f"🗺️ Path Planning Environment initialized:")
        print(f"   📍 {len(self.locations)} locations")
        print(f"   🛣️ Average connectivity: {np.mean([loc['connectivity'] for loc in self.locations.values()]):.1f}")
        print(f"   📊 Location types: {set(self.location_types.values())}")
    
    def build_path_planning_graph(self):
        """Build enhanced graph for RL path planning"""
        self.path_graph = nx.Graph()
        
        # Add nodes with path planning attributes
        for loc_id, loc_data in self.locations.items():
            self.path_graph.add_node(loc_id, 
                                   x=loc_data['x'], 
                                   y=loc_data['y'], 
                                   z=loc_data['z'],
                                   type=loc_data['type'],
                                   connectivity=loc_data['connectivity'])
        
        # Add weighted edges for path planning
        for loc_id, moves in self.movement_matrix.items():
            for move_dir, possible in moves.items():
                if possible == 1:
                    target_loc = self.find_adjacent_location(loc_id, move_dir)
                    if target_loc is not None:
                        # Calculate edge weight for path planning (Euclidean distance)
                        loc1 = self.locations[loc_id]
                        loc2 = self.locations[target_loc]
                        weight = np.sqrt((loc1['x'] - loc2['x'])**2 + 
                                       (loc1['y'] - loc2['y'])**2 + 
                                       (loc1['z'] - loc2['z'])**2)
                        self.path_graph.add_edge(loc_id, target_loc, weight=weight)
        
        print(f"🛣️ Path Planning Graph built: {len(self.path_graph.edges)} navigable connections")
        
        # Pre-compute path planning heuristics (optimized for large graphs)
        self.precompute_path_heuristics()
    
    def precompute_path_heuristics(self):
        """Pre-compute path planning heuristics for RL training (optimized)"""
        print("⚡ Optimizing graph for fast training startup...")
        
        # Skip expensive centrality calculation for large graphs
        # Use degree centrality as a faster approximation
        if len(self.path_graph.nodes) > 50000:
            print("   📊 Using degree centrality (fast) instead of betweenness centrality")
            degree_centrality = nx.degree_centrality(self.path_graph)
            self.centrality_scores = degree_centrality
        else:
            print("   📊 Computing betweenness centrality for optimal path planning")
            self.centrality_scores = nx.betweenness_centrality(self.path_graph)
        
        self.shortest_path_cache = {}  # Cache for frequently used paths
        
        # Identify critical nodes for path planning
        self.critical_nodes = [node for node, centrality in self.centrality_scores.items() 
                              if centrality > 0.1]
        
        print(f"🎯 Path Planning Heuristics computed:")
        print(f"   🔥 {len(self.critical_nodes)} critical navigation nodes identified")
    
    def build_adjacency_matrix(self):
        """Build adjacency matrix for path planning"""
        self.adjacency = {}
        
        for loc_id, moves in self.movement_matrix.items():
            self.adjacency[loc_id] = []
            
            for move_dir, possible in moves.items():
                if possible == 1:
                    target_loc = self.find_adjacent_location(loc_id, move_dir)
                    if target_loc is not None:
                        self.adjacency[loc_id].append(target_loc)
    
    def find_adjacent_location(self, current_id, direction):
        """Find adjacent location for path planning"""
        current = self.locations[current_id]
        
        directions = {
            'XP': (1, 0, 0), 'XN': (-1, 0, 0),
            'YP': (0, 1, 0), 'YN': (0, -1, 0),
            'ZP': (0, 0, 1), 'ZN': (0, 0, -1)
        }
        
        if direction not in directions:
            return None
            
        dx, dy, dz = directions[direction]
        target_x = current['x'] + dx
        target_y = current['y'] + dy
        target_z = current['z'] + dz
        
        # Find location with these coordinates
        for loc_id, loc_data in self.locations.items():
            if (abs(loc_data['x'] - target_x) < 0.1 and 
                abs(loc_data['y'] - target_y) < 0.1 and 
                abs(loc_data['z'] - target_z) < 0.1):
                return loc_id
        
        return None
    
    def reset(self):
        """Reset environment for new path planning episode"""
        self.step_count = 0
        self.agents = {}
        
        # Initialize agents with path planning objectives
        available_locations = [loc for loc, loc_type in self.location_types.items() 
                             if loc_type != 'OBSTACLE']
        
        start_locations = random.sample(available_locations, min(self.num_agents, len(available_locations)))
        
        for i in range(self.num_agents):
            # Assign challenging but reachable targets for path learning
            possible_targets = [loc for loc in available_locations 
                              if loc != start_locations[i] and 
                              self.euclidean_distance(start_locations[i], loc) > 2]
            
            if not possible_targets:
                possible_targets = [loc for loc in available_locations if loc != start_locations[i]]
            
            target = random.choice(possible_targets) if possible_targets else start_locations[i]
            
            # Use fast Euclidean distance as optimal path approximation
            # This removes the expensive NetworkX shortest_path calculation
            optimal_path_length = self.euclidean_distance(start_locations[i], target)
            
            self.agents[f'agent_{i}'] = {
                'current_location': start_locations[i],
                'target_location': target,
                'start_location': start_locations[i],
                'path_history': [start_locations[i]],
                'completed': False,
                'collisions': 0,
                'steps_taken': 0,
                'optimal_path_length': optimal_path_length,
                'actual_path_length': 0,
                'path_efficiency': 1.0,
                'route_learning_progress': 0.0
            }
        
        return self.get_path_planning_observations()
    
    def get_path_planning_observations(self):
        """Get enhanced observations for RL path planning"""
        observations = {}
        
        for agent_id, agent_data in self.agents.items():
            current_loc = self.locations[agent_data['current_location']]
            target_loc = self.locations[agent_data['target_location']]
            
            # Enhanced observation for path planning
            obs = [
                current_loc['x'],           # Current position
                current_loc['y'],
                current_loc['z'],
                target_loc['x'],            # Target position
                target_loc['y'],
                target_loc['z'],
                self.euclidean_distance(    # Distance to target
                    agent_data['current_location'], 
                    agent_data['target_location']
                ),
                agent_data['path_efficiency']  # Current path efficiency
            ]
            
            # Add other agents' positions and targets for coordination
            for other_id, other_data in self.agents.items():
                if other_id != agent_id:
                    other_current = self.locations[other_data['current_location']]
                    other_target = self.locations[other_data['target_location']]
                    obs.extend([
                        other_current['x'], other_current['y'], 
                        other_target['x'], other_target['y']
                    ])
            
            observations[agent_id] = np.array(obs, dtype=np.float32)
        
        return observations
    
    def step(self, actions):
        """Execute path planning step with RL learning"""
        rewards = {}
        dones = {}
        infos = {}
        
        # Execute actions for path planning
        new_positions = {}
        
        for agent_id, action in actions.items():
            if agent_id in self.agents and not self.agents[agent_id]['completed']:
                new_pos = self.execute_path_planning_action(agent_id, action)
                new_positions[agent_id] = new_pos
            else:
                new_positions[agent_id] = self.agents[agent_id]['current_location']
        
        # Check for collisions in path planning
        collision_agents = self.detect_collisions(new_positions)
        
        # Update positions and calculate path planning rewards
        for agent_id in self.agents.keys():
            if agent_id in collision_agents:
                # Collision penalty for path planning
                reward = -15  # Stronger penalty for poor path planning
                self.agents[agent_id]['collisions'] += 1
            else:
                # Move to new position and update path planning metrics
                old_pos = self.agents[agent_id]['current_location']
                new_pos = new_positions[agent_id]
                
                if new_pos != old_pos:
                    self.agents[agent_id]['current_location'] = new_pos
                    self.agents[agent_id]['path_history'].append(new_pos)
                    self.agents[agent_id]['steps_taken'] += 1
                    
                    # Update path planning metrics
                    self.update_path_efficiency(agent_id)
                
                # Calculate path planning reward
                reward = self.calculate_path_planning_reward(agent_id, old_pos, new_pos)
            
            rewards[agent_id] = reward
            
            # Check if agent reached target (path planning success)
            if self.agents[agent_id]['current_location'] == self.agents[agent_id]['target_location']:
                self.agents[agent_id]['completed'] = True
                # Bonus reward for successful path completion
                rewards[agent_id] += 50 + (20 * self.agents[agent_id]['path_efficiency'])
                dones[agent_id] = True
            else:
                dones[agent_id] = False
            
            # Enhanced info for path planning analysis
            infos[agent_id] = {
                'completed': self.agents[agent_id]['completed'],
                'collisions': self.agents[agent_id]['collisions'],
                'steps_taken': self.agents[agent_id]['steps_taken'],
                'path_efficiency': self.agents[agent_id]['path_efficiency'],
                'route_learning_progress': self.agents[agent_id]['route_learning_progress']
            }
        
        self.step_count += 1
        
        # Episode termination for path planning
        all_done = all(dones.values()) or self.step_count >= self.max_steps
        if all_done:
            for agent_id in dones.keys():
                dones[agent_id] = True
        
        observations = self.get_path_planning_observations()
        
        return observations, rewards, dones, infos
    
    def execute_path_planning_action(self, agent_id, action):
        """Execute action with path planning awareness"""
        current_pos = self.agents[agent_id]['current_location']
        
        action_map = ['XP', 'XN', 'YP', 'YN', 'ZP', 'ZN', 'STAY']
        
        if action == 6:  # STAY
            return current_pos
        
        move_direction = action_map[action]
        
        # Validate move for path planning
        if current_pos in self.movement_matrix:
            if self.movement_matrix[current_pos].get(move_direction, 0) == 1:
                new_pos = self.find_adjacent_location(current_pos, move_direction)
                if new_pos is not None and self.location_types.get(new_pos, 'PATH') != 'OBSTACLE':
                    return new_pos
        
        # Invalid move - stay in place (penalty in reward function)
        return current_pos
    
    def update_path_efficiency(self, agent_id):
        """Update path planning efficiency metrics"""
        agent = self.agents[agent_id]
        
        # Calculate actual path length so far
        path_length = 0
        for i in range(1, len(agent['path_history'])):
            path_length += self.euclidean_distance(
                agent['path_history'][i-1], 
                agent['path_history'][i]
            )
        
        agent['actual_path_length'] = path_length
        
        # Calculate path efficiency (optimal/actual)
        if path_length > 0:
            agent['path_efficiency'] = min(1.0, agent['optimal_path_length'] / path_length)
        else:
            agent['path_efficiency'] = 1.0
        
        # Calculate route learning progress
        distance_to_target = self.euclidean_distance(
            agent['current_location'], 
            agent['target_location']
        )
        initial_distance = self.euclidean_distance(
            agent['start_location'], 
            agent['target_location']
        )
        
        if initial_distance > 0:
            agent['route_learning_progress'] = max(0, 
                (initial_distance - distance_to_target) / initial_distance
            )
    
    def calculate_path_planning_reward(self, agent_id, old_pos, new_pos):
        """Calculate reward specifically designed for RL path planning"""
        agent = self.agents[agent_id]
        
        # Base step penalty (encourages efficiency)
        reward = -2
        
        # Distance-based reward for path planning
        old_distance = self.euclidean_distance(old_pos, agent['target_location'])
        new_distance = self.euclidean_distance(new_pos, agent['target_location'])
        
        distance_improvement = old_distance - new_distance
        
        if distance_improvement > 0:
            # Reward for getting closer (path planning progress)
            reward += 10 * distance_improvement
        elif distance_improvement < 0:
            # Penalty for moving away (poor path planning)
            reward += 5 * distance_improvement  # Negative value
        
        # Path efficiency reward
        efficiency_bonus = 5 * agent['path_efficiency']
        reward += efficiency_bonus
        
        # Penalty for staying in place without reason
        if old_pos == new_pos and new_pos != agent['target_location']:
            reward -= 8
        
        # Bonus for exploring new areas (encourages path discovery)
        if new_pos not in agent['path_history']:
            reward += 3
        
        # Penalty for revisiting locations (discourages loops)
        elif agent['path_history'].count(new_pos) > 2:
            reward -= 5
        
        # Critical node bonus (encourages learning key waypoints)
        if new_pos in self.critical_nodes:
            reward += 2
        
        return reward
    
    def euclidean_distance(self, pos1, pos2):
        """Calculate Euclidean distance for path planning"""
        if pos1 not in self.locations:
            print(f"⚠️ Position {pos1} not found in locations")
            return float('inf')
        if pos2 not in self.locations:
            print(f"⚠️ Position {pos2} not found in locations")
            return float('inf')
        
        loc1 = self.locations[pos1]
        loc2 = self.locations[pos2]
        
        # Add type checking
        if not isinstance(loc1, dict) or not isinstance(loc2, dict):
            print(f"⚠️ Location data issue: loc1={loc1}, loc2={loc2}")
            return float('inf')
        
        try:
            return np.sqrt(
                (loc1['x'] - loc2['x'])**2 + 
                (loc1['y'] - loc2['y'])**2 + 
                (loc1['z'] - loc2['z'])**2
            )
        except KeyError as e:
            print(f"⚠️ Key error in distance calculation: {e}")
            print(f"   loc1 keys: {list(loc1.keys()) if isinstance(loc1, dict) else 'not dict'}")
            print(f"   loc2 keys: {list(loc2.keys()) if isinstance(loc2, dict) else 'not dict'}")
            return float('inf')
    
    def detect_collisions(self, new_positions):
        """Detect collisions in path planning context"""
        position_counts = {}
        collision_agents = []
        
        for agent_id, pos in new_positions.items():
            if pos in position_counts:
                position_counts[pos].append(agent_id)
            else:
                position_counts[pos] = [agent_id]
        
        for pos, agents in position_counts.items():
            if len(agents) > 1:
                collision_agents.extend(agents)
        
        return collision_agents
    
    def get_path_planning_statistics(self):
        """Get comprehensive path planning performance statistics"""
        completed_agents = sum(1 for agent in self.agents.values() if agent['completed'])
        total_collisions = sum(agent['collisions'] for agent in self.agents.values())
        avg_steps = np.mean([agent['steps_taken'] for agent in self.agents.values()])
        avg_efficiency = np.mean([agent['path_efficiency'] for agent in self.agents.values()])
        avg_progress = np.mean([agent['route_learning_progress'] for agent in self.agents.values()])
        
        return {
            'completion_rate': completed_agents / self.num_agents,
            'total_collisions': total_collisions,
            'average_steps': avg_steps,
            'average_path_efficiency': avg_efficiency,
            'average_route_learning_progress': avg_progress,
            'completed_agents': completed_agents,
            'total_agents': self.num_agents
        }

print("🛣️ RL Path Planning Environment class defined!")