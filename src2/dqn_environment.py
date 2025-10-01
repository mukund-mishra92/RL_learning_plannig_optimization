import numpy as np
import pandas as pd
from database_connector import create_mock_location_data, load_location_data

class DQNPathPlanningEnvironment:
    def __init__(self, state_size=12, action_size=7, location_df=None, num_agents=3, max_steps=200):
        self.state_size = state_size
        self.action_size = action_size
        self.location_df = location_df if location_df is not None else create_mock_location_data(100)
        self.num_locations = len(self.location_df)
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.agent_states = [None] * num_agents
        self.agent_targets = [None] * num_agents
        self.steps = 0
        self.deadlock_memory = []
        self.reset()

    def reset(self):
        self.steps = 0
        self.agent_states = []
        self.agent_targets = []
        for _ in range(self.num_agents):
            start_idx = np.random.randint(0, self.num_locations)
            target_idx = np.random.randint(0, self.num_locations)
            while target_idx == start_idx:
                target_idx = np.random.randint(0, self.num_locations)
            start = self.location_df.iloc[start_idx][['x','y','z']].values
            target = self.location_df.iloc[target_idx][['x','y','z']].values
            self.agent_states.append(np.concatenate([start, target, np.zeros(self.state_size-6)]))
            self.agent_targets.append(target)
        return self.agent_states.copy()

    def step(self, actions):
        rewards = []
        next_states = []
        done_flags = []
        deadlock_flags = []
        for i in range(self.num_agents):
            state = self.agent_states[i].copy()
            # Simulate movement: action 0-5 moves in x/y/z directions, 6 = stay
            if actions[i] == 0: state[0] += 1  # +x
            elif actions[i] == 1: state[0] -= 1  # -x
            elif actions[i] == 2: state[1] += 1  # +y
            elif actions[i] == 3: state[1] -= 1  # -y
            elif actions[i] == 4: state[2] += 1  # +z
            elif actions[i] == 5: state[2] -= 1  # -z
            # else: stay
            # Add small random noise
            state[:3] += np.random.uniform(-0.2, 0.2, 3)
            # Reward: negative distance to target, bonus for reaching
            dist = np.linalg.norm(state[:3] - self.agent_targets[i])
            reward = -dist
            done = dist < 1.0 or self.steps >= self.max_steps
            # Deadlock detection: if two agents are very close and moving towards each other
            deadlock = False
            for j in range(self.num_agents):
                if i != j:
                    other_state = self.agent_states[j][:3]
                    other_target = self.agent_targets[j]
                    other_dist = np.linalg.norm(state[:3] - other_state)
                    if other_dist < 2.0:
                        # Check if moving towards each other
                        dir_to_target = self.agent_targets[i] - state[:3]
                        other_dir = other_target - other_state
                        if np.dot(dir_to_target, other_dir) < 0:
                            deadlock = True
                            self.deadlock_memory.append((i, j, self.steps))
            deadlock_flags.append(deadlock)
            rewards.append(reward - (10 if deadlock else 0))
            next_states.append(state)
            done_flags.append(done)
            self.agent_states[i] = state
        self.steps += 1
        return next_states, rewards, done_flags, {'deadlocks': deadlock_flags}
