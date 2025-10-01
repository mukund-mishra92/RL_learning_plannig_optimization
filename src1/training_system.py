# 🎯 Enhanced RL Path Planning Training System with Deadlock Awareness
"""
Comprehensive RL training system focused on path planning with deadlock prevention
"""

import numpy as np
import os
from .path_planning_environment import RLPathPlanningEnvironment

def rl_path_planning_training_system(location_df, agents, num_episodes=2000, 
                                   save_interval=500, use_transfer_learning=True, 
                                   policy_directory="./path_planning_policies/"):
    """
    Comprehensive RL training system focused on path planning with deadlock prevention
    
    This system trains agents to learn optimal routes while avoiding conflicts
    """
    
    # Create enhanced path planning environment
    env = RLPathPlanningEnvironment(location_df, num_agents=len(agents), max_steps=2000)
    
    # Enhanced training metrics for path planning
    training_metrics = {
        'episode_rewards': {agent_id: [] for agent_id in agents.keys()},
        'path_efficiencies': {agent_id: [] for agent_id in agents.keys()},
        'route_completion_rates': {agent_id: [] for agent_id in agents.keys()},
        'deadlock_incidents': {agent_id: [] for agent_id in agents.keys()},
        'path_discovery_progress': {agent_id: [] for agent_id in agents.keys()},
        'collision_counts': {agent_id: [] for agent_id in agents.keys()},
        'exploration_rates': {agent_id: [] for agent_id in agents.keys()},
        'transfer_learning_effectiveness': {agent_id: [] for agent_id in agents.keys()},
        'route_optimization_scores': {agent_id: [] for agent_id in agents.keys()}
    }
    
    # Load existing policies for transfer learning
    os.makedirs(policy_directory, exist_ok=True)
    
    if use_transfer_learning:
        print("🔄 Initializing Transfer Learning for Path Planning...")
        for agent_id, agent in agents.items():
            policy_file = os.path.join(policy_directory, f"{agent_id}_path_planning_policy.pkl")
            if os.path.exists(policy_file):
                success = agent.load_path_planning_policy(policy_file, transfer_learning=True)
                if success:
                    agent.fine_tune_mode(True, episodes=min(500, num_episodes//4))
                    print(f"   ✅ {agent_id}: Transfer learning activated")
                else:
                    print(f"   ⚠️ {agent_id}: Failed to load policy, starting fresh")
            else:
                print(f"   🆕 {agent_id}: No existing policy found, starting fresh")
    
    print(f"\n🚀 Starting RL Path Planning Training...")
    print(f"   📊 Episodes: {num_episodes}")
    print(f"   🤖 Agents: {len(agents)}")
    print(f"   🛣️ Environment: {len(env.locations)} locations")
    print(f"   💾 Save interval: {save_interval} episodes")
    
    best_performance = {agent_id: float('-inf') for agent_id in agents.keys()}
    
    for episode in range(1, num_episodes + 1):
        # Reset environment for new path planning episode
        observations = env.reset()
        episode_rewards = {agent_id: 0 for agent_id in agents.keys()}
        episode_steps = 0
        done = {agent_id: False for agent_id in agents.keys()}
        
        # Episode-specific metrics
        path_efficiencies = {agent_id: [] for agent_id in agents.keys()}
        deadlock_detections = {agent_id: 0 for agent_id in agents.keys()}
        
        while not all(done.values()) and episode_steps < env.max_steps:
            actions = {}
            
            # Get actions from all agents for path planning
            for agent_id, agent in agents.items():
                if not done.get(agent_id, False):
                    obs = observations[agent_id]
                    # Normal path planning action selection
                    action = agent.choose_action(obs)
                    actions[agent_id] = action
                else:
                    actions[agent_id] = 6  # STAY action for completed agents
            
            # Execute step in path planning environment
            next_observations, rewards, done, infos = env.step(actions)
            
            # Update agents with path planning learning
            for agent_id, agent in agents.items():
                if agent_id in rewards:
                    reward = rewards[agent_id]
                    episode_rewards[agent_id] += reward
                    
                    # RL update for path planning
                    agent.update_q_value(
                        observations[agent_id], 
                        actions[agent_id], 
                        reward, 
                        next_observations[agent_id], 
                        done[agent_id]
                    )
                    
                    # Track path planning metrics
                    info = infos.get(agent_id, {})
                    if 'path_efficiency' in info:
                        path_efficiencies[agent_id].append(info['path_efficiency'])
            
            observations = next_observations
            episode_steps += 1
        
        # Update exploration rates for path planning learning
        for agent in agents.values():
            agent.update_exploration_rate()
        
        # Collect episode statistics for path planning analysis
        
        for agent_id, agent in agents.items():
            # Store training metrics
            training_metrics['episode_rewards'][agent_id].append(episode_rewards[agent_id])
            training_metrics['path_efficiencies'][agent_id].append(
                np.mean(path_efficiencies[agent_id]) if path_efficiencies[agent_id] else 0
            )
            training_metrics['route_completion_rates'][agent_id].append(
                1 if done.get(agent_id, False) else 0
            )
            training_metrics['deadlock_incidents'][agent_id].append(deadlock_detections[agent_id])
            training_metrics['collision_counts'][agent_id].append(
                infos.get(agent_id, {}).get('collisions', 0)
            )
            training_metrics['exploration_rates'][agent_id].append(agent.exploration_rate)
            
            # Calculate route optimization score based on recent performance
            recent_rewards = training_metrics['episode_rewards'][agent_id][-10:]
            optimization_score = np.mean(recent_rewards) if recent_rewards else 0
            training_metrics['route_optimization_scores'][agent_id].append(optimization_score)
            
            # Store in agent's memory for learning
            if not hasattr(agent, 'episode_rewards'):
                agent.episode_rewards = []
            if not hasattr(agent, 'path_efficiency_history'):
                agent.path_efficiency_history = []
                
            agent.episode_rewards.append(episode_rewards[agent_id])
            if path_efficiencies[agent_id]:
                agent.path_efficiency_history.append(np.mean(path_efficiencies[agent_id]))
        
        # Progress reporting for path planning
        if episode % 100 == 0:
            avg_reward = np.mean([episode_rewards[agent_id] for agent_id in agents.keys()])
            avg_completion = np.mean([done.get(agent_id, False) for agent_id in agents.keys()])
            avg_efficiency = np.mean([np.mean(path_efficiencies[agent_id]) if path_efficiencies[agent_id] else 0 
                                    for agent_id in agents.keys()])
            
            print(f"Episode {episode:4d} | "
                  f"Avg Reward: {avg_reward:6.1f} | "
                  f"Completion: {avg_completion:.2%} | "
                  f"Path Efficiency: {avg_efficiency:.3f}")
            
            # Report transfer learning effectiveness
            if use_transfer_learning and episode <= num_episodes // 2:
                transfer_effectiveness = np.mean([len(agent.q_table) for agent in agents.values()])
                print(f"         Transfer Learning Knowledge: {transfer_effectiveness:.0f} states")
        
        # Save policies periodically for path planning
        if episode % save_interval == 0 or episode == num_episodes:
            print(f"\n💾 Saving path planning policies at episode {episode}...")
            
            for agent_id, agent in agents.items():
                policy_file = os.path.join(policy_directory, f"{agent_id}_path_planning_policy.pkl")
                agent.save_path_planning_policy(policy_file)
                
                # Track best performance for each agent
                current_performance = np.mean(training_metrics['episode_rewards'][agent_id][-100:])
                if current_performance > best_performance[agent_id]:
                    best_performance[agent_id] = current_performance
                    best_policy_file = os.path.join(policy_directory, f"{agent_id}_best_path_planning_policy.pkl")
                    agent.save_path_planning_policy(best_policy_file)
                    print(f"   🏆 New best policy saved for {agent_id}: {current_performance:.1f}")
        
        # Adaptive learning rate adjustment for path planning
        if episode % 1000 == 0 and episode < num_episodes:
            for agent in agents.values():
                if agent.learning_rate > 0.01:
                    agent.learning_rate *= 0.9  # Gradual learning rate decay
            print(f"   📉 Learning rates adjusted for fine-tuning")
    
    print(f"\n✅ RL Path Planning Training Complete!")
    print(f"   📊 Total episodes: {num_episodes}")
    print(f"   🎯 Final completion rates:")
    
    # Final performance summary
    final_stats = {}
    for agent_id, agent in agents.items():
        recent_completion = np.mean(training_metrics['route_completion_rates'][agent_id][-100:])
        recent_efficiency = np.mean(training_metrics['path_efficiencies'][agent_id][-100:])
        recent_rewards = np.mean(training_metrics['episode_rewards'][agent_id][-100:])
        
        final_stats[agent_id] = {
            'completion_rate': recent_completion,
            'path_efficiency': recent_efficiency,
            'average_reward': recent_rewards,
            'q_table_size': len(agent.q_table)
        }
        
        print(f"      {agent_id}: {recent_completion:.2%} completion, "
              f"{recent_efficiency:.3f} efficiency, "
              f"{len(agent.q_table)} states learned")
    
    return training_metrics, final_stats

print("🎯 Enhanced RL Path Planning Training System defined!")