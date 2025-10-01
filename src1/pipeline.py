# 🚀 Complete RL Path Planning Execution Pipeline
"""
Execute the complete RL-based path planning system with deadlock awareness and transfer learning
"""

import numpy as np
import os
from .database_connector import load_location_data, create_mock_location_data, connect_to_warehouse_database
from .path_planning_environment import RLPathPlanningEnvironment
from .rl_path_planning_agent import RLPathPlanningAgent
from .training_system import rl_path_planning_training_system
from .visualization import plot_rl_path_planning_results
from .config import get_config

def execute_complete_rl_path_planning_pipeline():
    """
    Execute the complete RL-based path planning system with deadlock awareness and transfer learning
    """
    
    # Load configuration
    from .config import TRAINING_CONFIG, AGENT_CONFIG
    agent_config = AGENT_CONFIG
    training_config = TRAINING_CONFIG
    
    print("🌟 INITIALIZING COMPLETE RL PATH PLANNING SYSTEM")
    print("="*70)
    
    # Step 1: Setup and Data Loading
    print("📊 Step 1: Loading warehouse data and setup...")
    
    # Force use of actual database data
    print("🔑 Connecting directly to database...")
    try:
        location_df = connect_to_warehouse_database()
        print(f"   ✅ Location data loaded: {len(location_df)} locations")
        
        # Process the data to ensure compatibility
        # Rename columns to match expected format
        column_mapping = {
            'LOCATION_ID': 'location_id',
            'X': 'x', 'Y': 'y', 'Z': 'z',
            'TYPE': 'type',
            'XP': 'XP', 'XN': 'XN',
            'YP': 'YP', 'YN': 'YN',
            'ZP': 'ZP', 'ZN': 'ZN'
        }
        
        # Apply column mapping if needed
        for old_col, new_col in column_mapping.items():
            if old_col in location_df.columns and new_col not in location_df.columns:
                location_df = location_df.rename(columns={old_col: new_col})
        
        print(f"   🗺️ Coordinate range: X({location_df['x'].min()}-{location_df['x'].max()}), "
              f"Y({location_df['y'].min()}-{location_df['y'].max()})")
        print(f"   📋 Data processing complete")
        
    except Exception as e:
        print(f"   ⚠️ Database connection failed: {e}")
        print("   🔄 Using fallback mock data...")
        location_df = create_mock_location_data()
    
    # Step 2: Environment Creation
    print("\n🏗️ Step 2: Creating RL Path Planning Environment...")
    
    # Initialize the path planning environment with config values
    env = RLPathPlanningEnvironment(location_df, num_agents=agent_config['num_agents'], max_steps=1500)
    
    print(f"   🛣️ Path planning environment created")
    print(f"   📍 Total navigable locations: {len(env.locations)}")
    print(f"   🔗 Path connections: {len(env.path_graph.edges)}")
    print(f"   🎯 Critical navigation nodes: {len(env.critical_nodes)}")
    
    # Step 3: Agent Initialization
    print("\n🤖 Step 3: Initializing RL Path Planning Agents...")
    
    # Create enhanced RL agents for path planning using config
    agents = {}
    for i in range(agent_config['num_agents']):
        agent_id = f'agent_{i}'  # Match environment agent ID format
        agent = RLPathPlanningAgent(
            agent_id=agent_id,
            state_size=agent_config['state_size'],
            action_size=agent_config['action_size'],
            learning_rate=agent_config['learning_rate'],
            discount_factor=agent_config['discount_factor'],
            exploration_rate=0.9
        )
        agents[agent_id] = agent
    
    print(f"   ✅ {len(agents)} RL path planning agents initialized")
    
    # Step 4: Training Execution
    print("\n🎯 Step 4: Executing RL Path Planning Training...")
    
    # Execute comprehensive training
    training_metrics, final_stats = rl_path_planning_training_system(
        location_df=location_df,
        agents=agents,
        num_episodes=training_config['num_episodes'],
        save_interval=training_config['save_interval'],
        use_transfer_learning=training_config['use_transfer_learning'],
        policy_directory=training_config['policy_directory']
    )
    
    print("   ✅ Training completed successfully!")
    
    # Step 5: Performance Analysis
    print("\n📈 Step 5: Analyzing Path Planning Performance...")
    
    # Generate comprehensive visualizations
    plot_rl_path_planning_results(training_metrics, final_stats, save_plots=True)
    
    # Step 6: Policy Testing and Validation
    print("\n🧪 Step 6: Testing Learned Path Planning Policies...")
    
    # Test agents in evaluation mode
    test_results = test_rl_path_planning_policies(env, agents, num_test_episodes=50)
    
    print("   📊 Policy testing completed")
    
    # Step 7: Transfer Learning Demonstration
    print("\n🔄 Step 7: Demonstrating Transfer Learning Capabilities...")
    
    # Create new agent and test transfer learning
    new_agent = RLPathPlanningAgent('transfer_test_agent', learning_rate=0.15)
    
    # Load existing policy for transfer learning
    policy_file = "./rl_path_planning_policies/agent_0_best_path_planning_policy.json"
    if os.path.exists(policy_file):
        success = new_agent.load_path_planning_policy(policy_file, transfer_learning=True)
        if success:
            print("   ✅ Transfer learning successfully demonstrated")
            # Quick training with transferred knowledge
            new_agent.fine_tune_mode(True, episodes=100)
            print("   🎯 Fine-tuning mode activated for transfer learning")
        else:
            print("   ⚠️ Transfer learning failed")
    else:
        print("   ⚠️ No existing policy found for transfer learning demo")
    
    # Final Summary
    print(f"\n🌟 RL PATH PLANNING SYSTEM EXECUTION COMPLETE!")
    print("="*70)
    print("📊 SYSTEM ACHIEVEMENTS:")
    
    # Calculate overall system performance
    total_episodes = len(training_metrics['episode_rewards']['agent_0'])
    avg_completion = np.mean([stats['completion_rate'] for stats in final_stats.values()])
    avg_efficiency = np.mean([stats['path_efficiency'] for stats in final_stats.values()])
    total_knowledge = sum([stats['q_table_size'] for stats in final_stats.values()])
    total_patterns = sum([stats['learned_patterns'] for stats in final_stats.values()])
    
    print(f"   🎯 Training Episodes: {total_episodes:,}")
    print(f"   🏆 Average Route Completion: {avg_completion:.2%}")
    print(f"   🛣️ Average Path Efficiency: {avg_efficiency:.3f}")
    print(f"   🧠 Total Knowledge Base: {total_knowledge:,} states")
    print(f"   ✅ Successful Patterns Learned: {total_patterns:,}")
    print(f"   🔄 Transfer Learning: {'✅ Active' if any(agent.transfer_learning_active for agent in agents.values()) else '❌ Not Active'}")
    print(f"   💾 Policies Saved: ✅ Available for future use")
    
    return {
        'environment': env,
        'agents': agents,
        'training_metrics': training_metrics,
        'final_stats': final_stats,
        'test_results': test_results
    }

def test_rl_path_planning_policies(env, agents, num_test_episodes=50):
    """
    Test learned RL path planning policies in evaluation mode
    """
    print(f"   🧪 Running {num_test_episodes} evaluation episodes...")
    
    test_results = {
        'completion_rates': [],
        'path_efficiencies': [],
        'average_rewards': [],
        'collision_counts': []
    }
    
    # Set agents to evaluation mode (no exploration)
    original_exploration_rates = {}
    for agent_id, agent in agents.items():
        original_exploration_rates[agent_id] = agent.exploration_rate
        agent.exploration_rate = 0.0  # Pure exploitation for testing
    
    for episode in range(num_test_episodes):
        observations = env.reset()
        episode_rewards = {agent_id: 0 for agent_id in agents.keys()}
        done = {agent_id: False for agent_id in agents.keys()}
        steps = 0
        
        while not all(done.values()) and steps < env.max_steps:
            actions = {}
            for agent_id, agent in agents.items():
                if not done.get(agent_id, False):
                    obs = observations[agent_id]
                    action = agent.choose_action(obs, training=False)
                    actions[agent_id] = action
                else:
                    actions[agent_id] = 6  # STAY
            
            observations, rewards, done, infos = env.step(actions)
            
            for agent_id in agents.keys():
                if agent_id in rewards:
                    episode_rewards[agent_id] += rewards[agent_id]
            
            steps += 1
        
        # Collect episode statistics
        stats = env.get_path_planning_statistics()
        test_results['completion_rates'].append(stats['completion_rate'])
        test_results['path_efficiencies'].append(stats['average_path_efficiency'])
        test_results['average_rewards'].append(np.mean(list(episode_rewards.values())))
        test_results['collision_counts'].append(stats['total_collisions'])
    
    # Restore original exploration rates
    for agent_id, agent in agents.items():
        agent.exploration_rate = original_exploration_rates[agent_id]
    
    # Print test results
    print(f"   📊 Test Results:")
    print(f"      🎯 Avg Completion Rate: {np.mean(test_results['completion_rates']):.2%}")
    print(f"      🛣️ Avg Path Efficiency: {np.mean(test_results['path_efficiencies']):.3f}")
    print(f"      🏆 Avg Reward: {np.mean(test_results['average_rewards']):.2f}")
    print(f"      🚫 Avg Collisions: {np.mean(test_results['collision_counts']):.1f}")
    
    return test_results

print("🚀 RL Path Planning Pipeline ready for execution!")