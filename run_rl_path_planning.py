# 🚀 Main Execution Script for RL Path Planning System
"""
Main script to run the RL-based Path Planning System with Deadlock Awareness

This script demonstrates how to use the modular RL path planning system.
"""

# Import the RL Path Planning System
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src1 import (
    execute_complete_rl_path_planning_pipeline,
    load_location_data,
    RLPathPlanningEnvironment,
    RLPathPlanningAgent,
    rl_path_planning_training_system,
    plot_rl_path_planning_results
)

def main():
    """
    Main execution function for RL Path Planning System
    """
    
    print("🎯 RL PATH PLANNING SYSTEM - MAIN EXECUTION")
    print("="*60)
    print("🛣️ Advanced RL-Based Path Planning with Deadlock Awareness")
    print("🔄 Transfer Learning & Multi-Agent Coordination")
    print("="*60)
    
    # Option 1: Execute complete pipeline (recommended)
    print("\n🚀 Option 1: Execute Complete Pipeline")
    print("This will run the full system including:")
    print("  • Database connection & data loading")
    print("  • Environment & agent initialization")
    print("  • RL training with transfer learning")
    print("  • Performance visualization")
    print("  • Policy testing & evaluation")
    
    choice = input("\n🤔 Execute complete pipeline? (y/n): ").lower()
    
    if choice == 'y':
        try:
            print("\n🌟 Executing complete RL Path Planning pipeline...")
            results = execute_complete_rl_path_planning_pipeline()
            
            print("\n✅ Pipeline execution completed successfully!")
            print("📊 Results summary:")
            print(f"   🏆 Trained {len(results['agents'])} agents")
            print(f"   📈 Environment with {len(results['environment'].locations)} locations")
            print(f"   💾 All policies saved for transfer learning")
            
            return results
            
        except Exception as e:
            print(f"\n❌ Error during execution: {e}")
            print("🔧 Please check your environment setup and try again.")
            return None
    
    else:
        print("\n🛠️ Manual Setup Mode")
        print("You can use individual components:")
        
        # Demonstrate manual usage
        try:
            print("\n1. 📊 Loading data...")
            location_df = load_location_data()
            
            print("2. 🏗️ Creating environment...")
            env = RLPathPlanningEnvironment(location_df, num_agents=2, max_steps=1000)
            
            print("3. 🤖 Creating agents...")
            agents = {
                'agent_1': RLPathPlanningAgent('agent_1', learning_rate=0.1),
                'agent_2': RLPathPlanningAgent('agent_2', learning_rate=0.1)
            }
            
            print("4. 🎯 Running training...")
            training_metrics, final_stats = rl_path_planning_training_system(
                location_df, agents, num_episodes=500, save_interval=100
            )
            
            print("5. 📊 Generating visualizations...")
            plot_rl_path_planning_results(training_metrics, final_stats)
            
            print("\n✅ Manual setup completed successfully!")
            
            return {
                'environment': env,
                'agents': agents,
                'training_metrics': training_metrics,
                'final_stats': final_stats
            }
            
        except Exception as e:
            print(f"\n❌ Error in manual setup: {e}")
            return None

def run_custom_training():
    """
    Run custom training with user-specified parameters
    """
    print("\n🎛️ CUSTOM TRAINING CONFIGURATION")
    print("="*50)
    
    try:
        # Get user parameters
        num_agents = int(input("🤖 Number of agents (1-5): ") or 3)
        num_episodes = int(input("📊 Number of training episodes (100-5000): ") or 1000)
        learning_rate = float(input("🧠 Learning rate (0.01-0.5): ") or 0.1)
        use_transfer = input("🔄 Use transfer learning? (y/n): ").lower() == 'y'
        
        print(f"\n🚀 Starting custom training...")
        print(f"   🤖 Agents: {num_agents}")
        print(f"   📊 Episodes: {num_episodes}")
        print(f"   🧠 Learning Rate: {learning_rate}")
        print(f"   🔄 Transfer Learning: {use_transfer}")
        
        # Load data and create environment
        location_df = load_location_data()
        env = RLPathPlanningEnvironment(location_df, num_agents=num_agents)
        
        # Create agents
        agents = {}
        for i in range(num_agents):
            agent_id = f'custom_agent_{i+1}'
            agents[agent_id] = RLPathPlanningAgent(agent_id, learning_rate=learning_rate)
        
        # Run training
        training_metrics, final_stats = rl_path_planning_training_system(
            location_df=location_df,
            agents=agents,
            num_episodes=num_episodes,
            use_transfer_learning=use_transfer
        )
        
        # Show results
        plot_rl_path_planning_results(training_metrics, final_stats)
        
        print("\n✅ Custom training completed!")
        return training_metrics, final_stats
        
    except Exception as e:
        print(f"\n❌ Error in custom training: {e}")
        return None, None

if __name__ == "__main__":
    print("🎯 RL Path Planning System - Modular Implementation")
    print("Choose execution mode:")
    print("1. 🚀 Complete Pipeline (Recommended)")
    print("2. 🛠️ Manual Setup")
    print("3. 🎛️ Custom Training")
    
    mode = input("\nEnter choice (1/2/3): ").strip()
    
    if mode == "1":
        results = main()
    elif mode == "2":
        results = main()  # Manual mode is handled in main()
    elif mode == "3":
        results = run_custom_training()
    else:
        print("🚀 Running default complete pipeline...")
        results = execute_complete_rl_path_planning_pipeline()
    
    print("\n🎉 Thank you for using RL Path Planning System!")