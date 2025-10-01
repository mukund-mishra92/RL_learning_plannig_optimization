# 📊 RL Path Planning Visualization and Analytics
"""
Comprehensive visualization for RL path planning performance and learning progress
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime

def plot_rl_path_planning_results(training_metrics, final_stats, save_plots=True):
    """
    Comprehensive visualization for RL path planning performance and learning progress
    """
    
    # Set up the plotting style
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 16))
    
    # Color scheme for agents
    agent_colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#7209B7']
    agent_ids = list(training_metrics['episode_rewards'].keys())
    colors = {agent_id: agent_colors[i % len(agent_colors)] for i, agent_id in enumerate(agent_ids)}
    
    # 1. Episode Rewards - Path Planning Learning Progress
    ax1 = plt.subplot(3, 3, 1)
    for agent_id in agent_ids:
        rewards = training_metrics['episode_rewards'][agent_id]
        if rewards:
            # Smooth the rewards for better visualization
            smoothed_rewards = pd.Series(rewards).rolling(window=50, min_periods=1).mean()
            ax1.plot(smoothed_rewards, label=f'{agent_id}', color=colors[agent_id], linewidth=2)
    
    ax1.set_title('🏆 Path Planning Learning Progress\n(Episode Rewards)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Path Efficiency Progress
    ax2 = plt.subplot(3, 3, 2)
    for agent_id in agent_ids:
        efficiencies = training_metrics['path_efficiencies'][agent_id]
        if efficiencies:
            smoothed_eff = pd.Series(efficiencies).rolling(window=50, min_periods=1).mean()
            ax2.plot(smoothed_eff, label=f'{agent_id}', color=colors[agent_id], linewidth=2)
    
    ax2.set_title('🛣️ Path Planning Efficiency\n(Route Optimization)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Path Efficiency')
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. Route Completion Rates
    ax3 = plt.subplot(3, 3, 3)
    for agent_id in agent_ids:
        completions = training_metrics['route_completion_rates'][agent_id]
        if completions:
            # Calculate rolling success rate
            rolling_completion = pd.Series(completions).rolling(window=100, min_periods=1).mean()
            ax3.plot(rolling_completion, label=f'{agent_id}', color=colors[agent_id], linewidth=2)
    
    ax3.set_title('🎯 Route Completion Success\n(Learning Effectiveness)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Completion Rate')
    ax3.set_ylim(0, 1)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. Deadlock Detection and Avoidance
    ax4 = plt.subplot(3, 3, 4)
    for agent_id in agent_ids:
        deadlocks = training_metrics['deadlock_incidents'][agent_id]
        if deadlocks:
            # Cumulative deadlock avoidance learning
            cumulative_deadlocks = pd.Series(deadlocks).cumsum()
            ax4.plot(cumulative_deadlocks, label=f'{agent_id}', color=colors[agent_id], linewidth=2)
    
    ax4.set_title('⚠️ Deadlock Detection Learning\n(Conflict Avoidance)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Cumulative Deadlock Detections')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # 5. Collision Reduction Over Time
    ax5 = plt.subplot(3, 3, 5)
    for agent_id in agent_ids:
        collisions = training_metrics['collision_counts'][agent_id]
        if collisions:
            # Rolling average of collisions (should decrease with learning)
            rolling_collisions = pd.Series(collisions).rolling(window=100, min_periods=1).mean()
            ax5.plot(rolling_collisions, label=f'{agent_id}', color=colors[agent_id], linewidth=2)
    
    ax5.set_title('🚫 Collision Avoidance Learning\n(Safety Improvement)', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Episode')
    ax5.set_ylabel('Average Collisions per Episode')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    
    # 6. Exploration vs Exploitation Balance
    ax6 = plt.subplot(3, 3, 6)
    for agent_id in agent_ids:
        exploration_rates = training_metrics['exploration_rates'][agent_id]
        if exploration_rates:
            ax6.plot(exploration_rates, label=f'{agent_id}', color=colors[agent_id], linewidth=2)
    
    ax6.set_title('🔍 Exploration vs Exploitation\n(Learning Strategy)', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Episode')
    ax6.set_ylabel('Exploration Rate')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)
    
    # 7. Final Performance Comparison
    ax7 = plt.subplot(3, 3, 7)
    metrics = ['completion_rate', 'path_efficiency', 'average_reward']
    x_pos = np.arange(len(agent_ids))
    
    # Normalize metrics for comparison
    for i, metric in enumerate(metrics):
        values = [final_stats[agent_id][metric] for agent_id in agent_ids]
        if metric == 'average_reward':
            # Normalize rewards to 0-1 scale for comparison
            min_val, max_val = min(values), max(values)
            if max_val != min_val:
                values = [(v - min_val) / (max_val - min_val) for v in values]
            else:
                values = [0.5] * len(values)
        
        ax7.bar(x_pos + i*0.25, values, 0.25, 
               label=metric.replace('_', ' ').title(), alpha=0.8)
    
    ax7.set_title('📈 Final Performance Comparison\n(Multi-Metric Analysis)', fontsize=14, fontweight='bold')
    ax7.set_xlabel('Agents')
    ax7.set_ylabel('Normalized Performance')
    ax7.set_xticks(x_pos + 0.25)
    ax7.set_xticklabels(agent_ids, rotation=45)
    ax7.legend(fontsize=10)
    ax7.grid(True, alpha=0.3)
    
    # 8. Learning Knowledge Acquisition
    ax8 = plt.subplot(3, 3, 8)
    knowledge_metrics = ['q_table_size', 'learned_patterns', 'deadlock_knowledge']
    x_pos = np.arange(len(agent_ids))
    
    for i, metric in enumerate(knowledge_metrics):
        values = [final_stats[agent_id][metric] for agent_id in agent_ids]
        ax8.bar(x_pos + i*0.25, values, 0.25, 
               label=metric.replace('_', ' ').title(), alpha=0.8)
    
    ax8.set_title('🧠 Knowledge Acquisition\n(Learning Depth)', fontsize=14, fontweight='bold')
    ax8.set_xlabel('Agents')
    ax8.set_ylabel('Knowledge Units')
    ax8.set_xticks(x_pos + 0.25)
    ax8.set_xticklabels(agent_ids, rotation=45)
    ax8.legend(fontsize=10)
    ax8.grid(True, alpha=0.3)
    
    # 9. Route Optimization Progress
    ax9 = plt.subplot(3, 3, 9)
    for agent_id in agent_ids:
        optimization_scores = training_metrics['route_optimization_scores'][agent_id]
        if optimization_scores:
            smoothed_scores = pd.Series(optimization_scores).rolling(window=50, min_periods=1).mean()
            ax9.plot(smoothed_scores, label=f'{agent_id}', color=colors[agent_id], linewidth=2)
    
    ax9.set_title('🎯 Route Optimization Mastery\n(Path Planning Excellence)', fontsize=14, fontweight='bold')
    ax9.set_xlabel('Episode')
    ax9.set_ylabel('Optimization Score')
    ax9.legend(fontsize=10)
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout(pad=3.0)
    
    if save_plots:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"rl_path_planning_analysis_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 Path planning analysis saved as: {filename}")
    
    plt.show()
    
    # Print detailed performance summary
    print("\n" + "="*80)
    print("🎯 FINAL RL PATH PLANNING PERFORMANCE SUMMARY")
    print("="*80)
    
    for agent_id in agent_ids:
        stats = final_stats[agent_id]
        print(f"\n🤖 Agent: {agent_id}")
        print(f"   🏆 Route Completion Rate: {stats['completion_rate']:.2%}")
        print(f"   🛣️ Path Efficiency: {stats['path_efficiency']:.3f}")
        print(f"   💯 Average Reward: {stats['average_reward']:.2f}")
        print(f"   🧠 States Learned: {stats['q_table_size']:,}")
        print(f"   ✅ Successful Patterns: {stats['learned_patterns']:,}")
        print(f"   ⚠️ Deadlock Knowledge: {stats['deadlock_knowledge']:,}")
        
        # Calculate learning efficiency
        if stats['q_table_size'] > 0:
            learning_efficiency = stats['learned_patterns'] / stats['q_table_size']
            print(f"   📈 Learning Efficiency: {learning_efficiency:.3f}")
    
    # Overall system performance
    avg_completion = np.mean([stats['completion_rate'] for stats in final_stats.values()])
    avg_efficiency = np.mean([stats['path_efficiency'] for stats in final_stats.values()])
    total_knowledge = sum([stats['q_table_size'] for stats in final_stats.values()])
    
    print(f"\n🌟 SYSTEM PERFORMANCE:")
    print(f"   🎯 Average Completion Rate: {avg_completion:.2%}")
    print(f"   🛣️ Average Path Efficiency: {avg_efficiency:.3f}")
    print(f"   🧠 Total Knowledge Base: {total_knowledge:,} states")
    print("="*80)

print("📊 RL Path Planning Visualization System defined!")